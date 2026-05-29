//! Module-level definition scan for TypeScript — the "find-*" step of `find-dup-defs`,
//! parallel + native, mirroring `py-canon::defs` over [`oxc_parser`].
//!
//! Walks each file's oxc AST **once** and lowers every definition straight to the engine's
//! [`Def`], with its canonical strings precomputed. Unlike `py-canon`, the canonical is computed
//! by re-parsing the def's source slice (TypeScript method slices and `.tsx`/JSX bodies parse
//! differently as the in-file node vs. a standalone slice, so a node-based canon would diverge);
//! for `fn_like` defs the cluster canonical is taken from the same single `analyze_functions`
//! call, so a function costs one re-parse rather than two.
//!
//! Surfaces these top-level kinds (the frontend lowers each to the engine's `Def`):
//!
//! * **`functions`** — `function foo()` / `async function`, plus module-level `const foo =
//!   (...) => {}` / `const foo = function(){}` (arrow / function-expression assigned to a `const`
//!   is the dominant "named function" form in idiomatic TS).
//! * **`methods`** — class methods (sync / async / generator), constructors, getters / setters.
//!   Qualified `ClassName.method`. Getter / setter get a role suffix (`.getter` / `.setter`) so
//!   they don't collide with each other in the name-gated pass.
//! * **`classes`** — `class Foo { ... }` (with or without `abstract`).
//! * **`type-aliases`** — `type X = ...`.
//! * **`interfaces`** — `interface X { ... }`. First-class kind so the directive layer can
//!   de-escalate them independently of `type-aliases`.
//! * **`constants`** — module-level `const NAME = ...` whose name is `UPPER_SNAKE_CASE` AND the
//!   initializer is not itself a function (those land in `functions` instead).
//!
//! `export` / `export default` are transparent wrappers: the inner declaration is what's
//! surfaced, with its own decorator-stripped text and span — keeping the canonical comparable
//! across `function foo()` and `export function foo()`.
#![allow(
    clippy::unnecessary_wraps, // type_alias_def / interface_def return Option<Def> for call-site symmetry with function_def / class_def (which DO conditionally return None); the symmetry pays for the lint.
    clippy::needless_raw_string_hashes, // test-fixture raw strings keep `r#"..."#` for visual consistency; some contain `"` literals and need the hashes anyway.
)]

use std::path::Path;
use std::sync::Arc;

use dup_defs_core::{Analysis, Def, KindSpec, LineMap};
use oxc_allocator::Allocator;
use oxc_ast::ast::{
    BindingPattern, Class, ClassElement, Declaration, Decorator, ExportDefaultDeclarationKind,
    Expression, FormalParameters, Function, FunctionBody, MethodDefinitionKind, PropertyKey,
    Statement, TSInterfaceDeclaration, TSTypeAliasDeclaration, VariableDeclaration,
    VariableDeclarationKind,
};
use oxc_parser::Parser;
use oxc_span::{GetSpan, SourceType};

use crate::canon::{analyze_functions, ast_canonical};
use crate::frontend::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};

#[inline]
fn u(x: u32) -> usize {
    x as usize
}

/// Non-blank line count of a def's source text. Same definition as `py-canon::count_loc` for
/// cross-language consistency.
fn count_loc(text: &str) -> usize {
    text.lines().filter(|l| !l.trim().is_empty()).count()
}

/// User-visible parameter count: every formal slot (`x`, `...rest`, defaults) counts once. The
/// `this: Foo` annotation in `function f(this: Foo, x: number)` is a type-only fake parameter
/// (analogous to Python's stripped `self`/`cls`) and does NOT count.
fn count_args(params: &FormalParameters<'_>) -> usize {
    params.items.len() + usize::from(params.rest.is_some())
}

/// True iff `name` is `UPPER_SNAKE_CASE`. Same rule as `py-canon` so `MAX_RETRIES` clusters
/// cross-language alongside Python `MAX_RETRIES`.
fn is_upper_snake(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// The byte offset just after the last decorator (skipping whitespace / single-line comments) —
/// the position of the `class` / `abstract` / `async` / `function` / `get` / `set` / method-name
/// keyword. Matches `py-canon::keyword_start` semantics.
fn keyword_start(source: &str, range_start: usize, last_decorator_end: Option<usize>) -> usize {
    let Some(mut i) = last_decorator_end else { return range_start };
    let bytes = source.as_bytes();
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        // `//` line comment between decorators and keyword — skip to EOL.
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        break;
    }
    i
}

fn last_decorator_end(decorators: &[Decorator<'_>]) -> Option<usize> {
    decorators.last().map(|d| u(d.span().end))
}

/// True when a function's body is effectively a no-op or single-token return — the canonical
/// dispatch-override / stub / placeholder shapes. Mirrors `py-canon::is_trivial_function_body`
/// for TS:
///
/// * `None` (declaration-only signature: overload / ambient) or empty `{}`.
/// * `return <Literal | this | Identifier | null | undefined>;` — trivial atom returns.
/// * `throw new <Identifier>(...)` for common "not implemented" / sentinel errors.
///
/// Falls through (still compared): `return this.x`, `return [this.x]`, `return foo()`.
fn is_trivial_function_body(body: Option<&FunctionBody<'_>>) -> bool {
    let Some(body) = body else { return true };
    if body.statements.is_empty() {
        return true;
    }
    body.statements.iter().all(|s| match s {
        Statement::ReturnStatement(r) => match &r.argument {
            None => true,
            Some(expr) => matches!(
                expr,
                Expression::NullLiteral(_)
                    | Expression::BooleanLiteral(_)
                    | Expression::NumericLiteral(_)
                    | Expression::StringLiteral(_)
                    | Expression::BigIntLiteral(_)
                    | Expression::RegExpLiteral(_)
                    | Expression::ThisExpression(_)
                    | Expression::Identifier(_)
            ),
        },
        Statement::ThrowStatement(t) => matches!(
            &t.argument,
            Expression::NewExpression(n) if matches!(
                &n.callee,
                Expression::Identifier(id) if matches!(
                    id.name.as_str(),
                    "Error" | "TypeError" | "RangeError" | "NotImplementedError"
                )
            )
        ),
        Statement::EmptyStatement(_) => true,
        _ => false,
    })
}

/// Precompute the canonical strings from a def's source slice. Non-body kinds (constants /
/// type-aliases) carry no canonical — the engine clusters them on `text_orig`. For `fn_like`
/// defs the cluster canonical is taken from the `analyze_functions` tuple (`.0`, identical to
/// `ast_canonical`), so a function re-parses once instead of twice; class methods (whose slice
/// doesn't re-parse as a standalone function) and non-callable body kinds fall back to
/// `ast_canonical`.
fn canon_for(kind: &KindSpec, text: &str) -> (Option<String>, Option<Analysis>) {
    if !kind.body {
        return (None, None);
    }
    if kind.fn_like {
        match analyze_functions(&[text.to_owned()]).into_iter().next().flatten() {
            Some((cc, xname, lines, size)) => {
                (Some(cc), Some(Analysis { xname_canonical: xname, type3_lines: lines, size }))
            }
            None => (Some(ast_canonical(text)), None),
        }
    } else {
        (Some(ast_canonical(text)), None)
    }
}

/// Build a [`Def`] for the definition spanning `source[start..end]` (TypeScript has no receiver
/// strip, so the canon input equals `text_orig`).
fn build_def(
    kind: &'static KindSpec,
    name: String,
    file: &Arc<str>,
    lines: &LineMap,
    source: &str,
    (start, end): (usize, usize),
    args: usize,
) -> Def {
    let text = source[start..end].to_owned();
    let loc = count_loc(&text);
    let (line, col) = lines.loc0(start);
    let (cluster_canonical, analysis) = canon_for(kind, &text);
    Def {
        lang: "ts",
        kind,
        name,
        file: Arc::clone(file),
        line,
        col,
        loc,
        args,
        text_orig: text,
        cluster_canonical,
        analysis,
    }
}

// ─────────────────────────── per-kind extractors ───────────────────────────

/// `function foo(...) { ... }` — top-level. Returns `None` for trivial-body / anonymous fns.
fn function_def(f: &Function<'_>, source: &str, lines: &LineMap, file: &Arc<str>) -> Option<Def> {
    let id = f.id.as_ref()?;
    if is_trivial_function_body(f.body.as_deref()) {
        return None;
    }
    let (start, end) = (u(f.span.start), u(f.span.end));
    Some(build_def(&FUNCTIONS, id.name.to_string(), file, lines, source, (start, end), count_args(&f.params)))
}

/// A class declaration as a whole — `class Foo { ... }`, decorators excluded.
fn class_def(c: &Class<'_>, source: &str, lines: &LineMap, file: &Arc<str>) -> Option<Def> {
    let id = c.id.as_ref()?;
    let start = keyword_start(source, u(c.span.start), last_decorator_end(&c.decorators));
    let end = u(c.span.end);
    Some(build_def(&CLASSES, id.name.to_string(), file, lines, source, (start, end), 0))
}

/// `type X = ...`.
fn type_alias_def(t: &TSTypeAliasDeclaration<'_>, source: &str, lines: &LineMap, file: &Arc<str>) -> Option<Def> {
    let (start, end) = (u(t.span.start), u(t.span.end));
    Some(build_def(&TYPE_ALIASES, t.id.name.to_string(), file, lines, source, (start, end), 0))
}

/// `interface X { ... }`.
fn interface_def(i: &TSInterfaceDeclaration<'_>, source: &str, lines: &LineMap, file: &Arc<str>) -> Option<Def> {
    let (start, end) = (u(i.span.start), u(i.span.end));
    Some(build_def(&INTERFACES, i.id.name.to_string(), file, lines, source, (start, end), 0))
}

/// A `const`/`let`/`var` declaration may carry several declarators (`const a = 1, b = 2`).
/// Each declarator with an identifier binding is surfaced:
/// - arrow / function-expression initializer ⇒ `functions` (dominant TS form).
/// - `const NAME` with `UPPER_SNAKE_CASE` name + non-function initializer ⇒ `constants`.
/// - destructuring patterns bind nothing nameable ⇒ skipped.
fn variable_decls(v: &VariableDeclaration<'_>, source: &str, lines: &LineMap, file: &Arc<str>, out: &mut Vec<Def>) {
    let is_const = matches!(v.kind, VariableDeclarationKind::Const);
    for decl in &v.declarations {
        let BindingPattern::BindingIdentifier(id) = &decl.id else { continue };
        let name = id.name.to_string();
        let Some(init) = &decl.init else { continue };
        match init {
            Expression::ArrowFunctionExpression(arrow) => {
                // Single-expression arrow body (`() => expr`) is one synthetic
                // ExpressionStatement equivalent; oxc still represents it as a FunctionBody with
                // a Return wrapping the expression. The trivial-body filter handles the
                // "return <atom>" case uniformly.
                if !arrow.expression && is_trivial_function_body(Some(arrow.body.as_ref())) {
                    continue;
                }
                let start = u(v.span.start); // include the `const`/`let`/`var` keyword
                let end = u(decl.span.end);
                out.push(build_def(&FUNCTIONS, name, file, lines, source, (start, end), count_args(&arrow.params)));
            }
            Expression::FunctionExpression(fexpr) => {
                if is_trivial_function_body(fexpr.body.as_deref()) {
                    continue;
                }
                let start = u(v.span.start);
                let end = u(decl.span.end);
                out.push(build_def(&FUNCTIONS, name, file, lines, source, (start, end), count_args(&fexpr.params)));
            }
            _ if is_const && is_upper_snake(&name) => {
                let start = u(decl.span.start);
                let end = u(decl.span.end);
                out.push(build_def(&CONSTANTS, name, file, lines, source, (start, end), 0));
            }
            _ => {}
        }
    }
}

/// Getter / setter ⇒ name-suffix so an accessor pair doesn't collide in the name-gated pass.
fn method_kind_suffix(kind: MethodDefinitionKind) -> Option<&'static str> {
    match kind {
        MethodDefinitionKind::Get => Some("getter"),
        MethodDefinitionKind::Set => Some("setter"),
        MethodDefinitionKind::Method | MethodDefinitionKind::Constructor => None,
    }
}

/// Best-effort name for a property key. Computed keys lump together as `<computed>` so
/// name-gated clustering doesn't blindly join, e.g., `[Symbol.iterator]` methods from unrelated
/// classes.
fn property_key_name(key: &PropertyKey<'_>) -> String {
    match key {
        PropertyKey::StaticIdentifier(id) => id.name.to_string(),
        PropertyKey::PrivateIdentifier(id) => format!("#{}", id.name),
        PropertyKey::StringLiteral(s) => s.value.to_string(),
        _ => "<computed>".to_owned(),
    }
}

/// Methods of one class, surfaced as `kind = "methods"` with class-qualified names.
fn class_method_defs(c: &Class<'_>, source: &str, lines: &LineMap, file: &Arc<str>, parent_chain: &str, out: &mut Vec<Def>) {
    let Some(class_id) = c.id.as_ref() else { return };
    let class_name = class_id.name.as_str();
    let parent = if parent_chain.is_empty() {
        class_name.to_owned()
    } else {
        format!("{parent_chain}.{class_name}")
    };
    for element in &c.body.body {
        if let ClassElement::MethodDefinition(m) = element {
            if is_trivial_function_body(m.value.body.as_deref()) {
                continue;
            }
            let start = keyword_start(source, u(m.span.start), last_decorator_end(&m.decorators));
            let end = u(m.span.end);
            let key_name = property_key_name(&m.key);
            let name = match method_kind_suffix(m.kind) {
                Some(role) => format!("{parent}.{key_name}.{role}"),
                None => format!("{parent}.{key_name}"),
            };
            let args = count_args(&m.value.params);
            out.push(build_def(&METHODS, name, file, lines, source, (start, end), args));
        }
    }
}

// ─────────────────────────── per-statement dispatch ───────────────────────────

fn process_top_stmt(stmt: &Statement<'_>, source: &str, lines: &LineMap, file: &Arc<str>, out: &mut Vec<Def>) {
    match stmt {
        Statement::FunctionDeclaration(f) => {
            if let Some(def) = function_def(f, source, lines, file) {
                out.push(def);
            }
        }
        Statement::ClassDeclaration(c) => {
            if let Some(def) = class_def(c, source, lines, file) {
                out.push(def);
            }
            class_method_defs(c, source, lines, file, "", out);
        }
        Statement::TSTypeAliasDeclaration(t) => {
            if let Some(def) = type_alias_def(t, source, lines, file) {
                out.push(def);
            }
        }
        Statement::TSInterfaceDeclaration(i) => {
            if let Some(def) = interface_def(i, source, lines, file) {
                out.push(def);
            }
        }
        Statement::VariableDeclaration(v) => variable_decls(v, source, lines, file, out),
        Statement::ExportNamedDeclaration(e) => {
            if let Some(decl) = &e.declaration {
                process_declaration(decl, source, lines, file, out);
            }
        }
        Statement::ExportDefaultDeclaration(e) => match &e.declaration {
            ExportDefaultDeclarationKind::FunctionDeclaration(f) => {
                if let Some(def) = function_def(f, source, lines, file) {
                    out.push(def);
                }
            }
            ExportDefaultDeclarationKind::ClassDeclaration(c) => {
                if let Some(def) = class_def(c, source, lines, file) {
                    out.push(def);
                }
                class_method_defs(c, source, lines, file, "", out);
            }
            _ => {}
        },
        _ => {}
    }
}

/// Inner-declaration walker — same cases as [`process_top_stmt`] minus the export wrappers.
fn process_declaration(decl: &Declaration<'_>, source: &str, lines: &LineMap, file: &Arc<str>, out: &mut Vec<Def>) {
    match decl {
        Declaration::FunctionDeclaration(f) => {
            if let Some(def) = function_def(f, source, lines, file) {
                out.push(def);
            }
        }
        Declaration::ClassDeclaration(c) => {
            if let Some(def) = class_def(c, source, lines, file) {
                out.push(def);
            }
            class_method_defs(c, source, lines, file, "", out);
        }
        Declaration::TSTypeAliasDeclaration(t) => {
            if let Some(def) = type_alias_def(t, source, lines, file) {
                out.push(def);
            }
        }
        Declaration::TSInterfaceDeclaration(i) => {
            if let Some(def) = interface_def(i, source, lines, file) {
                out.push(def);
            }
        }
        Declaration::VariableDeclaration(v) => variable_decls(v, source, lines, file, out),
        _ => {}
    }
}

// ─────────────────────────── parsing + driver ───────────────────────────

/// Scan one TypeScript source string → its definitions as [`Def`]s with canon precomputed. The
/// `file` path drives the parse mode (`.tsx` enables JSX) and is the shared `Arc` stamped onto
/// every def.
pub(crate) fn scan_source(source: &str, file: &Arc<str>) -> Vec<Def> {
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(Path::new(&**file)).unwrap_or_else(|_| SourceType::ts());
    let ret = Parser::new(&allocator, source, source_type).parse();
    if ret.panicked {
        return Vec::new();
    }
    let lines = LineMap::new(source);
    let mut defs: Vec<Def> = Vec::new();
    for stmt in &ret.program.body {
        process_top_stmt(stmt, source, &lines, file, &mut defs);
    }
    defs
}

#[cfg(test)]
mod tests {
    use super::scan_source;
    use std::sync::Arc;

    fn triples(src: &str, file: &str) -> Vec<(String, String, String)> {
        let f: Arc<str> = Arc::from(file);
        scan_source(src, &f).into_iter().map(|d| (d.kind.id.to_owned(), d.name, d.text_orig)).collect()
    }

    #[test]
    fn finds_top_level_functions_classes_types_interfaces() {
        let src = r#"
function topFn(x: number): number {
    return x + 1;
}

class C {
    method(x: number): number {
        return x + 1;
    }
}

type Ids = number[];

interface Repo {
    get(): number;
    set(x: number): void;
}

const MAX = 5;
const lower = 1;
"#;
        let got = triples(src, "test.ts");
        let names: Vec<&str> = got.iter().map(|(_, n, _)| n.as_str()).collect();
        assert!(names.contains(&"topFn"), "got: {names:?}");
        assert!(names.contains(&"C"), "got: {names:?}");
        assert!(names.contains(&"C.method"), "got: {names:?}");
        assert!(names.contains(&"Ids"), "got: {names:?}");
        assert!(names.contains(&"Repo"), "got: {names:?}");
        assert!(names.contains(&"MAX"), "got: {names:?}");
        assert!(!names.contains(&"lower"), "got: {names:?}");
    }

    #[test]
    fn arrow_const_surfaced_as_function() {
        let src = r#"
const fetch = async (x: number): Promise<number> => {
    return x + 1;
};
"#;
        let f: Arc<str> = Arc::from("test.ts");
        let got = scan_source(src, &f);
        let fetch = got.iter().find(|d| d.name == "fetch").expect("fetch arrow");
        assert_eq!(fetch.kind.id, "functions");
        assert_eq!(fetch.args, 1);
    }

    #[test]
    fn export_named_and_default_unwrap() {
        let src = r#"
export function exported(x: number): number {
    return x + 1;
}

export default class Default {
    method(): number {
        return 1 + 1;
    }
}
"#;
        let got = triples(src, "test.ts");
        let names: Vec<&str> = got.iter().map(|(_, n, _)| n.as_str()).collect();
        assert!(names.contains(&"exported"), "got: {names:?}");
        assert!(names.contains(&"Default"), "got: {names:?}");
        assert!(names.contains(&"Default.method"), "got: {names:?}");
    }

    #[test]
    fn trivial_returns_and_throw_not_implemented_skipped() {
        let src = r#"
class A {
    isX(): boolean { return false; }
    name(): string { return "a"; }
    self() { return this; }
    nullish() { return null; }
    empty(): void { return; }
    notImpl(): never { throw new Error("not implemented"); }
    getX() { return this._x + 1; }
    sources() { return [this._x]; }
    call() { return this.parent.fn(); }
}
"#;
        let got = triples(src, "test.ts");
        let methods: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        for skipped in ["A.isX", "A.name", "A.self", "A.nullish", "A.empty", "A.notImpl"] {
            assert!(!methods.contains(&skipped), "{skipped} should be skipped, got: {methods:?}");
        }
        for kept in ["A.getX", "A.sources", "A.call"] {
            assert!(methods.contains(&kept), "{kept} should be kept, got: {methods:?}");
        }
    }

    #[test]
    fn getter_and_setter_get_role_suffix() {
        let src = r#"
class C {
    get value(): number { return this._x + 1; }
    set value(v: number) { this._x = v; }
}
"#;
        let got = triples(src, "test.ts");
        let methods: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        assert!(methods.contains(&"C.value.getter"), "got: {methods:?}");
        assert!(methods.contains(&"C.value.setter"), "got: {methods:?}");
    }

    #[test]
    fn decorated_class_text_excludes_decorators() {
        let src = r#"
@Injectable()
class Service {
    do(x: number) { return x + 1; }
}
"#;
        let f: Arc<str> = Arc::from("test.ts");
        let got = scan_source(src, &f);
        let svc = got.iter().find(|d| d.name == "Service").expect("Service class");
        assert!(svc.text_orig.trim_start().starts_with("class "), "got: {:?}", svc.text_orig);
    }

    #[test]
    fn class_inside_function_methods_are_not_surfaced() {
        let src = r#"
function factory() {
    class Hidden {
        helper(): number { return 1 + 1; }
    }
    return Hidden;
}
"#;
        let got = triples(src, "test.ts");
        let methods: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        assert!(methods.is_empty(), "no methods expected, got: {methods:?}");
    }

    #[test]
    fn node_kinds_and_canon_presence() {
        // Body kinds carry a cluster canonical; raw-text kinds do not; callables carry analysis
        // (methods analyze to None — their slice isn't a standalone function — which is fine).
        let src = "export function f(x: number) { const y = x + 1; return y * 2; }\nexport const MAX_N = 7;\nexport interface I { a(): number; }\n";
        let f: Arc<str> = Arc::from("t.ts");
        let defs = scan_source(src, &f);
        let func = defs.iter().find(|d| d.name == "f").expect("fn");
        assert!(func.cluster_canonical.is_some() && func.analysis.is_some());
        let iface = defs.iter().find(|d| d.name == "I").expect("iface");
        assert!(iface.cluster_canonical.is_some() && iface.analysis.is_none());
        let konst = defs.iter().find(|d| d.name == "MAX_N").expect("const");
        assert!(konst.cluster_canonical.is_none() && konst.analysis.is_none());
    }
}
