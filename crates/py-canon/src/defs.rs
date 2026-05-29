//! Module-level definition scan for Python — the "find-*" step of `find-dup-defs`.
//!
//! Walks each file's Ruff AST **once** and lowers every tracked definition straight to the
//! engine's [`Def`], computing its canonical strings from the in-place AST node — no re-parse
//! of the def's source text. Surfaces **top-level only** definitions (not nested in any
//! function/class), `UPPER_CASE` constants, and **class methods** (qualified `ClassName.method`,
//! recursing into nested classes; classes hidden inside *functions* stay invisible). Decorators
//! are excluded from a def's text (the range starts at the `def`/`async`/`class` keyword).
//!
//! Per-kind canonicalization:
//! * **functions / classes** — node-based via [`analyze_stmt`](crate::canon::analyze_stmt) /
//!   [`cluster_canonical_node`](crate::canon::cluster_canonical_node) (byte-identical to
//!   re-parsing the decorator-stripped def text, but with zero extra parses).
//! * **methods** — the receiver (`self`/`cls`) is stripped at the *text* level (so a method's
//!   canonical lines up with an equivalent free function) and the stripped text is re-parsed
//!   once via [`analyze_functions`](crate::canon::analyze_functions); a node-level skip would
//!   need surgery in the `ast.dump` emitter and mishandle the `self, /` edge.
//! * **constants / type-aliases** — raw-text kinds; the engine clusters them on `text_orig`,
//!   so no canonical is computed.
//!
//! Modern syntax (PEP 695 `type` aliases / generics, PEP 701 f-strings) is handled by Ruff.

use std::sync::Arc;

use dup_defs_core::{Analysis, Def, KindSpec, LineMap};
use ruff_python_ast::{Expr, Parameters, Stmt};
use ruff_python_parser::parse_module;

use crate::canon::{analyze_functions, analyze_stmt, ast_canonical, cluster_canonical_node};
use crate::frontend::{kind_spec, METHODS};

/// Non-blank line count of a def's source text — the simplest "how big" metric the report can
/// surface. Blank/whitespace-only lines (including the line after a multi-line signature) are
/// excluded so a method with a deliberately spaced-out body doesn't read as twice as big as an
/// equivalent dense one.
fn count_loc(text: &str) -> usize {
    text.lines().filter(|l| !l.trim().is_empty()).count()
}

/// Total parameter count: posonly + args + kwonly + (`*args` if present) + (`**kwargs` if
/// present). For methods this includes the receiver (`self` / `cls`) — that's what the user
/// sees in their own code; the strip-receiver canonicalization step is invisible to this count.
fn count_args(p: &Parameters) -> usize {
    p.posonlyargs.len()
        + p.args.len()
        + p.kwonlyargs.len()
        + usize::from(p.vararg.is_some())
        + usize::from(p.kwarg.is_some())
}

fn is_upper(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// The name of a module-level `UPPER_CASE` constant assignment (`NAME = …` / `NAME: T = …`).
fn const_name(stmt: &Stmt) -> Option<String> {
    match stmt {
        Stmt::Assign(node) => match node.targets.as_slice() {
            [Expr::Name(name)] if is_upper(name.id.as_str()) => Some(name.id.as_str().to_owned()),
            _ => None,
        },
        Stmt::AnnAssign(node) => match node.target.as_ref() {
            Expr::Name(name) if is_upper(name.id.as_str()) => Some(name.id.as_str().to_owned()),
            _ => None,
        },
        _ => None,
    }
}

/// The byte offset of the `def`/`async`/`class` keyword — i.e. the def text *excluding* decorators.
/// With no decorators the statement range already starts at the keyword; with decorators we skip
/// past the last decorator and any intervening whitespace / comment lines to the keyword token.
fn keyword_start(source: &str, range_start: usize, last_decorator_end: Option<usize>) -> usize {
    let Some(mut i) = last_decorator_end else { return range_start };
    let bytes = source.as_bytes();
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        break;
    }
    i
}

/// True when a function's body is effectively a no-op or a single-token return — the canonical
/// "dispatch override" / "stub" / "placeholder" shapes. Surfaces:
///
/// * `...` / `pass` / docstring — overload sigs, abstractmethod, Protocol, `.pyi` stubs.
/// * `raise NotImplementedError[(…)]` — ABC interface declarations.
/// * `return <literal>` / `return <Name>` / `return` — single-token returns like `return False`,
///   `return None`, `return self`, `return 0`. These dominate cross-class virtual-dispatch
///   overrides (`def can_be_true_default(self) -> bool: return False`) — the BODIES collide
///   trivially across hundreds of unrelated classes, but the *intent* is per-method
///   specialisation, not a refactoring target.
///
/// Falls THROUGH (still compared): `return self.x` (Attribute), `return [self.x]` (List),
/// `return foo()` (Call) — these carry enough structure to be real refactor candidates.
fn is_trivial_function_body(body: &[ruff_python_ast::Stmt]) -> bool {
    body.iter().all(|s| match s {
        Stmt::Pass(_) => true,
        Stmt::Expr(e) => matches!(e.value.as_ref(), Expr::EllipsisLiteral(_) | Expr::StringLiteral(_)),
        Stmt::Raise(r) => match r.exc.as_deref() {
            Some(Expr::Name(name)) => name.id.as_str() == "NotImplementedError",
            Some(Expr::Call(call)) => matches!(call.func.as_ref(), Expr::Name(n) if n.id.as_str() == "NotImplementedError"),
            _ => false,
        },
        Stmt::Return(r) => match r.value.as_deref() {
            None => true,
            Some(e) => matches!(
                e,
                Expr::NoneLiteral(_)
                    | Expr::BooleanLiteral(_)
                    | Expr::NumberLiteral(_)
                    | Expr::StringLiteral(_)
                    | Expr::BytesLiteral(_)
                    | Expr::EllipsisLiteral(_)
                    | Expr::Name(_)
            ),
        },
        _ => false,
    })
}

/// If the first positional parameter of a method is `self` or `cls`, return the absolute byte
/// range covering that parameter plus its trailing `,` (and surrounding whitespace), so the
/// caller can splice it out of the method text. Returns `None` for `@staticmethod`-shaped
/// signatures whose first param has a normal name. Stripping the receiver makes a method's
/// xname canonical line up with a top-level function of the same body — without it, arity
/// alone (`(self, x)` vs `(x)`) drives the canonicals apart and the cross-name pass misses the
/// duplicate. A `def foo(self, /, x):` is left untouched here: removing `self` would leave a
/// stray `/` separator and break the post-strip re-parse; `analyze_functions` then returns
/// `None` for it, which is the existing graceful-degradation path.
fn method_receiver_strip_range(source: &str, params: &Parameters) -> Option<(usize, usize)> {
    // Bail on the `self, /` shape — see the doc note above.
    if !params.posonlyargs.is_empty() && params.posonlyargs.len() == 1 && params.args.is_empty() {
        // Sole posonly receiver — would leave `/` dangling. Skip.
        let only = &params.posonlyargs[0];
        let n = only.parameter.name.id.as_str();
        if n == "self" || n == "cls" {
            return None;
        }
    }
    let first = params.posonlyargs.first().or_else(|| params.args.first())?;
    let name = first.parameter.name.id.as_str();
    if name != "self" && name != "cls" {
        return None;
    }
    let param_start = usize::from(first.parameter.range.start());
    // `ParameterWithDefault.range` covers `name [: ann] [= default]`, i.e. the whole slot.
    let after_param = usize::from(first.range.end());
    let bytes = source.as_bytes();
    let mut i = after_param;
    // Eat any whitespace (including newlines) up to the next significant char.
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b',' {
        i += 1;
        // Trailing whitespace after the comma — eat it too so the next param hugs `(`.
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        Some((param_start, i))
    } else {
        // No trailing comma — `self` is the only positional param, e.g. `def foo(self):`.
        Some((param_start, after_param))
    }
}

/// The method's canonicalization-input text: the def source (`source[start..end]`, already
/// decorator-excluded) with the `self`/`cls` receiver spliced out, or the def source unchanged
/// when there's no strippable receiver. Mirrors the previous in-engine text manipulation.
pub(crate) fn apply_receiver_strip(source: &str, start: usize, end: usize, params: &Parameters) -> String {
    match method_receiver_strip_range(source, params) {
        Some((rs, re)) if rs >= start && re <= end => {
            let mut t = String::with_capacity(end - start - (re - rs));
            t.push_str(&source[start..rs]);
            t.push_str(&source[re..end]);
            t
        }
        _ => source[start..end].to_owned(),
    }
}

/// `@<name>.setter` / `@<name>.deleter` / `@<name>.getter` — accessor role suffix to attach to a
/// method's qualified name, so the property setter doesn't collide with the getter in the
/// name-gated pass. Setter and getter share a Python-level name (`Class.value`) but have
/// legitimately different bodies; without this disambiguation they cluster as `[ast sim ~0.65]`
/// false positives. `@property` (and any other plain `Name` decorator) returns `None` — the
/// getter keeps the bare `Class.method` name, and cross-class accessor duplicates still match
/// (`Foo.value.setter` ↔ `Bar.value.setter`).
fn property_accessor_suffix(decorators: &[ruff_python_ast::Decorator]) -> Option<&'static str> {
    for d in decorators {
        if let Expr::Attribute(attr) = &d.expression {
            match attr.attr.id.as_str() {
                "setter" => return Some("setter"),
                "deleter" => return Some("deleter"),
                "getter" => return Some("getter"),
                _ => {}
            }
        }
    }
    None
}

/// Classify one top-level statement → `(kind id, name, text_start, text_end, args)` if it is a
/// tracked definition. `text_start` is decorator-excluded (the keyword offset) for functions /
/// classes. `args` is the parameter count for functions and 0 for everything else. Trivial-body
/// functions are skipped — see [`is_trivial_function_body`].
pub(crate) fn classify(source: &str, stmt: &Stmt) -> Option<(&'static str, String, usize, usize, usize)> {
    match stmt {
        Stmt::FunctionDef(node) => {
            if is_trivial_function_body(&node.body) {
                return None;
            }
            let deco_end = node.decorator_list.last().map(|d| usize::from(d.range.end()));
            let start = keyword_start(source, usize::from(node.range.start()), deco_end);
            Some((
                "functions",
                node.name.id.as_str().to_owned(),
                start,
                usize::from(node.range.end()),
                count_args(&node.parameters),
            ))
        }
        Stmt::ClassDef(node) => {
            let deco_end = node.decorator_list.last().map(|d| usize::from(d.range.end()));
            let start = keyword_start(source, usize::from(node.range.start()), deco_end);
            Some(("classes", node.name.id.as_str().to_owned(), start, usize::from(node.range.end()), 0))
        }
        Stmt::TypeAlias(node) => match node.name.as_ref() {
            Expr::Name(name) => Some((
                "type-aliases",
                name.id.as_str().to_owned(),
                usize::from(node.range.start()),
                usize::from(node.range.end()),
                0,
            )),
            _ => None,
        },
        Stmt::Assign(node) => const_name(stmt)
            .map(|name| ("constants", name, usize::from(node.range.start()), usize::from(node.range.end()), 0)),
        Stmt::AnnAssign(node) => const_name(stmt)
            .map(|name| ("constants", name, usize::from(node.range.start()), usize::from(node.range.end()), 0)),
        _ => None,
    }
}

/// Tuple → [`Analysis`], shared by the function and method canon paths.
fn analysis_from(xname: String, lines: Vec<String>, size: usize) -> Analysis {
    Analysis { xname_canonical: xname, type3_lines: lines, size }
}

/// Canonicalize a **top-level** function/class node (no re-parse). Returns
/// `(cluster_canonical, analysis)`: functions get both off one node walk, classes get the
/// cluster canonical only (not callable), and raw-text kinds get neither.
fn top_def_canon(kind: &'static KindSpec, stmt: &Stmt, src: &str) -> (Option<String>, Option<Analysis>) {
    match kind.id {
        "functions" => match analyze_stmt(stmt, src) {
            Some((cc, xname, lines, size)) => (Some(cc), Some(analysis_from(xname, lines, size))),
            // A FunctionDef always analyzes; this branch only guards future kinds.
            None => (Some(cluster_canonical_node(stmt, src)), None),
        },
        "classes" => (Some(cluster_canonical_node(stmt, src)), None),
        _ => (None, None),
    }
}

/// Canonicalize a method from its receiver-stripped text. One re-parse (`analyze_functions`)
/// yields the cluster canonical (tuple `.0`, identical to `ast_canonical`) plus the analysis;
/// on the rare unparseable-strip edge (`self, /`) it falls back to `ast_canonical`'s raw-text
/// path with no analysis — byte-identical to the previous behavior.
fn method_canon(canon_text: &str) -> (Option<String>, Option<Analysis>) {
    match analyze_functions(&[canon_text.to_owned()]).into_iter().next().flatten() {
        Some((cc, xname, lines, size)) => (Some(cc), Some(analysis_from(xname, lines, size))),
        None => (Some(ast_canonical(canon_text)), None),
    }
}

/// Methods of one class as `Def`s (`kind = methods`, class-qualified names). Recurses into
/// nested classes (`Outer.Inner.foo`); classes hidden inside a *function* are never reached.
fn method_defs(source: &str, stmt: &Stmt, lines: &LineMap, file: &Arc<str>, parent_chain: &str, out: &mut Vec<Def>) {
    let Stmt::ClassDef(class) = stmt else { return };
    let class_name = class.name.id.as_str();
    let parent = if parent_chain.is_empty() { class_name.to_owned() } else { format!("{parent_chain}.{class_name}") };
    for inner in &class.body {
        match inner {
            Stmt::FunctionDef(node) => {
                if is_trivial_function_body(&node.body) {
                    continue;
                }
                let deco_end = node.decorator_list.last().map(|d| usize::from(d.range.end()));
                let start = keyword_start(source, usize::from(node.range.start()), deco_end);
                let end = usize::from(node.range.end());
                let (line, col) = lines.loc0(start);
                let method_name = node.name.id.as_str();
                let name = match property_accessor_suffix(&node.decorator_list) {
                    Some(role) => format!("{parent}.{method_name}.{role}"),
                    None => format!("{parent}.{method_name}"),
                };
                let text_orig = source[start..end].to_owned();
                let loc = count_loc(&text_orig);
                let args = count_args(&node.parameters);
                let canon_text = apply_receiver_strip(source, start, end, &node.parameters);
                let (cluster_canonical, analysis) = method_canon(&canon_text);
                out.push(Def {
                    lang: "py",
                    kind: &METHODS,
                    name,
                    file: Arc::clone(file),
                    line,
                    col,
                    loc,
                    args,
                    text_orig,
                    cluster_canonical,
                    analysis,
                });
            }
            Stmt::ClassDef(_) => method_defs(source, inner, lines, file, &parent, out),
            _ => {}
        }
    }
}

/// Scan one Python source string → its definitions as [`Def`]s with canon precomputed.
pub(crate) fn scan_source(source: &str, file: &Arc<str>) -> Vec<Def> {
    let Ok(parsed) = parse_module(source) else { return Vec::new() };
    let module = parsed.into_syntax();
    let lines = LineMap::new(source);
    let mut defs: Vec<Def> = Vec::new();
    for stmt in &module.body {
        if matches!(stmt, Stmt::ClassDef(_)) {
            method_defs(source, stmt, &lines, file, "", &mut defs);
        }
        let Some((kind_id, name, start, end, args)) = classify(source, stmt) else { continue };
        let (line, col) = lines.loc0(start);
        let text_orig = source[start..end].to_owned();
        let loc = count_loc(&text_orig);
        let kind = kind_spec(kind_id);
        let (cluster_canonical, analysis) = top_def_canon(kind, stmt, source);
        defs.push(Def {
            lang: "py",
            kind,
            name,
            file: Arc::clone(file),
            line,
            col,
            loc,
            args,
            text_orig,
            cluster_canonical,
            analysis,
        });
    }
    defs
}

#[cfg(test)]
mod tests {
    use super::{apply_receiver_strip, scan_source};
    use ruff_python_ast::Stmt;
    use ruff_python_parser::parse_module;
    use std::sync::Arc;

    /// `(kind id, name, text_orig)` triples — the extraction shape the old tests asserted on.
    fn triples(src: &str) -> Vec<(String, String, String)> {
        let file: Arc<str> = Arc::from("<test>");
        scan_source(src, &file).into_iter().map(|d| (d.kind.id.to_owned(), d.name, d.text_orig)).collect()
    }

    fn names(src: &str) -> Vec<String> {
        triples(src).into_iter().map(|(_, n, _)| n).collect()
    }

    fn method_names(src: &str) -> Vec<String> {
        triples(src).into_iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n).collect()
    }

    /// Receiver-stripped canon-input text of the first top-level method in `src` named `name`.
    /// Exercises [`apply_receiver_strip`] directly (the stripped form is no longer stored on a
    /// def — it feeds the canonical and is discarded).
    fn stripped_method_text(src: &str, name: &str) -> String {
        let module = parse_module(src).expect("parse").into_syntax();
        let Stmt::ClassDef(class) = &module.body[0] else { panic!("expected class") };
        for inner in &class.body {
            if let Stmt::FunctionDef(node) = inner {
                if node.name.id.as_str() == name {
                    // Skip decorators, exactly like the scan, so the stripped text starts at `def`.
                    let deco_end = node.decorator_list.last().map(|d| usize::from(d.range.end()));
                    let start = super::keyword_start(src, usize::from(node.range.start()), deco_end);
                    let end = usize::from(node.range.end());
                    return apply_receiver_strip(src, start, end, &node.parameters);
                }
            }
        }
        panic!("method {name} not found");
    }

    #[test]
    fn finds_top_level_kinds_and_class_methods() {
        let src = "MAX = 5\nlower = 1\n\ntype Ids = list[int]\n\n\ndef top():\n    def nested():\n        pass\n    return 1\n\n\nclass C:\n    def method(self):\n        return self.x + 1\n";
        let got = triples(src);
        let kinds: Vec<&str> = got.iter().map(|(k, _, _)| k.as_str()).collect();
        let ns: Vec<&str> = got.iter().map(|(_, n, _)| n.as_str()).collect();
        assert!(ns.contains(&"MAX") && ns.contains(&"Ids"));
        assert!(ns.contains(&"top") && ns.contains(&"C"));
        assert!(ns.contains(&"C.method"));
        assert!(!ns.contains(&"lower") && !ns.contains(&"nested") && !ns.contains(&"method"));
        assert_eq!(kinds.iter().filter(|k| **k == "functions").count(), 1);
        assert_eq!(kinds.iter().filter(|k| **k == "classes").count(), 1);
        assert_eq!(kinds.iter().filter(|k| **k == "methods").count(), 1);
    }

    #[test]
    fn class_methods_emitted_with_qualified_names_and_methods_kind() {
        let src = "class Foo:\n    def __init__(self, x):\n        self.x = x\n\n    async def fetch(self):\n        return self.x\n";
        let m = method_names(src);
        assert!(m.contains(&"Foo.__init__".to_owned()), "got methods: {m:?}");
        assert!(m.contains(&"Foo.fetch".to_owned()), "got methods: {m:?}");
        let init = triples(src).into_iter().find(|(_, n, _)| n == "Foo.__init__").expect("init");
        assert!(init.2.starts_with("def "), "method text should start at def, got: {:?}", init.2);
    }

    #[test]
    fn nested_class_methods_use_chained_parent_names() {
        let src = "class Outer:\n    def outer_m(self):\n        return self.x + 1\n\n    class Inner:\n        def inner_m(self):\n            return self.x + 2\n\n        class Deep:\n            def deep_m(self):\n                return self.x + 3\n";
        let m = method_names(src);
        assert!(m.contains(&"Outer.outer_m".to_owned()), "got methods: {m:?}");
        assert!(m.contains(&"Outer.Inner.inner_m".to_owned()), "got methods: {m:?}");
        assert!(m.contains(&"Outer.Inner.Deep.deep_m".to_owned()), "got methods: {m:?}");
    }

    #[test]
    fn single_token_return_dispatch_overrides_are_skipped() {
        let src = concat!(
            "class A:\n",
            "    def is_x(self) -> bool:\n        return False\n",
            "    def default(self):\n        return None\n",
            "    def name(self) -> str:\n        return \"a\"\n",
            "    def num(self) -> int:\n        return 0\n",
            "    def empty(self):\n        return\n",
            "    def self_(self):\n        return self\n",
            "    def get_x(self):\n        return self._x\n",
            "    def sources(self):\n        return [self._x]\n",
            "    def call(self):\n        return self.parent.fn()\n",
        );
        let m = method_names(src);
        for skipped in ["A.is_x", "A.default", "A.name", "A.num", "A.empty", "A.self_"] {
            assert!(!m.contains(&skipped.to_owned()), "{skipped} should be skipped, got: {m:?}");
        }
        for kept in ["A.get_x", "A.sources", "A.call"] {
            assert!(m.contains(&kept.to_owned()), "{kept} should be kept, got: {m:?}");
        }
    }

    #[test]
    fn raise_not_implemented_stubs_are_skipped() {
        let src = concat!(
            "class IFoo:\n",
            "    def do(self, x: int) -> int:\n        raise NotImplementedError\n\n",
            "    def go(self, x: int) -> int:\n        raise NotImplementedError('subclass me')\n\n",
            "    def real(self, x: int) -> int:\n        return x + 1\n",
        );
        assert_eq!(method_names(src), vec!["IFoo.real".to_owned()], "got: {:?}", method_names(src));
    }

    #[test]
    fn overload_and_abstract_stubs_are_skipped_real_impl_kept() {
        let src = concat!(
            "from typing import overload\n",
            "from abc import abstractmethod\n\n",
            "class C:\n",
            "    @overload\n    def foo(self, x: int) -> int: ...\n",
            "    @overload\n    def foo(self, x: str) -> str: ...\n",
            "    def foo(self, x):\n        return x + 1\n\n",
            "    @abstractmethod\n    def bar(self):\n        \"\"\"abstract.\"\"\"\n\n",
            "    @abstractmethod\n    def baz(self):\n        pass\n\n",
            "    def qux(self):\n        ...\n",
        );
        assert_eq!(method_names(src), vec!["C.foo".to_owned()], "expected only real impl, got: {:?}", method_names(src));
        let foo = triples(src).into_iter().find(|(_, n, _)| n == "C.foo").expect("real foo");
        assert!(foo.2.contains("return x + 1"), "expected the real impl, got: {:?}", foo.2);
    }

    #[test]
    fn loc_and_args_are_populated_from_original_source() {
        let src = concat!(
            "def free(a, b, *, c=3):\n    x = a + b\n    y = x * c\n    return y\n\n",
            "class C:\n    def method(self, x, y):\n        if x > y:\n            return x\n        return y\n",
        );
        let file: Arc<str> = Arc::from("<test>");
        let defs = scan_source(src, &file);
        let free = defs.iter().find(|d| d.name == "free").expect("free fn");
        assert_eq!(free.loc, 4, "free loc: {}", free.loc);
        assert_eq!(free.args, 3, "free args: {}", free.args);
        let method = defs.iter().find(|d| d.name == "C.method").expect("method");
        assert_eq!(method.loc, 4, "method loc: {}", method.loc);
        assert_eq!(method.args, 3, "method args (incl self): {}", method.args);
    }

    #[test]
    fn method_receiver_is_stripped_from_canon_input() {
        let src = concat!(
            "class C:\n",
            "    def one(self):\n        return self.x + 1\n\n",
            "    def two(self, x):\n        return x + 1\n\n",
            "    @classmethod\n    def three(cls, x):\n        return x * 2\n\n",
            "    @staticmethod\n    def four(x):\n        return x * 3\n",
        );
        assert!(stripped_method_text(src, "one").starts_with("def one():"), "{:?}", stripped_method_text(src, "one"));
        assert!(stripped_method_text(src, "two").starts_with("def two(x):"), "{:?}", stripped_method_text(src, "two"));
        assert!(stripped_method_text(src, "three").starts_with("def three(x):"), "{:?}", stripped_method_text(src, "three"));
        // `@staticmethod`-shaped first param (named `x`, not `self`/`cls`) is left alone.
        assert!(stripped_method_text(src, "four").starts_with("def four(x):"), "{:?}", stripped_method_text(src, "four"));
    }

    #[test]
    fn property_setter_and_deleter_get_role_suffix() {
        let src = concat!(
            "class C:\n",
            "    @property\n    def value(self):\n        return self._x\n\n",
            "    @value.setter\n    def value(self, v):\n        self._x = v\n\n",
            "    @value.deleter\n    def value(self):\n        del self._x\n",
        );
        let m = method_names(src);
        assert!(m.contains(&"C.value".to_owned()), "getter: {m:?}");
        assert!(m.contains(&"C.value.setter".to_owned()), "setter: {m:?}");
        assert!(m.contains(&"C.value.deleter".to_owned()), "deleter: {m:?}");
    }

    #[test]
    fn property_with_real_logic_is_kept() {
        let src = concat!(
            "class C:\n",
            "    @property\n    def value(self):\n        if self._cached is None:\n            self._cached = self._compute()\n        return self._cached\n\n",
            "    @value.setter\n    def value(self, v):\n        self._cached = v\n        self._dirty = True\n",
        );
        let m = method_names(src);
        assert!(m.contains(&"C.value".to_owned()), "getter: {m:?}");
        assert!(m.contains(&"C.value.setter".to_owned()), "setter: {m:?}");
    }

    #[test]
    fn class_hidden_inside_function_does_not_surface_methods() {
        let src = "def factory():\n    class Hidden:\n        def helper(self):\n            return 1\n    return Hidden\n";
        assert!(method_names(src).is_empty(), "no methods expected, got: {:?}", method_names(src));
    }

    #[test]
    fn decorated_method_text_excludes_decorators() {
        let src = "class C:\n    @staticmethod\n    def helper(x):\n        return x + 1\n";
        let helper = triples(src).into_iter().find(|(_, n, _)| n == "C.helper").expect("helper");
        assert!(helper.2.starts_with("def "), "decorated method text should start at def, got: {:?}", helper.2);
    }

    #[test]
    fn function_text_excludes_decorators() {
        let got = triples("import functools\n\n\n@functools.cache\ndef memo(x):\n    return x + 1\n");
        let func = got.into_iter().find(|(k, _, _)| k == "functions").expect("a function");
        assert!(func.2.starts_with("def "), "text should start at def, got: {:?}", func.2);
    }

    #[test]
    fn pep695_and_modern_syntax_file_is_scanned() {
        let src = "type Alias = list[int]\n\n\ndef worker[T](x: T) -> T:\n    msg = f\"got {x['k']}\"\n    return x\n\n\nclass Repo[T]:\n    pass\n";
        let ns = names(src);
        assert!(ns.contains(&"Alias".to_owned()), "type alias missing: {ns:?}");
        assert!(ns.contains(&"worker".to_owned()), "generic fn missing: {ns:?}");
        assert!(ns.contains(&"Repo".to_owned()), "generic class missing: {ns:?}");
    }

    #[test]
    fn node_canon_matches_slice_reparse_for_functions_and_classes() {
        use crate::canon::{analyze_functions, ast_canonical};
        let file: Arc<str> = Arc::from("<test>");
        for src in [
            "def add(a, b):\n    total = a + b\n    return total * 2\n",
            "@deco\ndef wrapped(x):\n    y = x + 1\n    return [y, x]\n",
            "class Repo:\n    def get(self, k):\n        return self.store[k] + 1\n",
        ] {
            // Only top-level functions/classes use the node-based path AND have `text_orig`
            // equal to the canon input (methods strip the receiver, so they're excluded here).
            for d in scan_source(src, &file).iter().filter(|d| matches!(d.kind.id, "functions" | "classes")) {
                assert_eq!(d.cluster_canonical.as_deref(), Some(ast_canonical(&d.text_orig).as_str()), "cc mismatch for {:?}", d.name);
                if d.kind.fn_like {
                    let slice = analyze_functions(std::slice::from_ref(&d.text_orig)).into_iter().next().flatten().expect("analyzes");
                    let a = d.analysis.as_ref().expect("fn_like has analysis");
                    assert_eq!(
                        (a.xname_canonical.as_str(), a.type3_lines.as_slice(), a.size),
                        (slice.1.as_str(), slice.2.as_slice(), slice.3),
                        "analysis mismatch for {:?}",
                        d.name
                    );
                }
            }
        }
    }
}
