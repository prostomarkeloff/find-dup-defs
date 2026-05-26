//! Module-level definition scan — the "find-*" step of dup-defs, ported off ast-grep.
//!
//! Replaces four `ast-grep scan --rule find-module-{functions,classes,constants,type-aliases}`
//! subprocess calls with one native, parallel pass. Matches the same semantics: **top-level
//! only** (not nested in any function/class), `UPPER_CASE` constants, decorators excluded from
//! a def's text (the range starts at the `def`/`async`/`class` keyword, like tree-sitter's
//! `function_definition`). Emits each def's kind / name / location / source text — the shape
//! the cross-file grouping step consumes.
//!
//! In addition to top-level definitions, **class methods** are surfaced as their own kind
//! `methods`, with a class-qualified name (`ClassName.method`, or `Outer.Inner.method` for
//! methods of nested classes). The qualification keeps the name-gated pass from clumping every
//! `__init__` / `__str__` across the codebase, while the cross-name and Type-3 passes (which
//! are name-agnostic) still catch a method copy-pasted across two different classes — the
//! typical "duplicate method" case. Classes nested inside *functions* are still skipped, to
//! preserve the "top-level only" principle for everything that isn't itself a class body.
//!
//! Parses with **ruff** (same parser the canonicalization uses), so modern syntax — PEP 695
//! `type` aliases / generics, PEP 701 f-strings — is handled; rustpython silently dropped any
//! file containing them, hiding every def it held from the dup-defs passes.

use std::fs;

use dup_defs_core::{Language, LineMap, ModuleDef};
use rayon::prelude::*;
use ruff_python_ast::{Expr, Parameters, Stmt};
use ruff_python_parser::parse_module;

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

/// Classify one top-level statement → `(kind, name, text_start, text_end, args)` if it is a
/// tracked definition. `text_start` is decorator-excluded (the keyword offset) for functions /
/// classes. `args` is the parameter count for functions and 0 for everything else (it has no
/// meaning on a class / constant / type-alias at the definition level). Trivial-body functions
/// are skipped — see [`is_trivial_function_body`] for the rationale.
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

/// Methods of one class, surfaced as `kind = "methods"` with class-qualified names. Recurses
/// into nested classes so a method of `class Outer: class Inner: def foo(self): …` shows up as
/// `Outer.Inner.foo`. Methods of classes hidden inside a *function* body are never reached —
/// the recursion descends through `class` definitions only.
fn class_method_defs(source: &str, stmt: &Stmt, lines: &LineMap, file: &str, parent_chain: &str) -> Vec<ModuleDef> {
    let Stmt::ClassDef(class) = stmt else { return Vec::new() };
    let class_name = class.name.id.as_str();
    let parent = if parent_chain.is_empty() { class_name.to_owned() } else { format!("{parent_chain}.{class_name}") };
    let mut out = Vec::new();
    for inner in &class.body {
        match inner {
            // `Stmt::FunctionDef` in ruff covers both `def` and `async def` (via an `is_async`
            // flag), so this branch picks up sync and async methods alike. Trivial-body stubs
            // (`@overload`, `@abstractmethod`, Protocol method declarations) are skipped here
            // for the same reason as in `classify` — see [`is_trivial_function_body`].
            Stmt::FunctionDef(node) => {
                if is_trivial_function_body(&node.body) {
                    continue;
                }
                let deco_end = node.decorator_list.last().map(|d| usize::from(d.range.end()));
                let start = keyword_start(source, usize::from(node.range.start()), deco_end);
                let end = usize::from(node.range.end());
                let (line, col) = lines.loc0(start);
                let method_name = node.name.id.as_str();
                // `@value.setter` / `@value.deleter` / `@value.getter` get a suffix so they
                // don't collide with the getter (`Class.value`) in the name-gated pass — see
                // [`property_accessor_suffix`].
                let name = match property_accessor_suffix(&node.decorator_list) {
                    Some(role) => format!("{parent}.{method_name}.{role}"),
                    None => format!("{parent}.{method_name}"),
                };
                // `text_orig` is what the user actually wrote (snippet-display path). `text`
                // gets the post-strip form for canonicalization downstream — see
                // [`method_receiver_strip_range`]. `loc` / `args` reflect the original source so
                // a method that's "really" 3 args still reports `args=3` in the report.
                let text_orig = source[start..end].to_owned();
                let loc = count_loc(&text_orig);
                let args = count_args(&node.parameters);
                let text = match method_receiver_strip_range(source, &node.parameters) {
                    Some((rs, re)) if rs >= start && re <= end => {
                        let mut t = String::with_capacity(end - start - (re - rs));
                        t.push_str(&source[start..rs]);
                        t.push_str(&source[re..end]);
                        t
                    }
                    _ => text_orig.clone(),
                };
                out.push(ModuleDef {
                    kind: "methods".to_owned(),
                    name,
                    file: file.to_owned(),
                    line,
                    col,
                    text,
                    text_orig,
                    loc,
                    args,
                    lang: Language::Python,
                });
            }
            Stmt::ClassDef(_) => {
                out.extend(class_method_defs(source, inner, lines, file, &parent));
            }
            _ => {}
        }
    }
    out
}

fn module_defs_from(source: &str, file: &str) -> Vec<ModuleDef> {
    let Ok(parsed) = parse_module(source) else { return Vec::new() };
    let module = parsed.into_syntax();
    let lines = LineMap::new(source);
    let mut defs: Vec<ModuleDef> = Vec::new();
    for stmt in &module.body {
        if matches!(stmt, Stmt::ClassDef(_)) {
            defs.extend(class_method_defs(source, stmt, &lines, file, ""));
        }
        let Some((kind, name, start, end, args)) = classify(source, stmt) else { continue };
        let (line, col) = lines.loc0(start);
        let text = source[start..end].to_owned();
        let loc = count_loc(&text);
        // For everything that isn't a method, the canonicalization input and the user-visible
        // source are the same; clone keeps the `text_orig` invariant uniform across kinds.
        let text_orig = text.clone();
        defs.push(ModuleDef {
            kind: kind.to_owned(),
            name,
            file: file.to_owned(),
            line,
            col,
            text,
            text_orig,
            loc,
            args,
            lang: Language::Python,
        });
    }
    defs
}

fn module_defs_in(file: &str) -> Vec<ModuleDef> {
    match fs::read_to_string(file) {
        Ok(source) => module_defs_from(&source, file),
        Err(_) => Vec::new(),
    }
}

/// `find_module_defs(files) -> Vec<ModuleDef>`: the dup-defs find step, native + parallel.
#[must_use]
pub fn find_module_defs(files: &[String]) -> Vec<ModuleDef> {
    let per_file: Vec<Vec<ModuleDef>> = files.par_iter().map(|f| module_defs_in(f)).collect();
    per_file.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::module_defs_from;

    fn triples(src: &str) -> Vec<(String, String, String)> {
        module_defs_from(src, "<test>").into_iter().map(|d| (d.kind, d.name, d.text)).collect()
    }

    #[test]
    fn finds_top_level_kinds_and_class_methods() {
        // Non-trivial bodies throughout — the trivial-body filter (overload/stub guard) would
        // otherwise drop the method and `top` is a closure host, so we make both do real work.
        let src = "MAX = 5\nlower = 1\n\ntype Ids = list[int]\n\n\ndef top():\n    def nested():\n        pass\n    return 1\n\n\nclass C:\n    def method(self):\n        return self.x + 1\n";
        let got = triples(src);
        let kinds: Vec<&str> = got.iter().map(|(k, _, _)| k.as_str()).collect();
        let names: Vec<&str> = got.iter().map(|(_, n, _)| n.as_str()).collect();
        // MAX (UPPER const), Ids (type-alias), top (fn), C (class), C.method (method with its
        // own `methods` kind). Excluded: lower (not UPPER), nested (closure inside a function).
        assert!(names.contains(&"MAX") && names.contains(&"Ids"));
        assert!(names.contains(&"top") && names.contains(&"C"));
        assert!(names.contains(&"C.method"));
        assert!(!names.contains(&"lower") && !names.contains(&"nested") && !names.contains(&"method"));
        assert_eq!(kinds.iter().filter(|k| **k == "functions").count(), 1);
        assert_eq!(kinds.iter().filter(|k| **k == "classes").count(), 1);
        assert_eq!(kinds.iter().filter(|k| **k == "methods").count(), 1);
    }

    #[test]
    fn class_methods_emitted_with_qualified_names_and_methods_kind() {
        let src = "class Foo:\n    def __init__(self, x):\n        self.x = x\n\n    async def fetch(self):\n        return self.x\n";
        let got = triples(src);
        // Sync and async methods of `Foo` alike, both with kind = "methods".
        let methods: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        assert!(methods.contains(&"Foo.__init__"), "got methods: {methods:?}");
        assert!(methods.contains(&"Foo.fetch"), "got methods: {methods:?}");
        // The decorator-stripping rule from `keyword_start` still applies to methods.
        let init = got.iter().find(|(_, n, _)| n == "Foo.__init__").expect("init method");
        assert!(init.2.starts_with("def "), "method text should start at def, got: {:?}", init.2);
    }

    #[test]
    fn nested_class_methods_use_chained_parent_names() {
        // Each method body is non-trivial (one real statement) so the trivial-body filter
        // leaves them all in — the point of this test is the name-chaining for nested classes.
        // BinOps in bodies so the trivial-body filter (which now also catches `return <atom>`)
        // doesn't strip these — the point of this test is name-chaining for nested classes.
        let src = "class Outer:\n    def outer_m(self):\n        return self.x + 1\n\n    class Inner:\n        def inner_m(self):\n            return self.x + 2\n\n        class Deep:\n            def deep_m(self):\n                return self.x + 3\n";
        let got = triples(src);
        let methods: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        // Direct, one-level, and two-levels-deep all surface with full parent chains.
        assert!(methods.contains(&"Outer.outer_m"), "got methods: {methods:?}");
        assert!(methods.contains(&"Outer.Inner.inner_m"), "got methods: {methods:?}");
        assert!(methods.contains(&"Outer.Inner.Deep.deep_m"), "got methods: {methods:?}");
    }

    #[test]
    fn single_token_return_dispatch_overrides_are_skipped() {
        // Virtual-dispatch override pattern: dozens of unrelated classes each provide a one-line
        // `return <atom>` specialisation. Bodies collide trivially in alpha-renamed canonical
        // form, but they're per-method semantics, not a refactoring target. Skip them.
        // Methods with structural bodies (Attribute / Call / List) must STAY.
        let src = concat!(
            "class A:\n",
            "    def is_x(self) -> bool:\n",
            "        return False\n",
            "    def default(self):\n",
            "        return None\n",
            "    def name(self) -> str:\n",
            "        return \"a\"\n",
            "    def num(self) -> int:\n",
            "        return 0\n",
            "    def empty(self):\n",
            "        return\n",
            "    def self_(self):\n",
            "        return self\n",
            "    def get_x(self):\n",
            "        return self._x\n",
            "    def sources(self):\n",
            "        return [self._x]\n",
            "    def call(self):\n",
            "        return self.parent.fn()\n",
        );
        let got = triples(src);
        let methods: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        // Skipped — trivial atom returns.
        for skipped in ["A.is_x", "A.default", "A.name", "A.num", "A.empty", "A.self_"] {
            assert!(!methods.contains(&skipped), "{skipped} should be skipped, got: {methods:?}");
        }
        // Kept — structural body (Attribute / List / Call).
        for kept in ["A.get_x", "A.sources", "A.call"] {
            assert!(methods.contains(&kept), "{kept} should be kept, got: {methods:?}");
        }
    }

    #[test]
    fn raise_not_implemented_stubs_are_skipped() {
        // ABC-style abstract methods with `raise NotImplementedError` are placeholders — same
        // role as `...`/`pass`/docstring stubs. Without filtering they collide with each other
        // across interfaces (same name, identical canonical bodies).
        let src = concat!(
            "class IFoo:\n",
            "    def do(self, x: int) -> int:\n",
            "        raise NotImplementedError\n",
            "\n",
            "    def go(self, x: int) -> int:\n",
            "        raise NotImplementedError('subclass me')\n",
            "\n",
            "    def real(self, x: int) -> int:\n",
            "        return x + 1\n",
        );
        let got = triples(src);
        let methods: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        // Only `real` survives — both `raise NotImplementedError` forms are skipped.
        assert_eq!(methods, vec!["IFoo.real"], "got: {methods:?}");
    }

    #[test]
    fn overload_and_abstract_stubs_are_skipped_real_impl_kept() {
        // `@overload` stubs and `@abstractmethod` declarations have trivial bodies (`...` or
        // `pass`, possibly with a docstring). Each repeats the method name; without filtering,
        // the name-gated pass clusters them as duplicates. The real implementation has a
        // non-trivial body and must survive the filter.
        let src = concat!(
            "from typing import overload\n",
            "from abc import abstractmethod\n",
            "\n",
            "class C:\n",
            "    @overload\n",
            "    def foo(self, x: int) -> int: ...\n",
            "    @overload\n",
            "    def foo(self, x: str) -> str: ...\n",
            "    def foo(self, x):\n",
            "        return x + 1\n",
            "\n",
            "    @abstractmethod\n",
            "    def bar(self):\n",
            "        \"\"\"abstract.\"\"\"\n",
            "\n",
            "    @abstractmethod\n",
            "    def baz(self):\n",
            "        pass\n",
            "\n",
            "    def qux(self):\n",
            "        ...\n",
        );
        let got = triples(src);
        let methods: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        // Only the real `foo` (with the actual body) is surfaced. `bar` / `baz` / `qux` are all
        // trivial-body and filtered out alongside the two overload stubs.
        assert_eq!(methods, vec!["C.foo"], "expected only the real impl, got: {methods:?}");
        let foo = got.iter().find(|(_, n, _)| n == "C.foo").expect("real foo");
        assert!(foo.2.contains("return x + 1"), "expected the real impl, got: {:?}", foo.2);
    }

    #[test]
    fn loc_and_args_are_populated_from_original_source() {
        let src = concat!(
            "def free(a, b, *, c=3):\n",
            "    x = a + b\n",
            "    y = x * c\n",
            "    return y\n",
            "\n",
            "class C:\n",
            "    def method(self, x, y):\n",
            "        if x > y:\n",
            "            return x\n",
            "        return y\n",
        );
        let defs = super::module_defs_from(src, "<test>");
        let free = defs.iter().find(|d| d.name == "free").expect("free fn");
        // `def free(a, b, *, c=3):` + 3 body lines = 4 non-blank lines; args = a + b + c.
        assert_eq!(free.loc, 4, "free loc: {}", free.loc);
        assert_eq!(free.args, 3, "free args: {}", free.args);

        let method = defs.iter().find(|d| d.name == "C.method").expect("method");
        // `loc` covers the original (pre-strip) text of 4 non-blank lines; `args` includes
        // `self` (the user-visible count, 3 total) even though the canonical text drops it.
        assert_eq!(method.loc, 4, "method loc: {}", method.loc);
        assert_eq!(method.args, 3, "method args (incl self): {}", method.args);
    }

    #[test]
    fn method_receiver_is_stripped_from_text() {
        // `self` / `cls` are erased from the emitted method text so cross-name canonicals line
        // up with free-function bodies. Single-param `self` and trailing-comma forms both work.
        let src = concat!(
            "class C:\n",
            "    def one(self):\n",
            "        return self.x + 1\n",
            "\n",
            "    def two(self, x):\n",
            "        return x + 1\n",
            "\n",
            "    @classmethod\n",
            "    def three(cls, x):\n",
            "        return x * 2\n",
            "\n",
            "    @staticmethod\n",
            "    def four(x):\n",
            "        return x * 3\n",
        );
        let got = triples(src);
        let body_of = |name: &str| {
            got.iter().find(|(_, n, _)| n == name).map(|(_, _, t)| t.clone()).expect("method")
        };
        // `self` / `cls` are gone from the signature line.
        assert!(body_of("C.one").starts_with("def one():"), "got: {:?}", body_of("C.one"));
        assert!(body_of("C.two").starts_with("def two(x):"), "got: {:?}", body_of("C.two"));
        assert!(body_of("C.three").starts_with("def three(x):"), "got: {:?}", body_of("C.three"));
        // `@staticmethod`-shaped first param (named `x`, not `self`/`cls`) is left alone.
        assert!(body_of("C.four").starts_with("def four(x):"), "got: {:?}", body_of("C.four"));
    }

    #[test]
    fn property_setter_and_deleter_get_role_suffix() {
        // Getter keeps the bare name; setter and deleter get `.setter` / `.deleter` suffixes
        // so they don't collide with the getter in the name-gated pass. Cross-class accessor
        // duplicates still cluster — `Foo.value.setter` ↔ `Bar.value.setter` matches by xname.
        let src = concat!(
            "class C:\n",
            "    @property\n",
            "    def value(self):\n",
            "        return self._x\n",
            "\n",
            "    @value.setter\n",
            "    def value(self, v):\n",
            "        self._x = v\n",
            "\n",
            "    @value.deleter\n",
            "    def value(self):\n",
            "        del self._x\n",
        );
        let got = triples(src);
        let methods: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        assert!(methods.contains(&"C.value"), "getter: {methods:?}");
        assert!(methods.contains(&"C.value.setter"), "setter: {methods:?}");
        assert!(methods.contains(&"C.value.deleter"), "deleter: {methods:?}");
    }

    #[test]
    fn property_with_real_logic_is_kept() {
        // A `@property` getter with actual logic — not just `return self._x` but anything
        // beyond a trivial body — must still be surfaced; the trivial-body filter only targets
        // bodies that are literally `...` / `pass` / docstring-only.
        let src = concat!(
            "class C:\n",
            "    @property\n",
            "    def value(self):\n",
            "        if self._cached is None:\n",
            "            self._cached = self._compute()\n",
            "        return self._cached\n",
            "\n",
            "    @value.setter\n",
            "    def value(self, v):\n",
            "        self._cached = v\n",
            "        self._dirty = True\n",
        );
        let got = triples(src);
        // Both getter and setter survive — getter as `C.value`, setter as `C.value.setter`.
        let names: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        assert!(names.contains(&"C.value"), "getter: {names:?}");
        assert!(names.contains(&"C.value.setter"), "setter: {names:?}");
    }

    #[test]
    fn class_hidden_inside_function_does_not_surface_methods() {
        // A class defined inside a function body is not a top-level statement; its methods must
        // stay invisible, to preserve the "top-level only" principle for non-class scopes.
        let src = "def factory():\n    class Hidden:\n        def helper(self):\n            return 1\n    return Hidden\n";
        let got = triples(src);
        let methods: Vec<&str> =
            got.iter().filter(|(k, _, _)| k == "methods").map(|(_, n, _)| n.as_str()).collect();
        assert!(methods.is_empty(), "no methods expected, got: {methods:?}");
    }

    #[test]
    fn decorated_method_text_excludes_decorators() {
        let src = "class C:\n    @staticmethod\n    def helper(x):\n        return x + 1\n";
        let got = triples(src);
        let helper = got.iter().find(|(_, n, _)| n == "C.helper").expect("helper method");
        assert!(helper.2.starts_with("def "), "decorated method text should start at def, got: {:?}", helper.2);
    }

    #[test]
    fn function_text_excludes_decorators() {
        // Body must be non-trivial — `return <atom>` is now part of the trivial-body filter,
        // so use `return value + 1` (BinOp) to ensure the function actually surfaces.
        let got = triples("import functools\n\n\n@functools.cache\ndef memo(x):\n    return x + 1\n");
        let func = got.iter().find(|(k, _, _)| k == "functions").expect("a function");
        assert!(func.2.starts_with("def "), "text should start at def, got: {:?}", func.2);
    }

    #[test]
    fn pep695_and_modern_syntax_file_is_scanned() {
        // A file mixing PEP 695 `type` aliases, PEP 695 generics, and PEP 701 f-strings — the
        // exact shapes rustpython choked on, silently dropping every def in the file.
        let src = "type Alias = list[int]\n\n\ndef worker[T](x: T) -> T:\n    msg = f\"got {x['k']}\"\n    return x\n\n\nclass Repo[T]:\n    pass\n";
        let names: Vec<String> = module_defs_from(src, "<test>").into_iter().map(|d| d.name).collect();
        assert!(names.contains(&"Alias".to_owned()), "type alias missing: {names:?}");
        assert!(names.contains(&"worker".to_owned()), "generic fn missing: {names:?}");
        assert!(names.contains(&"Repo".to_owned()), "generic class missing: {names:?}");
    }
}
