//! Module-level definition scan for Rust — the "find-*" step of `find-dup-defs`.
//!
//! Walks each file's `syn` AST **once** and lowers every tracked definition straight to the
//! engine's [`Def`], computing its canonical strings off the AST node (no re-parse). Surfaces:
//!
//! * **functions** — free `fn`s, qualified by their in-file module path (`a::b::foo`).
//! * **methods** — `impl` methods and trait default methods, qualified `Type::method` /
//!   `Trait::method` (with any module prefix). The `self` receiver is dropped in the canonical
//!   so a method lines up with an equivalent free function for the cross-name pass.
//! * **classes** — `struct` / `enum` / `union` (body-bearing nominal types).
//! * **interfaces** — `trait` (its associated-item shape).
//! * **constants** — `const` / `static` with an `UPPER_SNAKE` name.
//! * **type-aliases** — `type X = ...`.
//!
//! Descends into inline `mod foo { ... }` (qualifying with the module path) but never into a
//! function body, so an `impl` nested in a `fn` stays invisible — the "top-level only" rule the
//! Python / TypeScript frontends also follow. `macro_rules!` is not yet surfaced.
//!
//! Attributes (`#[derive(...)]`, `#[inline]`, doc comments) are excluded from a def's text (the
//! range starts at the `pub`/`fn`/`struct`/… keyword) and never enter the canonical.
#![allow(clippy::needless_raw_string_hashes)] // test fixtures keep `r#"..."#` for visual consistency

use std::sync::Arc;

use dup_defs_core::{Analysis, CanonDialect, Def, Facets, KindSpec, LineMap, ScanOpts};
use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::{Attribute, Block, Expr, FnArg, ImplItem, Item, Signature, Stmt, TraitItem, Type};

use crate::canon::{
    analyze_impl_fn, analyze_item_fn, analyze_trait_fn, enum_canon, struct_canon, trait_canon,
    union_canon, used_names, AnalyzedFn,
};
use dup_defs_core::{count_loc, is_upper_snake};

use crate::frontend::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};

/// Number of value parameters, excluding a `self` receiver (the analog of TS not counting
/// `this`).
fn count_args(sig: &Signature) -> usize {
    sig.inputs.iter().filter(|a| matches!(a, FnArg::Typed(_))).count()
}

/// Byte offset of the def keyword (`pub`/`fn`/`struct`/…) — the def text *excluding* attributes.
/// From just after the last attribute, skip whitespace and `//` / `/* */` comments to the first
/// real token. With no attributes the item span already starts at the keyword.
fn keyword_start(source: &str, span_start: usize, last_attr_end: Option<usize>) -> usize {
    let Some(mut i) = last_attr_end else { return span_start };
    let bytes = source.as_bytes();
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            continue;
        }
        break;
    }
    i
}

/// `(keyword_start, end)` byte range of a def given its attributes and full span.
fn def_range(source: &str, attrs: &[Attribute], span: proc_macro2::Span) -> (usize, usize) {
    let range = span.byte_range();
    let last_attr_end = attrs.last().map(|a| a.span().byte_range().end);
    (keyword_start(source, range.start, last_attr_end), range.end)
}

/// True when a fn/method body is a no-op, single-token stub, or one-line formatter/predicate —
/// the shapes that aren't refactor clusters, mirroring (and extending, for Rust idioms) the
/// Python / TypeScript trivial-body filters: empty `{}`, a single literal / bare identifier tail,
/// `return <atom>`, or a single trivial macro ([`is_trivial_macro`] — `todo!`/`panic!` stubs plus
/// one-line `write!`/`writeln!` formatters and `matches!` predicates). Field access (`self.x`),
/// calls, and anything structural fall through (still compared).
fn is_trivial_block(block: &Block) -> bool {
    block.stmts.iter().all(|s| match s {
        Stmt::Expr(e, _) => is_trivial_expr(e),
        // `todo!();` is the stub `todo!()` is, with a semicolon.
        Stmt::Macro(m) => is_trivial_macro(&m.mac),
        _ => false,
    })
}

fn is_trivial_expr(e: &Expr) -> bool {
    match e {
        Expr::Lit(_) => true,
        // `match *self {}` — the body a trait method gets on an uninhabited type (`Infallible`, an
        // empty enum): there is no value to run it on, the `unreachable!()` the type system writes.
        Expr::Match(m) => m.arms.is_empty(),
        // A bare path (`x`, `self`, `None`) — single segment, no leading `::`.
        Expr::Path(p) => p.qself.is_none() && p.path.segments.len() == 1 && p.path.leading_colon.is_none(),
        Expr::Macro(m) => is_trivial_macro(&m.mac),
        Expr::Return(r) => match r.expr.as_deref() {
            None | Some(Expr::Lit(_) | Expr::Path(_)) => true,
            Some(Expr::Macro(m)) => is_trivial_macro(&m.mac),
            Some(_) => false,
        },
        _ => false,
    }
}

/// A macro invocation that, as a function's whole body, marks it trivial — not a refactor
/// cluster. Covers stub macros (`todo!` / `unimplemented!` / `panic!` / `unreachable!`), one-line
/// formatters (`write!` / `writeln!` — a `Display`/`Debug` impl that's a single `write!(f, …)`),
/// and one-line predicates (`matches!`). These dominated the cross-name false positives on the
/// Rust corpora (`*::fmt` ×21 in tokio, `is_*` ×14 in actix-web).
fn is_trivial_macro(mac: &syn::Macro) -> bool {
    let last = mac.path.segments.last().map(|s| s.ident.to_string());
    matches!(
        last.as_deref(),
        Some("todo" | "unimplemented" | "panic" | "unreachable" | "write" | "writeln" | "matches")
    )
}

/// Last path segment of a type, for qualifying `impl` methods (`impl Foo<T>` → `Foo`).
fn type_name(ty: &Type) -> String {
    match ty {
        Type::Path(p) => p.path.segments.last().map_or_else(|| "<ty>".to_owned(), |s| s.ident.to_string()),
        Type::Reference(r) => type_name(&r.elem),
        _ => "<ty>".to_owned(),
    }
}

/// True when an attribute hands the function to something that calls it — a test harness, a runtime,
/// a router — so the calls the corpus shows are not all the calls there are.
fn registered(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        let name = a.path().segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
        !matches!(
            name.as_str(),
            "doc" | "inline" | "allow" | "warn" | "deny" | "forbid" | "expect" | "cfg" | "cfg_attr" | "must_use"
                | "cold" | "track_caller" | "deprecated" | "skip" | "automatically_derived"
        )
    })
}

fn qualify(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}::{name}")
    }
}

/// What a definition contributes beyond its location: its canonical forms and, for a callable, the
/// facets the perspective passes read.
///
/// One value rather than three parameters on [`Builder::push`], because the three are not
/// independent — a body-only kind has a cluster canonical and nothing else, a callable has all of
/// them — and passing them separately means every non-callable call site spells out two `None`s and
/// a `default()` that say nothing.
struct DefCanon {
    cluster: Option<String>,
    analysis: Option<Analysis>,
    facets: Facets,
}

impl DefCanon {
    /// A body-bearing kind clustered by its canonical text alone (struct, enum, trait).
    fn body(cluster: Option<String>) -> Self {
        Self { cluster, analysis: None, facets: Facets::default() }
    }

    /// A callable: canonical forms, statement stream, and the paths it reaches.
    fn callable(analyzed: AnalyzedFn, block: &syn::Block, imports: &HashMap<String, Arc<str>>) -> Self {
        let AnalyzedFn { cluster_canonical, xname_canonical, type3_lines, type_refs, type3_units, statements, size } = analyzed;
        Self {
            cluster: Some(cluster_canonical),
            analysis: Some(Analysis {
                xname_canonical,
                type3_lines,
                type_refs,
                type3_units,
                size,
                canon_dialect: CanonDialect::Rust,
            }),
            facets: Facets { statements, reaches: reached_paths(block, imports), ..Facets::default() },
        }
    }
}

/// The dotted paths one callable reaches: for every path head it mentions that the file `use`d, the
/// whole path that name stands for.
///
/// Resolved here rather than in the engine because only the frontend knows what its language calls a
/// path and what a `use` binds. What the engine gets is the answer, in the one dotted form every
/// language normalizes to.
fn reached_paths(block: &syn::Block, imports: &HashMap<String, Arc<str>>) -> Vec<Arc<str>> {
    if imports.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<Arc<str>> =
        used_names(block).iter().filter_map(|name| imports.get(name).map(Arc::clone)).collect();
    out.sort();
    out.dedup();
    out
}

/// One callable as the decisions pass reads it (see [`crate::decides`]), with its identity: what the
/// corpus's calls name it by. `prefix` is the inline module it stands in, `self_ty` what `Self`
/// names; `open` — reached by calls the corpus cannot show.
#[allow(clippy::too_many_arguments)] // each is a different fact about the one definition
fn decided(
    canon: &mut DefCanon,
    scope: &crate::decides::FileScope,
    prefix: &str,
    name: &str,
    sig: &Signature,
    block: &Block,
    self_ty: Option<&str>,
    open: bool,
) {
    let read = crate::decides::decides(sig, block, scope, &scope.here(prefix), self_ty, open);
    let facets = &mut canon.facets;
    facets.decides = read.atoms;
    facets.identity = Some(scope.identity(name));
    facets.params = read.params;
    facets.calls = read.calls;
    facets.mentions = read.mentions;
    facets.loose_keywords = read.loose;
}

/// One builder so every push site stays uniform.
struct Builder<'a> {
    source: &'a str,
    lines: &'a LineMap<'a>,
    file: &'a Arc<str>,
    /// Whether this run reads per-def facets (statement stream, reached paths, decides/identity).
    /// Off, every perspective-pass input is skipped and the defs carry empty facets.
    facets: bool,
    /// What each name this file `use`s stands for — resolved once, read by every callable. Empty
    /// when `facets` is off: reached paths are a facets-only fact.
    imports: HashMap<String, Arc<str>>,
    /// Which lenses this run asked for; empty when it did not ask.
    lenses: Vec<dup_defs_core::lens::Lens>,
    /// Names the file itself introduced — what every lens erases before it looks.
    file_names: std::collections::HashSet<String>,
    /// What the file's names stand for, as the decisions pass writes them. `None` when `facets` is
    /// off: the file walk that builds it is skipped too.
    decides: Option<crate::decides::FileScope>,
    out: &'a mut Vec<Def>,
}

impl Builder<'_> {
    /// One lens [`Def`] for this definition, when the run asked for lenses and the projection has
    /// enough to say. The record, the thinness floor and the scoring are the shared module's.
    fn push_lens(
        &mut self,
        name: &str,
        attrs: &[Attribute],
        span: proc_macro2::Span,
        facts: &dup_defs_core::lens::LensFacts,
    ) {
        if self.lenses.is_empty() {
            return;
        }
        let (start, end) = def_range(self.source, attrs, span);
        let (line, col) = self.lines.loc0(start);
        let loc = count_loc(&self.source[start..end]);
        let site = dup_defs_core::lens::DefSite { name, file: self.file, line, col, loc };
        if let Some(def) =
            dup_defs_core::lens::lens_def("rs", CanonDialect::Rust, &site, &self.lenses, facts)
        {
            self.out.push(def);
        }
    }

    fn push(
        &mut self,
        kind: &'static KindSpec,
        name: String,
        attrs: &[Attribute],
        span: proc_macro2::Span,
        args: usize,
        canon: DefCanon,
    ) {
        let DefCanon { cluster: cluster_canonical, analysis, facets } = canon;
        let (start, end) = def_range(self.source, attrs, span);
        let (line, col) = self.lines.loc0(start);
        let text_orig = self.source[start..end].to_owned();
        let loc = count_loc(&text_orig);
        self.out.push(Def {
            lang: "rs",
            kind,
            name,
            file: Arc::clone(self.file),
            line,
            col,
            loc,
            args,
            text_orig,
            cluster_canonical,
            analysis,
            thickness: None,
            facets,
        });
    }
}

fn walk_items(items: &[Item], prefix: &str, b: &mut Builder) {
    for item in items {
        walk_item(item, prefix, b);
    }
}

#[allow(clippy::too_many_lines)] // one match over the Item variants reads better straight-through
fn walk_item(item: &Item, prefix: &str, b: &mut Builder) {
    match item {
        Item::Fn(f) => {
            if is_trivial_block(&f.block) {
                return;
            }
            if !b.lenses.is_empty() {
                let scope = crate::canon::scope_canonical(&f.sig, &f.block, &b.file_names);
                let facts =
                    crate::lenses::callable_facts(&f.sig, &f.block, &f.attrs, &b.file_names, scope);
                b.push_lens(&qualify(prefix, &f.sig.ident.to_string()), &f.attrs, f.span(), &facts);
            }
            let name = qualify(prefix, &f.sig.ident.to_string());
            let mut canon = DefCanon::callable(analyze_item_fn(f, b.facets), &f.block, &b.imports);
            if let Some(scope) = &b.decides {
                decided(&mut canon, scope, prefix, &name, &f.sig, &f.block, None, registered(&f.attrs));
            }
            b.push(
                &FUNCTIONS,
                name,
                &f.attrs,
                f.span(),
                count_args(&f.sig),
                canon,
            );
        }
        Item::Struct(s) => {
            if !b.lenses.is_empty() {
                let facts = crate::lenses::struct_facts(s, &b.file_names);
                b.push_lens(&qualify(prefix, &s.ident.to_string()), &s.attrs, s.span(), &facts);
            }
            let name = qualify(prefix, &s.ident.to_string());
            let mut canon = DefCanon::body(Some(struct_canon(s)));
            // A struct is a record: its literal is a constructor, called with every field by name.
            if let Some(scope) = &b.decides {
                canon.facets.identity = Some(scope.identity(&name));
                if prefix.is_empty() {
                    if let Some(fields) = scope.record(&s.ident.to_string()) {
                        canon.facets.params = fields.iter().map(|(f, _)| Arc::from(f.as_str())).collect();
                    }
                }
            }
            b.push(&CLASSES, name, &s.attrs, s.span(), 0, canon);
        }
        Item::Enum(e) => {
            if !b.lenses.is_empty() {
                let facts = crate::lenses::enum_facts(e, &b.file_names);
                b.push_lens(&qualify(prefix, &e.ident.to_string()), &e.attrs, e.span(), &facts);
            }
            b.push(&CLASSES, qualify(prefix, &e.ident.to_string()), &e.attrs, e.span(), 0, DefCanon::body(Some(enum_canon(e))));
        }
        Item::Union(u) => {
            b.push(&CLASSES, qualify(prefix, &u.ident.to_string()), &u.attrs, u.span(), 0, DefCanon::body(Some(union_canon(u))));
        }
        Item::Trait(t) => {
            b.push(&INTERFACES, qualify(prefix, &t.ident.to_string()), &t.attrs, t.span(), 0, DefCanon::body(Some(trait_canon(t))));
            let owner = qualify(prefix, &t.ident.to_string());
            for ti in &t.items {
                if let TraitItem::Fn(tf) = ti {
                    if tf.default.is_none() || is_trivial_block(tf.default.as_ref().unwrap()) {
                        continue;
                    }
                    let body = tf.default.as_ref().expect("checked above");
                    if !b.lenses.is_empty() {
                        let scope = crate::canon::scope_canonical(&tf.sig, body, &b.file_names);
                        let facts =
                            crate::lenses::callable_facts(&tf.sig, body, &tf.attrs, &b.file_names, scope);
                        b.push_lens(&format!("{owner}::{}", tf.sig.ident), &tf.attrs, tf.span(), &facts);
                    }
                    let name = format!("{owner}::{}", tf.sig.ident);
                    let mut canon =
                        DefCanon::callable(analyze_trait_fn(tf, b.facets).expect("has a default body"), body, &b.imports);
                    // A trait's method is called through the trait, by whatever holds an implementor.
                    if let Some(scope) = &b.decides {
                        decided(&mut canon, scope, prefix, &name, &tf.sig, body, None, true);
                    }
                    b.push(
                        &METHODS,
                        name,
                        &tf.attrs,
                        tf.span(),
                        count_args(&tf.sig),
                        canon,
                    );
                }
            }
        }
        Item::Impl(im) => {
            let self_ty = type_name(&im.self_ty);
            let owner = qualify(prefix, &self_ty);
            for ii in &im.items {
                if let ImplItem::Fn(f) = ii {
                    if is_trivial_block(&f.block) {
                        continue;
                    }
                    if !b.lenses.is_empty() {
                        let scope = crate::canon::scope_canonical(&f.sig, &f.block, &b.file_names);
                        let facts =
                            crate::lenses::callable_facts(&f.sig, &f.block, &f.attrs, &b.file_names, scope);
                        b.push_lens(&format!("{owner}::{}", f.sig.ident), &f.attrs, f.span(), &facts);
                    }
                    let name = format!("{owner}::{}", f.sig.ident);
                    let mut canon = DefCanon::callable(analyze_impl_fn(f, b.facets), &f.block, &b.imports);
                    let open = im.trait_.is_some() || registered(&f.attrs);
                    if let Some(scope) = &b.decides {
                        decided(&mut canon, scope, prefix, &name, &f.sig, &f.block, Some(&self_ty), open);
                    }
                    b.push(
                        &METHODS,
                        name,
                        &f.attrs,
                        f.span(),
                        count_args(&f.sig),
                        canon,
                    );
                }
            }
        }
        Item::Const(c) if is_upper_snake(&c.ident.to_string()) => {
            b.push(&CONSTANTS, qualify(prefix, &c.ident.to_string()), &c.attrs, c.span(), 0, DefCanon::body(None));
        }
        Item::Static(s) if is_upper_snake(&s.ident.to_string()) => {
            b.push(&CONSTANTS, qualify(prefix, &s.ident.to_string()), &s.attrs, s.span(), 0, DefCanon::body(None));
        }
        Item::Type(t) => {
            b.push(&TYPE_ALIASES, qualify(prefix, &t.ident.to_string()), &t.attrs, t.span(), 0, DefCanon::body(None));
        }
        Item::Mod(m) => {
            if let Some((_, items)) = &m.content {
                let inner = qualify(prefix, &m.ident.to_string());
                walk_items(items, &inner, b);
            }
        }
        _ => {}
    }
}

/// Scan one Rust source string → its definitions as [`Def`]s with canon precomputed. Returns an
/// empty vec if the file doesn't parse (syn is not error-recovering — a single bad file drops
/// out rather than poisoning the run).
#[must_use]
pub fn scan_source(source: &str, file: &Arc<str>, opts: &ScanOpts) -> Vec<Def> {
    let Ok(ast) = syn::parse_file(source) else { return Vec::new() };
    let lines = LineMap::new(source);
    let mut out = Vec::new();
    let facets = opts.facets;
    // The file's `use` table, resolved once: what each bound name stands for. Facets-only data —
    // skipped when no pass reads reached paths.
    let imports: HashMap<String, Arc<str>> = if facets {
        crate::canon::file_imports(&ast).into_iter().collect()
    } else {
        HashMap::new()
    };
    let lenses = dup_defs_core::lens::enabled_lenses(opts);
    // Every lens reads the file's own bound names, so they are collected once per file.
    let file_names = (!lenses.is_empty()).then(|| crate::lenses::file_bound_names(&ast));
    let mut b = Builder {
        source,
        lines: &lines,
        file,
        facets,
        imports,
        lenses,
        file_names: file_names.unwrap_or_default(),
        decides: facets.then(|| crate::decides::FileScope::new(file, &ast)),
        out: &mut out,
    };
    walk_items(&ast.items, "", &mut b);
    // Collapse `#[cfg(...)]`-gated siblings: two items with the same (kind, qualified name) in one
    // file only compile when they're mutually-exclusive `cfg` alternatives (`#[cfg(unix)] fn x`
    // + `#[cfg(windows)] fn x`, the `BLOCK_CAP` const ×3 under target/loom cfgs) — one logical
    // definition, not a duplicate. Keep the first; cross-file duplicates (separate scans) are
    // untouched, so genuine cross-file copy-paste still clusters.
    let mut seen = std::collections::HashSet::new();
    out.retain(|d| seen.insert((d.kind.id, d.name.clone())));
    out
}

#[cfg(test)]
mod tests {
    use super::scan_source;
    use dup_defs_core::ScanOpts;
    use std::sync::Arc;

    fn defs(src: &str) -> Vec<(String, String)> {
        let f: Arc<str> = Arc::from("t.rs");
        scan_source(src, &f, &ScanOpts::default()).into_iter().map(|d| (d.kind.id.to_owned(), d.name)).collect()
    }

    fn names_of_kind(src: &str, kind: &str) -> Vec<String> {
        defs(src).into_iter().filter(|(k, _)| k == kind).map(|(_, n)| n).collect()
    }

    fn lens_record(src: &str, name: &str) -> Vec<String> {
        let f: Arc<str> = Arc::from("t.rs");
        let kinds = vec!["lenses".to_owned()];
        let opts = ScanOpts { kinds: Some(&kinds), facets: true };
        scan_source(src, &f, &opts)
            .into_iter()
            .find(|d| d.kind.id == "lenses" && d.name == name)
            .map(|d| d.analysis.expect("lens analysis").type3_lines)
            .unwrap_or_default()
    }

    #[test]
    fn lenses_are_absent_unless_the_run_asks_for_them() {
        let src = "fn f(x: u8) -> u8 {\n    if x > 0 {\n        g(x);\n    }\n    x\n}\n";
        let f: Arc<str> = Arc::from("t.rs");
        let defs = scan_source(src, &f, &ScanOpts::default());
        assert!(defs.iter().all(|d| d.kind.id != "lenses"), "the default run stays byte-identical");
    }

    #[test]
    fn a_rust_callable_projects_through_every_lens_it_can_answer() {
        let src = "fn f(x: u8) -> Result<u8, MyError> {\n    let _guard = lock();\n    for i in 0..x {\n        emit(i);\n    }\n    if x == 0 {\n        return Err(MyError::Empty);\n    }\n    Ok(x)\n}\n";
        let facts = lens_record(src, "f");
        assert!(facts.contains(&"outgoing:emit".to_owned()), "{facts:?}");
        assert!(facts.contains(&"effects:lock".to_owned()), "{facts:?}");
        assert!(facts.contains(&"control:for".to_owned()), "{facts:?}");
        assert!(facts.contains(&"control:+return".to_owned()), "nesting rides on the tag: {facts:?}");
        // The path is kept whole: the enum is the failure family, the variant the failure.
        assert!(facts.contains(&"failures:MyError::Empty".to_owned()), "{facts:?}");
        assert!(facts.contains(&"signature:ret:Result".to_owned()), "{facts:?}");
        assert!(facts.iter().any(|f| f.starts_with("scope:")), "the widest rung rides as one fact: {facts:?}");
    }

    #[test]
    fn a_binding_never_read_again_is_what_rust_holds_open() {
        // Rust has no `with`. Its answer to "what does it hold open" is a guard — a binding whose
        // value is never read again and whose only job is to live until the scope ends. Structural,
        // not a list of blessed names: the same call bound to a name that IS read is not a guard.
        let guard = "fn f() -> u8 {\n    let _g = lock();\n    0\n}\n";
        let used = "fn f() -> u8 {\n    let g = lock();\n    g.value()\n}\n";
        assert!(lens_record(guard, "f").contains(&"resources:lock".to_owned()), "{:?}", lens_record(guard, "f"));
        assert!(!lens_record(used, "f").contains(&"resources:lock".to_owned()), "{:?}", lens_record(used, "f"));
    }

    #[test]
    fn a_struct_declares_a_shape_as_a_set() {
        let a = "#[derive(Clone)]\nstruct S { a: u8, b: Vec<String> }\n";
        let b = "#[derive(Clone)]\nstruct S { b: Vec<String>, a: u8 }\n";
        // A declaration is a set: the same struct with its fields reordered is the same struct.
        assert_eq!(lens_record(a, "S"), lens_record(b, "S"));
        assert!(lens_record(a, "S").contains(&"decorators:derive".to_owned()));
    }

    #[test]
    fn a_callable_reports_its_statement_stream_with_nesting() {
        let src = "fn f(xs: &[u8]) -> u8 {\n    for x in xs {\n        g(*x);\n    }\n    0\n}\n";
        let f: Arc<str> = Arc::from("t.rs");
        let defs = scan_source(src, &f, &ScanOpts::default());
        let d = defs.iter().find(|d| d.name == "f").expect("f");
        let depths: Vec<u16> = d.facets.statements.iter().map(|s| s.depth).collect();
        // Header at 0, body at 1, and the call inside the loop at 2. Rust's control flow is
        // expressions, so the walk descends into them rather than reading the whole `for` — loop
        // and body together — as one line.
        assert_eq!(depths, vec![0, 1, 2, 1], "statements: {:?}", d.facets.statements);
        assert!(d.facets.statements[0].line.starts_with("Func("), "the head is the definition itself");
        assert!(d.facets.statements[1].line.starts_with("For("), "the loop contributes a header");
    }

    #[test]
    fn the_type3_lines_are_the_stream_and_their_parts_are_units() {
        // An edit inside a loop costs the statement it touched, not the loop; and the header leads,
        // because the Type-3 pass reads every function's first line as its interface.
        // Written in its normal form (`0 < r.v`, not `r.v > 0`), so the lines and the stream agree.
        let src = "fn f(rows: &[Row]) -> usize {\n    for r in rows {\n        if r.ok && 0 < r.v {\n            out.push(r.render(true));\n        }\n    }\n    0\n}\n";
        let f: Arc<str> = Arc::from("t.rs");
        let defs = scan_source(src, &f, &ScanOpts::default());
        let d = defs.iter().find(|d| d.name == "f").expect("f");
        let a = d.analysis.as_ref().expect("analysis");
        let stream: Vec<&str> = d.facets.statements.iter().map(|s| s.line.as_str()).collect();
        // The body is the stream's; the header is written the way the Python frontend writes its
        // own, the types as slots and reported apart — while the stream keeps them as spelled.
        assert_eq!(a.type3_lines[1..].iter().map(String::as_str).collect::<Vec<_>>(), stream[1..]);
        assert_eq!(a.type3_lines[0], "fn _fn(_v0: TyRef('', TySlice(Ty(_T0, )))) -> Ty(_T1, )");
        assert_eq!(a.type_refs, ["Row", "usize"]);
        assert!(stream[0].contains("Ty('Row', )"), "{stream:?}");
        let units_of = |line: u32| -> Vec<&str> {
            a.type3_units.iter().filter(|(l, _)| *l == line).map(|(_, u)| u.as_str()).collect()
        };
        // The `if` header: the comparison and the field reads are parts; the whole test is the line.
        let test_parts = units_of(2);
        assert!(test_parts.iter().any(|u| u.starts_with("Bin(Int('0'), '<', Field(")), "{test_parts:?}");
        assert!(!test_parts.iter().any(|u| u.contains("'&&'")), "the test itself is the line: {test_parts:?}");
        // The push: the inner method call is a part of the push's own line.
        assert!(units_of(3).iter().any(|u| u.starts_with("Method(Path('_v1'), 'render'")), "{:?}", units_of(3));
    }

    #[test]
    fn an_if_else_chain_keeps_its_branches_as_siblings() {
        let src = "fn f(n: u8) -> u8 {\n    if n > 1 {\n        g();\n    } else if n > 0 {\n        h();\n    } else {\n        k();\n    }\n    n\n}\n";
        let f: Arc<str> = Arc::from("t.rs");
        let defs = scan_source(src, &f, &ScanOpts::default());
        let d = defs.iter().find(|d| d.name == "f").expect("f");
        let shape: Vec<(u16, String)> = d
            .facets
            .statements
            .iter()
            .map(|s| (s.depth, s.line.split('(').next().unwrap_or("").to_owned()))
            .collect();
        // An `else if` is a sibling of its `else`, not a level deeper — what the source means, and
        // what Python's `elif` renders as, so the two languages' streams stay comparable.
        assert_eq!(
            shape,
            vec![
                (0, "Func".into()),
                (1, "If".into()),
                (2, "ExprStmt".into()),
                (1, "Else".into()),
                (1, "If".into()),
                (2, "ExprStmt".into()),
                (1, "Else".into()),
                (2, "ExprStmt".into()),
                (1, "Tail".into()),
            ],
            "statements: {:?}",
            d.facets.statements
        );
    }

    #[test]
    fn a_callable_reports_the_use_paths_it_reaches() {
        let src = "use a::b::c;\nuse d::e as z;\nuse q::*;\n\nfn f() -> u8 {\n    c()\n}\n";
        let f: Arc<str> = Arc::from("t.rs");
        let defs = scan_source(src, &f, &ScanOpts::default());
        let d = defs.iter().find(|d| d.name == "f").expect("f");
        let reaches: Vec<&str> = d.facets.reaches.iter().map(|r| &**r).collect();
        // `c` is used and its whole path reported; `z` is bound and never touched; the glob binds
        // nothing this can attribute a use to, and inventing reach for it would be a guess.
        assert_eq!(reaches, vec!["a.b.c"]);
    }

    #[test]
    fn surfaces_each_kind() {
        let src = r#"
pub const MAX_RETRIES: u32 = 5;
static GREETING: &str = "hi";
type Ids = Vec<u64>;

pub fn compute(values: &[i32], weight: i32) -> i32 {
    let mut total = 0;
    for v in values {
        total += v * weight;
    }
    total
}

pub struct Repo { store: Vec<u8> }

pub enum State { On, Off(u8) }

pub trait Fetch {
    fn get(&self, id: u64) -> u64;
    fn describe(&self) -> String {
        let n = self.get(0);
        format!("repo with {}", n)
    }
}

impl Repo {
    pub fn fetch_item(&self, id: usize) -> u8 {
        let rec = self.store[id];
        rec + 1
    }
}
"#;
        let d = defs(src);
        let has = |k: &str, n: &str| d.iter().any(|(kk, nn)| kk == k && nn == n);
        assert!(has("constants", "MAX_RETRIES"), "{d:?}");
        assert!(has("constants", "GREETING"), "{d:?}");
        assert!(has("type-aliases", "Ids"), "{d:?}");
        assert!(has("functions", "compute"), "{d:?}");
        assert!(has("classes", "Repo"), "{d:?}");
        assert!(has("classes", "State"), "{d:?}");
        assert!(has("interfaces", "Fetch"), "{d:?}");
        assert!(has("methods", "Repo::fetch_item"), "{d:?}");
        // trait default method surfaces; the bodiless `get` signature does not.
        assert!(has("methods", "Fetch::describe"), "{d:?}");
        assert!(!has("methods", "Fetch::get"), "bodiless sig should not be a method: {d:?}");
    }

    #[test]
    fn module_path_qualifies_functions_and_methods() {
        let src = "mod a {\n  pub fn helper(x: i32) -> i32 { let y = x + 1; y }\n  pub mod b {\n    pub fn helper(x: i32) -> i32 { let y = x + 2; y }\n  }\n  pub struct T;\n  impl T { pub fn run(&self, x: i32) -> i32 { let y = x + 3; y } }\n}\n";
        let fns = names_of_kind(src, "functions");
        assert!(fns.contains(&"a::helper".to_owned()), "{fns:?}");
        assert!(fns.contains(&"a::b::helper".to_owned()), "{fns:?}");
        let methods = names_of_kind(src, "methods");
        assert!(methods.contains(&"a::T::run".to_owned()), "{methods:?}");
    }

    #[test]
    fn trivial_bodies_skipped() {
        let src = r#"
fn empty() {}
fn lit() -> bool { true }
fn ident(x: i32) -> i32 { x }
fn stub() -> u32 { todo!() }
fn unimpl() -> u32 { unimplemented!() }
fn disp(f: &mut Fmt) -> Result { write!(f, "channel closed") }
fn disp2(f: &mut Fmt) -> Result { writeln!(f, "{}", self.0) }
fn pred(&self) -> bool { matches!(self, Foo::A | Foo::B) }
fn ret_macro() -> bool { return matches!(1, 1); }
fn real(x: i32) -> i32 { let y = x + 1; y * 2 }
"#;
        let fns = names_of_kind(src, "functions");
        // One-line write!/writeln!/matches! formatter & predicate bodies are dropped alongside
        // the todo!/unimplemented! stubs — only the structural body survives.
        assert_eq!(fns, vec!["real".to_owned()], "only the structural body survives: {fns:?}");
    }

    #[test]
    fn impl_nested_in_fn_not_surfaced() {
        let src = "fn factory() -> u8 {\n    struct Hidden;\n    impl Hidden { fn helper(&self) -> u8 { let x = 1; x + 1 } }\n    7\n}\n";
        let methods = names_of_kind(src, "methods");
        assert!(methods.is_empty(), "nested impl methods must not surface: {methods:?}");
        // `Hidden` struct is also inside the fn body → not surfaced.
        assert!(!defs(src).iter().any(|(k, _)| k == "classes"), "nested struct must not surface");
    }

    #[test]
    fn cfg_gated_siblings_collapse_to_one() {
        // The classic Rust pattern: one logical item defined N times under mutually-exclusive
        // cfgs. They must surface once, not as an N-member "duplicate" cluster.
        let src = concat!(
            "#[cfg(target_pointer_width = \"64\")]\npub const BLOCK_CAP: usize = 32;\n",
            "#[cfg(not(target_pointer_width = \"64\"))]\npub const BLOCK_CAP: usize = 16;\n",
            "#[cfg(unix)]\nfn platform(x: i32) -> i32 { let y = x + 1; y }\n",
            "#[cfg(windows)]\nfn platform(x: i32) -> i32 { let y = x - 1; y }\n",
        );
        let d = defs(src);
        assert_eq!(d.iter().filter(|(k, n)| k == "constants" && n == "BLOCK_CAP").count(), 1, "{d:?}");
        assert_eq!(d.iter().filter(|(k, n)| k == "functions" && n == "platform").count(), 1, "{d:?}");
    }

    #[test]
    fn lowercase_const_skipped() {
        // Only UPPER_SNAKE consts are surfaced (the constant convention).
        let src = "const lower_thing: u32 = 1;\nconst REAL_MAX: u32 = 9;\n";
        let consts = names_of_kind(src, "constants");
        assert_eq!(consts, vec!["REAL_MAX".to_owned()], "{consts:?}");
    }

    #[test]
    fn method_receiver_stripped_aligns_with_free_fn() {
        // A method and a free fn with the same body produce the same xname canonical (receiver
        // dropped), so the cross-name pass can pair them. We check the cluster canonicals of the
        // method body vs the free fn match structurally by comparing analysis presence + that the
        // method's canon does not mention a `self` param slot.
        let src = "struct S;\nimpl S { fn add(&self, a: i32, b: i32) -> i32 { let t = a + b; t } }\nfn add_free(a: i32, b: i32) -> i32 { let t = a + b; t }\n";
        let f: Arc<str> = Arc::from("t.rs");
        let all = scan_source(src, &f, &ScanOpts::default());
        let method = all.iter().find(|d| d.name == "S::add").expect("method");
        let free = all.iter().find(|d| d.name == "add_free").expect("free fn");
        // xname canonicals are equal once the receiver is dropped and names are alpha-renamed.
        assert_eq!(
            method.analysis.as_ref().map(|a| &a.xname_canonical),
            free.analysis.as_ref().map(|a| &a.xname_canonical),
            "method (receiver-stripped) should alpha-equal the free fn"
        );
    }
}

#[cfg(test)]
mod type_slot_tests {
    use super::scan_source;
    use dup_defs_core::{Analysis, ScanOpts};
    use std::sync::Arc;

    fn analysis(src: &str, name: &str) -> Analysis {
        let f: Arc<str> = Arc::from("t.rs");
        scan_source(src, &f, &ScanOpts::default())
            .into_iter()
            .find(|d| d.name == name)
            .and_then(|d| d.analysis)
            .expect(name)
    }

    #[test]
    fn the_exact_canonical_keeps_types_the_near_copy_lines_slot_them() {
        // A Rust type picks the code that runs, so the exact pass keeps `Token` and `Claims` apart;
        // the near-copy lines read them as vocabulary, so the bodies agree line for line.
        let a = analysis("fn f(p: Token) -> Option<u32> { let v = p.get(\"uid\"); v }\n", "f");
        let b = analysis("fn g(q: Claims) -> Option<u32> { let v = q.get(\"uid\"); v }\n", "g");
        assert_ne!(a.xname_canonical, b.xname_canonical);
        assert_eq!(a.type3_lines, b.type3_lines);
        assert_eq!(a.type_refs, ["Token", "Option", "u32"]);
    }

    #[test]
    fn a_declared_type_parameter_is_bound_not_vocabulary() {
        let a = analysis("fn first<T: Clone>(items: &[T]) -> T { items[0].clone() }\n", "first");
        assert_eq!(a.type_refs, Vec::<String>::new());
        assert!(a.type3_lines[0].starts_with("fn _fn<_T0>("), "{:?}", a.type3_lines[0]);
    }
}

#[cfg(test)]
mod macro_tests {
    use super::scan_source;
    use dup_defs_core::ScanOpts;
    use std::sync::Arc;

    fn names(src: &str) -> Vec<String> {
        let f: Arc<str> = Arc::from("t.rs");
        scan_source(src, &f, &ScanOpts::default()).into_iter().map(|d| d.name).collect()
    }

    fn xname(src: &str, name: &str) -> String {
        let f: Arc<str> = Arc::from("t.rs");
        scan_source(src, &f, &ScanOpts::default())
            .into_iter()
            .find(|d| d.name == name)
            .and_then(|d| d.analysis)
            .map(|a| a.xname_canonical)
            .expect(name)
    }

    #[test]
    fn a_macro_shows_its_arguments() {
        // Two tests that build different values and assert different things were one canonical
        // while a macro was its path alone.
        let a = "fn t1() { let v = Lang(vec![\"fr\".parse().unwrap()]); assert_eq!(v.ranked(), vec![1]); }\n";
        let b = "fn t2() { let v = Lang(vec![\"de\".parse().unwrap(), \"en\".parse().unwrap()]); assert!(v.ranked().is_empty()); }\n";
        assert_ne!(xname(a, "t1"), xname(b, "t2"));
        assert!(xname(a, "t1").contains("Macro('vec', [Method("), "{}", xname(a, "t1"));
        // A body that is not a list of expressions stays opaque, as it was.
        let repeat = "fn z(n: usize) -> Vec<u8> { let v = vec![0; n]; v }\n";
        assert!(xname(repeat, "z").contains("Macro('vec')"), "{}", xname(repeat, "z"));
    }

    #[test]
    fn an_empty_match_and_a_stub_statement_are_trivial() {
        let src = "impl Body for Infallible {\n    fn size(&self) -> BodySize { match *self {} }\n    fn todo(&self) { todo!(); }\n    fn real(&self) -> u8 { self.0 + 1 }\n}\n";
        let found = names(src);
        assert!(!found.iter().any(|n| n.ends_with("::size") || n.ends_with("::todo")), "{found:?}");
        assert!(found.iter().any(|n| n.ends_with("::real")), "{found:?}");
    }
}
