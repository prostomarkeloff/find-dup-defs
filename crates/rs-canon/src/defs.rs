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

use dup_defs_core::{Analysis, Def, KindSpec, LineMap};
use syn::spanned::Spanned;
use syn::{Attribute, Block, Expr, FnArg, ImplItem, Item, Signature, Stmt, TraitItem, Type};

use crate::canon::{
    analyze_impl_fn, analyze_item_fn, analyze_trait_fn, enum_canon, struct_canon, trait_canon,
    union_canon, AnalyzedFn,
};
use crate::frontend::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};

/// Non-blank line count of a def's source text.
fn count_loc(text: &str) -> usize {
    text.lines().filter(|l| !l.trim().is_empty()).count()
}

/// `UPPER_SNAKE` (the constant convention) — same rule as the other frontends.
fn is_upper_snake(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

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
        _ => false,
    })
}

fn is_trivial_expr(e: &Expr) -> bool {
    match e {
        Expr::Lit(_) => true,
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

fn qualify(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}::{name}")
    }
}

fn analysis_from((cluster, xname, lines, size): AnalyzedFn) -> (String, Analysis) {
    (cluster, Analysis { xname_canonical: xname, type3_lines: lines, size })
}

/// One builder so every push site stays uniform.
struct Builder<'a> {
    source: &'a str,
    lines: &'a LineMap<'a>,
    file: &'a Arc<str>,
    out: &'a mut Vec<Def>,
}

impl Builder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        kind: &'static KindSpec,
        name: String,
        attrs: &[Attribute],
        span: proc_macro2::Span,
        args: usize,
        cluster_canonical: Option<String>,
        analysis: Option<Analysis>,
    ) {
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
            let (cluster, analysis) = analysis_from(analyze_item_fn(f));
            b.push(
                &FUNCTIONS,
                qualify(prefix, &f.sig.ident.to_string()),
                &f.attrs,
                f.span(),
                count_args(&f.sig),
                Some(cluster),
                Some(analysis),
            );
        }
        Item::Struct(s) => {
            b.push(&CLASSES, qualify(prefix, &s.ident.to_string()), &s.attrs, s.span(), 0, Some(struct_canon(s)), None);
        }
        Item::Enum(e) => {
            b.push(&CLASSES, qualify(prefix, &e.ident.to_string()), &e.attrs, e.span(), 0, Some(enum_canon(e)), None);
        }
        Item::Union(u) => {
            b.push(&CLASSES, qualify(prefix, &u.ident.to_string()), &u.attrs, u.span(), 0, Some(union_canon(u)), None);
        }
        Item::Trait(t) => {
            b.push(&INTERFACES, qualify(prefix, &t.ident.to_string()), &t.attrs, t.span(), 0, Some(trait_canon(t)), None);
            let owner = qualify(prefix, &t.ident.to_string());
            for ti in &t.items {
                if let TraitItem::Fn(tf) = ti {
                    if tf.default.is_none() || is_trivial_block(tf.default.as_ref().unwrap()) {
                        continue;
                    }
                    let (cluster, analysis) = analysis_from(analyze_trait_fn(tf).unwrap());
                    b.push(
                        &METHODS,
                        format!("{owner}::{}", tf.sig.ident),
                        &tf.attrs,
                        tf.span(),
                        count_args(&tf.sig),
                        Some(cluster),
                        Some(analysis),
                    );
                }
            }
        }
        Item::Impl(im) => {
            let owner = qualify(prefix, &type_name(&im.self_ty));
            for ii in &im.items {
                if let ImplItem::Fn(f) = ii {
                    if is_trivial_block(&f.block) {
                        continue;
                    }
                    let (cluster, analysis) = analysis_from(analyze_impl_fn(f));
                    b.push(
                        &METHODS,
                        format!("{owner}::{}", f.sig.ident),
                        &f.attrs,
                        f.span(),
                        count_args(&f.sig),
                        Some(cluster),
                        Some(analysis),
                    );
                }
            }
        }
        Item::Const(c) if is_upper_snake(&c.ident.to_string()) => {
            b.push(&CONSTANTS, qualify(prefix, &c.ident.to_string()), &c.attrs, c.span(), 0, None, None);
        }
        Item::Static(s) if is_upper_snake(&s.ident.to_string()) => {
            b.push(&CONSTANTS, qualify(prefix, &s.ident.to_string()), &s.attrs, s.span(), 0, None, None);
        }
        Item::Type(t) => {
            b.push(&TYPE_ALIASES, qualify(prefix, &t.ident.to_string()), &t.attrs, t.span(), 0, None, None);
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
pub fn scan_source(source: &str, file: &Arc<str>) -> Vec<Def> {
    let Ok(ast) = syn::parse_file(source) else { return Vec::new() };
    let lines = LineMap::new(source);
    let mut out = Vec::new();
    let mut b = Builder { source, lines: &lines, file, out: &mut out };
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
    use std::sync::Arc;

    fn defs(src: &str) -> Vec<(String, String)> {
        let f: Arc<str> = Arc::from("t.rs");
        scan_source(src, &f).into_iter().map(|d| (d.kind.id.to_owned(), d.name)).collect()
    }

    fn names_of_kind(src: &str, kind: &str) -> Vec<String> {
        defs(src).into_iter().filter(|(k, _)| k == kind).map(|(_, n)| n).collect()
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
        let all = scan_source(src, &f);
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
