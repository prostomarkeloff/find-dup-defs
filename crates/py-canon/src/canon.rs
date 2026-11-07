//! Native dup-defs canonicalization via ruff's parser + AST — no CPython `ast.*`, rayon-parallel,
//! so it runs fast on stock CPython while staying byte-compatible with the reference.
//!
//! The canonical string is **CPython `ast.dump(node, annotate_fields=False)`** reproduced from the
//! ruff AST: same node names, same ASDL field order, same `show_empty=False` rule (trailing
//! `None`/`[]` dropped; an emptied field before a present one switches the rest to `name=` keyword
//! form), and Python `repr` for literals. Reproducing that exact shape is what keeps the downstream
//! difflib ratios (clustering) and the alpha-renamed equality (cross-name) aligned with CPython's
//! `ast` — a terser form drifts the ratios. Two passes, both off one `Collect` of bound locals:
//! clustering keeps identifiers, cross-name alpha-renames them (`_v{n}`) and blanks the def name.
//!
//! Cross-name also renames every **type reference in annotation position** to a `_T{n}` slot
//! ([`dup_defs_core::type_slot`]): a parameter / return / variable annotation, and a type
//! parameter's name, bound and default. Python attaches no behaviour to an annotation, so two bodies
//! differing only in declared types run the same code — the textbook Type-2 clone. A *call* inside an
//! annotation is a value, not a type (the typing grammar admits no calls), so `Annotated[T,
//! Depends(f)]` keeps `Depends(f)` verbatim: that metadata is read at runtime, and it is exactly what
//! must keep two such definitions apart. Here, and only here, the name-agnostic canonical departs from
//! CPython's `ast.dump`.
#![allow(clippy::doc_markdown)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

/// `(cluster_canonical, xname_canonical, type3_lines, node_count)` — the analysis tuple the scan
/// reads to build a `Def`'s cluster canonical + `Analysis`. `py-canon`'s own type (was shared via
/// `dup-defs-core`, now local since the engine consumes `Def`, not this tuple).
pub use dup_defs_core::AnalyzedFn;
use dup_defs_core::Statement;
use rayon::prelude::*;
use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{
    self as ast, BoolOp, CmpOp, Comprehension, ExceptHandler, Expr, ExprContext, Keyword, Number,
    Operator, Parameter, Parameters, Stmt, UnaryOp, WithItem,
};
use ruff_python_parser::parse_module;
use ruff_text_size::Ranged;

// ---------------------------------------------------------------------------
// Bound-locals collection (cross-name rename targets) — unchanged in spirit: params + assignment /
// loop / with / except / import / comprehension / walrus targets. Free names carry behaviour, kept.
// ---------------------------------------------------------------------------

/// Collect names *bound* anywhere in the function (cross-name rename set).
#[derive(Default)]
struct Collect {
    bound: HashSet<String>,
}

impl Collect {
    /// The top function's own params (posonly + args + kwonly + `*args` + `**kwargs`). CPython's
    /// `_collect_locals` adds these explicitly and does NOT collect *nested* defs' params.
    fn add_params(&mut self, params: &Parameters) {
        for x in params.posonlyargs.iter().chain(params.args.iter()).chain(params.kwonlyargs.iter()) {
            self.bound.insert(x.parameter.name.id.as_str().to_owned());
        }
        if let Some(vararg) = &params.vararg {
            self.bound.insert(vararg.name.id.as_str().to_owned());
        }
        if let Some(kwarg) = &params.kwarg {
            self.bound.insert(kwarg.name.id.as_str().to_owned());
        }
    }

    fn add_target(&mut self, expr: &Expr) {
        match expr {
            Expr::Name(name) => {
                self.bound.insert(name.id.as_str().to_owned());
            }
            Expr::Starred(starred) => self.add_target(&starred.value),
            Expr::Tuple(tuple) => tuple.elts.iter().for_each(|elt| self.add_target(elt)),
            Expr::List(list) => list.elts.iter().for_each(|elt| self.add_target(elt)),
            _ => {} // Attribute / Subscript targets bind no local name
        }
    }
}

impl<'a> Visitor<'a> for Collect {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::Assign(node) => node.targets.iter().for_each(|t| self.add_target(t)),
            Stmt::AnnAssign(node) => self.add_target(&node.target),
            Stmt::AugAssign(node) => self.add_target(&node.target),
            Stmt::For(node) => self.add_target(&node.target),
            Stmt::Import(node) => {
                for alias in &node.names {
                    let name = match &alias.asname {
                        Some(asname) => asname.id.as_str().to_owned(),
                        None => alias.name.id.as_str().split('.').next().unwrap_or("").to_owned(),
                    };
                    self.bound.insert(name);
                }
            }
            Stmt::ImportFrom(node) => {
                for alias in &node.names {
                    let name = alias.asname.as_ref().unwrap_or(&alias.name);
                    self.bound.insert(name.id.as_str().to_owned());
                }
            }
            _ => {}
        }
        visitor::walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::Named(named) => self.add_target(&named.target),
            // Lambda params bind locals at any depth (CPython collects posonly+args+kwonly only —
            // not `*args`/`**kwargs`). Nested `def` params are deliberately NOT collected; only the
            // top function's params (added via `add_params`) and lambda params count.
            Expr::Lambda(lam) => {
                if let Some(p) = &lam.parameters {
                    for x in p.posonlyargs.iter().chain(p.args.iter()).chain(p.kwonlyargs.iter()) {
                        self.bound.insert(x.parameter.name.id.as_str().to_owned());
                    }
                }
            }
            _ => {}
        }
        visitor::walk_expr(self, expr);
    }

    fn visit_comprehension(&mut self, comprehension: &'a Comprehension) {
        self.add_target(&comprehension.target);
        visitor::walk_comprehension(self, comprehension);
    }

    fn visit_except_handler(&mut self, except_handler: &'a ExceptHandler) {
        let ExceptHandler::ExceptHandler(handler) = except_handler;
        if let Some(name) = &handler.name {
            self.bound.insert(name.id.as_str().to_owned());
        }
        visitor::walk_except_handler(self, except_handler);
    }

    fn visit_with_item(&mut self, with_item: &'a WithItem) {
        if let Some(vars) = &with_item.optional_vars {
            self.add_target(vars);
        }
        visitor::walk_with_item(self, with_item);
    }
}

// ---------------------------------------------------------------------------
// Python repr helpers — match CPython's `repr` for the literal kinds `ast.dump` embeds.
// ---------------------------------------------------------------------------

/// CPython `repr(str)`: single quotes unless that forces escaping a lone `'` while `"` is free;
/// `\\ \n \r \t` and C0/DEL controls escaped, printable (incl. non-ASCII) kept verbatim.
fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if needs_escape(c) => {
                let cp = c as u32;
                if cp < 0x100 {
                    let _ = write!(out, "\\x{cp:02x}");
                } else if cp < 0x10000 {
                    let _ = write!(out, "\\u{cp:04x}");
                } else {
                    let _ = write!(out, "\\U{cp:08x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Chars CPython's `repr` escapes as `\xNN`/`\uNNNN` (non-printable): C0/C1 controls plus the
/// common zero-width / BOM / line-separator format chars. (Full `str.isprintable` parity would
/// need Unicode tables; these cover what shows up in source-literal text.)
fn needs_escape(c: char) -> bool {
    c.is_control()
        || matches!(c, '\u{ad}' | '\u{feff}' | '\u{2028}' | '\u{2029}' | '\u{2060}')
        || ('\u{200b}'..='\u{200f}').contains(&c)
        // non-`U+0020` Unicode spaces (category Zs) — `str.isprintable()` is False for these
        || matches!(c, '\u{a0}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{1680}')
        || ('\u{2000}'..='\u{200a}').contains(&c)
}

/// CPython `repr(bytes)`: `b'…'`, printable ASCII kept, everything else `\xNN`.
fn repr_bytes(bytes: &[u8]) -> String {
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') { b'"' } else { b'\'' };
    let mut out = String::from("b");
    out.push(quote as char);
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            b if b == quote => {
                out.push('\\');
                out.push(b as char);
            }
            0x20..=0x7e => out.push(b as char),
            b => {
                let _ = write!(out, "\\x{b:02x}");
            }
        }
    }
    out.push(quote as char);
    out
}

/// CPython `repr(float)`: shortest round-trip, switching to scientific notation when the decimal
/// exponent is `< -4` or `>= 16` (Python's threshold), with `e±NN` (≥2 exponent digits) and a `.0`
/// forced on whole values in fixed notation. Rust's `{:e}` provides the shortest mantissa/exponent.
fn repr_float(value: f64) -> String {
    if value.is_infinite() {
        return if value < 0.0 { "-inf".to_owned() } else { "inf".to_owned() };
    }
    if value.is_nan() {
        return "nan".to_owned();
    }
    let sci = format!("{value:e}"); // e.g. "1e-6", "1.5e0", "-2.5e16"
    let (mantissa, exp) = match sci.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (sci.as_str(), 0),
    };
    if (-4..16).contains(&exp) {
        let fixed = format!("{value}");
        if fixed.contains('.') {
            fixed
        } else {
            format!("{fixed}.0")
        }
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        let abs = exp.unsigned_abs();
        format!("{mantissa}e{sign}{abs:02}")
    }
}

#[allow(clippy::float_cmp)] // complex literals carry an exact 0.0 real part
pub(crate) fn repr_number(num: &Number) -> String {
    match num {
        Number::Int(value) => format!("{value}"),
        Number::Float(value) => repr_float(*value),
        Number::Complex { real, imag } => {
            if *real == 0.0 {
                format!("{imag}j")
            } else {
                format!("({real}+{imag}j)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The serializer: one node at a time, mirroring CPython `ast.dump` `_format`.
// ---------------------------------------------------------------------------

/// One field's contribution under the `show_empty=False` rule. A field is a RENDER INSTRUCTION,
/// not a pre-rendered string: children write directly into the shared buffer, so a left-nested
/// chain (binop / attribute / call) costs one pass over its text — the old shape copied each
/// child's full string into its parent at every level, quadratic in nesting depth.
enum F<'a> {
    /// A present value — emitted positionally, or `name=` once keyword mode is on.
    /// Verbatim text.
    S(&'a str),
    /// A pre-rendered owned leaf (repr'd literal, formatted number) — copied once.
    P(String),
    /// An optional pre-rendered leaf: `None` switches the remaining fields to keyword form.
    O(Option<String>),
    /// An empty list (`[]`): buffered then dropped, unless a later present field flushes it back.
    Empty,
    /// An optional field that is `None`: dropped, and switches the remaining fields to keyword form.
    Skip,
    /// A child expression, dumped now.
    E(&'a Expr),
    /// An optional child expression.
    OE(Option<&'a Expr>),
    /// An annotation: the child with the type-position flag up.
    A(&'a Expr),
    /// An optional annotation.
    AO(Option<&'a Expr>),
    /// A small tag node (`Add()`, `Store()`, `Eq()` …).
    Tag(Tag),
    /// The `arguments(...)` node of a def or lambda.
    Args(Option<&'a Parameters>),
    /// The optional `vararg`/`kwarg` param — one `arg(...)` node, not a list.
    ParamOpt(Option<&'a Parameter>),
    /// A keyword's optional name (the repr'd arg slot or `#name#` mark).
    Kwo(Option<&'a str>, bool),
    /// A statement body.
    B(&'a [Stmt], bool),
    /// The nested-If `orelse` of a flat elif/else chain.
    OR(&'a [ast::ElifElseClause]),
    /// A list shape (see [`List`]).
    L(List<'a>),
    /// An optional f-string format spec (its own `JoinedStr`).
    FS(Option<&'a ast::InterpolatedStringElements>),
    /// A repr'd renamed local (`'_v0'` or the quoted verbatim name).
    Rid(&'a str),
    /// An optional repr'd renamed local.
    Rido(Option<&'a str>),
    /// A repr'd renamed attribute.
    Rattr(&'a str),
    /// A repr'd type parameter name (its `_T{n}` slot when renaming).
    Rtp(&'a str),
    /// One pattern node.
    PT(&'a ast::Pattern),
    /// An optional pattern node.
    Pto(Option<&'a ast::Pattern>),
}

/// A small tag node: the operator/context/comparison keywords `ast.dump` spells as leaf nodes.
#[derive(Clone, Copy)]
enum Tag {
    Ctx(ExprContext),
    Op(Operator),
    UO(UnaryOp),
    BO(BoolOp),
    CO(CmpOp),
}

/// A bracketed list rendered item by item into the shared buffer. All references — cheap to copy.
#[derive(Clone, Copy)]
enum List<'a> {
    Exprs(&'a [Expr]),
    /// Dict keys: an absent key renders as `None`.
    ExprsOrNone(&'a [ast::DictItem]),
    /// Dict values.
    DictValues(&'a [ast::DictItem]),
    Params(&'a [ast::ParameterWithDefault]),
    /// Keyword-only defaults: the expression, or `None` where absent.
    KwDefaults(&'a [ast::ParameterWithDefault]),
    /// The defaults that exist, posonlyargs then args.
    Defaults(&'a [ast::ParameterWithDefault], &'a [ast::ParameterWithDefault]),
    Comprehensions(&'a [Comprehension]),
    Keywords(&'a [Keyword], bool),
    Patterns(&'a [ast::Pattern]),
    KwdPatterns(&'a [ast::PatternKeyword]),
    Reprs(&'a [&'a str]),
    MatchCases(&'a [ast::MatchCase]),
    WithItems(&'a [ast::WithItem]),
    Handlers(&'a [ExceptHandler]),
    Aliases(&'a [ast::Alias]),
    Decorators(&'a [ast::Decorator]),
    TypeParams(&'a [ast::TypeParam]),
    CmpOps(&'a [CmpOp]),
    /// An f-string's parts (the values of a `JoinedStr`).
    FStringParts(&'a ast::FStringValue),
    /// A format-spec's elements (the values of its own `JoinedStr`).
    FStringElements(&'a ast::InterpolatedStringElements),
}

impl List<'_> {
    fn is_empty(&self) -> bool {
        match self {
            List::Exprs(v) => v.is_empty(),
            List::ExprsOrNone(v) | List::DictValues(v) => v.is_empty(),
            List::Params(v) | List::KwDefaults(v) => v.is_empty(),
            List::Defaults(a, b) => a.iter().chain(b.iter()).all(|x| x.default.is_none()),
            List::Comprehensions(v) => v.is_empty(),
            List::Keywords(v, _) => v.is_empty(),
            List::Patterns(v) => v.is_empty(),
            List::KwdPatterns(v) => v.is_empty(),
            List::Reprs(v) => v.is_empty(),
            List::MatchCases(v) => v.is_empty(),
            List::WithItems(v) => v.is_empty(),
            List::Handlers(v) => v.is_empty(),
            List::Aliases(v) => v.is_empty(),
            List::Decorators(v) => v.is_empty(),
            List::TypeParams(v) => v.is_empty(),
            List::CmpOps(v) => v.is_empty(),
            List::FStringParts(v) => !v.into_iter().any(|part| match part {
                ast::FStringPart::Literal(s) => !s.value.is_empty(),
                ast::FStringPart::FString(_) => true,
            }),
            List::FStringElements(v) => !v.iter().any(|el| match el {
                ast::InterpolatedStringElement::Literal(lit) => !lit.value.is_empty(),
                ast::InterpolatedStringElement::Interpolation(_) => true,
            }),
        }
    }
}

/// The separator + `name=`/pending-flush prefix for one field of [`Dump::node`] — the streaming
/// form of the old collect-then-join argument list (same bytes, one pass over the buffer).
fn field_prefix(out: &mut String, first: &mut bool, pending: &mut usize, keywords: bool, fname: &str) {
    if !*first {
        out.push_str(", ");
    }
    *first = false;
    if keywords {
        out.push_str(fname);
        out.push('=');
    } else {
        for _ in 0..*pending {
            out.push_str("[]");
            out.push_str(", ");
        }
        *pending = 0;
    }
}

/// Write one `_v{n}` slot: the existing slot for `name`, or the next one. The slot text goes
/// straight into the buffer — no per-reference allocation (a bound local is usually referenced
/// more than once).
#[allow(clippy::cast_possible_truncation)] // a def's distinct bound-name count is far below u32::MAX
fn write_slot(out: &mut String, map: &mut HashMap<String, u32>, name: &str) {
    let slot = if let Some(&s) = map.get(name) {
        s
    } else {
        let s = map.len() as u32;
        map.insert(name.to_owned(), s);
        s
    };
    let _ = write!(out, "_v{slot}");
}

/// The `_T{n}` counterpart of [`write_slot`] for type references.
#[allow(clippy::cast_possible_truncation)] // a def's distinct type-reference count is far below u32::MAX
fn write_tslot(out: &mut String, map: &mut HashMap<String, u32>, name: &str) {
    let slot = if let Some(&s) = map.get(name) {
        s
    } else {
        let s = map.len() as u32;
        map.insert(name.to_owned(), s);
        s
    };
    let _ = write!(out, "_T{slot}");
}

/// Whether an `elif_orelse` field renders to nothing: no clauses, or one bare `else` whose body
/// is empty (the old `flist` of an empty Vec was dropped by `show_empty=False`).
fn orelse_empty(clauses: &[ast::ElifElseClause]) -> bool {
    match clauses.split_first() {
        None => true,
        Some((first, _)) => first.test.is_none() && first.body.is_empty(),
    }
}

/// CPython-`ast.dump(annotate_fields=False)` serializer over the ruff AST. `locals = Some(set)`
/// ⇒ cross-name mode (rename bound locals → `_v{n}`, blank the def name); `None` ⇒ clustering
/// mode (names preserved). `count` mirrors `sum(1 for _ in ast.walk(node))` for the dup-defs size.
struct Dump<'a> {
    count: usize,
    /// The single output buffer every serializer method writes into — children append their text
    /// and the parent wraps it in place, so nothing is ever re-copied up the tree.
    out: String,
    src: &'a str,
    locals: Option<&'a HashSet<String>>,
    map: HashMap<String, u32>,
    /// Cross-name blanks the *top* function's name to `_fn` once; nested defs keep their names
    /// (CPython sets `fn.name = "_fn"` on the top node only).
    blanked: bool,
    /// The *top* def's own decorators are excluded (the dup-defs def text starts at the keyword, so
    /// the text-based canonical never had them). Consumed by the first def **before** its body is
    /// walked, so nested defs keep their decorators. Lets a node-based canonical (from a file AST,
    /// decorators present) match the text-based one byte-for-byte.
    top_def_pending: bool,
    /// **Use-profile mode.** `Some(name)` ⇒ this dump canonicalizes a *use site* of the definition
    /// called `name`: every `Name(name)` becomes the anchor `_t0`, and every attribute read directly
    /// off it (`name.foo`) is emitted marked as `#foo#` for the caller to renumber positionally.
    /// `None` (every existing caller) ⇒ byte-identical to the historical behaviour.
    anchor: Option<&'a str>,
    /// **Module-scope mode.** Names the module itself binds — its imports, its top-level
    /// definitions and assignments, and the fields its classes declare. They are renamed wherever
    /// they occur, *including in attribute position* (`row.stored_at`), which the local set never
    /// reaches. What survives is the grammar of talking to things the module did not define:
    /// `session.get(…)`, `.filter(…).delete()`. `None` ⇒ historical behaviour.
    scope: Option<&'a HashSet<String>>,
    /// Type reference → its `_T{n}` slot, numbered in order of first appearance.
    tmap: HashMap<String, u32>,
    /// Inside an annotation and not inside a call within it — where a name is a type reference.
    in_type: bool,
}

impl<'a> Dump<'a> {
    fn new(src: &'a str, locals: Option<&'a HashSet<String>>) -> Self {
        Self {
            count: 0,
            out: String::new(),
            src,
            locals,
            map: HashMap::new(),
            blanked: false,
            top_def_pending: true,
            anchor: None,
            scope: None,
            tmap: HashMap::new(),
            in_type: false,
        }
    }

    /// Use-profile mode: as [`Dump::new`], but with `anchor` as the definition whose use site is
    /// being canonicalized. See the [`Dump::anchor`] field.
    fn with_anchor(src: &'a str, locals: Option<&'a HashSet<String>>, anchor: &'a str) -> Self {
        Self { anchor: Some(anchor), ..Dump::new(src, locals) }
    }

    /// Module-scope mode: `locals` carries the function's own bound names, `scope` the module's —
    /// both rename in name position, only `scope` in attribute position.
    fn with_scope(
        src: &'a str,
        locals: &'a HashSet<String>,
        scope: &'a HashSet<String>,
    ) -> Self {
        Self { scope: Some(scope), ..Dump::new(src, Some(locals)) }
    }

    /// In cross-name mode, rewrite a bound local to its positional `_v{n}` placeholder.
    ///
    /// A name is bound if the definition binds it OR, in module-scope mode, if the module does:
    /// the two sets are consulted in turn rather than merged, because merging them meant copying
    /// the module's whole scope into every function's local set, once per function.
    fn rename_id(&mut self, id: &str) {
        let bound = [self.locals, self.scope].into_iter().flatten().find(|set| set.contains(id));
        if bound.is_some() {
            let Dump { out, map, .. } = self;
            write_slot(out, map, id);
        } else {
            self.out.push_str(id);
        }
    }

    /// An attribute whose name the *module* declares (a field of one of its classes) is a slot too:
    /// `row.stored_at` and `entry.refreshed` name the same position in two copies of one shape.
    /// Attributes of things the module did not define (`session.get`) are left alone — they are the
    /// grammar the shape is written in.
    fn rename_attr(&mut self, attr: &str) {
        if self.scope.is_some_and(|set| set.contains(attr)) {
            let Dump { out, map, .. } = self;
            write_slot(out, map, attr);
        } else {
            self.out.push_str(attr);
        }
    }

    /// An annotation, dumped as a type expression: in a name-agnostic dump its type references
    /// become `_T{n}` slots (see the module docs). The clustering dump keeps them verbatim.
    fn annotation(&mut self, expr: &Expr) {
        if self.locals.is_none() {
            self.expr(expr);
            return;
        }
        let outer = std::mem::replace(&mut self.in_type, true);
        self.expr(expr);
        self.in_type = outer;
    }

    /// A type parameter's own name: declared here, referenced from the annotations, so it takes the
    /// same slot they do.
    fn type_param_name(&mut self, name: &str) {
        if self.locals.is_some() {
            let Dump { out, tmap, .. } = self;
            write_tslot(out, tmap, name);
        } else {
            self.out.push_str(name);
        }
    }

    /// Emit one AST node `name(field, …)` applying the `show_empty=False` + keyword-switch rule.
    /// Children render straight into the shared buffer in field order — nothing is pre-rendered,
    /// nothing is copied up the tree.
    #[allow(clippy::too_many_lines)] // one arm per field kind — the kind list IS the method
    fn node(&mut self, name: &str, fields: Vec<(&str, F)>) {
        self.count += 1;
        self.out.push_str(name);
        self.out.push('(');
        let mut keywords = false;
        let mut pending = 0usize; // buffered empty lists, flushed iff a positional value follows
        let mut first = true;
        for (fname, f) in fields {
            match f {
                F::Skip => keywords = true,
                F::Empty => {
                    if !keywords {
                        pending += 1;
                    }
                }
                F::S(s) => {
                    field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                    self.out.push_str(s);
                }
                F::P(s) => {
                    field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                    self.out.push_str(&s);
                }
                F::O(o) => match o {
                    Some(s) => {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.out.push_str(&s);
                    }
                    None => keywords = true,
                },
                F::E(e) => {
                    field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                    self.expr(e);
                }
                F::OE(o) => match o {
                    Some(e) => {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.expr(e);
                    }
                    None => keywords = true,
                },
                F::A(e) => {
                    field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                    self.annotation(e);
                }
                F::AO(o) => match o {
                    Some(e) => {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.annotation(e);
                    }
                    None => keywords = true,
                },
                F::Tag(t) => {
                    field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                    self.tag(t);
                }
                F::Args(p) => {
                    field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                    self.arguments(p);
                }
                F::ParamOpt(o) => match o {
                    Some(x) => {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.arg_node(x);
                    }
                    None => keywords = true,
                },
                F::Kwo(o, on_anchor) => match o {
                    Some(s) => {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.out.push('\'');
                        if on_anchor {
                            self.out.push('#');
                            self.out.push_str(s);
                            self.out.push('#');
                        } else {
                            self.rename_attr(s);
                        }
                        self.out.push('\'');
                    }
                    None => keywords = true,
                },
                F::B(b, strip) => {
                    if b.is_empty() {
                        if !keywords {
                            pending += 1;
                        }
                    } else {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.body(b, strip);
                    }
                }
                F::OR(c) => {
                    if orelse_empty(c) {
                        if !keywords {
                            pending += 1;
                        }
                    } else {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.elif_orelse(c);
                    }
                }
                F::L(l) => {
                    if l.is_empty() {
                        if !keywords {
                            pending += 1;
                        }
                    } else {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.list(&l);
                    }
                }
                F::FS(o) => match o {
                    Some(e) => {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.formatspec_dump(e);
                    }
                    None => keywords = true,
                },
                F::Rid(s) => {
                    field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                    self.out.push('\'');
                    self.rename_id(s);
                    self.out.push('\'');
                }
                F::Rido(o) => match o {
                    Some(s) => {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.out.push('\'');
                        self.rename_id(s);
                        self.out.push('\'');
                    }
                    None => keywords = true,
                },
                F::Rattr(s) => {
                    field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                    self.out.push('\'');
                    self.rename_attr(s);
                    self.out.push('\'');
                }
                F::Rtp(s) => {
                    field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                    self.out.push('\'');
                    self.type_param_name(s);
                    self.out.push('\'');
                }
                F::PT(p) => {
                    field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                    self.pattern(p);
                }
                F::Pto(o) => match o {
                    Some(p) => {
                        field_prefix(&mut self.out, &mut first, &mut pending, keywords, fname);
                        self.pattern(p);
                    }
                    None => keywords = true,
                },
            }
        }
        self.out.push(')');
    }

    fn tag(&mut self, tag: Tag) {
        let name = match tag {
            Tag::Ctx(ctx) => match ctx {
                ExprContext::Store => "Store",
                ExprContext::Del => "Del",
                ExprContext::Load | ExprContext::Invalid => "Load",
            },
            Tag::Op(op) => match op {
                Operator::Add => "Add",
                Operator::Sub => "Sub",
                Operator::Mult => "Mult",
                Operator::MatMult => "MatMult",
                Operator::Div => "Div",
                Operator::Mod => "Mod",
                Operator::Pow => "Pow",
                Operator::LShift => "LShift",
                Operator::RShift => "RShift",
                Operator::BitOr => "BitOr",
                Operator::BitXor => "BitXor",
                Operator::BitAnd => "BitAnd",
                Operator::FloorDiv => "FloorDiv",
            },
            Tag::UO(op) => match op {
                UnaryOp::Invert => "Invert",
                UnaryOp::Not => "Not",
                UnaryOp::UAdd => "UAdd",
                UnaryOp::USub => "USub",
            },
            Tag::BO(op) => match op {
                BoolOp::And => "And",
                BoolOp::Or => "Or",
            },
            Tag::CO(op) => match op {
                CmpOp::Eq => "Eq",
                CmpOp::NotEq => "NotEq",
                CmpOp::Lt => "Lt",
                CmpOp::LtE => "LtE",
                CmpOp::Gt => "Gt",
                CmpOp::GtE => "GtE",
                CmpOp::Is => "Is",
                CmpOp::IsNot => "IsNot",
                CmpOp::In => "In",
                CmpOp::NotIn => "NotIn",
            },
        };
        self.node(name, vec![]);
    }

    /// `arg(arg, annotation?, type_comment?)` — the param name is a rename target.
    fn arg_node(&mut self, param: &Parameter) {
        self.node(
            "arg",
            vec![
                ("arg", F::Rid(param.name.id.as_str())),
                ("annotation", F::AO(param.annotation.as_deref())),
                ("type_comment", F::Skip),
            ],
        );
    }

    /// `arguments(posonlyargs, args, vararg?, kwonlyargs, kw_defaults, kwarg?, defaults)`.
    fn arguments(&mut self, params: Option<&Parameters>) {
        let Some(p) = params else {
            self.node(
                "arguments",
                vec![
                    ("posonlyargs", F::Empty),
                    ("args", F::Empty),
                    ("vararg", F::Skip),
                    ("kwonlyargs", F::Empty),
                    ("kw_defaults", F::Empty),
                    ("kwarg", F::Skip),
                    ("defaults", F::Empty),
                ],
            );
            return;
        };
        self.node(
            "arguments",
            vec![
                ("posonlyargs", F::L(List::Params(&p.posonlyargs))),
                ("args", F::L(List::Params(&p.args))),
                ("vararg", F::ParamOpt(p.vararg.as_deref())),
                ("kwonlyargs", F::L(List::Params(&p.kwonlyargs))),
                ("kw_defaults", F::L(List::KwDefaults(&p.kwonlyargs))),
                ("kwarg", F::ParamOpt(p.kwarg.as_deref())),
                ("defaults", F::L(List::Defaults(&p.posonlyargs, &p.args))),
            ],
        );
    }

    /// `on_anchor` ⇒ this keyword names the anchored definition's own declared surface (it is an
    /// argument of a call on the anchor itself), so its name is a slot and gets marked `#name#` —
    /// the same treatment `X.name` receives. `X(key=…)` and `X.key` are the same attribute reached
    /// through different grammar; erasing one but not the other would split a profile in half.
    fn keyword_at(&mut self, kw: &Keyword, on_anchor: bool) {
        let arg = kw.arg.as_ref().map(|id| id.id.as_str());
        self.node(
            "keyword",
            vec![
                ("arg", F::Kwo(arg, on_anchor)),
                ("value", F::E(&kw.value)),
            ],
        );
    }

    fn comprehension(&mut self, comp: &Comprehension) {
        self.node(
            "comprehension",
            vec![
                ("target", F::E(&comp.target)),
                ("iter", F::E(&comp.iter)),
                ("ifs", F::L(List::Exprs(&comp.ifs))),
                ("is_async", F::P(format!("{}", i32::from(comp.is_async)))),
            ],
        );
    }

    /// An f-string → CPython `JoinedStr([Constant | FormattedValue, …])`. Iterates PARTS (so plain
    /// `StringLiteral` parts in implicit concatenation are kept), merges consecutive literals into
    /// one `Constant`, and bakes a `{x=}` debug interpolation as `Constant('x=')` + `FormattedValue`
    /// — exactly as CPython's parser does.
    fn fstring_dump(&mut self, fvalue: &ast::FStringValue) {
        self.node("JoinedStr", vec![("values", F::L(List::FStringParts(fvalue)))]);
    }

    /// A format-spec → its own `JoinedStr` (CPython wraps format specs the same way).
    fn formatspec_dump(&mut self, elements: &ast::InterpolatedStringElements) {
        self.node("JoinedStr", vec![("values", F::L(List::FStringElements(elements)))]);
    }

    fn fstring_elements(
        &mut self,
        elements: &[ast::InterpolatedStringElement],
        literal: &mut String,
        first: &mut bool,
    ) {
        for element in elements {
            match element {
                ast::InterpolatedStringElement::Literal(lit) => literal.push_str(&lit.value),
                ast::InterpolatedStringElement::Interpolation(interp) => {
                    if let Some(dbg) = &interp.debug_text {
                        let range = interp.expression.range();
                        let expr_src =
                            self.src.get(usize::from(range.start())..usize::from(range.end())).unwrap_or("");
                        let _ = write!(literal, "{}{expr_src}{}", dbg.leading, dbg.trailing);
                    }
                    if !literal.is_empty() {
                        dup_defs_core::comma_sep(&mut self.out, first);
                        self.node("Constant", vec![("value", F::P(repr_str(literal)))]);
                        literal.clear();
                    }
                    let mut conversion = interp.conversion as i8;
                    if conversion == -1 && interp.debug_text.is_some() && interp.format_spec.is_none() {
                        conversion = 114; // bare `{x=}` implies `!r`
                    }
                    dup_defs_core::comma_sep(&mut self.out, first);
                    self.node(
                        "FormattedValue",
                        vec![
                            ("value", F::E(&interp.expression)),
                            ("conversion", F::P(format!("{conversion}"))),
                            (
                                "format_spec",
                                F::FS(interp.format_spec.as_ref().map(|s| &s.elements)),
                            ),
                        ],
                    );
                }
            }
        }
    }

    fn type_params(params: Option<&ast::TypeParams>) -> F<'_> {
        match params {
            None => F::Empty,
            Some(tps) => F::L(List::TypeParams(&tps.type_params)),
        }
    }

    fn type_param(&mut self, tp: &ast::TypeParam) {
        match tp {
            ast::TypeParam::TypeVar(t) => {
                self.node(
                    "TypeVar",
                    vec![
                        ("name", F::Rtp(t.name.id.as_str())),
                        ("bound", F::AO(t.bound.as_deref())),
                        ("default_value", F::AO(t.default.as_deref())),
                    ],
                );
            }
            ast::TypeParam::ParamSpec(t) => {
                self.node(
                    "ParamSpec",
                    vec![("name", F::Rtp(t.name.id.as_str())), ("default_value", F::AO(t.default.as_deref()))],
                );
            }
            ast::TypeParam::TypeVarTuple(t) => {
                self.node(
                    "TypeVarTuple",
                    vec![("name", F::Rtp(t.name.id.as_str())), ("default_value", F::AO(t.default.as_deref()))],
                );
            }
        }
    }

    fn alias_node(&mut self, alias: &ast::Alias) {
        self.node(
            "alias",
            vec![
                ("name", F::P(repr_str(alias.name.id.as_str()))),
                ("asname", F::O(alias.asname.as_ref().map(|n| repr_str(n.id.as_str())))),
            ],
        );
    }

    fn pattern(&mut self, pat: &ast::Pattern) {
        match pat {
            ast::Pattern::MatchValue(p) => {
                self.node("MatchValue", vec![("value", F::E(&p.value))]);
            }
            ast::Pattern::MatchSingleton(p) => {
                let value = match p.value {
                    ast::Singleton::None => "None",
                    ast::Singleton::True => "True",
                    ast::Singleton::False => "False",
                };
                self.node("MatchSingleton", vec![("value", F::S(value))]);
            }
            ast::Pattern::MatchSequence(p) => {
                self.node("MatchSequence", vec![("patterns", F::L(List::Patterns(&p.patterns)))]);
            }
            ast::Pattern::MatchMapping(p) => {
                self.node(
                    "MatchMapping",
                    vec![
                        ("keys", F::L(List::Exprs(&p.keys))),
                        ("patterns", F::L(List::Patterns(&p.patterns))),
                        ("rest", F::O(p.rest.as_ref().map(|id| repr_str(id.id.as_str())))),
                    ],
                );
            }
            ast::Pattern::MatchClass(p) => {
                self.node(
                    "MatchClass",
                    vec![
                        ("cls", F::E(&p.cls)),
                        ("patterns", F::L(List::Patterns(&p.arguments.patterns))),
                        (
                            "kwd_attrs",
                            F::L(List::Reprs(
                                &p.arguments.keywords.iter().map(|k| k.attr.id.as_str()).collect::<Vec<_>>(),
                            )),
                        ),
                        ("kwd_patterns", F::L(List::KwdPatterns(&p.arguments.keywords))),
                    ],
                );
            }
            ast::Pattern::MatchStar(p) => {
                self.node("MatchStar", vec![("name", F::O(p.name.as_ref().map(|id| repr_str(id.id.as_str()))))]);
            }
            ast::Pattern::MatchAs(p) => {
                self.node(
                    "MatchAs",
                    vec![
                        ("pattern", F::Pto(p.pattern.as_deref())),
                        ("name", F::O(p.name.as_ref().map(|id| repr_str(id.id.as_str())))),
                    ],
                );
            }
            ast::Pattern::MatchOr(p) => {
                self.node("MatchOr", vec![("patterns", F::L(List::Patterns(&p.patterns)))]);
            }
        }
    }

    fn match_case(&mut self, case: &ast::MatchCase) {
        self.node(
            "match_case",
            vec![
                ("pattern", F::PT(&case.pattern)),
                ("guard", F::OE(case.guard.as_deref())),
                ("body", F::B(&case.body, false)),
            ],
        );
    }

    fn withitem(&mut self, item: &ast::WithItem) {
        self.node(
            "withitem",
            vec![
                ("context_expr", F::E(&item.context_expr)),
                ("optional_vars", F::OE(item.optional_vars.as_deref())),
            ],
        );
    }

    fn handler(&mut self, h: &ExceptHandler) {
        let ExceptHandler::ExceptHandler(handler) = h;
        self.node(
            "ExceptHandler",
            vec![
                ("type", F::OE(handler.type_.as_deref())),
                ("name", F::Rido(handler.name.as_ref().map(|n| n.id.as_str()))),
                ("body", F::B(&handler.body, false)),
            ],
        );
    }

    /// A statement body; `strip` drops a leading string-literal docstring (def/class bodies only),
    /// replacing an emptied body with `[Pass()]` exactly like CPython's `_strip_docstring`.
    fn body(&mut self, stmts: &[Stmt], strip: bool) {
        self.out.push('[');
        let mut first = true;
        if strip && stmts.first().is_some_and(is_docstring) {
            let rest = &stmts[1..];
            if rest.is_empty() {
                self.node("Pass", vec![]);
            } else {
                for s in rest {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.stmt(s);
                }
            }
        } else {
            for s in stmts {
                dup_defs_core::comma_sep(&mut self.out, &mut first);
                self.stmt(s);
            }
        }
        self.out.push(']');
    }

    /// Rebuild CPython's nested-`If` `orelse` from ruff's flat `elif_else_clauses`, as one
    /// bracketed list rendered into the buffer.
    fn elif_orelse(&mut self, clauses: &[ast::ElifElseClause]) {
        self.out.push('[');
        let mut first = true;
        if let Some((first_c, rest)) = clauses.split_first() {
            match &first_c.test {
                Some(test) => {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.node(
                        "If",
                        vec![
                            ("test", F::E(test)),
                            ("body", F::B(&first_c.body, false)),
                            ("orelse", F::OR(rest)),
                        ],
                    );
                }
                None => {
                    // A bare `else`: its body flattens into the orelse list.
                    for s in &first_c.body {
                        dup_defs_core::comma_sep(&mut self.out, &mut first);
                        self.stmt(s);
                    }
                }
            }
        }
        self.out.push(']');
    }

    /// One bracketed list rendered item by item into the buffer.
    #[allow(clippy::too_many_lines)] // one arm per list shape — the shape list IS the method
    fn list(&mut self, list: &List<'_>) {
        self.out.push('[');
        let mut first = true;
        match *list {
            List::Exprs(items) => {
                for e in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.expr(e);
                }
            }
            List::ExprsOrNone(items) => {
                for it in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    match &it.key {
                        Some(k) => self.expr(k),
                        None => self.out.push_str("None"),
                    }
                }
            }
            List::DictValues(items) => {
                for it in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.expr(&it.value);
                }
            }
            List::Params(items) => {
                for x in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.arg_node(&x.parameter);
                }
            }
            List::KwDefaults(items) => {
                for x in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    match &x.default {
                        Some(d) => self.expr(d),
                        None => self.out.push_str("None"),
                    }
                }
            }
            List::Defaults(a, b) => {
                for x in a.iter().chain(b) {
                    if let Some(d) = &x.default {
                        dup_defs_core::comma_sep(&mut self.out, &mut first);
                        self.expr(d);
                    }
                }
            }
            List::Comprehensions(items) => {
                for c in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.comprehension(c);
                }
            }
            List::Keywords(items, on_anchor) => {
                for k in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.keyword_at(k, on_anchor);
                }
            }
            List::Patterns(items) => {
                for p in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.pattern(p);
                }
            }
            List::KwdPatterns(items) => {
                for k in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.pattern(&k.pattern);
                }
            }
            List::Reprs(items) => {
                for s in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.out.push_str(&repr_str(s));
                }
            }
            List::MatchCases(items) => {
                for c in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.match_case(c);
                }
            }
            List::WithItems(items) => {
                for w in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.withitem(w);
                }
            }
            List::Handlers(items) => {
                for h in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.handler(h);
                }
            }
            List::Aliases(items) => {
                for a in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.alias_node(a);
                }
            }
            List::Decorators(items) => {
                for d in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.expr(&d.expression);
                }
            }
            List::TypeParams(items) => {
                for t in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.type_param(t);
                }
            }
            List::CmpOps(items) => {
                for &o in items {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.tag(Tag::CO(o));
                }
            }
            List::FStringParts(fvalue) => {
                let mut literal = String::new();
                for part in fvalue {
                    match part {
                        ast::FStringPart::Literal(s) => literal.push_str(&s.value),
                        ast::FStringPart::FString(fs) => {
                            self.fstring_elements(&fs.elements, &mut literal, &mut first);
                        }
                    }
                }
                if !literal.is_empty() {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.node("Constant", vec![("value", F::P(repr_str(&literal)))]);
                }
            }
            List::FStringElements(elements) => {
                let mut literal = String::new();
                self.fstring_elements(elements, &mut literal, &mut first);
                if !literal.is_empty() {
                    dup_defs_core::comma_sep(&mut self.out, &mut first);
                    self.node("Constant", vec![("value", F::P(repr_str(&literal)))]);
                }
            }
        }
        self.out.push(']');
    }

    #[allow(clippy::too_many_lines)]
    fn expr(&mut self, expr: &Expr) {
        // A type reference is one slot however it is spelled: `typing.Optional` names one type, not
        // an attribute read. The anchor keeps its own slot even in type position.
        if self.in_type {
            if let Some(reference) = type_reference(expr).filter(|r| self.anchor != Some(r.as_str())) {
                let id = dup_defs_core::type_slot(&mut self.tmap, &reference);
                self.node(
                    "Name",
                    vec![
                        ("id", F::P(repr_str(&id))),
                        ("ctx", F::Tag(Tag::Ctx(ExprContext::Load))),
                    ],
                );
                return;
            }
        }
        match expr {
            Expr::Name(n) => {
                // In use-profile mode the anchored definition's own name is the slot, not a free
                // name: it is exactly the identity we are erasing one level up.
                let on_anchor = self.anchor == Some(n.id.as_str());
                self.node(
                    "Name",
                    vec![
                        ("id", if on_anchor { F::S("'_t0'") } else { F::Rid(n.id.as_str()) }),
                        ("ctx", F::Tag(Tag::Ctx(n.ctx))),
                    ],
                );
            }
            Expr::Attribute(a) => {
                // An attribute read straight off the anchor (`X.foo`) is part of the anchor's own
                // declared surface, so it is a slot too — marked here, renumbered positionally by
                // the caller once the whole profile is assembled and canonically ordered.
                let on_anchor = self
                    .anchor
                    .is_some_and(|a0| matches!(a.value.as_ref(), Expr::Name(n) if n.id.as_str() == a0));
                let mark = on_anchor.then(|| repr_str(&format!("#{}#", a.attr.id.as_str())));
                self.node(
                    "Attribute",
                    vec![
                        ("value", F::E(&a.value)),
                        ("attr", mark.map_or_else(|| F::Rattr(a.attr.id.as_str()), F::P)),
                        ("ctx", F::Tag(Tag::Ctx(a.ctx))),
                    ],
                );
            }
            Expr::Call(c) => {
                // A call *on the anchor* (`X(a=…, b=…)`) names the anchor's declared surface
                // through its keywords — the constructor form of `X.a`. See `keyword_at`.
                let on_anchor = self
                    .anchor
                    .is_some_and(|a0| matches!(c.func.as_ref(), Expr::Name(n) if n.id.as_str() == a0));
                // A call is a value even inside an annotation — `Depends(f)` in `Annotated[T, …]`.
                let outer = std::mem::replace(&mut self.in_type, false);
                self.node(
                    "Call",
                    vec![
                        ("func", F::E(&c.func)),
                        ("args", F::L(List::Exprs(&c.arguments.args))),
                        ("keywords", F::L(List::Keywords(&c.arguments.keywords, on_anchor))),
                    ],
                );
                self.in_type = outer;
            }
            Expr::BinOp(b) => {
                self.node(
                    "BinOp",
                    vec![
                        ("left", F::E(&b.left)),
                        ("op", F::Tag(Tag::Op(b.op))),
                        ("right", F::E(&b.right)),
                    ],
                );
            }
            Expr::UnaryOp(u) => {
                self.node(
                    "UnaryOp",
                    vec![("op", F::Tag(Tag::UO(u.op))), ("operand", F::E(&u.operand))],
                );
            }
            Expr::BoolOp(b) => {
                self.node(
                    "BoolOp",
                    vec![("op", F::Tag(Tag::BO(b.op))), ("values", F::L(List::Exprs(&b.values)))],
                );
            }
            Expr::Compare(c) => {
                self.node(
                    "Compare",
                    vec![
                        ("left", F::E(&c.left)),
                        ("ops", F::L(List::CmpOps(&c.ops))),
                        ("comparators", F::L(List::Exprs(&c.comparators))),
                    ],
                );
            }
            Expr::NumberLiteral(n) => {
                self.node("Constant", vec![("value", F::P(repr_number(&n.value)))]);
            }
            Expr::StringLiteral(s) => {
                self.node("Constant", vec![("value", F::P(repr_str(s.value.to_str())))]);
            }
            Expr::BytesLiteral(b) => {
                let bytes: Vec<u8> = b.value.bytes().collect();
                self.node("Constant", vec![("value", F::P(repr_bytes(&bytes)))]);
            }
            Expr::BooleanLiteral(b) => {
                self.node("Constant", vec![("value", F::S(if b.value { "True" } else { "False" }))]);
            }
            // IpyEscapeCommand has no CPython analogue (Jupyter only); a `None` Constant placeholder
            // it that never occurs in real modules.
            Expr::NoneLiteral(_) | Expr::IpyEscapeCommand(_) => {
                self.node("Constant", vec![("value", F::S("None"))]);
            }
            Expr::EllipsisLiteral(_) => self.node("Constant", vec![("value", F::S("Ellipsis"))]),
            Expr::Subscript(s) => {
                self.node(
                    "Subscript",
                    vec![
                        ("value", F::E(&s.value)),
                        ("slice", F::E(&s.slice)),
                        ("ctx", F::Tag(Tag::Ctx(s.ctx))),
                    ],
                );
            }
            Expr::Starred(s) => {
                self.node(
                    "Starred",
                    vec![("value", F::E(&s.value)), ("ctx", F::Tag(Tag::Ctx(s.ctx)))],
                );
            }
            Expr::List(l) => {
                self.node(
                    "List",
                    vec![("elts", F::L(List::Exprs(&l.elts))), ("ctx", F::Tag(Tag::Ctx(l.ctx)))],
                );
            }
            Expr::Tuple(t) => {
                self.node(
                    "Tuple",
                    vec![("elts", F::L(List::Exprs(&t.elts))), ("ctx", F::Tag(Tag::Ctx(t.ctx)))],
                );
            }
            Expr::Set(s) => {
                self.node("Set", vec![("elts", F::L(List::Exprs(&s.elts)))]);
            }
            Expr::Dict(d) => {
                self.node(
                    "Dict",
                    vec![
                        ("keys", F::L(List::ExprsOrNone(&d.items))),
                        ("values", F::L(List::DictValues(&d.items))),
                    ],
                );
            }
            Expr::Slice(s) => {
                self.node(
                    "Slice",
                    vec![
                        ("lower", F::OE(s.lower.as_deref())),
                        ("upper", F::OE(s.upper.as_deref())),
                        ("step", F::OE(s.step.as_deref())),
                    ],
                );
            }
            Expr::If(i) => {
                self.node(
                    "IfExp",
                    vec![
                        ("test", F::E(&i.test)),
                        ("body", F::E(&i.body)),
                        ("orelse", F::E(&i.orelse)),
                    ],
                );
            }
            Expr::Lambda(l) => {
                self.node(
                    "Lambda",
                    vec![("args", F::Args(l.parameters.as_deref())), ("body", F::E(&l.body))],
                );
            }
            Expr::Named(n) => {
                self.node(
                    "NamedExpr",
                    vec![("target", F::E(&n.target)), ("value", F::E(&n.value))],
                );
            }
            Expr::Await(a) => {
                self.node("Await", vec![("value", F::E(&a.value))]);
            }
            Expr::Yield(y) => {
                self.node("Yield", vec![("value", F::OE(y.value.as_deref()))]);
            }
            Expr::YieldFrom(y) => {
                self.node("YieldFrom", vec![("value", F::E(&y.value))]);
            }
            Expr::ListComp(c) => {
                self.node(
                    "ListComp",
                    vec![
                        ("elt", F::E(&c.elt)),
                        ("generators", F::L(List::Comprehensions(&c.generators))),
                    ],
                );
            }
            Expr::SetComp(c) => {
                self.node(
                    "SetComp",
                    vec![
                        ("elt", F::E(&c.elt)),
                        ("generators", F::L(List::Comprehensions(&c.generators))),
                    ],
                );
            }
            Expr::Generator(c) => {
                self.node(
                    "GeneratorExp",
                    vec![
                        ("elt", F::E(&c.elt)),
                        ("generators", F::L(List::Comprehensions(&c.generators))),
                    ],
                );
            }
            Expr::DictComp(c) => {
                self.node(
                    "DictComp",
                    vec![
                        ("key", F::E(&c.key)),
                        ("value", F::E(&c.value)),
                        ("generators", F::L(List::Comprehensions(&c.generators))),
                    ],
                );
            }
            Expr::FString(f) => self.fstring_dump(&f.value),
            Expr::TString(_) => {
                // PEP 750 template strings (3.14, vanishingly rare): CPython's `Interpolation.str`
                // field can't be reproduced from the ruff AST, so emit a minimal `TemplateStr()`.
                self.node("TemplateStr", vec![("values", F::Empty)]);
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::FunctionDef(f) => {
                let strip_deco = self.top_def_pending; // capture + clear BEFORE walking the body
                self.top_def_pending = false;
                let cpy = if f.is_async { "AsyncFunctionDef" } else { "FunctionDef" };
                let name = if self.locals.is_some() && !self.blanked {
                    self.blanked = true;
                    "_fn".to_owned()
                } else {
                    f.name.id.as_str().to_owned()
                };
                let decorators: &[ast::Decorator] = if strip_deco { &[] } else { &f.decorator_list };
                let type_params = Self::type_params(f.type_params.as_deref());
                self.node(
                    cpy,
                    vec![
                        ("name", F::P(repr_str(&name))),
                        ("args", F::Args(Some(&f.parameters))),
                        ("body", F::B(&f.body, true)),
                        ("decorator_list", F::L(List::Decorators(decorators))),
                        ("returns", F::AO(f.returns.as_deref())),
                        ("type_comment", F::Skip),
                        ("type_params", type_params),
                    ],
                );
            }
            Stmt::ClassDef(c) => {
                let strip_deco = self.top_def_pending; // capture + clear BEFORE walking the body
                self.top_def_pending = false;
                // Cross-name mode blanks the *top* definition's own name so two structurally equal
                // definitions under different names collapse to one canonical. `FunctionDef` has
                // always done this; a class shape needs the same treatment, and `blanked` keeps it
                // to the top node (a class nested inside an already-blanked function is untouched).
                let name = if self.locals.is_some() && !self.blanked {
                    self.blanked = true;
                    "_fn".to_owned()
                } else {
                    c.name.id.as_str().to_owned()
                };
                let (bases, keywords) = match &c.arguments {
                    Some(a) => (&a.args[..], &a.keywords[..]),
                    None => (&[][..], &[][..]),
                };
                let decorators: &[ast::Decorator] = if strip_deco { &[] } else { &c.decorator_list };
                let type_params = Self::type_params(c.type_params.as_deref());
                self.node(
                    "ClassDef",
                    vec![
                        ("name", F::P(repr_str(&name))),
                        ("bases", F::L(List::Exprs(bases))),
                        ("keywords", F::L(List::Keywords(keywords, false))),
                        ("body", F::B(&c.body, true)),
                        ("decorator_list", F::L(List::Decorators(decorators))),
                        ("type_params", type_params),
                    ],
                );
            }
            Stmt::Return(r) => {
                self.node("Return", vec![("value", F::OE(r.value.as_deref()))]);
            }
            Stmt::Delete(d) => {
                self.node("Delete", vec![("targets", F::L(List::Exprs(&d.targets)))]);
            }
            Stmt::Assign(a) => {
                self.node(
                    "Assign",
                    vec![
                        ("targets", F::L(List::Exprs(&a.targets))),
                        ("value", F::E(&a.value)),
                        ("type_comment", F::Skip),
                    ],
                );
            }
            Stmt::AugAssign(a) => {
                self.node(
                    "AugAssign",
                    vec![
                        ("target", F::E(&a.target)),
                        ("op", F::Tag(Tag::Op(a.op))),
                        ("value", F::E(&a.value)),
                    ],
                );
            }
            Stmt::AnnAssign(a) => {
                self.node(
                    "AnnAssign",
                    vec![
                        ("target", F::E(&a.target)),
                        ("annotation", F::A(&a.annotation)),
                        ("value", F::OE(a.value.as_deref())),
                        ("simple", F::P(format!("{}", i32::from(a.simple)))),
                    ],
                );
            }
            Stmt::TypeAlias(t) => {
                let type_params = Self::type_params(t.type_params.as_deref());
                self.node(
                    "TypeAlias",
                    vec![
                        ("name", F::E(&t.name)),
                        ("type_params", type_params),
                        ("value", F::E(&t.value)),
                    ],
                );
            }
            Stmt::For(f) => {
                let cpy = if f.is_async { "AsyncFor" } else { "For" };
                self.node(
                    cpy,
                    vec![
                        ("target", F::E(&f.target)),
                        ("iter", F::E(&f.iter)),
                        ("body", F::B(&f.body, false)),
                        ("orelse", F::B(&f.orelse, false)),
                        ("type_comment", F::Skip),
                    ],
                );
            }
            Stmt::While(w) => {
                self.node(
                    "While",
                    vec![
                        ("test", F::E(&w.test)),
                        ("body", F::B(&w.body, false)),
                        ("orelse", F::B(&w.orelse, false)),
                    ],
                );
            }
            Stmt::If(i) => {
                self.node(
                    "If",
                    vec![
                        ("test", F::E(&i.test)),
                        ("body", F::B(&i.body, false)),
                        ("orelse", F::OR(&i.elif_else_clauses)),
                    ],
                );
            }
            Stmt::With(w) => {
                let cpy = if w.is_async { "AsyncWith" } else { "With" };
                self.node(
                    cpy,
                    vec![
                        ("items", F::L(List::WithItems(&w.items))),
                        ("body", F::B(&w.body, false)),
                        ("type_comment", F::Skip),
                    ],
                );
            }
            Stmt::Raise(r) => {
                self.node(
                    "Raise",
                    vec![("exc", F::OE(r.exc.as_deref())), ("cause", F::OE(r.cause.as_deref()))],
                );
            }
            Stmt::Try(t) => {
                let cpy = if t.is_star { "TryStar" } else { "Try" };
                self.node(
                    cpy,
                    vec![
                        ("body", F::B(&t.body, false)),
                        ("handlers", F::L(List::Handlers(&t.handlers))),
                        ("orelse", F::B(&t.orelse, false)),
                        ("finalbody", F::B(&t.finalbody, false)),
                    ],
                );
            }
            Stmt::Assert(a) => {
                self.node(
                    "Assert",
                    vec![("test", F::E(&a.test)), ("msg", F::OE(a.msg.as_deref()))],
                );
            }
            Stmt::Import(i) => {
                self.node("Import", vec![("names", F::L(List::Aliases(&i.names)))]);
            }
            Stmt::ImportFrom(i) => {
                self.node(
                    "ImportFrom",
                    vec![
                        ("module", F::O(i.module.as_ref().map(|m| repr_str(m.id.as_str())))),
                        ("names", F::L(List::Aliases(&i.names))),
                        ("level", F::P(format!("{}", i.level))),
                    ],
                );
            }
            Stmt::Global(g) => {
                self.node(
                    "Global",
                    vec![("names", F::L(List::Reprs(&g.names.iter().map(|n| n.id.as_str()).collect::<Vec<_>>())))],
                );
            }
            Stmt::Nonlocal(n) => {
                self.node(
                    "Nonlocal",
                    vec![("names", F::L(List::Reprs(&n.names.iter().map(|x| x.id.as_str()).collect::<Vec<_>>())))],
                );
            }
            Stmt::Expr(e) => {
                self.node("Expr", vec![("value", F::E(&e.value))]);
            }
            Stmt::Match(m) => {
                self.node(
                    "Match",
                    vec![
                        ("subject", F::E(&m.subject)),
                        ("cases", F::L(List::MatchCases(&m.cases))),
                    ],
                );
            }
            // IpyEscapeCommand (Jupyter only) has no CPython node; treat as Pass — never occurs.
            Stmt::Pass(_) | Stmt::IpyEscapeCommand(_) => self.node("Pass", vec![]),
            Stmt::Break(_) => self.node("Break", vec![]),
            Stmt::Continue(_) => self.node("Continue", vec![]),
        }
    }
}

fn is_docstring(stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Expr(e) if matches!(e.value.as_ref(), Expr::StringLiteral(_)))
}

// ---------------------------------------------------------------------------
// Unparse — CPython 3.14 `ast.unparse` (`_ast_unparse.Unparser`) reproduced over the ruff AST.
// Used ONLY to produce the Type-3 (ECScan) `lines` (= `ast.unparse(normalized_fn).splitlines()`
// stripped), so the IDF/cosine matches the CPython `ast` reference exactly. Precedence is threaded as a
// `ctx` argument (each node is visited once after its parent sets its precedence), reproducing
// CPython's `set_precedence`/`require_parens` parenthesisation. Indentation is omitted (lines are
// stripped anyway); newlines fall only at statement/clause boundaries.
// ---------------------------------------------------------------------------

mod prec {
    pub const NAMED_EXPR: u8 = 1;
    pub const TUPLE: u8 = 2;
    pub const YIELD: u8 = 3;
    pub const TEST: u8 = 4;
    pub const OR: u8 = 5;
    pub const AND: u8 = 6;
    pub const NOT: u8 = 7;
    pub const CMP: u8 = 8;
    pub const EXPR: u8 = 9;
    pub const BOR: u8 = 9;
    pub const BXOR: u8 = 10;
    pub const BAND: u8 = 11;
    pub const SHIFT: u8 = 12;
    pub const ARITH: u8 = 13;
    pub const TERM: u8 = 14;
    pub const FACTOR: u8 = 15;
    pub const POWER: u8 = 16;
    pub const AWAIT: u8 = 17;
    pub const ATOM: u8 = 18;
}

fn next_prec(p: u8) -> u8 {
    if p < prec::ATOM {
        p + 1
    } else {
        prec::ATOM
    }
}

fn operator_tag(op: Operator) -> &'static str {
    match op {
        Operator::Add => "+",
        Operator::Sub => "-",
        Operator::Mult => "*",
        Operator::MatMult => "@",
        Operator::Div => "/",
        Operator::Mod => "%",
        Operator::Pow => "**",
        Operator::LShift => "<<",
        Operator::RShift => ">>",
        Operator::BitOr => "|",
        Operator::BitXor => "^",
        Operator::BitAnd => "&",
        Operator::FloorDiv => "//",
    }
}

fn binop_prec(op: Operator) -> u8 {
    match op {
        Operator::Add | Operator::Sub => prec::ARITH,
        Operator::Mult | Operator::MatMult | Operator::Div | Operator::Mod | Operator::FloorDiv => prec::TERM,
        Operator::LShift | Operator::RShift => prec::SHIFT,
        Operator::BitOr => prec::BOR,
        Operator::BitXor => prec::BXOR,
        Operator::BitAnd => prec::BAND,
        Operator::Pow => prec::POWER,
    }
}

fn cmpop_tag(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "==",
        CmpOp::NotEq => "!=",
        CmpOp::Lt => "<",
        CmpOp::LtE => "<=",
        CmpOp::Gt => ">",
        CmpOp::GtE => ">=",
        CmpOp::Is => "is",
        CmpOp::IsNot => "is not",
        CmpOp::In => "in",
        CmpOp::NotIn => "not in",
    }
}

/// Python `str.isprintable` (approx): control + the common format/separator chars are not printable.
fn is_printable(c: char) -> bool {
    c == ' ' || !needs_escape(c)
}

/// One char as Python's `unicode_escape` would render it (used for non-printable / backslash).
fn unicode_escape(c: char) -> String {
    match c {
        '\\' => "\\\\".to_owned(),
        '\n' => "\\n".to_owned(),
        '\r' => "\\r".to_owned(),
        '\t' => "\\t".to_owned(),
        c => {
            let cp = c as u32;
            if cp < 0x100 {
                format!("\\x{cp:02x}")
            } else if cp < 0x10000 {
                format!("\\u{cp:04x}")
            } else {
                format!("\\U{cp:08x}")
            }
        }
    }
}

/// CPython unparse `escape_char`: keep `\n`/`\t` literal unless `esw`, else escape backslash + any
/// non-printable char (used for f-string literal parts; plain string Constants use `repr` instead).
fn escape_char(c: char, esw: bool) -> String {
    if !esw && (c == '\n' || c == '\t') {
        return c.to_string();
    }
    if c == '\\' || !is_printable(c) {
        return unicode_escape(c);
    }
    c.to_string()
}

/// Append `s` as an f-string literal chunk: double `{`/`}`, then escape each char (esw=true).
fn fstring_literal_into(body: &mut String, s: &str) {
    let doubled = s.replace('{', "{{").replace('}', "}}");
    for c in doubled.chars() {
        body.push_str(&escape_char(c, true));
    }
}

fn fstring_literal_escaped(s: &str) -> String {
    let mut body = String::new();
    fstring_literal_into(&mut body, s);
    body
}

struct Unparse<'a> {
    out: String,
    src: &'a str,
    locals: Option<&'a HashSet<String>>,
    map: HashMap<String, u32>,
    blanked: bool,
    /// Block nesting of the statement being written. Emitted as leading spaces, which every
    /// existing caller trims away; [`unparse_depths`] is the one that reads it back.
    indent: usize,
    /// `Some` when writing the Type-3 lines: type reference → `_T{n}` slot, handed
    /// over from the [`Dump`] as `map` is. `None` writes types as spelled — the statement stream.
    tmap: Option<HashMap<String, u32>>,
    /// Inside an annotation and not inside a call within it — see [`Dump::in_type`].
    in_type: bool,
    /// Every type reference turned into a slot, in order of appearance — [`AnalyzedFn::type_refs`].
    mentions: Vec<String>,
    /// Type parameters the definition declares: renamed like any type, but bound here, so not
    /// vocabulary it shares with anything.
    declared_types: HashSet<String>,
    /// `Some` when writing the Type-3 lines: every sub-expression worth a unit of its own, rendered
    /// exactly as it appears inside its line, with the index of that line —
    /// [`AnalyzedFn::type3_units`].
    units: Option<Vec<(u32, String)>>,
    /// Logical lines started so far ([`Self::fill`] calls) — the index a captured unit is filed under.
    fills: u32,
    /// How many expressions enclose the one being written: `1` for a statement's own expression.
    expr_depth: u32,
}

/// Whether an expression is a Type-3 unit of its own: a node with structure, not a bare name or a
/// literal (which carry no agreement by themselves) nor a slice or a starred operand (which are not
/// expressions a reader would call a part).
fn is_unit(expr: &Expr) -> bool {
    match expr {
        Expr::Call(_)
        | Expr::Attribute(_)
        | Expr::Subscript(_)
        | Expr::BinOp(_)
        | Expr::BoolOp(_)
        | Expr::Compare(_)
        | Expr::UnaryOp(_)
        | Expr::If(_)
        | Expr::Lambda(_)
        | Expr::ListComp(_)
        | Expr::SetComp(_)
        | Expr::DictComp(_)
        | Expr::Generator(_)
        | Expr::Await(_)
        | Expr::Yield(_)
        | Expr::YieldFrom(_)
        | Expr::Named(_)
        | Expr::FString(_) => true,
        Expr::List(l) => !l.elts.is_empty(),
        Expr::Tuple(t) => !t.elts.is_empty(),
        Expr::Set(s) => !s.elts.is_empty(),
        Expr::Dict(d) => !d.items.is_empty(),
        _ => false,
    }
}

impl Unparse<'_> {
    /// Write an expression and, when collecting units, keep what was written for it as one.
    /// `buffered` regions swap the output buffer, so the slice is taken from whichever buffer the
    /// expression was written into — the text is the same.
    fn expr(&mut self, expr: &Expr, ctx: u8) {
        let start = self.out.len();
        self.expr_depth += 1;
        self.expr_inner(expr, ctx);
        self.expr_depth -= 1;
        // A proper sub-expression only: the statement's own expression is the line, already a unit,
        // and counting it again would double the weight of exactly the text an edit changes.
        if self.units.is_some() && self.expr_depth > 0 && !self.in_type && is_unit(expr) {
            let unit = self.out[start..].to_owned();
            let line = self.fills.saturating_sub(1);
            if let Some(units) = self.units.as_mut() {
                units.push((line, unit));
            }
        }
    }

    fn rename_id(&mut self, id: &str) -> String {
        dup_defs_core::alpha_rename(&mut self.map, self.locals, id)
    }

    /// An annotation, written as a type expression — the line form of [`Dump::annotation`] when
    /// writing the Type-3 lines, verbatim otherwise.
    fn annotation(&mut self, expr: &Expr, ctx: u8) {
        if self.tmap.is_none() {
            self.expr(expr, ctx);
            return;
        }
        let outer = std::mem::replace(&mut self.in_type, true);
        self.expr(expr, ctx);
        self.in_type = outer;
    }

    /// A type parameter's own name — the line form of [`Dump::type_param_name`].
    fn type_param_name(&mut self, name: &str) {
        let name = match self.tmap.as_mut() {
            Some(tmap) => {
                self.declared_types.insert(name.to_owned());
                dup_defs_core::type_slot(tmap, name)
            }
            None => name.to_owned(),
        };
        self.write(&name);
    }

    /// Start a new logical line (CPython `fill`): a newline unless we're at the start, then the
    /// current block nesting as leading spaces.
    fn fill(&mut self, text: &str) {
        self.fill_fmt(format_args!("{text}"));
    }

    /// [`fill`] for a formatted value, written straight into `out` (no `format!` temp String).
    fn fill_fmt(&mut self, args: std::fmt::Arguments<'_>) {
        self.fills += 1;
        if !self.out.is_empty() {
            self.out.push('\n');
        }
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        let _ = self.out.write_fmt(args);
    }

    fn write(&mut self, text: &str) {
        self.out.push_str(text);
    }

    /// Render `f(self)` into a fresh buffer and return it (CPython `buffered`).
    fn buffered(&mut self, f: impl FnOnce(&mut Self)) -> String {
        let saved = std::mem::take(&mut self.out);
        f(self);
        std::mem::replace(&mut self.out, saved)
    }

    fn buffered_expr(&mut self, e: &Expr, ctx: u8) -> String {
        self.buffered(|s| s.expr(e, ctx))
    }

    /// A def/class body: skip a leading docstring (→ `pass` if that empties it), then each stmt.
    fn body(&mut self, stmts: &[Stmt], strip: bool) {
        let slice = if strip && stmts.first().is_some_and(is_docstring) { &stmts[1..] } else { stmts };
        self.indent += 1;
        if slice.is_empty() {
            self.fill("pass");
        } else {
            for stmt in slice {
                self.stmt(stmt);
            }
        }
        self.indent -= 1;
    }

    #[allow(clippy::too_many_lines)]
    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::FunctionDef(f) => {
                let kw = if f.is_async { "async def " } else { "def " };
                let name = if self.locals.is_some() && !self.blanked {
                    self.blanked = true;
                    "_fn".to_owned()
                } else {
                    f.name.id.as_str().to_owned()
                };
                self.fill_fmt(format_args!("{kw}{name}"));
                self.type_params(f.type_params.as_deref());
                self.write("(");
                self.arguments(Some(&f.parameters));
                self.write(")");
                if let Some(returns) = &f.returns {
                    self.write(" -> ");
                    self.annotation(returns, prec::TEST);
                }
                self.write(":");
                self.body(&f.body, true);
            }
            Stmt::ClassDef(c) => {
                self.fill_fmt(format_args!("class {}", c.name.id.as_str()));
                self.type_params(c.type_params.as_deref());
                let has_args = c.arguments.as_ref().is_some_and(|a| !a.args.is_empty() || !a.keywords.is_empty());
                if has_args {
                    self.write("(");
                    if let Some(a) = &c.arguments {
                        let mut first = true;
                        for base in &a.args {
                            if !first {
                                self.write(", ");
                            }
                            first = false;
                            self.expr(base, prec::TEST);
                        }
                        for kw in &a.keywords {
                            if !first {
                                self.write(", ");
                            }
                            first = false;
                            self.keyword(kw);
                        }
                    }
                    self.write(")");
                }
                self.write(":");
                self.body(&c.body, true);
            }
            Stmt::Return(r) => {
                self.fill("return");
                if let Some(v) = &r.value {
                    self.write(" ");
                    self.expr(v, prec::TEST);
                }
            }
            Stmt::Delete(d) => {
                self.fill("del ");
                self.comma_exprs(&d.targets);
            }
            Stmt::Assign(a) => {
                self.fill("");
                for target in &a.targets {
                    self.expr(target, prec::TUPLE);
                    self.write(" = ");
                }
                self.expr(&a.value, prec::TEST);
            }
            Stmt::AugAssign(a) => {
                self.fill("");
                self.expr(&a.target, prec::TEST);
                let _ = write!(self.out, " {}= ", operator_tag(a.op));
                self.expr(&a.value, prec::TEST);
            }
            Stmt::AnnAssign(a) => {
                self.fill("");
                let parens = !a.simple && matches!(a.target.as_ref(), Expr::Name(_));
                if parens {
                    self.write("(");
                }
                self.expr(&a.target, prec::TEST);
                if parens {
                    self.write(")");
                }
                self.write(": ");
                self.annotation(&a.annotation, prec::TEST);
                if let Some(v) = &a.value {
                    self.write(" = ");
                    self.expr(v, prec::TEST);
                }
            }
            Stmt::For(f) => {
                self.fill(if f.is_async { "async for " } else { "for " });
                self.expr(&f.target, prec::TUPLE);
                self.write(" in ");
                self.expr(&f.iter, prec::TEST);
                self.write(":");
                self.body(&f.body, false);
                if !f.orelse.is_empty() {
                    self.fill("else:");
                    self.body(&f.orelse, false);
                }
            }
            Stmt::While(w) => {
                self.fill("while ");
                self.expr(&w.test, prec::TEST);
                self.write(":");
                self.body(&w.body, false);
                if !w.orelse.is_empty() {
                    self.fill("else:");
                    self.body(&w.orelse, false);
                }
            }
            Stmt::If(i) => {
                self.fill("if ");
                self.expr(&i.test, prec::TEST);
                self.write(":");
                self.body(&i.body, false);
                for clause in &i.elif_else_clauses {
                    if let Some(test) = &clause.test {
                        self.fill("elif ");
                        self.expr(test, prec::TEST);
                        self.write(":");
                        self.body(&clause.body, false);
                    } else {
                        self.fill("else:");
                        self.body(&clause.body, false);
                    }
                }
            }
            Stmt::With(w) => {
                self.fill(if w.is_async { "async with " } else { "with " });
                let mut first = true;
                for item in &w.items {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    self.expr(&item.context_expr, prec::TEST);
                    if let Some(v) = &item.optional_vars {
                        self.write(" as ");
                        self.expr(v, prec::TEST);
                    }
                }
                self.write(":");
                self.body(&w.body, false);
            }
            Stmt::Raise(r) => {
                self.fill("raise");
                if let Some(exc) = &r.exc {
                    self.write(" ");
                    self.expr(exc, prec::TEST);
                    if let Some(cause) = &r.cause {
                        self.write(" from ");
                        self.expr(cause, prec::TEST);
                    }
                }
            }
            Stmt::Try(t) => {
                self.fill("try:");
                self.body(&t.body, false);
                for handler in &t.handlers {
                    let ExceptHandler::ExceptHandler(h) = handler;
                    self.fill(if t.is_star { "except*" } else { "except" });
                    if let Some(typ) = &h.type_ {
                        self.write(" ");
                        self.expr(typ, prec::TEST);
                    }
                    if let Some(name) = &h.name {
                        self.write(" as ");
                        let renamed = self.rename_id(name.id.as_str());
                        self.write(&renamed);
                    }
                    self.write(":");
                    self.body(&h.body, false);
                }
                if !t.orelse.is_empty() {
                    self.fill("else:");
                    self.body(&t.orelse, false);
                }
                if !t.finalbody.is_empty() {
                    self.fill("finally:");
                    self.body(&t.finalbody, false);
                }
            }
            Stmt::Assert(a) => {
                self.fill("assert ");
                self.expr(&a.test, prec::TEST);
                if let Some(msg) = &a.msg {
                    self.write(", ");
                    self.expr(msg, prec::TEST);
                }
            }
            Stmt::Import(i) => {
                self.fill("import ");
                let mut first = true;
                for alias in &i.names {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    self.alias(alias);
                }
            }
            Stmt::ImportFrom(i) => {
                self.fill("from ");
                for _ in 0..i.level {
                    self.write(".");
                }
                if let Some(m) = &i.module {
                    self.write(m.id.as_str());
                }
                self.write(" import ");
                let mut first = true;
                for alias in &i.names {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    self.alias(alias);
                }
            }
            Stmt::Global(g) => {
                self.fill("global ");
                let mut first = true;
                for n in &g.names {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    self.write(n.id.as_str());
                }
            }
            Stmt::Nonlocal(n) => {
                self.fill("nonlocal ");
                let mut first = true;
                for x in &n.names {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    self.write(x.id.as_str());
                }
            }
            Stmt::TypeAlias(t) => {
                self.fill("type ");
                self.expr(&t.name, prec::TEST);
                self.type_params(t.type_params.as_deref());
                self.write(" = ");
                self.expr(&t.value, prec::TEST);
            }
            Stmt::Expr(e) => {
                self.fill("");
                self.expr(&e.value, prec::YIELD);
            }
            Stmt::Match(m) => {
                self.fill("match ");
                self.expr(&m.subject, prec::TEST);
                self.write(":");
                for case in &m.cases {
                    self.fill("case ");
                    self.pattern(&case.pattern);
                    if let Some(guard) = &case.guard {
                        self.write(" if ");
                        self.expr(guard, prec::TEST);
                    }
                    self.write(":");
                    self.body(&case.body, false);
                }
            }
            Stmt::Pass(_) | Stmt::IpyEscapeCommand(_) => self.fill("pass"),
            Stmt::Break(_) => self.fill("break"),
            Stmt::Continue(_) => self.fill("continue"),
        }
    }

    fn comma_exprs(&mut self, exprs: &[Expr]) {
        let mut first = true;
        for e in exprs {
            if !first {
                self.write(", ");
            }
            first = false;
            self.expr(e, prec::TEST);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn expr_inner(&mut self, expr: &Expr, ctx: u8) {
        if self.in_type {
            if let (Some(reference), Some(tmap)) = (type_reference(expr), self.tmap.as_mut()) {
                let slot = dup_defs_core::type_slot(tmap, &reference);
                if !self.declared_types.contains(&reference) {
                    self.mentions.push(reference);
                }
                self.write(&slot);
                return;
            }
        }
        match expr {
            Expr::Name(n) => {
                let id = self.rename_id(n.id.as_str());
                self.write(&id);
            }
            Expr::NumberLiteral(n) => self.write(&repr_number(&n.value)),
            Expr::StringLiteral(s) => self.write(&repr_str(s.value.to_str())),
            Expr::BytesLiteral(b) => {
                let bytes: Vec<u8> = b.value.bytes().collect();
                self.write(&repr_bytes(&bytes));
            }
            Expr::BooleanLiteral(b) => self.write(if b.value { "True" } else { "False" }),
            // NoneLiteral and the (Jupyter-only, never-occurring) IpyEscapeCommand both render `None`.
            Expr::NoneLiteral(_) | Expr::IpyEscapeCommand(_) => self.write("None"),
            Expr::EllipsisLiteral(_) => self.write("..."),
            Expr::FString(f) => {
                // Iterate PARTS (not `.elements()`, which skips plain `StringLiteral` parts in
                // implicit concatenation like `f"a{x}" "b"`), then merge as CPython does.
                let mut body = String::new();
                for part in &f.value {
                    match part {
                        ast::FStringPart::Literal(s) => fstring_literal_into(&mut body, &s.value),
                        ast::FStringPart::FString(fs) => {
                            for element in &fs.elements {
                                let chunk = self.element_str(element);
                                body.push_str(&chunk);
                            }
                        }
                    }
                }
                self.write_fstring(&body, "f");
            }
            Expr::TString(t) => {
                let mut body = String::new();
                for tstr in &t.value {
                    for element in &tstr.elements {
                        let chunk = self.element_str(element);
                        body.push_str(&chunk);
                    }
                }
                self.write_fstring(&body, "t");
            }
            Expr::Attribute(a) => {
                self.expr(&a.value, prec::ATOM);
                self.write(".");
                self.write(a.attr.id.as_str());
            }
            Expr::Call(c) => {
                let outer = std::mem::replace(&mut self.in_type, false);
                self.expr(&c.func, prec::ATOM);
                self.write("(");
                let mut first = true;
                for arg in &c.arguments.args {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    self.expr(arg, prec::TEST);
                }
                for kw in &c.arguments.keywords {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    self.keyword(kw);
                }
                self.write(")");
                self.in_type = outer;
            }
            Expr::Subscript(s) => {
                self.expr(&s.value, prec::ATOM);
                self.write("[");
                match s.slice.as_ref() {
                    Expr::Tuple(t) if !t.elts.is_empty() => self.items_view(&t.elts),
                    other => self.expr(other, prec::TEST),
                }
                self.write("]");
            }
            Expr::Starred(s) => {
                self.write("*");
                self.expr(&s.value, prec::EXPR);
            }
            Expr::List(l) => {
                self.write("[");
                self.comma_exprs(&l.elts);
                self.write("]");
            }
            Expr::Tuple(t) => {
                let parens = t.elts.is_empty() || ctx > prec::TUPLE;
                if parens {
                    self.write("(");
                }
                self.items_view(&t.elts);
                if parens {
                    self.write(")");
                }
            }
            Expr::Set(s) => {
                if s.elts.is_empty() {
                    self.write("{*()}");
                } else {
                    self.write("{");
                    self.comma_exprs(&s.elts);
                    self.write("}");
                }
            }
            Expr::Dict(d) => {
                self.write("{");
                let mut first = true;
                for item in &d.items {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    if let Some(k) = &item.key {
                        self.expr(k, prec::TEST);
                        self.write(": ");
                        self.expr(&item.value, prec::TEST);
                    } else {
                        self.write("**");
                        self.expr(&item.value, prec::EXPR);
                    }
                }
                self.write("}");
            }
            Expr::Slice(s) => {
                if let Some(l) = &s.lower {
                    self.expr(l, prec::TEST);
                }
                self.write(":");
                if let Some(u) = &s.upper {
                    self.expr(u, prec::TEST);
                }
                if let Some(step) = &s.step {
                    self.write(":");
                    self.expr(step, prec::TEST);
                }
            }
            Expr::BoolOp(b) => {
                let (op, p) = match b.op {
                    BoolOp::And => (" and ", prec::AND),
                    BoolOp::Or => (" or ", prec::OR),
                };
                let parens = ctx > p;
                if parens {
                    self.write("(");
                }
                let mut level = p;
                let mut first = true;
                for v in &b.values {
                    if !first {
                        self.write(op);
                    }
                    first = false;
                    level = next_prec(level);
                    self.expr(v, level);
                }
                if parens {
                    self.write(")");
                }
            }
            Expr::BinOp(b) => {
                let op = operator_tag(b.op);
                let p = binop_prec(b.op);
                let parens = ctx > p;
                if parens {
                    self.write("(");
                }
                let (lp, rp) = if matches!(b.op, Operator::Pow) { (next_prec(p), p) } else { (p, next_prec(p)) };
                self.expr(&b.left, lp);
                let _ = write!(self.out, " {op} ");
                self.expr(&b.right, rp);
                if parens {
                    self.write(")");
                }
            }
            Expr::UnaryOp(u) => {
                let (op, p) = match u.op {
                    UnaryOp::Invert => ("~", prec::FACTOR),
                    UnaryOp::Not => ("not", prec::NOT),
                    UnaryOp::UAdd => ("+", prec::FACTOR),
                    UnaryOp::USub => ("-", prec::FACTOR),
                };
                let parens = ctx > p;
                if parens {
                    self.write("(");
                }
                self.write(op);
                if p != prec::FACTOR {
                    self.write(" ");
                }
                self.expr(&u.operand, p);
                if parens {
                    self.write(")");
                }
            }
            Expr::Compare(c) => {
                let parens = ctx > prec::CMP;
                if parens {
                    self.write("(");
                }
                self.expr(&c.left, next_prec(prec::CMP));
                for (op, comp) in c.ops.iter().zip(c.comparators.iter()) {
                    let _ = write!(self.out, " {} ", cmpop_tag(*op));
                    self.expr(comp, next_prec(prec::CMP));
                }
                if parens {
                    self.write(")");
                }
            }
            Expr::If(i) => {
                let parens = ctx > prec::TEST;
                if parens {
                    self.write("(");
                }
                self.expr(&i.body, next_prec(prec::TEST));
                self.write(" if ");
                self.expr(&i.test, next_prec(prec::TEST));
                self.write(" else ");
                self.expr(&i.orelse, prec::TEST);
                if parens {
                    self.write(")");
                }
            }
            Expr::Lambda(l) => {
                let parens = ctx > prec::TEST;
                if parens {
                    self.write("(");
                }
                self.write("lambda");
                let args = self.buffered(|s| s.arguments(l.parameters.as_deref()));
                if !args.is_empty() {
                    self.write(" ");
                    self.write(&args);
                }
                self.write(": ");
                self.expr(&l.body, prec::TEST);
                if parens {
                    self.write(")");
                }
            }
            Expr::Named(n) => {
                let parens = ctx > prec::NAMED_EXPR;
                if parens {
                    self.write("(");
                }
                self.expr(&n.target, prec::ATOM);
                self.write(" := ");
                self.expr(&n.value, prec::ATOM);
                if parens {
                    self.write(")");
                }
            }
            Expr::Await(a) => {
                let parens = ctx > prec::AWAIT;
                if parens {
                    self.write("(");
                }
                self.write("await ");
                self.expr(&a.value, prec::ATOM);
                if parens {
                    self.write(")");
                }
            }
            Expr::Yield(y) => {
                let parens = ctx > prec::YIELD;
                if parens {
                    self.write("(");
                }
                self.write("yield");
                if let Some(v) = &y.value {
                    self.write(" ");
                    self.expr(v, prec::ATOM);
                }
                if parens {
                    self.write(")");
                }
            }
            Expr::YieldFrom(y) => {
                let parens = ctx > prec::YIELD;
                if parens {
                    self.write("(");
                }
                self.write("yield from ");
                self.expr(&y.value, prec::ATOM);
                if parens {
                    self.write(")");
                }
            }
            Expr::ListComp(c) => {
                self.write("[");
                self.expr(&c.elt, prec::TEST);
                for g in &c.generators {
                    self.comprehension(g);
                }
                self.write("]");
            }
            Expr::SetComp(c) => {
                self.write("{");
                self.expr(&c.elt, prec::TEST);
                for g in &c.generators {
                    self.comprehension(g);
                }
                self.write("}");
            }
            Expr::Generator(c) => {
                self.write("(");
                self.expr(&c.elt, prec::TEST);
                for g in &c.generators {
                    self.comprehension(g);
                }
                self.write(")");
            }
            Expr::DictComp(c) => {
                self.write("{");
                self.expr(&c.key, prec::TEST);
                self.write(": ");
                self.expr(&c.value, prec::TEST);
                for g in &c.generators {
                    self.comprehension(g);
                }
                self.write("}");
            }
        }
    }

    fn items_view(&mut self, elts: &[Expr]) {
        if elts.len() == 1 {
            self.expr(&elts[0], prec::TEST);
            self.write(",");
        } else {
            self.comma_exprs(elts);
        }
    }

    fn comprehension(&mut self, comp: &Comprehension) {
        self.write(if comp.is_async { " async for " } else { " for " });
        self.expr(&comp.target, prec::TUPLE);
        self.write(" in ");
        self.expr(&comp.iter, next_prec(prec::TEST));
        for cond in &comp.ifs {
            self.write(" if ");
            self.expr(cond, next_prec(prec::TEST));
        }
    }

    fn keyword(&mut self, kw: &Keyword) {
        match &kw.arg {
            Some(name) => {
                self.write(name.id.as_str());
                self.write("=");
            }
            None => self.write("**"),
        }
        self.expr(&kw.value, prec::TEST);
    }

    fn alias(&mut self, alias: &ast::Alias) {
        self.write(alias.name.id.as_str());
        if let Some(asname) = &alias.asname {
            self.write(" as ");
            self.write(asname.id.as_str());
        }
    }

    fn arg(&mut self, param: &Parameter) {
        let name = self.rename_id(param.name.id.as_str());
        self.write(&name);
        if let Some(annotation) = &param.annotation {
            self.write(": ");
            self.annotation(annotation, prec::TEST);
        }
    }

    fn arguments(&mut self, params: Option<&Parameters>) {
        let Some(p) = params else { return };
        let mut first = true;
        let posonly = p.posonlyargs.len();
        for (index, x) in p.posonlyargs.iter().chain(p.args.iter()).enumerate() {
            if first {
                first = false;
            } else {
                self.write(", ");
            }
            self.arg(&x.parameter);
            if let Some(default) = &x.default {
                self.write("=");
                self.expr(default, prec::TEST);
            }
            if index + 1 == posonly {
                self.write(", /");
            }
        }
        if p.vararg.is_some() || !p.kwonlyargs.is_empty() {
            if first {
                first = false;
            } else {
                self.write(", ");
            }
            self.write("*");
            if let Some(vararg) = &p.vararg {
                self.arg(vararg);
            }
        }
        for x in &p.kwonlyargs {
            self.write(", ");
            self.arg(&x.parameter);
            if let Some(default) = &x.default {
                self.write("=");
                self.expr(default, prec::TEST);
            }
        }
        if let Some(kwarg) = &p.kwarg {
            if first {
                first = false;
            } else {
                self.write(", ");
            }
            self.write("**");
            self.arg(kwarg);
        }
        let _ = first;
    }

    fn type_params(&mut self, params: Option<&ast::TypeParams>) {
        let Some(tps) = params else { return };
        if tps.type_params.is_empty() {
            return;
        }
        self.write("[");
        let mut first = true;
        for tp in &tps.type_params {
            if !first {
                self.write(", ");
            }
            first = false;
            match tp {
                ast::TypeParam::TypeVar(t) => {
                    self.type_param_name(t.name.id.as_str());
                    if let Some(bound) = &t.bound {
                        self.write(": ");
                        self.annotation(bound, prec::TEST);
                    }
                    if let Some(default) = &t.default {
                        self.write(" = ");
                        self.annotation(default, prec::TEST);
                    }
                }
                ast::TypeParam::TypeVarTuple(t) => {
                    self.write("*");
                    self.type_param_name(t.name.id.as_str());
                }
                ast::TypeParam::ParamSpec(t) => {
                    self.write("**");
                    self.type_param_name(t.name.id.as_str());
                }
            }
        }
        self.write("]");
    }

    /// Choose a quote (CPython `_ftstring_helper`: ALL_QUOTES order, skip any appearing in the body,
    /// prefer one whose char ≠ the body's last char; escape a forced final triple-quote char) and
    /// write `f'…'` / `t'…'`. Body already has literals escaped and interpolations rendered.
    fn write_fstring(&mut self, body: &str, prefix: &str) {
        let mut candidates: Vec<&str> =
            ["'", "\"", "\"\"\"", "'''"].into_iter().filter(|q| !body.contains(*q)).collect();
        self.write(prefix);
        let Some(&first) = candidates.first() else {
            // Body contains every quote form (pathological); best-effort triple-single.
            self.write("'''");
            self.write(body);
            self.write("'''");
            return;
        };
        // Stable: quotes whose first char ≠ body's last char sort first (avoids escaping a final quote).
        candidates.sort_by_key(|q| body.chars().last() == q.chars().next());
        let quote = *candidates.first().unwrap_or(&first);
        self.write(quote);
        if quote.len() == 3 && body.chars().last() == quote.chars().next() {
            let mut escaped = body.to_owned();
            let last = escaped.pop().unwrap_or(' ');
            escaped.push('\\');
            escaped.push(last);
            self.write(&escaped);
        } else {
            self.write(body);
        }
        self.write(quote);
    }

    /// One f-string element → its rendered chunk. A `{x=}` debug interpolation prepends the source
    /// text (`leading + <expr source> + trailing`) as a literal, exactly as CPython bakes it.
    fn element_str(&mut self, element: &ast::InterpolatedStringElement) -> String {
        match element {
            ast::InterpolatedStringElement::Literal(lit) => fstring_literal_escaped(&lit.value),
            ast::InterpolatedStringElement::Interpolation(interp) => {
                let mut out = String::new();
                if let Some(dbg) = &interp.debug_text {
                    let range = interp.expression.range();
                    let expr_src =
                        self.src.get(usize::from(range.start())..usize::from(range.end())).unwrap_or("");
                    out.push_str(&fstring_literal_escaped(&format!("{}{expr_src}{}", dbg.leading, dbg.trailing)));
                }
                out.push_str(&self.interpolation(interp));
                out
            }
        }
    }

    fn interpolation(&mut self, interp: &ast::InterpolatedElement) -> String {
        let mut out = String::from("{");
        let value = self.buffered_expr(&interp.expression, next_prec(prec::TEST));
        if value.starts_with('{') {
            out.push(' ');
        }
        out.push_str(&value);
        let mut conversion = match interp.conversion {
            ast::ConversionFlag::Str => Some('s'),
            ast::ConversionFlag::Ascii => Some('a'),
            ast::ConversionFlag::Repr => Some('r'),
            ast::ConversionFlag::None => None,
        };
        // A bare `{x=}` (debug, no explicit conversion / format spec) implies `!r`, like CPython.
        if conversion.is_none() && interp.debug_text.is_some() && interp.format_spec.is_none() {
            conversion = Some('r');
        }
        if let Some(c) = conversion {
            out.push('!');
            out.push(c);
        }
        if let Some(spec) = &interp.format_spec {
            out.push(':');
            for element in &spec.elements {
                match element {
                    ast::InterpolatedStringElement::Literal(lit) => {
                        let mut v = lit.value.replace('{', "{{").replace('}', "}}");
                        v = v.replace('\\', "\\\\").replace('\'', "\\'").replace('"', "\\\"").replace('\n', "\\n");
                        out.push_str(&v);
                    }
                    ast::InterpolatedStringElement::Interpolation(inner) => {
                        out.push_str(&self.interpolation(inner));
                    }
                }
            }
        }
        out.push('}');
        out
    }

    fn pattern(&mut self, pat: &ast::Pattern) {
        match pat {
            ast::Pattern::MatchValue(p) => self.expr(&p.value, prec::TEST),
            ast::Pattern::MatchSingleton(p) => self.write(match p.value {
                ast::Singleton::None => "None",
                ast::Singleton::True => "True",
                ast::Singleton::False => "False",
            }),
            ast::Pattern::MatchSequence(p) => {
                self.write("[");
                let mut first = true;
                for x in &p.patterns {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    self.pattern(x);
                }
                self.write("]");
            }
            ast::Pattern::MatchMapping(p) => {
                self.write("{");
                let mut first = true;
                for (k, v) in p.keys.iter().zip(p.patterns.iter()) {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    self.expr(k, prec::TEST);
                    self.write(": ");
                    self.pattern(v);
                }
                if let Some(rest) = &p.rest {
                    if !p.keys.is_empty() {
                        self.write(", ");
                    }
                    let _ = write!(self.out, "**{}", rest.id.as_str());
                }
                self.write("}");
            }
            ast::Pattern::MatchClass(p) => {
                self.expr(&p.cls, prec::ATOM);
                self.write("(");
                let mut first = true;
                for x in &p.arguments.patterns {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    self.pattern(x);
                }
                for kw in &p.arguments.keywords {
                    if !first {
                        self.write(", ");
                    }
                    first = false;
                    let _ = write!(self.out, "{}=", kw.attr.id.as_str());
                    self.pattern(&kw.pattern);
                }
                self.write(")");
            }
            ast::Pattern::MatchStar(p) => {
                let name = p.name.as_ref().map_or("_", |n| n.id.as_str());
                let _ = write!(self.out, "*{name}");
            }
            ast::Pattern::MatchAs(p) => match (&p.pattern, &p.name) {
                (None, None) => self.write("_"),
                (None, Some(name)) => self.write(name.id.as_str()),
                (Some(inner), name) => {
                    self.pattern(inner);
                    if let Some(name) = name {
                        let _ = write!(self.out, " as {}", name.id.as_str());
                    }
                }
            },
            ast::Pattern::MatchOr(p) => {
                let mut first = true;
                for x in &p.patterns {
                    if !first {
                        self.write(" | ");
                    }
                    first = false;
                    self.pattern(x);
                }
            }
        }
    }
}

/// A type reference as written — `X`, `typing.Optional` — or `None` when `expr` is not a name or a
/// chain of attribute reads off one. A dotted path names one type, so it becomes one slot.
fn type_reference(expr: &Expr) -> Option<String> {
    let (root, attrs) = crate::decides::chain(expr);
    let root = root?;
    Some(std::iter::once(root).chain(attrs).collect::<Vec<_>>().join("."))
}

/// What one unparse of a function yields: its lines with their block depth, and — in the Type-3
/// mode (`tmap` given) — the type references turned into slots and the sub-expression units.
struct Unparsed {
    lines: Vec<(usize, String)>,
    type_refs: Vec<String>,
    units: Vec<(u32, String)>,
}

/// Unparse the alpha-renamed function into stripped, non-empty lines keeping each line's **block
/// nesting**, plus the type references the unparse turned into slots and the sub-expression units
/// (neither when `tmap` is `None`, which writes types as spelled — the statement stream).
///
/// Order alone cannot tell `for _v0 in _v1: / foo() / bar()` from the same three statements where
/// `bar()` runs *after* the loop instead of inside it — flattened, both are one sequence. Depth is
/// what says which block a statement belongs to, and therefore whether a line following a run
/// continues the path that run described or is what happens when that path was not taken.
fn unparse_depths(
    stmt: &Stmt,
    src: &str,
    locals: &HashSet<String>,
    map: HashMap<String, u32>,
    tmap: Option<HashMap<String, u32>>,
) -> Unparsed {
    let units = tmap.as_ref().map(|_| Vec::new());
    let mut up = Unparse {
        out: String::new(),
        src,
        locals: Some(locals),
        map,
        blanked: false,
        indent: 0,
        tmap,
        in_type: false,
        mentions: Vec::new(),
        declared_types: HashSet::new(),
        units,
        fills: 0,
        expr_depth: 0,
    };
    up.stmt(stmt);
    let lines = up
        .out
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| ((l.len() - l.trim_start().len()) / 4, l.trim().to_owned()))
        .collect();
    Unparsed { lines, type_refs: up.mentions, units: up.units.unwrap_or_default() }
}

// ---------------------------------------------------------------------------
// Public entry points (unchanged signatures).
// ---------------------------------------------------------------------------

/// CPython-`ast.dump`-shaped canonical of the leading def in `text` (names preserved, docstrings
/// stripped), or the raw text if it does not parse / has no statements. Single-text entry point.
#[must_use]
pub fn ast_canonical(text: &str) -> String {
    cluster_canonical(text)
}

/// CPython-`ast.dump`-shaped canonical of the leading def in `text` (names preserved, docstrings
/// stripped), or the raw text if it does not parse / has no statements. Used by the clustering pass.
fn cluster_canonical(text: &str) -> String {
    let Ok(parsed) = parse_module(text) else {
        return text.to_string();
    };
    let module = parsed.into_syntax();
    let Some(stmt) = module.body.first() else {
        return text.to_string();
    };
    let mut dump = Dump::new(text, None);
    dump.stmt(stmt);
    dump.out
}

/// Batch canonicalize def texts (functions / classes / …) in parallel — replaces the Python
/// `ast_canonical` loop. Returns one canonical string per input, in order.
#[must_use]
pub fn ast_canonical_many(texts: &[String]) -> Vec<String> {
    texts.par_iter().map(|text| cluster_canonical(text)).collect()
}

/// Name-agnostic forms of one function: `(alpha-renamed canonical, per-statement renamed lines,
/// node count)`, or `None` if `text` is not a single function definition.
fn normalize_one(text: &str) -> Option<(String, Vec<String>, usize)> {
    let parsed = parse_module(text).ok()?;
    let module = parsed.into_syntax();
    let stmt = module.body.first()?;
    if !matches!(stmt, Stmt::FunctionDef(_)) {
        return None; // ruff folds async into StmtFunctionDef; classes/others are out of scope here
    }
    // Two normal forms, as in `analyze_stmt`: the lines keep every statement, the canonical inlines.
    let statement_form = crate::normal::normal_form(stmt, false);
    let line_locals = collect_locals(&statement_form);
    let typed = unparse_depths(&statement_form, text, &line_locals, HashMap::new(), Some(HashMap::new()));
    let lines = typed.lines.into_iter().map(|(_, line)| line).collect();

    let exact_form = crate::normal::with_inlining(&statement_form);
    let locals = collect_locals(&exact_form); // top fn's params (nested defs' params are not locals)
    let mut dump = Dump::new(text, Some(&locals));
    dump.stmt(&exact_form);
    let canonical = std::mem::take(&mut dump.out);
    let size = dump.count;
    Some((canonical, lines, size))
}

/// Batch alpha-rename canonicalize function texts in parallel — replaces the Python `_analyze`
/// (cross-name + Type-3 canonicalization). `None` entries are non-function texts.
#[must_use]
pub fn normalize_functions(texts: &[String]) -> Vec<Option<(String, Vec<String>, usize)>> {
    texts.par_iter().map(|text| normalize_one(text)).collect()
}

/// Full dup-defs analysis of one function FROM AN AST NODE (no re-parse): `(cluster_canonical,
/// xname_canonical, lines, size)`, or `None` if `stmt` is not a function def. `src` is the source the
/// node's ranges index into (for f-string `{x=}` debug text). The node's own (top) decorators are
/// excluded — matching the decorator-stripped def *text* the text-based path canonicalized — so a
/// canonical built from a file's AST node is byte-identical to one built by re-parsing the def text.
pub(crate) fn analyze_stmt(stmt: &Stmt, src: &str, facets: bool) -> Option<AnalyzedFn> {
    let Stmt::FunctionDef(_) = stmt else {
        return None;
    };
    let mut cluster_dump = Dump::new(src, None);
    cluster_dump.stmt(stmt);
    let cluster_canonical = std::mem::take(&mut cluster_dump.out);

    // The statement stream keeps the definition as written: converge reads it to say where two
    // definitions stop agreeing, and a rewritten line would point the reader at code that is not
    // there. A run that never reads facets skips the whole walk.
    //
    // The slot map comes from a names-renamed dump of the same tree: the statement unparse must
    // number bound locals in the DUMP walk's order, or the stream text drifts byte-wise (measured:
    // the unparse's own numbering reorders `_vN` slots on nested defs). The dump's render itself
    // is now linear, so the pass is cheap — only the map is read.
    let statements = if facets {
        let locals = collect_locals(stmt);
        let mut dump = Dump::new(src, Some(&locals));
        dump.stmt(stmt);
        statement_stream(unparse_depths(stmt, src, &locals, dump.map, None).lines)
    } else {
        Vec::new()
    };

    // The name-agnostic canonicals are taken from the normal form (`normal.rs`): what the exact and
    // Type-3 passes compare is the meaning the rules reach, not the spelling. The Type-3 lines come
    // from the form that keeps every statement; the exact canonical from the one that also inlines
    // single-use temporaries — see `normal_form` for why the two passes read two forms. Slot numbers
    // in the lines are their own (the pass renumbers per line), so the lines need no dump of theirs.
    let statement_form = crate::normal::normal_form(stmt, false);
    let line_locals = collect_locals(&statement_form);
    let typed = unparse_depths(&statement_form, src, &line_locals, HashMap::new(), Some(HashMap::new()));
    let type3_lines = typed.lines.into_iter().map(|(_, line)| line).collect();

    let exact_form = crate::normal::with_inlining(&statement_form);
    let locals = collect_locals(&exact_form);
    let mut dump = Dump::new(src, Some(&locals));
    dump.stmt(&exact_form);
    let xname_canonical = std::mem::take(&mut dump.out);
    let size = dump.count;
    Some(AnalyzedFn {
        cluster_canonical,
        xname_canonical,
        type3_lines,
        type_refs: typed.type_refs,
        type3_units: typed.units,
        statements,
        size,
    })
}

/// The unparser's `(depth, line)` pairs as the engine's statement stream.
///
/// One line per statement, as the unparser walks into nested blocks, with the types spelled as
/// written: the Type-3 lines are the same statements with their types as slots (reported apart as
/// [`AnalyzedFn::type_refs`]), a second walk because the two readers need different things from an
/// annotation — converge parts two definitions on the names they use, a type among them.
fn statement_stream(pairs: Vec<(usize, String)>) -> Vec<Statement> {
    pairs
        .into_iter()
        .map(|(depth, line)| Statement { line, depth: u16::try_from(depth).unwrap_or(u16::MAX) })
        .collect()
}

/// Names-preserved cluster canonical of any def node (functions AND classes), decorators of the top
/// node excluded. Node-based counterpart of `cluster_canonical`, used by the scan to canonicalize
/// classes without a re-parse.
pub(crate) fn cluster_canonical_node(stmt: &Stmt, src: &str) -> String {
    let mut dump = Dump::new(src, None);
    dump.stmt(stmt);
    dump.out
}

/// Bound-local set of a definition — the same alpha-rename target set the body canon uses
/// ([`Collect`]), exposed so the use-profile canon renames a site's locals identically. For a
/// `FunctionDef` this is params + every assignment/loop/with/except/import/walrus/comprehension
/// target; for any other statement, just the targets it binds.
pub(crate) fn collect_locals(stmt: &Stmt) -> HashSet<String> {
    let mut collect = Collect::default();
    if let Stmt::FunctionDef(func) = stmt {
        collect.add_params(&func.parameters);
    }
    collect.visit_stmt(stmt);
    collect.bound
}

/// Canonicalize one **use site** of `anchor`: the statement's tag plus its *direct* expressions
/// (the caller stops at nested-statement boundaries, so a compound statement contributes its
/// header, not its body). Locals rename to `_v{n}` exactly as in the body canon, the anchor itself
/// becomes `_t0`, and attributes read off the anchor come back marked `#name#` for the caller to
/// renumber once the whole profile is canonically ordered. Returns `(canonical, node_count)`.
pub(crate) fn canon_site(
    tag: &str,
    exprs: &[&Expr],
    src: &str,
    locals: &HashSet<String>,
    anchor: &str,
) -> (String, usize) {
    let mut dump = Dump::with_anchor(src, Some(locals), anchor);
    dump.out.push_str(tag);
    dump.out.push('(');
    let mut first = true;
    for e in exprs {
        dup_defs_core::comma_sep(&mut dump.out, &mut first);
        dump.expr(e);
    }
    dump.out.push(')');
    (dump.out, dump.count)
}

/// The module-scope canonical alone — what the `scope` lens contributes. Its predecessor rendered
/// the definition three ways (a names-preserved dump, this one, and a full unparse) and the lens
/// read one of them; the other two were computed per function to be dropped.
pub(crate) fn scope_canonical(stmt: &Stmt, src: &str, scope: &HashSet<String>) -> String {
    let mut collect = Collect::default();
    if let Stmt::FunctionDef(func) = stmt {
        collect.add_params(&func.parameters);
    }
    collect.visit_stmt(stmt);
    let mut dump = Dump::with_scope(src, &collect.bound, scope);
    dump.stmt(stmt);
    dump.out
}

/// Every name this definition mentions — the raw material for [`Facets::reaches`].
///
/// Deliberately *not* filtered by boundness: a name that shadows an import inside one function is
/// rare, and treating the shadow as evidence the definition does not reach the module would be a
/// silent false negative, which is the expensive direction here. The frontend intersects this with
/// what its file imported; a name nothing imported simply matches nothing.
///
/// [`Facets::reaches`]: dup_defs_core::Facets::reaches
pub(crate) fn used_names(stmt: &Stmt) -> HashSet<String> {
    #[derive(Default)]
    struct Names {
        seen: HashSet<String>,
    }
    impl<'a> Visitor<'a> for Names {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if let Expr::Name(name) = expr {
                self.seen.insert(name.id.as_str().to_owned());
            }
            visitor::walk_expr(self, expr);
        }
    }
    let mut names = Names::default();
    names.visit_stmt(stmt);
    names.seen
}

/// Every name the module binds at top level: imports, `def`/`class` names, module-level assignment
/// targets, and the fields declared in its class bodies. Purely structural — "did this file
/// introduce this name" — with no notion of what any of them mean.
pub(crate) fn module_bound_names(module: &[Stmt]) -> HashSet<String> {
    let mut collect = Collect::default();
    for stmt in module {
        match stmt {
            Stmt::FunctionDef(f) => {
                collect.bound.insert(f.name.id.as_str().to_owned());
            }
            Stmt::ClassDef(c) => {
                collect.bound.insert(c.name.id.as_str().to_owned());
                // Class body assignments are the fields the module declares.
                for inner in &c.body {
                    match inner {
                        Stmt::Assign(a) => a.targets.iter().for_each(|t| collect.add_target(t)),
                        Stmt::AnnAssign(a) => collect.add_target(&a.target),
                        _ => {}
                    }
                }
            }
            Stmt::Assign(a) => a.targets.iter().for_each(|t| collect.add_target(t)),
            Stmt::AnnAssign(a) => collect.add_target(&a.target),
            Stmt::Import(_) | Stmt::ImportFrom(_) => collect.visit_stmt(stmt),
            _ => {}
        }
    }
    collect.bound
}

pub(crate) fn analyze_one(text: &str) -> Option<AnalyzedFn> {
    let module = parse_module(text).ok()?.into_syntax();
    let stmt = module.body.first()?;
    analyze_stmt(stmt, text, true)
}

/// Batch `analyze_one` in parallel — one parse per function, all dup-defs canonical forms at once.
#[must_use]
pub fn analyze_functions(texts: &[String]) -> Vec<Option<AnalyzedFn>> {
    texts.par_iter().map(|text| analyze_one(text)).collect()
}

// ---------------------------------------------------------------------------
// Imports: what a module declares it reaches into
// ---------------------------------------------------------------------------

/// One name a file binds by importing, and the dotted path that name stands for.
///
/// The path is what was **written**, joined whole: `from a.b import c` binds `c` to `a.b.c`, and
/// `from a.b import c as d` binds `d` to that same `a.b.c`. Recording the whole path rather than
/// the module is what lets two files that reach one thing through different doors — one importing
/// the module, the other importing a member of it — meet on a **prefix** of the path instead of on
/// a name. That prefix is the referent they share, and the corpus states it itself; nothing here
/// decides which paths are worth caring about.
///
/// Relative imports keep their leading dots, so a `.` path is not silently unified with an absolute
/// one that happens to end the same way.
#[derive(Debug, Clone)]
pub(crate) struct Import {
    /// The name the file can use afterwards.
    pub local: String,
    /// Dotted path, as written.
    pub path: String,
}

/// Every import in `src`, including those written inside a function body.
///
/// Function-local imports are kept because they are a definition reaching for something just as
/// much as a file-level one is — often more pointedly, since a local import is usually there to
/// break a cycle around one specific call.
#[must_use]
pub(crate) fn module_imports(src: &str) -> Vec<Import> {
    let Ok(parsed) = parse_module(src) else { return Vec::new() };
    let mut out = Vec::new();
    walk_imports(&parsed.into_syntax().body, &mut out);
    out
}

fn walk_imports(body: &[Stmt], out: &mut Vec<Import>) {
    for stmt in body {
        match stmt {
            Stmt::Import(node) => {
                for alias in &node.names {
                    let path = alias.name.id.as_str().to_owned();
                    let local = match &alias.asname {
                        Some(asname) => asname.id.as_str().to_owned(),
                        // Plain `import a.b.c` binds only `a`, and a use reads `a.b.c.f` — so the
                        // name bound stands for the head of the path, not the whole of it.
                        None => path.split('.').next().unwrap_or("").to_owned(),
                    };
                    let path = if alias.asname.is_some() { path } else { local.clone() };
                    out.push(Import { local, path });
                }
            }
            Stmt::ImportFrom(node) => {
                let mut prefix = ".".repeat(node.level as usize);
                if let Some(module) = &node.module {
                    prefix.push_str(module.id.as_str());
                }
                for alias in &node.names {
                    let member = alias.name.id.as_str();
                    if member == "*" {
                        continue; // binds no name this can attribute a use to
                    }
                    let local = alias.asname.as_ref().unwrap_or(&alias.name).id.as_str().to_owned();
                    let path =
                        if prefix.ends_with('.') { format!("{prefix}{member}") } else { format!("{prefix}.{member}") };
                    out.push(Import { local, path });
                }
            }
            _ => walk_imports(stmt_body(stmt), out),
        }
    }
}

/// Statements nested directly inside `stmt`, so the walk reaches imports written in a function or
/// under a `try`.
fn stmt_body(stmt: &Stmt) -> &[Stmt] {
    match stmt {
        Stmt::FunctionDef(node) => &node.body,
        Stmt::ClassDef(node) => &node.body,
        Stmt::If(node) => &node.body,
        Stmt::With(node) => &node.body,
        Stmt::While(node) => &node.body,
        Stmt::For(node) => &node.body,
        Stmt::Try(node) => &node.body,
        _ => &[],
    }
}

#[cfg(test)]
mod import_tests {
    use super::module_imports;

    fn pairs(src: &str) -> Vec<(String, String)> {
        module_imports(src).into_iter().map(|i| (i.local, i.path)).collect()
    }

    #[test]
    fn from_import_binds_the_whole_dotted_path() {
        // The path is what makes two files meet: importing the module and importing a member of it
        // are different strings that share a prefix, and the prefix is the thing both are about.
        assert_eq!(
            pairs("from a.b import c\n"),
            vec![("c".to_owned(), "a.b.c".to_owned())]
        );
        assert_eq!(
            pairs("from a.b.c import d\n"),
            vec![("d".to_owned(), "a.b.c.d".to_owned())]
        );
    }

    #[test]
    fn an_alias_renames_the_binding_but_not_the_path() {
        assert_eq!(
            pairs("from a.b import c as z\n"),
            vec![("z".to_owned(), "a.b.c".to_owned())]
        );
    }

    #[test]
    fn plain_import_binds_only_the_head() {
        // `import a.b.c` makes `a` the name in scope; a use reads `a.b.c.f`, so the bound name
        // stands for the head and nothing deeper can be attributed to it here.
        assert_eq!(pairs("import a.b.c\n"), vec![("a".to_owned(), "a".to_owned())]);
        assert_eq!(pairs("import a.b.c as d\n"), vec![("d".to_owned(), "a.b.c".to_owned())]);
    }

    #[test]
    fn relative_imports_keep_their_dots() {
        // Otherwise `.models` from two different packages would unify into one referent.
        assert_eq!(pairs("from . import x\n"), vec![("x".to_owned(), ".x".to_owned())]);
        assert_eq!(pairs("from ..pkg import y\n"), vec![("y".to_owned(), "..pkg.y".to_owned())]);
    }

    #[test]
    fn star_binds_no_name_to_attribute_a_use_to() {
        let bound = pairs("from a.b import *\n");
        assert!(bound.is_empty(), "a star import binds no name, got: {bound:?}");
    }

    #[test]
    fn imports_inside_a_body_are_reached() {
        // A local import is a definition reaching for something just as much as a file-level one —
        // often more pointedly, since it is usually there to break a cycle around one call.
        let src = "def f():\n    from a.b import c\n    return c\n";
        assert_eq!(pairs(src), vec![("c".to_owned(), "a.b.c".to_owned())]);
        let guarded = "if TYPE_CHECKING:\n    from a.b import c\n";
        assert_eq!(pairs(guarded), vec![("c".to_owned(), "a.b.c".to_owned())]);
    }

    #[test]
    fn unparsable_source_yields_nothing_rather_than_panicking() {
        let bound = pairs("def (:\n");
        assert!(bound.is_empty(), "unparsable source binds nothing, got: {bound:?}");
    }
}

#[cfg(test)]
mod type_slot_tests {
    use super::{analyze_one, AnalyzedFn};

    fn analyzed(src: &str) -> AnalyzedFn {
        analyze_one(src).expect("a function definition")
    }

    #[test]
    fn bodies_differing_only_in_declared_types_are_one_canonical() {
        // The Type-2 clone by definition: Python runs the same code whatever the annotations say.
        let a = analyzed("def f(p: Token) -> int | None:\n    v = p.get('uid')\n    return v\n");
        let b = analyzed("def g(q: Claims) -> int | None:\n    v = q.get('uid')\n    return v\n");
        assert_eq!(a.xname_canonical, b.xname_canonical);
    }

    #[test]
    fn types_are_renamed_consistently_not_erased() {
        // Parameterized matching: one bijection has to carry the types of one onto the other's.
        let same = analyzed("def f(a: X, b: X) -> None:\n    pass\n");
        let mixed = analyzed("def f(a: X, b: Y) -> None:\n    pass\n");
        assert_ne!(same.xname_canonical, mixed.xname_canonical);
    }

    #[test]
    fn a_call_inside_an_annotation_is_a_value_and_stays() {
        // Metadata like `Depends(...)` is read at runtime — exactly what keeps two otherwise equal
        // handlers apart.
        let member = analyzed("def h(u: Annotated[User, Depends(as_member)]) -> None:\n    pass\n");
        let admin = analyzed("def h(u: Annotated[User, Depends(as_admin)]) -> None:\n    pass\n");
        assert_ne!(member.xname_canonical, admin.xname_canonical);
        assert!(member.xname_canonical.contains("as_member"));
        assert!(!member.xname_canonical.contains("'User'"));
    }

    #[test]
    fn type3_lines_slot_the_types_and_report_them_apart() {
        let a = analyzed("def f(p: Token) -> int | None:\n    x: models.Row = load(p)\n    return x\n");
        assert_eq!(a.type_refs, ["Token", "int", "models.Row"]);
        // Slot numbers follow the lines' own walk — the signature before the body, as with `_v{n}`;
        // the engine renumbers each unit on its own anyway.
        assert_eq!(a.type3_lines[0], "def _fn(_v0: _T0) -> _T1 | None:");
        assert_eq!(a.type3_lines[1], "_v1: _T2 = load(_v0)");
        // The statement stream keeps them as written: converge parts definitions on names, a type
        // among them.
        assert_eq!(a.statements[0].line, "def _fn(_v0: Token) -> int | None:");
    }

    #[test]
    fn a_declared_type_parameter_is_bound_not_vocabulary() {
        let a = analyzed("def first[T](items: list[T]) -> T:\n    return items[0]\n");
        assert_eq!(a.type_refs, ["list"]);
        let b = analyzed("def head[U](rows: list[U]) -> U:\n    return rows[0]\n");
        assert_eq!(a.xname_canonical, b.xname_canonical);
    }
}

