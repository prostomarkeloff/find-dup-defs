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
#![allow(clippy::doc_markdown)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use dup_defs_core::AnalyzedFn;
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
fn repr_number(num: &Number) -> String {
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

/// One field's contribution under the `show_empty=False` rule.
enum F {
    /// A present value (already formatted) — emitted positionally, or `name=` once keyword mode is on.
    P(String),
    /// An empty list (`[]`): buffered then dropped, unless a later present field flushes it back.
    Empty,
    /// An optional field that is `None`: dropped, and switches the remaining fields to keyword form.
    Skip,
}

fn opt(value: Option<String>) -> F {
    value.map_or(F::Skip, F::P)
}

#[allow(clippy::needless_pass_by_value)] // items is a fresh per-field Vec, consumed by the join
fn flist(items: Vec<String>) -> F {
    if items.is_empty() {
        F::Empty
    } else {
        F::P(format!("[{}]", items.join(", ")))
    }
}

/// CPython-`ast.dump(annotate_fields=False)` serializer over the ruff AST. `locals = Some(set)`
/// ⇒ cross-name mode (rename bound locals → `_v{n}`, blank the def name); `None` ⇒ clustering
/// mode (names preserved). `count` mirrors `sum(1 for _ in ast.walk(node))` for the dup-defs size.
struct Dump<'a> {
    count: usize,
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
}

impl<'a> Dump<'a> {
    fn new(src: &'a str, locals: Option<&'a HashSet<String>>) -> Self {
        Self { count: 0, src, locals, map: HashMap::new(), blanked: false, top_def_pending: true }
    }

    /// In cross-name mode, rewrite a bound local to its positional `_v{n}` placeholder.
    #[allow(clippy::cast_possible_truncation)] // a function's distinct local count is far below u32::MAX
    fn rename_id(&mut self, id: &str) -> String {
        if let Some(locals) = self.locals {
            if locals.contains(id) {
                let next = self.map.len() as u32;
                let slot = *self.map.entry(id.to_owned()).or_insert(next);
                return format!("_v{slot}");
            }
        }
        id.to_owned()
    }

    /// Emit one AST node `name(field, …)` applying the `show_empty=False` + keyword-switch rule.
    fn node(&mut self, name: &str, fields: Vec<(&str, F)>) -> String {
        self.count += 1;
        let mut args: Vec<String> = Vec::new();
        let mut pending = 0usize; // buffered empty lists, flushed iff a positional field follows
        let mut keywords = false;
        for (fname, field) in fields {
            match field {
                F::Skip => keywords = true,
                F::Empty => {
                    if !keywords {
                        pending += 1;
                    }
                }
                F::P(value) => {
                    if keywords {
                        args.push(format!("{fname}={value}"));
                    } else {
                        for _ in 0..pending {
                            args.push("[]".to_owned());
                        }
                        pending = 0;
                        args.push(value);
                    }
                }
            }
        }
        format!("{name}({})", args.join(", "))
    }

    fn ctx(&mut self, ctx: ExprContext) -> String {
        let name = match ctx {
            ExprContext::Store => "Store",
            ExprContext::Del => "Del",
            ExprContext::Load | ExprContext::Invalid => "Load",
        };
        self.node(name, vec![])
    }

    fn operator(&mut self, op: Operator) -> String {
        let name = match op {
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
        };
        self.node(name, vec![])
    }

    fn unaryop(&mut self, op: UnaryOp) -> String {
        let name = match op {
            UnaryOp::Invert => "Invert",
            UnaryOp::Not => "Not",
            UnaryOp::UAdd => "UAdd",
            UnaryOp::USub => "USub",
        };
        self.node(name, vec![])
    }

    fn boolop(&mut self, op: BoolOp) -> String {
        let name = match op {
            BoolOp::And => "And",
            BoolOp::Or => "Or",
        };
        self.node(name, vec![])
    }

    fn cmpop(&mut self, op: CmpOp) -> String {
        let name = match op {
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
        };
        self.node(name, vec![])
    }

    /// `arg(arg, annotation?, type_comment?)` — the param name is a rename target.
    fn arg_node(&mut self, param: &Parameter) -> String {
        let name = self.rename_id(param.name.id.as_str());
        let annotation = param.annotation.as_ref().map(|a| self.expr(a));
        self.node(
            "arg",
            vec![("arg", F::P(repr_str(&name))), ("annotation", opt(annotation)), ("type_comment", F::Skip)],
        )
    }

    /// `arguments(posonlyargs, args, vararg?, kwonlyargs, kw_defaults, kwarg?, defaults)`.
    fn arguments(&mut self, params: Option<&Parameters>) -> String {
        let Some(p) = params else {
            return self.node(
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
        };
        let posonly: Vec<String> = p.posonlyargs.iter().map(|x| self.arg_node(&x.parameter)).collect();
        let args: Vec<String> = p.args.iter().map(|x| self.arg_node(&x.parameter)).collect();
        let vararg = p.vararg.as_ref().map(|x| self.arg_node(x));
        let kwonly: Vec<String> = p.kwonlyargs.iter().map(|x| self.arg_node(&x.parameter)).collect();
        let kw_defaults: Vec<String> = p
            .kwonlyargs
            .iter()
            .map(|x| match &x.default {
                Some(d) => self.expr(d),
                None => "None".to_owned(),
            })
            .collect();
        let kwarg = p.kwarg.as_ref().map(|x| self.arg_node(x));
        let defaults: Vec<String> = p
            .posonlyargs
            .iter()
            .chain(p.args.iter())
            .filter_map(|x| x.default.as_ref())
            .map(|d| self.expr(d))
            .collect();
        self.node(
            "arguments",
            vec![
                ("posonlyargs", flist(posonly)),
                ("args", flist(args)),
                ("vararg", opt(vararg)),
                ("kwonlyargs", flist(kwonly)),
                ("kw_defaults", flist(kw_defaults)),
                ("kwarg", opt(kwarg)),
                ("defaults", flist(defaults)),
            ],
        )
    }

    fn keyword(&mut self, kw: &Keyword) -> String {
        let arg = kw.arg.as_ref().map(|id| repr_str(id.id.as_str()));
        let value = self.expr(&kw.value);
        self.node("keyword", vec![("arg", opt(arg)), ("value", F::P(value))])
    }

    fn comprehension(&mut self, comp: &Comprehension) -> String {
        let target = self.expr(&comp.target);
        let iter = self.expr(&comp.iter);
        let ifs: Vec<String> = comp.ifs.iter().map(|i| self.expr(i)).collect();
        let is_async = i32::from(comp.is_async);
        self.node(
            "comprehension",
            vec![
                ("target", F::P(target)),
                ("iter", F::P(iter)),
                ("ifs", flist(ifs)),
                ("is_async", F::P(format!("{is_async}"))),
            ],
        )
    }

    /// An f-string → CPython `JoinedStr([Constant | FormattedValue, …])`. Iterates PARTS (so plain
    /// `StringLiteral` parts in implicit concatenation are kept), merges consecutive literals into
    /// one `Constant`, and bakes a `{x=}` debug interpolation as `Constant('x=')` + `FormattedValue`
    /// — exactly as CPython's parser does.
    fn fstring_dump(&mut self, fvalue: &ast::FStringValue) -> String {
        let mut literal = String::new();
        let mut values: Vec<String> = Vec::new();
        for part in fvalue {
            match part {
                ast::FStringPart::Literal(s) => literal.push_str(&s.value),
                ast::FStringPart::FString(fs) => self.fstring_elements(&fs.elements, &mut literal, &mut values),
            }
        }
        if !literal.is_empty() {
            values.push(self.node("Constant", vec![("value", F::P(repr_str(&literal)))]));
        }
        self.node("JoinedStr", vec![("values", flist(values))])
    }

    /// A format-spec → its own `JoinedStr` (CPython wraps format specs the same way).
    fn formatspec_dump(&mut self, elements: &[ast::InterpolatedStringElement]) -> String {
        let mut literal = String::new();
        let mut values: Vec<String> = Vec::new();
        self.fstring_elements(elements, &mut literal, &mut values);
        if !literal.is_empty() {
            values.push(self.node("Constant", vec![("value", F::P(repr_str(&literal)))]));
        }
        self.node("JoinedStr", vec![("values", flist(values))])
    }

    fn fstring_elements(
        &mut self,
        elements: &[ast::InterpolatedStringElement],
        literal: &mut String,
        values: &mut Vec<String>,
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
                        values.push(self.node("Constant", vec![("value", F::P(repr_str(literal)))]));
                        literal.clear();
                    }
                    let value = self.expr(&interp.expression);
                    let mut conversion = interp.conversion as i8;
                    if conversion == -1 && interp.debug_text.is_some() && interp.format_spec.is_none() {
                        conversion = 114; // bare `{x=}` implies `!r`
                    }
                    let format_spec = interp.format_spec.as_ref().map(|s| self.formatspec_dump(&s.elements));
                    values.push(self.node(
                        "FormattedValue",
                        vec![
                            ("value", F::P(value)),
                            ("conversion", F::P(format!("{conversion}"))),
                            ("format_spec", opt(format_spec)),
                        ],
                    ));
                }
            }
        }
    }

    fn type_params(&mut self, params: Option<&ast::TypeParams>) -> F {
        match params {
            None => F::Empty,
            Some(tps) => {
                let items: Vec<String> = tps.type_params.iter().map(|t| self.type_param(t)).collect();
                flist(items)
            }
        }
    }

    fn type_param(&mut self, tp: &ast::TypeParam) -> String {
        match tp {
            ast::TypeParam::TypeVar(t) => {
                let bound = t.bound.as_ref().map(|b| self.expr(b));
                let default = t.default.as_ref().map(|d| self.expr(d));
                self.node(
                    "TypeVar",
                    vec![
                        ("name", F::P(repr_str(t.name.id.as_str()))),
                        ("bound", opt(bound)),
                        ("default_value", opt(default)),
                    ],
                )
            }
            ast::TypeParam::ParamSpec(t) => {
                let default = t.default.as_ref().map(|d| self.expr(d));
                self.node(
                    "ParamSpec",
                    vec![("name", F::P(repr_str(t.name.id.as_str()))), ("default_value", opt(default))],
                )
            }
            ast::TypeParam::TypeVarTuple(t) => {
                let default = t.default.as_ref().map(|d| self.expr(d));
                self.node(
                    "TypeVarTuple",
                    vec![("name", F::P(repr_str(t.name.id.as_str()))), ("default_value", opt(default))],
                )
            }
        }
    }

    fn alias_node(&mut self, alias: &ast::Alias) -> String {
        let asname = alias.asname.as_ref().map(|n| repr_str(n.id.as_str()));
        self.node("alias", vec![("name", F::P(repr_str(alias.name.id.as_str()))), ("asname", opt(asname))])
    }

    fn pattern(&mut self, pat: &ast::Pattern) -> String {
        match pat {
            ast::Pattern::MatchValue(p) => {
                let value = self.expr(&p.value);
                self.node("MatchValue", vec![("value", F::P(value))])
            }
            ast::Pattern::MatchSingleton(p) => {
                let value = match p.value {
                    ast::Singleton::None => "None",
                    ast::Singleton::True => "True",
                    ast::Singleton::False => "False",
                };
                self.node("MatchSingleton", vec![("value", F::P(value.to_owned()))])
            }
            ast::Pattern::MatchSequence(p) => {
                let patterns: Vec<String> = p.patterns.iter().map(|x| self.pattern(x)).collect();
                self.node("MatchSequence", vec![("patterns", flist(patterns))])
            }
            ast::Pattern::MatchMapping(p) => {
                let keys: Vec<String> = p.keys.iter().map(|k| self.expr(k)).collect();
                let patterns: Vec<String> = p.patterns.iter().map(|x| self.pattern(x)).collect();
                let rest = p.rest.as_ref().map(|id| repr_str(id.id.as_str()));
                self.node(
                    "MatchMapping",
                    vec![("keys", flist(keys)), ("patterns", flist(patterns)), ("rest", opt(rest))],
                )
            }
            ast::Pattern::MatchClass(p) => {
                let cls = self.expr(&p.cls);
                let patterns: Vec<String> = p.arguments.patterns.iter().map(|x| self.pattern(x)).collect();
                let kwd_attrs: Vec<String> =
                    p.arguments.keywords.iter().map(|k| repr_str(k.attr.id.as_str())).collect();
                let kwd_patterns: Vec<String> =
                    p.arguments.keywords.iter().map(|k| self.pattern(&k.pattern)).collect();
                self.node(
                    "MatchClass",
                    vec![
                        ("cls", F::P(cls)),
                        ("patterns", flist(patterns)),
                        ("kwd_attrs", flist(kwd_attrs)),
                        ("kwd_patterns", flist(kwd_patterns)),
                    ],
                )
            }
            ast::Pattern::MatchStar(p) => {
                let name = p.name.as_ref().map(|id| repr_str(id.id.as_str()));
                self.node("MatchStar", vec![("name", opt(name))])
            }
            ast::Pattern::MatchAs(p) => {
                let pattern = p.pattern.as_ref().map(|x| self.pattern(x));
                let name = p.name.as_ref().map(|id| repr_str(id.id.as_str()));
                self.node("MatchAs", vec![("pattern", opt(pattern)), ("name", opt(name))])
            }
            ast::Pattern::MatchOr(p) => {
                let patterns: Vec<String> = p.patterns.iter().map(|x| self.pattern(x)).collect();
                self.node("MatchOr", vec![("patterns", flist(patterns))])
            }
        }
    }

    fn match_case(&mut self, case: &ast::MatchCase) -> String {
        let pattern = self.pattern(&case.pattern);
        let guard = case.guard.as_ref().map(|g| self.expr(g));
        let body = self.body(&case.body, false);
        self.node("match_case", vec![("pattern", F::P(pattern)), ("guard", opt(guard)), ("body", flist(body))])
    }

    /// A statement body; `strip` drops a leading string-literal docstring (def/class bodies only),
    /// replacing an emptied body with `[Pass()]` exactly like CPython's `_strip_docstring`.
    fn body(&mut self, stmts: &[Stmt], strip: bool) -> Vec<String> {
        if strip && stmts.first().is_some_and(is_docstring) {
            let rest = &stmts[1..];
            if rest.is_empty() {
                return vec![self.node("Pass", vec![])];
            }
            return rest.iter().map(|s| self.stmt(s)).collect();
        }
        stmts.iter().map(|s| self.stmt(s)).collect()
    }

    /// Rebuild CPython's nested-`If` `orelse` from ruff's flat `elif_else_clauses`.
    fn elif_orelse(&mut self, clauses: &[ast::ElifElseClause]) -> Vec<String> {
        let Some((first, rest)) = clauses.split_first() else { return Vec::new() };
        match &first.test {
            Some(test) => {
                let test_s = self.expr(test);
                let body = self.body(&first.body, false);
                let orelse = self.elif_orelse(rest);
                vec![self.node(
                    "If",
                    vec![("test", F::P(test_s)), ("body", flist(body)), ("orelse", flist(orelse))],
                )]
            }
            None => self.body(&first.body, false),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Name(n) => {
                let id = self.rename_id(n.id.as_str());
                let ctx = self.ctx(n.ctx);
                self.node("Name", vec![("id", F::P(repr_str(&id))), ("ctx", F::P(ctx))])
            }
            Expr::Attribute(a) => {
                let value = self.expr(&a.value);
                let ctx = self.ctx(a.ctx);
                self.node(
                    "Attribute",
                    vec![("value", F::P(value)), ("attr", F::P(repr_str(a.attr.id.as_str()))), ("ctx", F::P(ctx))],
                )
            }
            Expr::Call(c) => {
                let func = self.expr(&c.func);
                let args: Vec<String> = c.arguments.args.iter().map(|a| self.expr(a)).collect();
                let keywords: Vec<String> = c.arguments.keywords.iter().map(|k| self.keyword(k)).collect();
                self.node("Call", vec![("func", F::P(func)), ("args", flist(args)), ("keywords", flist(keywords))])
            }
            Expr::BinOp(b) => {
                let left = self.expr(&b.left);
                let op = self.operator(b.op);
                let right = self.expr(&b.right);
                self.node("BinOp", vec![("left", F::P(left)), ("op", F::P(op)), ("right", F::P(right))])
            }
            Expr::UnaryOp(u) => {
                let op = self.unaryop(u.op);
                let operand = self.expr(&u.operand);
                self.node("UnaryOp", vec![("op", F::P(op)), ("operand", F::P(operand))])
            }
            Expr::BoolOp(b) => {
                let op = self.boolop(b.op);
                let values: Vec<String> = b.values.iter().map(|v| self.expr(v)).collect();
                self.node("BoolOp", vec![("op", F::P(op)), ("values", flist(values))])
            }
            Expr::Compare(c) => {
                let left = self.expr(&c.left);
                let ops: Vec<String> = c.ops.iter().map(|o| self.cmpop(*o)).collect();
                let comparators: Vec<String> = c.comparators.iter().map(|e| self.expr(e)).collect();
                self.node(
                    "Compare",
                    vec![("left", F::P(left)), ("ops", flist(ops)), ("comparators", flist(comparators))],
                )
            }
            Expr::NumberLiteral(n) => self.node("Constant", vec![("value", F::P(repr_number(&n.value)))]),
            Expr::StringLiteral(s) => {
                self.node("Constant", vec![("value", F::P(repr_str(s.value.to_str())))])
            }
            Expr::BytesLiteral(b) => {
                let bytes: Vec<u8> = b.value.bytes().collect();
                self.node("Constant", vec![("value", F::P(repr_bytes(&bytes)))])
            }
            Expr::BooleanLiteral(b) => {
                let value = if b.value { "True" } else { "False" };
                self.node("Constant", vec![("value", F::P(value.to_owned()))])
            }
            // IpyEscapeCommand has no CPython analogue (Jupyter only); a `None` Constant placeholder
            // it that never occurs in real modules.
            Expr::NoneLiteral(_) | Expr::IpyEscapeCommand(_) => {
                self.node("Constant", vec![("value", F::P("None".to_owned()))])
            }
            Expr::EllipsisLiteral(_) => self.node("Constant", vec![("value", F::P("Ellipsis".to_owned()))]),
            Expr::Subscript(s) => {
                let value = self.expr(&s.value);
                let slice = self.expr(&s.slice);
                let ctx = self.ctx(s.ctx);
                self.node(
                    "Subscript",
                    vec![("value", F::P(value)), ("slice", F::P(slice)), ("ctx", F::P(ctx))],
                )
            }
            Expr::Starred(s) => {
                let value = self.expr(&s.value);
                let ctx = self.ctx(s.ctx);
                self.node("Starred", vec![("value", F::P(value)), ("ctx", F::P(ctx))])
            }
            Expr::List(l) => {
                let elts: Vec<String> = l.elts.iter().map(|e| self.expr(e)).collect();
                let ctx = self.ctx(l.ctx);
                self.node("List", vec![("elts", flist(elts)), ("ctx", F::P(ctx))])
            }
            Expr::Tuple(t) => {
                let elts: Vec<String> = t.elts.iter().map(|e| self.expr(e)).collect();
                let ctx = self.ctx(t.ctx);
                self.node("Tuple", vec![("elts", flist(elts)), ("ctx", F::P(ctx))])
            }
            Expr::Set(s) => {
                let elts: Vec<String> = s.elts.iter().map(|e| self.expr(e)).collect();
                self.node("Set", vec![("elts", flist(elts))])
            }
            Expr::Dict(d) => {
                let keys: Vec<String> = d
                    .items
                    .iter()
                    .map(|i| i.key.as_ref().map_or_else(|| "None".to_owned(), |k| self.expr(k)))
                    .collect();
                let values: Vec<String> = d.items.iter().map(|i| self.expr(&i.value)).collect();
                self.node("Dict", vec![("keys", flist(keys)), ("values", flist(values))])
            }
            Expr::Slice(s) => {
                let lower = s.lower.as_ref().map(|e| self.expr(e));
                let upper = s.upper.as_ref().map(|e| self.expr(e));
                let step = s.step.as_ref().map(|e| self.expr(e));
                self.node("Slice", vec![("lower", opt(lower)), ("upper", opt(upper)), ("step", opt(step))])
            }
            Expr::If(i) => {
                let test = self.expr(&i.test);
                let body = self.expr(&i.body);
                let orelse = self.expr(&i.orelse);
                self.node("IfExp", vec![("test", F::P(test)), ("body", F::P(body)), ("orelse", F::P(orelse))])
            }
            Expr::Lambda(l) => {
                let args = self.arguments(l.parameters.as_deref());
                let body = self.expr(&l.body);
                self.node("Lambda", vec![("args", F::P(args)), ("body", F::P(body))])
            }
            Expr::Named(n) => {
                let target = self.expr(&n.target);
                let value = self.expr(&n.value);
                self.node("NamedExpr", vec![("target", F::P(target)), ("value", F::P(value))])
            }
            Expr::Await(a) => {
                let value = self.expr(&a.value);
                self.node("Await", vec![("value", F::P(value))])
            }
            Expr::Yield(y) => {
                let value = y.value.as_ref().map(|v| self.expr(v));
                self.node("Yield", vec![("value", opt(value))])
            }
            Expr::YieldFrom(y) => {
                let value = self.expr(&y.value);
                self.node("YieldFrom", vec![("value", F::P(value))])
            }
            Expr::ListComp(c) => {
                let elt = self.expr(&c.elt);
                let generators: Vec<String> = c.generators.iter().map(|g| self.comprehension(g)).collect();
                self.node("ListComp", vec![("elt", F::P(elt)), ("generators", flist(generators))])
            }
            Expr::SetComp(c) => {
                let elt = self.expr(&c.elt);
                let generators: Vec<String> = c.generators.iter().map(|g| self.comprehension(g)).collect();
                self.node("SetComp", vec![("elt", F::P(elt)), ("generators", flist(generators))])
            }
            Expr::Generator(c) => {
                let elt = self.expr(&c.elt);
                let generators: Vec<String> = c.generators.iter().map(|g| self.comprehension(g)).collect();
                self.node("GeneratorExp", vec![("elt", F::P(elt)), ("generators", flist(generators))])
            }
            Expr::DictComp(c) => {
                let key = self.expr(&c.key);
                let value = self.expr(&c.value);
                let generators: Vec<String> = c.generators.iter().map(|g| self.comprehension(g)).collect();
                self.node(
                    "DictComp",
                    vec![("key", F::P(key)), ("value", F::P(value)), ("generators", flist(generators))],
                )
            }
            Expr::FString(f) => self.fstring_dump(&f.value),
            Expr::TString(_) => {
                // PEP 750 template strings (3.14, vanishingly rare): CPython's `Interpolation.str`
                // field can't be reproduced from the ruff AST, so emit a minimal `TemplateStr()`.
                self.node("TemplateStr", vec![("values", F::Empty)])
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn stmt(&mut self, stmt: &Stmt) -> String {
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
                let args = self.arguments(Some(&f.parameters));
                let body = self.body(&f.body, true);
                let decorators: Vec<String> =
                    if strip_deco { Vec::new() } else { f.decorator_list.iter().map(|d| self.expr(&d.expression)).collect() };
                let returns = f.returns.as_ref().map(|r| self.expr(r));
                let type_params = self.type_params(f.type_params.as_deref());
                self.node(
                    cpy,
                    vec![
                        ("name", F::P(repr_str(&name))),
                        ("args", F::P(args)),
                        ("body", flist(body)),
                        ("decorator_list", flist(decorators)),
                        ("returns", opt(returns)),
                        ("type_comment", F::Skip),
                        ("type_params", type_params),
                    ],
                )
            }
            Stmt::ClassDef(c) => {
                let strip_deco = self.top_def_pending; // capture + clear BEFORE walking the body
                self.top_def_pending = false;
                let name = c.name.id.as_str().to_owned();
                let (bases, keywords) = match &c.arguments {
                    Some(a) => {
                        let bases = a.args.iter().map(|b| self.expr(b)).collect();
                        let keywords = a.keywords.iter().map(|k| self.keyword(k)).collect();
                        (bases, keywords)
                    }
                    None => (Vec::new(), Vec::new()),
                };
                let body = self.body(&c.body, true);
                let decorators: Vec<String> =
                    if strip_deco { Vec::new() } else { c.decorator_list.iter().map(|d| self.expr(&d.expression)).collect() };
                let type_params = self.type_params(c.type_params.as_deref());
                self.node(
                    "ClassDef",
                    vec![
                        ("name", F::P(repr_str(&name))),
                        ("bases", flist(bases)),
                        ("keywords", flist(keywords)),
                        ("body", flist(body)),
                        ("decorator_list", flist(decorators)),
                        ("type_params", type_params),
                    ],
                )
            }
            Stmt::Return(r) => {
                let value = r.value.as_ref().map(|v| self.expr(v));
                self.node("Return", vec![("value", opt(value))])
            }
            Stmt::Delete(d) => {
                let targets: Vec<String> = d.targets.iter().map(|t| self.expr(t)).collect();
                self.node("Delete", vec![("targets", flist(targets))])
            }
            Stmt::Assign(a) => {
                let targets: Vec<String> = a.targets.iter().map(|t| self.expr(t)).collect();
                let value = self.expr(&a.value);
                self.node(
                    "Assign",
                    vec![("targets", flist(targets)), ("value", F::P(value)), ("type_comment", F::Skip)],
                )
            }
            Stmt::AugAssign(a) => {
                let target = self.expr(&a.target);
                let op = self.operator(a.op);
                let value = self.expr(&a.value);
                self.node(
                    "AugAssign",
                    vec![("target", F::P(target)), ("op", F::P(op)), ("value", F::P(value))],
                )
            }
            Stmt::AnnAssign(a) => {
                let target = self.expr(&a.target);
                let annotation = self.expr(&a.annotation);
                let value = a.value.as_ref().map(|v| self.expr(v));
                let simple = i32::from(a.simple);
                self.node(
                    "AnnAssign",
                    vec![
                        ("target", F::P(target)),
                        ("annotation", F::P(annotation)),
                        ("value", opt(value)),
                        ("simple", F::P(format!("{simple}"))),
                    ],
                )
            }
            Stmt::TypeAlias(t) => {
                let name = self.expr(&t.name);
                let type_params = self.type_params(t.type_params.as_deref());
                let value = self.expr(&t.value);
                self.node(
                    "TypeAlias",
                    vec![("name", F::P(name)), ("type_params", type_params), ("value", F::P(value))],
                )
            }
            Stmt::For(f) => {
                let cpy = if f.is_async { "AsyncFor" } else { "For" };
                let target = self.expr(&f.target);
                let iter = self.expr(&f.iter);
                let body = self.body(&f.body, false);
                let orelse = self.body(&f.orelse, false);
                self.node(
                    cpy,
                    vec![
                        ("target", F::P(target)),
                        ("iter", F::P(iter)),
                        ("body", flist(body)),
                        ("orelse", flist(orelse)),
                        ("type_comment", F::Skip),
                    ],
                )
            }
            Stmt::While(w) => {
                let test = self.expr(&w.test);
                let body = self.body(&w.body, false);
                let orelse = self.body(&w.orelse, false);
                self.node(
                    "While",
                    vec![("test", F::P(test)), ("body", flist(body)), ("orelse", flist(orelse))],
                )
            }
            Stmt::If(i) => {
                let test = self.expr(&i.test);
                let body = self.body(&i.body, false);
                let orelse = self.elif_orelse(&i.elif_else_clauses);
                self.node(
                    "If",
                    vec![("test", F::P(test)), ("body", flist(body)), ("orelse", flist(orelse))],
                )
            }
            Stmt::With(w) => {
                let cpy = if w.is_async { "AsyncWith" } else { "With" };
                let items: Vec<String> = w
                    .items
                    .iter()
                    .map(|item| {
                        let context_expr = self.expr(&item.context_expr);
                        let optional_vars = item.optional_vars.as_ref().map(|v| self.expr(v));
                        self.node(
                            "withitem",
                            vec![("context_expr", F::P(context_expr)), ("optional_vars", opt(optional_vars))],
                        )
                    })
                    .collect();
                let body = self.body(&w.body, false);
                self.node(cpy, vec![("items", flist(items)), ("body", flist(body)), ("type_comment", F::Skip)])
            }
            Stmt::Raise(r) => {
                let exc = r.exc.as_ref().map(|e| self.expr(e));
                let cause = r.cause.as_ref().map(|c| self.expr(c));
                self.node("Raise", vec![("exc", opt(exc)), ("cause", opt(cause))])
            }
            Stmt::Try(t) => {
                let cpy = if t.is_star { "TryStar" } else { "Try" };
                let body = self.body(&t.body, false);
                let handlers: Vec<String> = t
                    .handlers
                    .iter()
                    .map(|h| {
                        let ExceptHandler::ExceptHandler(handler) = h;
                        let typ = handler.type_.as_ref().map(|e| self.expr(e));
                        let name = handler.name.as_ref().map(|n| repr_str(&self.rename_id(n.id.as_str())));
                        let body = self.body(&handler.body, false);
                        self.node(
                            "ExceptHandler",
                            vec![("type", opt(typ)), ("name", opt(name)), ("body", flist(body))],
                        )
                    })
                    .collect();
                let orelse = self.body(&t.orelse, false);
                let finalbody = self.body(&t.finalbody, false);
                self.node(
                    cpy,
                    vec![
                        ("body", flist(body)),
                        ("handlers", flist(handlers)),
                        ("orelse", flist(orelse)),
                        ("finalbody", flist(finalbody)),
                    ],
                )
            }
            Stmt::Assert(a) => {
                let test = self.expr(&a.test);
                let msg = a.msg.as_ref().map(|m| self.expr(m));
                self.node("Assert", vec![("test", F::P(test)), ("msg", opt(msg))])
            }
            Stmt::Import(i) => {
                let names: Vec<String> = i.names.iter().map(|a| self.alias_node(a)).collect();
                self.node("Import", vec![("names", flist(names))])
            }
            Stmt::ImportFrom(i) => {
                let module = i.module.as_ref().map(|m| repr_str(m.id.as_str()));
                let names: Vec<String> = i.names.iter().map(|a| self.alias_node(a)).collect();
                self.node(
                    "ImportFrom",
                    vec![("module", opt(module)), ("names", flist(names)), ("level", F::P(format!("{}", i.level)))],
                )
            }
            Stmt::Global(g) => {
                let names: Vec<String> = g.names.iter().map(|n| repr_str(n.id.as_str())).collect();
                self.node("Global", vec![("names", flist(names))])
            }
            Stmt::Nonlocal(n) => {
                let names: Vec<String> = n.names.iter().map(|x| repr_str(x.id.as_str())).collect();
                self.node("Nonlocal", vec![("names", flist(names))])
            }
            Stmt::Expr(e) => {
                let value = self.expr(&e.value);
                self.node("Expr", vec![("value", F::P(value))])
            }
            Stmt::Match(m) => {
                let subject = self.expr(&m.subject);
                let cases: Vec<String> = m.cases.iter().map(|c| self.match_case(c)).collect();
                self.node("Match", vec![("subject", F::P(subject)), ("cases", flist(cases))])
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
}

impl Unparse<'_> {
    #[allow(clippy::cast_possible_truncation)]
    fn rename_id(&mut self, id: &str) -> String {
        if let Some(locals) = self.locals {
            if locals.contains(id) {
                let next = self.map.len() as u32;
                let slot = *self.map.entry(id.to_owned()).or_insert(next);
                return format!("_v{slot}");
            }
        }
        id.to_owned()
    }

    /// Start a new logical line (CPython `fill`): a newline unless we're at the start.
    fn fill(&mut self, text: &str) {
        if !self.out.is_empty() {
            self.out.push('\n');
        }
        self.out.push_str(text);
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
        if slice.is_empty() {
            self.fill("pass");
            return;
        }
        for stmt in slice {
            self.stmt(stmt);
        }
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
                self.fill(&format!("{kw}{name}"));
                self.type_params(f.type_params.as_deref());
                self.write("(");
                self.arguments(Some(&f.parameters));
                self.write(")");
                if let Some(returns) = &f.returns {
                    self.write(" -> ");
                    self.expr(returns, prec::TEST);
                }
                self.write(":");
                self.body(&f.body, true);
            }
            Stmt::ClassDef(c) => {
                self.fill(&format!("class {}", c.name.id.as_str()));
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
                self.write(&format!(" {}= ", operator_tag(a.op)));
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
                self.expr(&a.annotation, prec::TEST);
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
                self.write(&g.names.iter().map(|n| n.id.as_str().to_owned()).collect::<Vec<_>>().join(", "));
            }
            Stmt::Nonlocal(n) => {
                self.fill("nonlocal ");
                self.write(&n.names.iter().map(|x| x.id.as_str().to_owned()).collect::<Vec<_>>().join(", "));
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
    fn expr(&mut self, expr: &Expr, ctx: u8) {
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
                self.write(&format!(" {op} "));
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
                    self.write(&format!(" {} ", cmpop_tag(*op)));
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
            self.expr(annotation, prec::TEST);
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
                    self.write(t.name.id.as_str());
                    if let Some(bound) = &t.bound {
                        self.write(": ");
                        self.expr(bound, prec::TEST);
                    }
                    if let Some(default) = &t.default {
                        self.write(" = ");
                        self.expr(default, prec::TEST);
                    }
                }
                ast::TypeParam::TypeVarTuple(t) => {
                    self.write("*");
                    self.write(t.name.id.as_str());
                }
                ast::TypeParam::ParamSpec(t) => {
                    self.write("**");
                    self.write(t.name.id.as_str());
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
                    self.write(&format!("**{}", rest.id.as_str()));
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
                    self.write(&format!("{}=", kw.attr.id.as_str()));
                    self.pattern(&kw.pattern);
                }
                self.write(")");
            }
            ast::Pattern::MatchStar(p) => {
                let name = p.name.as_ref().map_or("_", |n| n.id.as_str());
                self.write(&format!("*{name}"));
            }
            ast::Pattern::MatchAs(p) => match (&p.pattern, &p.name) {
                (None, None) => self.write("_"),
                (None, Some(name)) => self.write(name.id.as_str()),
                (Some(inner), name) => {
                    self.pattern(inner);
                    if let Some(name) = name {
                        self.write(&format!(" as {}", name.id.as_str()));
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

/// Unparse the alpha-renamed function and split into stripped, non-empty lines (the Type-3 units).
fn unparse_lines(stmt: &Stmt, src: &str, locals: &HashSet<String>, map: HashMap<String, u32>) -> Vec<String> {
    let mut up = Unparse { out: String::new(), src, locals: Some(locals), map, blanked: false };
    up.stmt(stmt);
    up.out.lines().map(str::trim).filter(|l| !l.is_empty()).map(ToOwned::to_owned).collect()
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
    Dump::new(text, None).stmt(stmt)
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
    let Stmt::FunctionDef(func) = stmt else {
        return None; // ruff folds async into StmtFunctionDef; classes/others are out of scope here
    };

    let mut collect = Collect::default();
    collect.add_params(&func.parameters); // top fn's params (nested defs' params are not locals)
    collect.visit_stmt(stmt);
    let locals = collect.bound;

    let mut dump = Dump::new(text, Some(&locals));
    let canonical = dump.stmt(stmt);
    let size = dump.count;

    // Type-3 shingle units: `ast.unparse(normalized_fn).splitlines()` stripped — reproduced exactly
    // (CPython 3.14 unparse) so the ECScan IDF/cosine matches the CPython `ast` reference bit-for-bit. The
    // rename map from the dump pass is reused so the `_v{n}` numbering is identical to the canonical.
    let lines = unparse_lines(stmt, text, &locals, dump.map);
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
pub(crate) fn analyze_stmt(stmt: &Stmt, src: &str) -> Option<AnalyzedFn> {
    let Stmt::FunctionDef(func) = stmt else {
        return None;
    };
    let cluster_canonical = Dump::new(src, None).stmt(stmt);

    let mut collect = Collect::default();
    collect.add_params(&func.parameters);
    collect.visit_stmt(stmt);
    let locals = collect.bound;

    let mut dump = Dump::new(src, Some(&locals));
    let xname_canonical = dump.stmt(stmt);
    let size = dump.count;
    let lines = unparse_lines(stmt, src, &locals, dump.map);
    Some((cluster_canonical, xname_canonical, lines, size))
}

/// Names-preserved cluster canonical of any def node (functions AND classes), decorators of the top
/// node excluded. Node-based counterpart of `cluster_canonical`, used by the scan to canonicalize
/// classes without a re-parse.
pub(crate) fn cluster_canonical_node(stmt: &Stmt, src: &str) -> String {
    Dump::new(src, None).stmt(stmt)
}

fn analyze_one(text: &str) -> Option<AnalyzedFn> {
    let module = parse_module(text).ok()?.into_syntax();
    let stmt = module.body.first()?;
    analyze_stmt(stmt, text)
}

/// Batch `analyze_one` in parallel — one parse per function, all dup-defs canonical forms at once.
#[must_use]
pub fn analyze_functions(texts: &[String]) -> Vec<Option<AnalyzedFn>> {
    texts.par_iter().map(|text| analyze_one(text)).collect()
}
