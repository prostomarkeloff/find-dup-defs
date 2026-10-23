//! Structural canonicalization of Go definitions over the tree-sitter syntax tree.
//!
//! Mirrors the role `rs-canon::canon` plays for Rust: a compact, internally consistent **s-expr** per
//! definition that `difflib-fast` compares for name-gated similarity, and that the cross-name pass
//! `Eq`-checks once locals are alpha-renamed. The tag vocabulary is the Rust frontend's wherever the
//! construct is the same one (`Func`, `Params`, `Block`, `Let`, `ExprStmt`, `If`, `For`, `Call`,
//! `Method`, `Field`, `Bin`, …), so the engine's s-expr readers — converge's vocabulary split, the
//! unit counter — read Go the way they already read Rust. Two modes, driven by [`Dump::locals`]:
//!
//! * **cluster** (`locals = None`) — identifiers pass through verbatim; the names-preserved
//!   canonical the name-gated pass clusters on.
//! * **xname** (`locals = Some(set)`) — value bindings (parameters, named results, `:=`, `var`,
//!   `range` and `select` bindings, a type switch's alias) are renumbered `_v{n}` by first
//!   occurrence and the top definition's own name is blanked to `_fn`.
//!
//! A method's receiver is dropped from the parameters and written `self` wherever the body names it,
//! whatever the source called it: `func (s *Server) Close()` and `func (c *Client) Close()` spell the
//! same shape, and a method lines up with an equivalent free function for the cross-name pass — the
//! analog of the Rust frontend dropping `&self`.
//!
//! A selector on an imported package (`logx.Errorf`) is a path, `Path('logx.Errorf')`; on any other
//! value it is a field read or a method call. Go spells both `a.b`, and only the file's import table
//! tells them apart.
#![allow(
    clippy::too_many_lines, // the stmt/expr/type matches enumerate grammar kinds; splitting just scatters one shape
    clippy::match_same_arms // distinct kinds intentionally share an emission for clarity
)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use dup_defs_core::{AnalyzedFn, Statement, UNFILED};
use tree_sitter::Node;

use crate::tree::{field, fields, has_token, kids, statements, text, unquote};

// ───────────────────────────── the parts of a callable ─────────────────────────────

/// A function or method, as the grammar splits it — the input every analysis below reads.
#[derive(Clone, Copy)]
pub(crate) struct Callable<'t> {
    /// The receiver's name, for a method that names it (`func (s *S)`); `None` for a function, an
    /// anonymous receiver (`func (*S)`), or `_`.
    pub receiver: Option<&'t str>,
    pub type_params: Option<Node<'t>>,
    pub params: Option<Node<'t>>,
    pub results: Option<Node<'t>>,
    pub body: Node<'t>,
}

// ───────────────────────────── bound-locals collector ─────────────────────────────

/// Every value binding the callable introduces — the rename set for xname mode, and in every mode
/// the set that tells a local from an imported package: parameters, named results, `:=`, `var` and
/// local `const` names, `range` / `select` bindings declared with `:=`, a type switch's alias.
///
/// Inside a function literal its declarations count and its parameters do not — the rule the
/// TypeScript frontend keeps for arrow functions (and the Rust one for closures), so a callback reads
/// the same in either language.
#[derive(Default)]
struct Collect {
    bound: HashSet<String>,
}

impl Collect {
    fn add(&mut self, name: &str) {
        if name != "_" {
            self.bound.insert(name.to_owned());
        }
    }

    fn params(&mut self, list: Option<Node<'_>>, src: &str) {
        let Some(list) = list else { return };
        if list.kind() != "parameter_list" {
            return; // an unnamed single result type binds nothing
        }
        for decl in kids(list) {
            for name in fields(decl, "name") {
                self.add(text(name, src));
            }
        }
    }

    fn binds(&mut self, list: Node<'_>, src: &str) {
        for k in kids(list) {
            if k.kind() == "identifier" {
                self.add(text(k, src));
            }
        }
    }

    fn walk(&mut self, n: Node<'_>, src: &str) {
        match n.kind() {
            "short_var_declaration" => {
                if let Some(left) = field(n, "left") {
                    self.binds(left, src);
                }
                if let Some(right) = field(n, "right") {
                    self.walk(right, src);
                }
            }
            "var_spec" | "const_spec" => {
                for name in fields(n, "name") {
                    self.add(text(name, src));
                }
                if let Some(value) = field(n, "value") {
                    self.walk(value, src);
                }
            }
            "range_clause" | "receive_statement" => {
                if has_token(n, ":=") {
                    if let Some(left) = field(n, "left") {
                        self.binds(left, src);
                    }
                }
                if let Some(right) = field(n, "right") {
                    self.walk(right, src);
                }
            }
            "type_switch_statement" => {
                if let Some(alias) = field(n, "alias") {
                    self.binds(alias, src);
                }
                for k in kids(n) {
                    self.walk(k, src);
                }
            }
            // A function literal's parameters are its own; its body's declarations are walked.
            "func_literal" => {
                if let Some(body) = field(n, "body") {
                    self.walk(body, src);
                }
            }
            _ => {
                for k in kids(n) {
                    self.walk(k, src);
                }
            }
        }
    }
}

/// The bound locals of one callable (its receiver excluded — that one is written `self`).
fn locals_of(c: &Callable<'_>, src: &str) -> HashSet<String> {
    let mut collect = Collect::default();
    collect.params(c.params, src);
    collect.params(c.results, src);
    collect.walk(c.body, src);
    if let Some(receiver) = c.receiver {
        collect.bound.remove(receiver);
    }
    collect.bound
}

// ───────────────────────────── s-expr emitter ─────────────────────────────

pub(crate) struct Dump<'a> {
    src: &'a str,
    /// What each name the file imports binds — a selector on one of them is a package member.
    imports: &'a HashMap<String, Arc<str>>,
    /// Every name the callable binds, in every mode: a local shadowing a package is a value.
    bound: &'a HashSet<String>,
    /// The receiver's own name, written `self`.
    receiver: Option<&'a str>,
    /// `None` = cluster mode (names verbatim); `Some(set)` = xname mode (bound locals → `_v{n}`).
    locals: Option<&'a HashSet<String>>,
    map: HashMap<String, u32>,
    /// Blank the *top* callable's own name to `_fn` exactly once (xname mode).
    blanked: bool,
    /// Node-emit count — the cross-name "substance" gate.
    pub(crate) count: usize,
    /// `Some` while writing the Type-3 lines: every sub-expression worth a unit of its own, with the
    /// index of the line it sits in — [`AnalyzedFn::type3_units`].
    units: Option<Vec<(u32, String)>>,
    unit_line: u32,
    /// How many expressions enclose the one being written: `0` for a statement's own expression.
    expr_depth: u32,
    /// `Some` while writing the Type-3 lines: type reference → `_T{n}`. See [`Dump::type_name`].
    tmap: Option<HashMap<String, u32>>,
    /// Every type reference turned into a slot, in order — [`AnalyzedFn::type_refs`].
    type_refs: Vec<String>,
    /// Type parameters the definition declares: slotted like any type, but bound, not vocabulary.
    declared_types: HashSet<String>,
}

impl<'a> Dump<'a> {
    pub(crate) fn new(
        src: &'a str,
        imports: &'a HashMap<String, Arc<str>>,
        bound: &'a HashSet<String>,
        receiver: Option<&'a str>,
        locals: Option<&'a HashSet<String>>,
    ) -> Self {
        Self {
            src,
            imports,
            bound,
            receiver,
            locals,
            map: HashMap::new(),
            blanked: false,
            count: 0,
            units: None,
            unit_line: 0,
            expr_depth: 0,
            tmap: None,
            type_refs: Vec::new(),
            declared_types: HashSet::new(),
        }
    }

    fn text(&self, n: Node<'_>) -> &'a str {
        text(n, self.src)
    }

    /// Whether a bare name is an imported package rather than a value.
    fn is_package(&self, name: &str) -> bool {
        self.imports.contains_key(name) && !self.bound.contains(name) && self.receiver != Some(name)
    }

    fn rename(&mut self, name: &str) -> String {
        dup_defs_core::alpha_rename(&mut self.map, self.locals, name)
    }

    pub(crate) fn node(&mut self, tag: &str, fields: &[String]) -> String {
        self.count += 1;
        let mut s = String::with_capacity(tag.len() + 2 + fields.iter().map(String::len).sum::<usize>() + fields.len() * 2);
        s.push_str(tag);
        s.push('(');
        for (i, f) in fields.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(f);
        }
        s.push(')');
        s
    }

    fn list(&mut self, tag: &str, items: &[String]) -> String {
        let joined = items.join(", ");
        self.node(tag, &[joined])
    }

    pub(crate) fn lit(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('\'');
        for c in s.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '\'' => out.push_str("\\'"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c => out.push(c),
            }
        }
        out.push('\'');
        out
    }

    /// A type's name: its `_T{n}` slot while writing the Type-3 lines (reported apart unless the
    /// definition declared it), as spelled otherwise.
    ///
    /// 🔴 Only the Type-3 lines slot types — the Rust frontend's rule, for the Rust frontend's reason.
    /// A Go type picks the code that runs: the method set a call resolves against, the interface a
    /// value satisfies. Two bodies differing only in their types may not be one definition, so the
    /// exact pass keeps them apart; the near-copy pass reads a type as vocabulary like any other.
    fn type_name(&mut self, name: &str) -> String {
        let Some(tmap) = self.tmap.as_mut() else { return Self::lit(name) };
        if !self.declared_types.contains(name) {
            self.type_refs.push(name.to_owned());
        }
        dup_defs_core::type_slot(tmap, name)
    }

    /// A binding site: `Bind(_vN)`, or `Wild()` for the blank identifier.
    fn bind(&mut self, name: &str) -> String {
        if name == "_" {
            return self.node("Wild", &[]);
        }
        let renamed = self.rename(name);
        self.node("Bind", &[Self::lit(&renamed)])
    }

    /// A name read as a value.
    fn ident(&mut self, name: &str) -> String {
        if name == "_" {
            return self.node("Wild", &[]);
        }
        if self.receiver == Some(name) {
            return self.node("Path", &[Self::lit("self")]);
        }
        let renamed = self.rename(name);
        self.node("Path", &[Self::lit(&renamed)])
    }

    // ────── callables ──────

    /// `is_top` blanks the definition's own name in xname mode. The receiver is not a parameter.
    pub(crate) fn func(&mut self, name: &str, c: &Callable<'_>, with_body: bool, is_top: bool) -> String {
        let name = if is_top && self.locals.is_some() && !self.blanked {
            self.blanked = true;
            "_fn".to_owned()
        } else {
            self.rename(name)
        };
        let params = self.params(c.params, "Params");
        let results = self.results(c.results);
        let body = if with_body { self.block(c.body) } else { String::new() };
        self.node("Func", &[Self::lit(&name), params, results, body])
    }

    /// A parameter list, one `Param` per name — `a, b int` is two parameters of one type.
    fn params(&mut self, list: Option<Node<'_>>, tag: &str) -> String {
        let mut items = Vec::new();
        for decl in list.map(kids).unwrap_or_default() {
            let variadic = decl.kind() == "variadic_parameter_declaration";
            let ty = field(decl, "type").map_or_else(String::new, |t| self.ty(t));
            let ty = if variadic { self.node("TyVariadic", &[ty]) } else { ty };
            let names = fields(decl, "name");
            if names.is_empty() {
                items.push(self.node("Param", &[String::new(), ty]));
                continue;
            }
            for name in names {
                let name = self.text(name);
                let bound = self.bind(name);
                items.push(self.node("Param", &[bound, ty.clone()]));
            }
        }
        self.list(tag, &items)
    }

    /// A result: one type, or a parenthesized list whose names are bound locals.
    fn results(&mut self, results: Option<Node<'_>>) -> String {
        match results {
            None => String::new(),
            Some(list) if list.kind() == "parameter_list" => self.params(Some(list), "Results"),
            Some(ty) => self.ty(ty),
        }
    }

    /// The header of the Type-3 lines, in the language's own order: `func _fn[_T0](_v0 Ty(_T1, )) Ty(_T2, )`.
    ///
    /// 🔴 Not the `Func(…)` node the stream opens with. The Type-3 pass reads every function's first
    /// line as context and weighs it like any other; an s-expr header spells the parse tree's grammar
    /// (`Params`, `Param`, `Bind`, `Results`), which only headers spell, so every one of those tokens
    /// was rare and the header outweighed the body of every short function (the Rust frontend's finding).
    fn compact_header(&mut self, c: &Callable<'_>) -> String {
        let mut out = String::from("func _fn");
        if let Some(tps) = c.type_params {
            let mut declared = Vec::new();
            for decl in kids(tps) {
                for name in fields(decl, "name") {
                    let name = self.text(name);
                    self.declared_types.insert(name.to_owned());
                    declared.push(self.type_name(name));
                }
            }
            if !declared.is_empty() {
                let _ = write!(out, "[{}]", declared.join(", "));
            }
        }
        let params = self.header_params(c.params);
        let _ = write!(out, "({params})");
        match c.results {
            None => {}
            Some(list) if list.kind() == "parameter_list" => {
                let results = self.header_params(Some(list));
                let _ = write!(out, " ({results})");
            }
            Some(ty) => {
                let ty = self.ty(ty);
                let _ = write!(out, " {ty}");
            }
        }
        out
    }

    fn header_params(&mut self, list: Option<Node<'_>>) -> String {
        let mut items = Vec::new();
        for decl in list.map(kids).unwrap_or_default() {
            let variadic = decl.kind() == "variadic_parameter_declaration";
            let ty = field(decl, "type").map_or_else(String::new, |t| self.ty(t));
            let ty = if variadic { format!("...{ty}") } else { ty };
            let names = fields(decl, "name");
            if names.is_empty() {
                items.push(ty);
                continue;
            }
            for name in names {
                let name = self.text(name);
                let name = if name == "_" { "_".to_owned() } else { self.rename(name) };
                items.push(format!("{name} {ty}"));
            }
        }
        items.join(", ")
    }

    // ────── statements / blocks ──────

    pub(crate) fn block(&mut self, block: Node<'_>) -> String {
        let items: Vec<String> = statements(block).into_iter().map(|s| self.stmt(s)).collect();
        self.list("Block", &items)
    }

    /// An expression list as a binding site: names become `Bind`s, a list of two or more a tuple.
    fn binds(&mut self, list: Node<'_>) -> String {
        let items: Vec<String> = kids(list)
            .into_iter()
            .map(|k| if k.kind() == "identifier" { self.bind(self.text(k)) } else { self.expr(k) })
            .collect();
        self.tuple_of("PatTuple", items)
    }

    /// An expression list read as values, a list of two or more a tuple.
    fn exprs(&mut self, list: Node<'_>) -> String {
        let items: Vec<String> = kids(list).into_iter().map(|k| self.expr(k)).collect();
        self.tuple_of("Tuple", items)
    }

    fn tuple_of(&mut self, tag: &str, mut items: Vec<String>) -> String {
        if items.len() == 1 {
            items.pop().unwrap_or_default()
        } else {
            self.list(tag, &items)
        }
    }

    /// One `var` / `const` spec inside a body: `Let(PatType(binds, ty), values)`.
    fn spec(&mut self, spec: Node<'_>, tag: &str) -> String {
        let names: Vec<String> = fields(spec, "name").into_iter().map(|n| self.bind(self.text(n))).collect();
        let pat = self.tuple_of("PatTuple", names);
        let pat = match field(spec, "type") {
            Some(t) => {
                let ty = self.ty(t);
                self.node("PatType", &[pat, ty])
            }
            None => pat,
        };
        let value = field(spec, "value").map_or_else(String::new, |v| self.exprs(v));
        self.node(tag, &[pat, value])
    }

    fn specs(&mut self, decl: Node<'_>, spec_kind: &str, tag: &str) -> String {
        let mut specs = Vec::new();
        for k in kids(decl) {
            if k.kind() == spec_kind {
                specs.push(k);
            } else {
                specs.extend(kids(k).into_iter().filter(|s| s.kind() == spec_kind));
            }
        }
        let mut items: Vec<String> = specs.into_iter().map(|s| self.spec(s, tag)).collect();
        if items.len() == 1 {
            items.pop().unwrap_or_default()
        } else {
            self.list("Decls", &items)
        }
    }

    pub(crate) fn stmt(&mut self, n: Node<'_>) -> String {
        match n.kind() {
            "short_var_declaration" => {
                let left = field(n, "left").map_or_else(String::new, |l| self.binds(l));
                let right = field(n, "right").map_or_else(String::new, |r| self.exprs(r));
                self.node("Let", &[left, right])
            }
            "var_declaration" => self.specs(n, "var_spec", "Let"),
            "const_declaration" => self.specs(n, "const_spec", "LocalConst"),
            "type_declaration" => self.node("NestedItem", &[]),
            "assignment_statement" => {
                let op = field(n, "operator").map_or("=", |o| o.kind());
                let left = field(n, "left").map_or_else(String::new, |l| self.exprs(l));
                let right = field(n, "right").map_or_else(String::new, |r| self.exprs(r));
                if op == "=" {
                    self.node("Assign", &[left, right])
                } else {
                    self.node("Bin", &[left, Self::lit(op), right])
                }
            }
            "inc_statement" | "dec_statement" => {
                let tag = if n.kind() == "inc_statement" { "Inc" } else { "Dec" };
                let e = kids(n).first().map_or_else(String::new, |e| self.expr(*e));
                self.node(tag, &[e])
            }
            "send_statement" => {
                let ch = field(n, "channel").map_or_else(String::new, |c| self.expr(c));
                let v = field(n, "value").map_or_else(String::new, |v| self.expr(v));
                self.node("Send", &[ch, v])
            }
            "expression_statement" => {
                let e = kids(n).first().map_or_else(String::new, |e| self.expr(*e));
                self.node("ExprStmt", &[e])
            }
            "return_statement" => {
                let values = kids(n).first().map_or_else(String::new, |l| self.exprs(*l));
                self.node("Return", &[values])
            }
            "go_statement" | "defer_statement" => {
                let tag = if n.kind() == "go_statement" { "Go" } else { "Defer" };
                let e = kids(n).first().map_or_else(String::new, |e| self.expr(*e));
                self.node(tag, &[e])
            }
            "break_statement" | "continue_statement" | "goto_statement" => {
                let tag = match n.kind() {
                    "break_statement" => "Break",
                    "continue_statement" => "Continue",
                    _ => "Goto",
                };
                let label = kids(n).first().map_or_else(String::new, |l| Self::lit(self.text(*l)));
                self.node(tag, &[label])
            }
            "fallthrough_statement" => self.node("Fallthrough", &[]),
            "if_statement" => {
                let init = field(n, "initializer").map_or_else(String::new, |i| self.stmt(i));
                let cond = field(n, "condition").map_or_else(String::new, |c| self.expr(c));
                let then = field(n, "consequence").map_or_else(String::new, |b| self.block(b));
                let els = match field(n, "alternative") {
                    Some(alt) if alt.kind() == "if_statement" => self.stmt(alt),
                    Some(alt) => self.block(alt),
                    None => String::new(),
                };
                self.node("If", &[init, cond, then, els])
            }
            "for_statement" => {
                let body = field(n, "body").map_or_else(String::new, |b| self.block(b));
                match for_head(n) {
                    None => self.node("Loop", &[body]),
                    Some(h) if h.kind() == "range_clause" => {
                        let (pat, iter) = self.range_parts(h);
                        self.node("For", &[pat, iter, body])
                    }
                    Some(h) if h.kind() == "for_clause" => {
                        let (init, cond, update) = self.clause_parts(h);
                        self.node("ForC", &[init, cond, update, body])
                    }
                    Some(cond) => {
                        let cond = self.expr(cond);
                        self.node("While", &[cond, body])
                    }
                }
            }
            "expression_switch_statement" => {
                let init = field(n, "initializer").map_or_else(String::new, |i| self.stmt(i));
                let value = field(n, "value").map_or_else(String::new, |v| self.expr(v));
                let cases: Vec<String> = kids(n)
                    .into_iter()
                    .filter_map(|c| match c.kind() {
                        "expression_case" => {
                            let values = field(c, "value").map_or_else(String::new, |v| self.exprs(v));
                            let body = self.clause_block(c);
                            Some(self.node("Case", &[values, body]))
                        }
                        "default_case" => {
                            let body = self.clause_block(c);
                            Some(self.node("Default", &[body]))
                        }
                        _ => None,
                    })
                    .collect();
                let cases = cases.join(", ");
                self.node("Switch", &[init, value, cases])
            }
            "type_switch_statement" => {
                let init = field(n, "initializer").map_or_else(String::new, |i| self.stmt(i));
                let alias = field(n, "alias").map_or_else(String::new, |a| self.binds(a));
                let value = field(n, "value").map_or_else(String::new, |v| self.expr(v));
                let cases: Vec<String> = kids(n)
                    .into_iter()
                    .filter_map(|c| match c.kind() {
                        "type_case" => {
                            let types = self.case_types(c);
                            let body = self.clause_block(c);
                            Some(self.node("TypeCase", &[types, body]))
                        }
                        "default_case" => {
                            let body = self.clause_block(c);
                            Some(self.node("Default", &[body]))
                        }
                        _ => None,
                    })
                    .collect();
                let cases = cases.join(", ");
                self.node("TypeSwitch", &[init, alias, value, cases])
            }
            "select_statement" => {
                let cases: Vec<String> = kids(n)
                    .into_iter()
                    .filter_map(|c| match c.kind() {
                        "communication_case" => {
                            let comm = field(c, "communication").map_or_else(String::new, |m| self.stmt(m));
                            let body = self.clause_block(c);
                            Some(self.node("CommCase", &[comm, body]))
                        }
                        "default_case" => {
                            let body = self.clause_block(c);
                            Some(self.node("Default", &[body]))
                        }
                        _ => None,
                    })
                    .collect();
                self.list("Select", &cases)
            }
            "receive_statement" => {
                let left = match field(n, "left") {
                    Some(l) if has_token(n, ":=") => self.binds(l),
                    Some(l) => self.exprs(l),
                    None => String::new(),
                };
                let right = field(n, "right").map_or_else(String::new, |r| self.expr(r));
                self.node("RecvStmt", &[left, right])
            }
            "block" => {
                let body = self.block(n);
                self.node("BlockExpr", &[body])
            }
            "labeled_statement" => {
                let label = field(n, "label").map_or_else(String::new, |l| Self::lit(self.text(l)));
                let inner = kids(n)
                    .into_iter()
                    .find(|k| k.kind() != "label_name")
                    .map_or_else(String::new, |s| self.stmt(s));
                self.node("Labeled", &[label, inner])
            }
            "empty_statement" => self.node("Empty", &[]),
            other => self.node(&format!("Unknown_{other}"), &[]),
        }
    }

    /// A `case` clause's statements, as a block.
    fn clause_block(&mut self, clause: Node<'_>) -> String {
        let items: Vec<String> = statements(clause).into_iter().map(|s| self.stmt(s)).collect();
        self.list("Block", &items)
    }

    /// The types a type switch's `case` names.
    fn case_types(&mut self, clause: Node<'_>) -> String {
        let types: Vec<String> = fields(clause, "type").into_iter().map(|t| self.ty(t)).collect();
        types.join(", ")
    }

    /// `for k, v := range xs` → the bindings and what is ranged over.
    fn range_parts(&mut self, clause: Node<'_>) -> (String, String) {
        let pat = match field(clause, "left") {
            Some(l) if has_token(clause, ":=") => self.binds(l),
            Some(l) => self.exprs(l),
            None => String::new(),
        };
        let iter = field(clause, "right").map_or_else(String::new, |r| self.expr(r));
        (pat, iter)
    }

    /// `for init; cond; update` → its three parts, each possibly empty.
    fn clause_parts(&mut self, clause: Node<'_>) -> (String, String, String) {
        let init = field(clause, "initializer").map_or_else(String::new, |i| self.stmt(i));
        let cond = field(clause, "condition").map_or_else(String::new, |c| self.expr(c));
        let update = field(clause, "update").map_or_else(String::new, |u| self.stmt(u));
        (init, cond, update)
    }

    // ────── expressions ──────

    /// An expression — and, when units are being collected, kept as one if it is a proper part of
    /// its line: the statement's own expression is the line, already a unit. A wrapper (parentheses,
    /// `&`) is not a level of its own; what it wraps is the part.
    pub(crate) fn expr(&mut self, n: Node<'_>) -> String {
        if is_wrapper(n) {
            return self.expr_inner(n);
        }
        self.expr_depth += 1;
        let out = self.expr_inner(n);
        self.expr_depth -= 1;
        if self.expr_depth > 0 && self.is_unit(n) {
            if let Some(units) = self.units.as_mut() {
                units.push((self.unit_line, out.clone()));
            }
        }
        out
    }

    fn expr_inner(&mut self, n: Node<'_>) -> String {
        match n.kind() {
            "identifier" => self.ident(self.text(n)),
            "true" | "false" => self.node("Bool", &[Self::lit(n.kind())]),
            "nil" => self.node("Nil", &[]),
            "iota" => self.node("Iota", &[]),
            "int_literal" => self.node("Int", &[Self::lit(self.text(n))]),
            "float_literal" => self.node("Float", &[Self::lit(self.text(n))]),
            "imaginary_literal" => self.node("Imag", &[Self::lit(self.text(n))]),
            "rune_literal" => self.node("Char", &[Self::lit(unquote(self.text(n)))]),
            "interpreted_string_literal" | "raw_string_literal" => self.node("Str", &[Self::lit(unquote(self.text(n)))]),
            "binary_expression" => {
                let l = field(n, "left").map_or_else(String::new, |e| self.expr(e));
                let op = field(n, "operator").map_or("?op", |o| o.kind());
                let r = field(n, "right").map_or_else(String::new, |e| self.expr(e));
                self.node("Bin", &[l, Self::lit(op), r])
            }
            "unary_expression" => {
                let op = field(n, "operator").map_or("?un", |o| o.kind());
                let e = field(n, "operand").map_or_else(String::new, |e| self.expr(e));
                match op {
                    "&" => self.node("Ref", &[e]),
                    "<-" => self.node("Recv", &[e]),
                    op => self.node("Unary", &[Self::lit(op), e]),
                }
            }
            "parenthesized_expression" => kids(n).first().map_or_else(String::new, |e| self.expr(*e)),
            "call_expression" => self.call(n),
            "selector_expression" => {
                let operand = field(n, "operand");
                let name = field(n, "field").map_or("", |f| self.text(f));
                if let Some(pkg) = operand.filter(|o| o.kind() == "identifier").map(|o| self.text(o)) {
                    if self.is_package(pkg) {
                        return self.node("Path", &[Self::lit(&format!("{pkg}.{name}"))]);
                    }
                }
                let base = operand.map_or_else(String::new, |o| self.expr(o));
                self.node("Field", &[base, Self::lit(name)])
            }
            "index_expression" => {
                let base = field(n, "operand").map_or_else(String::new, |e| self.expr(e));
                let idx = field(n, "index").map_or_else(String::new, |e| self.expr(e));
                self.node("Index", &[base, idx])
            }
            "slice_expression" => {
                let base = field(n, "operand").map_or_else(String::new, |e| self.expr(e));
                let lo = field(n, "start").map_or_else(String::new, |e| self.expr(e));
                let hi = field(n, "end").map_or_else(String::new, |e| self.expr(e));
                let cap = field(n, "capacity").map_or_else(String::new, |e| self.expr(e));
                self.node("Slice", &[base, lo, hi, cap])
            }
            "type_assertion_expression" => {
                let e = field(n, "operand").map_or_else(String::new, |e| self.expr(e));
                let t = field(n, "type").map_or_else(String::new, |t| self.ty(t));
                self.node("Assert", &[e, t])
            }
            "type_conversion_expression" => {
                let e = field(n, "operand").map_or_else(String::new, |e| self.expr(e));
                let t = field(n, "type").map_or_else(String::new, |t| self.ty(t));
                self.node("Cast", &[e, t])
            }
            "type_instantiation_expression" => {
                let t = field(n, "type").map_or_else(String::new, |t| self.ty(t));
                let args: Vec<String> =
                    kids(n).into_iter().skip(1).map(|a| self.ty(a)).collect();
                let args = self.list("TypeArgs", &args);
                self.node("Inst", &[t, args])
            }
            "composite_literal" => {
                let t = field(n, "type").map_or_else(String::new, |t| self.ty(t));
                let elems = field(n, "body").map_or_else(String::new, |b| self.elements(b));
                self.node("Composite", &[t, elems])
            }
            // An elided inner literal (`[]Point{{1, 2}}`) — its type is the enclosing one's element.
            "literal_value" => {
                let elems = self.elements(n);
                self.node("Composite", &[String::new(), elems])
            }
            "func_literal" => {
                let params = self.params(field(n, "parameters"), "Params");
                let results = self.results(field(n, "result"));
                let body = field(n, "body").map_or_else(String::new, |b| self.block(b));
                self.node("Closure", &[params, results, body])
            }
            "variadic_argument" => {
                let e = kids(n).first().map_or_else(String::new, |e| self.expr(*e));
                self.node("Spread", &[e])
            }
            kind if is_type_kind(kind) => self.ty(n),
            other => self.node(&format!("Unknown_{other}"), &[]),
        }
    }

    /// A call: `Method(receiver, 'name', args)` when it calls through a value — the Rust frontend's
    /// method call — and `Call(callee, args)` when the callee is a name or a package member.
    fn call(&mut self, n: Node<'_>) -> String {
        let func = field(n, "function");
        let mut args: Vec<String> =
            field(n, "arguments").map(kids).unwrap_or_default().into_iter().map(|a| self.expr(a)).collect();
        if let Some(targs) = field(n, "type_arguments") {
            let types: Vec<String> = kids(targs).into_iter().map(|t| self.ty(t)).collect();
            args.push(self.list("TypeArgs", &types));
        }
        let args = args.join(", ");
        if let Some(sel) = func.filter(|f| f.kind() == "selector_expression") {
            let operand = field(sel, "operand");
            let on_package = operand
                .filter(|o| o.kind() == "identifier")
                .is_some_and(|o| self.is_package(self.text(o)));
            if !on_package {
                let recv = operand.map_or_else(String::new, |o| self.expr(o));
                let method = field(sel, "field").map_or("", |f| self.text(f));
                return self.node("Method", &[recv, Self::lit(method), args]);
            }
        }
        let callee = func.map_or_else(String::new, |f| self.expr(f));
        self.node("Call", &[callee, args])
    }

    /// A composite literal's elements. A key that is a bare name is a struct field: `FieldVal('name',
    /// v)`, the Rust frontend's struct-literal field, its name kept as spelled. Any other key is an
    /// expression — a map's — and the element is `KV(key, v)`. The two differ in what a varying key
    /// means: another field is another member (reflection), another map key is just another value.
    fn elements(&mut self, body: Node<'_>) -> String {
        let items: Vec<String> = kids(body)
            .into_iter()
            .map(|el| match el.kind() {
                "keyed_element" => {
                    let key = field(el, "key");
                    let name = key.and_then(|k| kids(k).into_iter().next()).filter(|i| i.kind() == "identifier");
                    // The key is written before the value, so a local in it is numbered in source order.
                    let (tag, key) = match (name, key) {
                        (Some(name), _) => ("FieldVal", Self::lit(self.text(name))),
                        (None, Some(k)) => ("KV", self.element(k)),
                        (None, None) => ("KV", String::new()),
                    };
                    let value = field(el, "value").map_or_else(String::new, |v| self.element(v));
                    self.node(tag, &[key, value])
                }
                _ => self.element(el),
            })
            .collect();
        items.join(", ")
    }

    fn element(&mut self, el: Node<'_>) -> String {
        match kids(el).first() {
            Some(inner) => self.expr(*inner),
            None => String::new(),
        }
    }

    // ────── types (summarized) ──────

    pub(crate) fn ty(&mut self, n: Node<'_>) -> String {
        match n.kind() {
            "type_identifier" | "qualified_type" | "identifier" => {
                let name = self.type_name(self.text(n));
                self.node("Ty", &[name, String::new()])
            }
            "generic_type" => {
                let name = field(n, "type").map_or_else(String::new, |t| self.type_name(self.text(t)));
                let args: Vec<String> =
                    field(n, "type_arguments").map(kids).unwrap_or_default().into_iter().map(|a| self.ty(a)).collect();
                let args = args.join(", ");
                self.node("Ty", &[name, args])
            }
            "pointer_type" => {
                let inner = kids(n).first().map_or_else(String::new, |t| self.ty(*t));
                self.node("TyPtr", &[inner])
            }
            "slice_type" => {
                let inner = field(n, "element").map_or_else(String::new, |t| self.ty(t));
                self.node("TySlice", &[inner])
            }
            "array_type" | "implicit_length_array_type" => {
                let inner = field(n, "element").map_or_else(String::new, |t| self.ty(t));
                self.node("TyArray", &[inner])
            }
            "map_type" => {
                let k = field(n, "key").map_or_else(String::new, |t| self.ty(t));
                let v = field(n, "value").map_or_else(String::new, |t| self.ty(t));
                self.node("TyMap", &[k, v])
            }
            "channel_type" => {
                let spelled = self.text(n);
                let dir = if spelled.starts_with("<-") {
                    "recv"
                } else if spelled.trim_start_matches("chan").trim_start().starts_with("<-") {
                    "send"
                } else {
                    ""
                };
                let v = field(n, "value").map_or_else(String::new, |t| self.ty(t));
                self.node("TyChan", &[Self::lit(dir), v])
            }
            "function_type" => {
                let params = self.params(field(n, "parameters"), "Params");
                let results = self.results(field(n, "result"));
                self.node("TyFn", &[params, results])
            }
            "struct_type" => {
                let fields = struct_fields(self, n);
                self.node("TyStruct", &[fields])
            }
            "interface_type" => {
                let items = interface_items(self, n);
                self.node("TyIface", &[items])
            }
            "parenthesized_type" => kids(n).first().map_or_else(String::new, |t| self.ty(*t)),
            "negated_type" => {
                let inner = kids(n).first().map_or_else(String::new, |t| self.ty(*t));
                self.node("TyApprox", &[inner])
            }
            // A constraint union (`~int | ~string`) and a type argument's element.
            "type_elem" | "type_constraint" => {
                let mut parts: Vec<String> = kids(n).into_iter().map(|t| self.ty(t)).collect();
                if parts.len() == 1 {
                    parts.pop().unwrap_or_default()
                } else {
                    self.list("TyUnion", &parts)
                }
            }
            _ => self.node("Ty_Other", &[]),
        }
    }

    /// Whether an expression is a Type-3 unit of its own: a node with structure — a call, a field
    /// read, an operator application, a literal with elements, a function literal — not a bare name,
    /// a package member, or a literal, which carry no agreement by themselves. The rule the Python
    /// and Rust frontends keep.
    fn is_unit(&self, n: Node<'_>) -> bool {
        match n.kind() {
            "call_expression"
            | "index_expression"
            | "slice_expression"
            | "binary_expression"
            | "unary_expression"
            | "type_assertion_expression"
            | "type_conversion_expression"
            | "type_instantiation_expression"
            | "func_literal" => true,
            "selector_expression" => !field(n, "operand")
                .filter(|o| o.kind() == "identifier")
                .is_some_and(|o| self.is_package(self.text(o))),
            "composite_literal" => field(n, "body").is_some_and(|b| !kids(b).is_empty()),
            "literal_value" => !kids(n).is_empty(),
            _ => false,
        }
    }
}

/// Syntax around an expression that is not a part of its own: what it wraps is the part.
fn is_wrapper(n: Node<'_>) -> bool {
    match n.kind() {
        "parenthesized_expression" => true,
        "unary_expression" => field(n, "operator").is_some_and(|o| o.kind() == "&"),
        _ => false,
    }
}

/// Whether a grammar kind is a type — what an argument list may hold in place of an expression
/// (`make([]int, n)`, `new(T)` parses `T` as a name).
fn is_type_kind(kind: &str) -> bool {
    matches!(
        kind,
        "array_type"
            | "channel_type"
            | "function_type"
            | "generic_type"
            | "implicit_length_array_type"
            | "interface_type"
            | "map_type"
            | "negated_type"
            | "parenthesized_type"
            | "pointer_type"
            | "qualified_type"
            | "slice_type"
            | "struct_type"
            | "type_identifier"
    )
}

/// The node a `for` statement tests: a `range` clause, a three-part clause, a bare condition, or
/// nothing (`for {}`).
fn for_head(n: Node<'_>) -> Option<Node<'_>> {
    let body = field(n, "body").map(|b| b.id());
    kids(n).into_iter().find(|k| Some(k.id()) != body)
}

// ───────────────────────────── declaration canonicals ─────────────────────────────

/// A struct's fields, one per name; an embedded field is `Embed(type)`. Tags are left out: two
/// structs that hold the same fields are the same shape however they serialize.
fn struct_fields(d: &mut Dump<'_>, struct_type: Node<'_>) -> String {
    let mut items = Vec::new();
    for list in kids(struct_type) {
        for decl in kids(list).into_iter().filter(|k| k.kind() == "field_declaration") {
            let ty = field(decl, "type").map_or_else(String::new, |t| d.ty(t));
            let names = fields(decl, "name");
            if names.is_empty() {
                items.push(d.node("Embed", &[ty]));
                continue;
            }
            for name in names {
                let name = Dump::lit(d.text(name));
                items.push(d.node("Field", &[name, ty.clone()]));
            }
        }
    }
    d.list("Named", &items)
}

/// An interface's method set and embedded types, with method signatures only.
fn interface_items(d: &mut Dump<'_>, iface: Node<'_>) -> String {
    let items: Vec<String> = kids(iface)
        .into_iter()
        .map(|item| {
            if item.kind() == "method_elem" {
                let name = field(item, "name").map_or_else(String::new, |n| Dump::lit(d.text(n)));
                let params = d.params(field(item, "parameters"), "Params");
                let results = d.results(field(item, "result"));
                d.node("IfaceFn", &[name, params, results])
            } else {
                let ty = d.ty(item);
                d.node("IfaceEmbed", &[ty])
            }
        })
        .collect();
    items.join(", ")
}

/// `g{n}` — the number of type parameters a declaration takes (their names are not compared).
fn generics_tag(type_params: Option<Node<'_>>) -> String {
    let n: usize = type_params.map(kids).unwrap_or_default().iter().map(|d| fields(*d, "name").len()).sum();
    format!("g{n}")
}

/// Names-preserved structural canonical of a `type X struct { … }` — the cluster pass's input for
/// `classes`.
#[must_use]
pub(crate) fn struct_canon(src: &str, name: &str, type_params: Option<Node<'_>>, struct_type: Node<'_>) -> String {
    let (imports, bound) = (HashMap::new(), HashSet::new());
    let mut d = Dump::new(src, &imports, &bound, None, None);
    let fields = struct_fields(&mut d, struct_type);
    d.node("Struct", &[Dump::lit(name), generics_tag(type_params), fields])
}

/// Names-preserved structural canonical of a `type X interface { … }` — the cluster pass's input for
/// `interfaces`.
#[must_use]
pub(crate) fn interface_canon(src: &str, name: &str, type_params: Option<Node<'_>>, iface: Node<'_>) -> String {
    let (imports, bound) = (HashMap::new(), HashSet::new());
    let mut d = Dump::new(src, &imports, &bound, None, None);
    let items = interface_items(&mut d, iface);
    d.node("Interface", &[Dump::lit(name), generics_tag(type_params), items])
}

// ───────────────────────────── callable analysis ─────────────────────────────

/// Every canonical form of one callable, off its syntax tree — no re-parse.
pub(crate) fn analyze(src: &str, imports: &HashMap<String, Arc<str>>, name: &str, c: &Callable<'_>) -> AnalyzedFn {
    let bound = locals_of(c, src);
    let cluster = Dump::new(src, imports, &bound, c.receiver, None).func(name, c, true, true);

    let statements = {
        let mut dump = Dump::new(src, imports, &bound, c.receiver, Some(&bound));
        // The definition's own header opens the stream at depth 0 and the body starts at depth 1 — the
        // contract every frontend agrees on.
        let head = dump.func(name, c, false, true);
        let mut walk = StreamWalk { dump: &mut dump, out: Vec::new() };
        walk.push(0, head);
        walk.stmts(c.body, 1);
        walk.out
    };

    let (type3_lines, type3_units, type_refs) = {
        let mut dump = Dump::new(src, imports, &bound, c.receiver, Some(&bound));
        dump.tmap = Some(HashMap::new());
        dump.units = Some(Vec::new());
        dump.unit_line = UNFILED;
        let head = dump.compact_header(c);
        let mut walk = StreamWalk { dump: &mut dump, out: Vec::new() };
        walk.push(0, head);
        walk.stmts(c.body, 1);
        let lines = walk.out.into_iter().map(|s| s.line).collect();
        (lines, dump.units.unwrap_or_default(), dump.type_refs)
    };

    let mut xd = Dump::new(src, imports, &bound, c.receiver, Some(&bound));
    let xname = xd.func(name, c, true, true);
    let size = xd.count;
    AnalyzedFn { cluster_canonical: cluster, xname_canonical: xname, type3_lines, type_refs, type3_units, statements, size }
}

// ───────────────────────────── statement stream ─────────────────────────────

/// One line per statement at **every** nesting level, in source order — the engine's
/// [`Facets::statements`](dup_defs_core::Facets::statements), and with the types slotted, the Type-3
/// lines.
///
/// A compound statement contributes a **header** naming what it tests (`If(cond)`, `Switch(value)`,
/// `For(binds, iter)`) with its body one level deeper — the shape the Rust and Python frontends
/// produce, so the languages' streams compare. Go's `if x := f(); x != nil` is two steps written on
/// one line: the initializer is emitted as its own statement before the `If`, so it lines up with the
/// same code written `x := f()` on the line above.
struct StreamWalk<'a, 'b> {
    dump: &'b mut Dump<'a>,
    out: Vec<Statement>,
}

impl StreamWalk<'_, '_> {
    /// A line is rendered whole before it is pushed, so the units still unfiled when it arrives are
    /// exactly its own.
    fn push(&mut self, depth: u16, line: String) {
        let at = u32::try_from(self.out.len()).unwrap_or(UNFILED - 1);
        if let Some(units) = self.dump.units.as_mut() {
            for unit in units.iter_mut().rev().take_while(|u| u.0 == UNFILED) {
                unit.0 = at;
            }
        }
        self.out.push(Statement { line, depth });
    }

    fn stmts(&mut self, holder: Node<'_>, depth: u16) {
        for s in statements(holder) {
            self.stmt(s, depth);
        }
    }

    fn init(&mut self, n: Node<'_>, depth: u16) {
        if let Some(init) = field(n, "initializer") {
            let line = self.dump.stmt(init);
            self.push(depth, line);
        }
    }

    fn stmt(&mut self, n: Node<'_>, depth: u16) {
        match n.kind() {
            "if_statement" => self.if_chain(n, depth),
            "for_statement" => {
                let head = match for_head(n) {
                    None => self.dump.node("Loop", &[]),
                    Some(h) if h.kind() == "range_clause" => {
                        let (pat, iter) = self.dump.range_parts(h);
                        self.dump.node("For", &[pat, iter])
                    }
                    Some(h) if h.kind() == "for_clause" => {
                        let (init, cond, update) = self.dump.clause_parts(h);
                        self.dump.node("ForC", &[init, cond, update])
                    }
                    Some(cond) => {
                        let cond = self.dump.expr(cond);
                        self.dump.node("While", &[cond])
                    }
                };
                self.push(depth, head);
                if let Some(body) = field(n, "body") {
                    self.stmts(body, depth + 1);
                }
            }
            "expression_switch_statement" => {
                self.init(n, depth);
                let value = field(n, "value").map_or_else(String::new, |v| self.dump.expr(v));
                let head = self.dump.node("Switch", &[value]);
                self.push(depth, head);
                for case in kids(n) {
                    let head = match case.kind() {
                        "expression_case" => {
                            let values = field(case, "value").map_or_else(String::new, |v| self.dump.exprs(v));
                            self.dump.node("Case", &[values])
                        }
                        "default_case" => self.dump.node("Default", &[]),
                        _ => continue,
                    };
                    self.push(depth + 1, head);
                    self.stmts(case, depth + 2);
                }
            }
            "type_switch_statement" => {
                self.init(n, depth);
                let alias = field(n, "alias").map_or_else(String::new, |a| self.dump.binds(a));
                let value = field(n, "value").map_or_else(String::new, |v| self.dump.expr(v));
                let head = self.dump.node("TypeSwitch", &[alias, value]);
                self.push(depth, head);
                for case in kids(n) {
                    let head = match case.kind() {
                        "type_case" => {
                            let types = self.dump.case_types(case);
                            self.dump.node("TypeCase", &[types])
                        }
                        "default_case" => self.dump.node("Default", &[]),
                        _ => continue,
                    };
                    self.push(depth + 1, head);
                    self.stmts(case, depth + 2);
                }
            }
            "select_statement" => {
                let head = self.dump.node("Select", &[]);
                self.push(depth, head);
                for case in kids(n) {
                    let head = match case.kind() {
                        "communication_case" => {
                            let comm = field(case, "communication").map_or_else(String::new, |m| self.dump.stmt(m));
                            self.dump.node("CommCase", &[comm])
                        }
                        "default_case" => self.dump.node("Default", &[]),
                        _ => continue,
                    };
                    self.push(depth + 1, head);
                    self.stmts(case, depth + 2);
                }
            }
            "block" => {
                let head = self.dump.node("Block", &[]);
                self.push(depth, head);
                self.stmts(n, depth + 1);
            }
            // A label names a statement; the statement is the step.
            "labeled_statement" => {
                if let Some(inner) = kids(n).into_iter().find(|k| k.kind() != "label_name") {
                    self.stmt(inner, depth);
                }
            }
            "empty_statement" => {}
            _ => {
                let line = self.dump.stmt(n);
                self.push(depth, line);
            }
        }
    }

    /// `if` / `else if` / `else`: an `else if` is a sibling of its `else`, not a level deeper — what
    /// the source means, and what Python's `elif` and the Rust frontend's chains render as.
    fn if_chain(&mut self, n: Node<'_>, depth: u16) {
        self.init(n, depth);
        let cond = field(n, "condition").map_or_else(String::new, |c| self.dump.expr(c));
        let head = self.dump.node("If", &[cond]);
        self.push(depth, head);
        if let Some(then) = field(n, "consequence") {
            self.stmts(then, depth + 1);
        }
        if let Some(alt) = field(n, "alternative") {
            let head = self.dump.node("Else", &[]);
            self.push(depth, head);
            if alt.kind() == "if_statement" {
                self.if_chain(alt, depth);
            } else {
                self.stmts(alt, depth + 1);
            }
        }
    }
}

// ───────────────────────────── reach ─────────────────────────────

/// What each package name a file's imports bind stands for, as the engine's dotted path
/// (`github.com/a/b/logx` → `github.com.a.b.logx`).
///
/// A blank import (`_ "embed"`) binds no name; a dot import (`. "x"`) binds names this cannot
/// enumerate, and guessing would invent reach that is not there — both are skipped. An unaliased
/// import binds the package's own name, which only the imported package's source states; the
/// convention it nearly always follows (the last path element, a major-version element skipped, a
/// `go-` prefix or `.vN` suffix dropped) is read instead, and a package that breaks it simply goes
/// unattributed.
#[must_use]
pub(crate) fn file_imports(root: Node<'_>, src: &str) -> HashMap<String, Arc<str>> {
    let mut out = HashMap::new();
    for decl in kids(root).into_iter().filter(|k| k.kind() == "import_declaration") {
        let mut specs = Vec::new();
        for k in kids(decl) {
            if k.kind() == "import_spec" {
                specs.push(k);
            } else {
                specs.extend(kids(k).into_iter().filter(|s| s.kind() == "import_spec"));
            }
        }
        for spec in specs {
            let Some(path) = field(spec, "path").map(|p| unquote(text(p, src))) else { continue };
            let binding = match field(spec, "name") {
                Some(name) if name.kind() == "package_identifier" => text(name, src).to_owned(),
                Some(_) => continue,
                None => default_package_name(path),
            };
            if binding.is_empty() {
                continue;
            }
            let segments: Vec<&str> = path.split('/').collect();
            out.insert(binding, dup_defs_core::reach::reach_path(&segments));
        }
    }
    out
}

/// The name an unaliased import conventionally binds.
fn default_package_name(path: &str) -> String {
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let is_major = |s: &str| s.len() > 1 && s.starts_with('v') && s[1..].chars().all(|c| c.is_ascii_digit());
    let last = match segments.as_slice() {
        [.., prev, last] if is_major(last) => prev,
        [.., last] => last,
        [] => return String::new(),
    };
    let mut name = *last;
    if let Some((stem, version)) = name.rsplit_once(".v") {
        if !version.is_empty() && version.chars().all(|c| c.is_ascii_digit()) {
            name = stem;
        }
    }
    let name = name.strip_prefix("go-").unwrap_or(name);
    let name = name.strip_suffix("-go").unwrap_or(name);
    name.replace(['-', '.'], "_")
}

/// The dotted paths one callable reaches: every package it names through a selector or a qualified
/// type, resolved through the file's imports.
///
/// Not filtered by boundness, as in the Rust frontend: a local shadowing a package inside one
/// function is rare, and reading the shadow as "this definition does not reach that package" is a
/// silent false negative, the expensive direction here.
pub(crate) fn reached_paths(body: Node<'_>, src: &str, imports: &HashMap<String, Arc<str>>) -> Vec<Arc<str>> {
    fn walk(n: Node<'_>, src: &str, imports: &HashMap<String, Arc<str>>, out: &mut Vec<Arc<str>>) {
        let head = match n.kind() {
            "selector_expression" => field(n, "operand").filter(|o| o.kind() == "identifier"),
            "qualified_type" => field(n, "package"),
            _ => None,
        };
        if let Some(path) = head.and_then(|h| imports.get(text(h, src))) {
            out.push(Arc::clone(path));
        }
        for k in kids(n) {
            walk(k, src, imports, out);
        }
    }
    if imports.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    walk(body, src, imports, &mut out);
    out.sort();
    out.dedup();
    out
}
