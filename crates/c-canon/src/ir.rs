//! An owned syntax tree for C bodies, lowered once from tree-sitter's.
//!
//! tree-sitter's tree is read-only: a node is a view into a parse that cannot be edited. The normal
//! form ([`crate::normal`]) rewrites a body — flips a branch, lifts a guard, inlines a temporary — so
//! the canonicals are written from this tree instead, which can be. It holds exactly what the
//! canonicals read and nothing else: names and literals as owned text, no positions, no comments.
//!
//! Three things C needs that the grammar does not hand over ready-made:
//!
//! * **Types are split in two.** `int *a[4]` puts `int` in the specifier and "array of four pointers
//!   to" in the declarator, read inside out; [`declarate`] puts them back together.
//! * **A body's own preprocessor conditionals** stay — `#ifdef X … #else … #endif` inside a function
//!   is part of how it is written, and two copies of it carry the same ones — as [`Stmt::PreIf`],
//!   each branch with its statements.
//! * **Macros used as statement keywords** — `list_for_each(p, head) { … }` — are a call the parser
//!   had to end with an invented `;`, followed by a block. Read as written that is a call and an
//!   unrelated block; it is a loop, [`Stmt::MacroLoop`].
//!
//! Lowering is depth-limited ([`MAX_DEPTH`]): machine-written C nests an expression thousands of levels
//! deep (a table's sum, an `else if` ladder of every opcode), and every walk downstream — this one,
//! the canonicals, the engine reading them — recurses. A subtree past the limit is one `Unknown`.

use std::cell::Cell;

use tree_sitter::Node;

use crate::tree::{field, fields, kids, missing_semicolon, squashed, text, unquote};

/// How deep lowering follows nested statements and expressions before writing the rest as one
/// opaque node.
pub(crate) const MAX_DEPTH: usize = 200;

thread_local! {
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// One level of lowering depth, released when dropped; `None` past [`MAX_DEPTH`].
struct Level;

impl Level {
    fn enter() -> Option<Self> {
        DEPTH.with(|d| {
            if d.get() >= MAX_DEPTH {
                None
            } else {
                d.set(d.get() + 1);
                Some(Level)
            }
        })
    }
}

impl Drop for Level {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Type {
    /// A type name as spelled — `int`, `unsigned long`, `size_t`, `struct foo`.
    Named(String),
    Ptr(Box<Type>),
    Array(Box<Type>, Option<Box<Expr>>),
    /// A function type: parameters, whether it is variadic, what it returns.
    Func(Vec<Param>, bool, Box<Type>),
    /// A `struct` / `union` written out where it is used.
    Record { kind: &'static str, name: Option<String>, fields: Vec<FieldDecl> },
    /// An `enum` written out where it is used.
    Enum { name: Option<String>, items: Vec<(String, Option<Expr>)> },
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Param {
    pub name: Option<String>,
    pub ty: Type,
}

/// One member of a `struct` / `union`: `None` for an anonymous member.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FieldDecl {
    pub name: Option<String>,
    pub ty: Type,
    pub bits: Option<Expr>,
}

/// One declarator of a local declaration: `int a = 1, *b;` is two.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Spec {
    pub name: String,
    pub ty: Type,
    pub init: Option<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Stmt {
    /// A local declaration, one spec per declarator.
    Decl(Vec<Spec>),
    /// A `struct`, `typedef` or nested function declared inside a body.
    NestedType,
    /// `x = e`, `x += e` written as a statement.
    Assign { op: String, lhs: Expr, rhs: Expr },
    /// `x++;` / `--x;` — as a statement the two are one.
    IncDec { inc: bool, target: Expr },
    Expr(Expr),
    Return(Option<Expr>),
    /// `break` / `continue` / `goto`, with the label a `goto` names.
    Jump { kind: Jump, label: Option<String> },
    If { cond: Expr, then: Vec<Stmt>, els: Option<Box<Else>> },
    While { cond: Expr, body: Vec<Stmt> },
    DoWhile { body: Vec<Stmt>, cond: Expr },
    For { init: Option<Box<Stmt>>, cond: Option<Expr>, update: Option<Box<Stmt>>, body: Vec<Stmt> },
    Switch { value: Expr, cases: Vec<Case> },
    Block(Vec<Stmt>),
    Labeled { label: String, stmt: Option<Box<Stmt>> },
    /// `#if …` / `#elif …` / `#else` inside a body: each branch's condition as written (`else` for
    /// the last) and its statements.
    PreIf { conds: Vec<String>, branches: Vec<Vec<Stmt>> },
    /// A macro used as a loop or block header: `list_for_each(p, head) { … }`.
    MacroLoop { head: Expr, body: Vec<Stmt> },
    Asm,
    Empty,
    Unknown(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Jump {
    Break,
    Continue,
    Goto,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Else {
    Block(Vec<Stmt>),
    /// `else if …` — always a [`Stmt::If`].
    If(Stmt),
}

/// A `case` (`value` set) or the `default` label (`value` `None`), with the statements up to the next
/// label — C's cases fall through, so where one ends is where the next begins.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Case {
    pub value: Option<Expr>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Expr {
    Name(String),
    /// `Int` / `Float` / `Char` / `Str`, with the literal's text (quotes off for the last two).
    Lit(&'static str, String),
    Bool(bool),
    Null,
    Binary { op: String, l: Box<Expr>, r: Box<Expr> },
    /// Every prefix operator: `!`, `-`, `+`, `~`, and `*` / `&` on a pointer.
    Unary { op: String, e: Box<Expr> },
    /// `x++`, `--x` inside an expression.
    Update { inc: bool, prefix: bool, e: Box<Expr> },
    /// An assignment used as a value: `while ((c = next()) != EOF)`.
    Assign { op: String, l: Box<Expr>, r: Box<Expr> },
    Cond { c: Box<Expr>, t: Option<Box<Expr>>, f: Box<Expr> },
    Comma(Box<Expr>, Box<Expr>),
    Call { f: Box<Expr>, args: Vec<Expr> },
    /// `a.b` and `a->b` alike — the same member read, through a value or a pointer.
    Field { e: Box<Expr>, field: String },
    Index { e: Box<Expr>, i: Box<Expr> },
    Cast { ty: Type, e: Box<Expr> },
    SizeofType(Type),
    SizeofValue(Box<Expr>),
    Alignof(Type),
    Offsetof(Type, String),
    /// `(struct p){ .x = 1 }`.
    Compound { ty: Type, elems: Vec<Elem> },
    /// A brace initializer: `{ 1, 2 }`, `{ .x = 1 }`.
    InitList(Vec<Elem>),
    /// GNU's statement expression, `({ …; v; })`.
    GnuBlock(Vec<Stmt>),
    Paren(Box<Expr>),
    Generic(Box<Expr>),
    Asm,
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Elem {
    Plain(Expr),
    /// `.name = v` — a member, named.
    Field(String, Expr),
    /// `[i] = v` — an element, by index.
    Index(Expr, Expr),
}

/// A function, owned: what every canonical reads.
#[derive(Clone, Debug)]
pub(crate) struct Func {
    pub params: Vec<Param>,
    pub variadic: bool,
    pub ret: Type,
    pub body: Vec<Stmt>,
}

/// A function definition as the grammar splits it — what is lowered to a [`Func`], and what the lenses
/// and the reach walk read directly.
#[derive(Clone)]
pub(crate) struct FnParts<'t> {
    /// The name, or the macro call that builds it (`FUNC(put_pixels)`), spelled as written.
    pub name: String,
    pub params: Node<'t>,
    pub ret: Type,
    pub body: Node<'t>,
    /// `static` — visible in its own file only.
    pub is_static: bool,
    /// The storage classes and function specifiers written on it (`static`, `inline`), in order.
    pub specifiers: Vec<String>,
}

// ───────────────────────────── declarations ─────────────────────────────

/// The parts of a function definition; `None` when its declarator names no function.
pub(crate) fn fn_parts<'t>(def: Node<'t>, src: &str) -> Option<FnParts<'t>> {
    let body = field(def, "body")?;
    let declarator = field(def, "declarator")?;
    // `static av_cold int f(…)` read without the macro's definition: the parser takes the macro for
    // the type and leaves the real one in an error beside it. The type is the word in the error.
    let base = kids(def)
        .into_iter()
        .find(|k| k.is_error() && k.end_byte() <= declarator.start_byte())
        .map(|e| squashed(e, src))
        .filter(|t| !t.is_empty() && t.split(' ').all(|w| w.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '*')))
        .map_or_else(|| field(def, "type").map_or(Type::Other, |t| lower_spec(t, src)), Type::Named);
    let (name, params, ret) = fn_declarator(base, declarator, src)?;
    let specifiers: Vec<String> = kids(def)
        .into_iter()
        .filter(|k| matches!(k.kind(), "storage_class_specifier" | "type_qualifier"))
        .map(|k| text(k, src).to_owned())
        .collect();
    let is_static = specifiers.iter().any(|s| s == "static");
    Some(FnParts { name, params, ret, body, is_static, specifiers })
}

/// The name, the parameter list and the return type a function's declarator spells.
///
/// The return type wraps around the name from outside in: `int *f(int)` returns a pointer, and
/// `void (*signal(int, handler))(int)` returns a pointer to a function — so every declarator met on
/// the way to the one holding the name adds to the type returned. A function declarator whose own
/// declarator is another function declarator is not C at all but a macro building the name
/// (`static void FUNC(put_pixels)(uint8_t *dst)`); the name is that macro call, as written.
fn fn_declarator<'t>(base: Type, d: Node<'t>, src: &str) -> Option<(String, Node<'t>, Type)> {
    match d.kind() {
        "pointer_declarator" => fn_declarator(Type::Ptr(Box::new(base)), field(d, "declarator")?, src),
        "array_declarator" => fn_declarator(Type::Array(Box::new(base), None), field(d, "declarator")?, src),
        "parenthesized_declarator" | "attributed_declarator" => {
            let inner = kids(d).into_iter().find(|k| is_declarator(k.kind()))?;
            fn_declarator(base, inner, src)
        }
        "function_declarator" => {
            let inner = field(d, "declarator")?;
            let params = field(d, "parameters")?;
            match inner.kind() {
                "identifier" => Some((text(inner, src).to_owned(), params, base)),
                "function_declarator" => Some((squashed(inner, src), params, base)),
                _ => {
                    let (variadic, list) = lower_params(Some(params), src);
                    fn_declarator(Type::Func(list, variadic, Box::new(base)), inner, src)
                }
            }
        }
        _ => None,
    }
}

fn is_declarator(kind: &str) -> bool {
    matches!(
        kind,
        "identifier"
            | "field_identifier"
            | "type_identifier"
            | "primitive_type"
            | "pointer_declarator"
            | "array_declarator"
            | "function_declarator"
            | "parenthesized_declarator"
            | "attributed_declarator"
            | "abstract_pointer_declarator"
            | "abstract_array_declarator"
            | "abstract_function_declarator"
            | "abstract_parenthesized_declarator"
    )
}

/// A type specifier: `int`, `unsigned long`, `struct foo`, or a `struct` / `enum` written out.
pub(crate) fn lower_spec(n: Node<'_>, src: &str) -> Type {
    match n.kind() {
        "primitive_type" | "type_identifier" | "sized_type_specifier" | "macro_type_specifier" => {
            Type::Named(squashed(n, src))
        }
        "struct_specifier" | "union_specifier" => {
            let kind = if n.kind() == "struct_specifier" { "struct" } else { "union" };
            let name = field(n, "name").map(|t| text(t, src).to_owned());
            match field(n, "body") {
                Some(body) => Type::Record { kind, name, fields: lower_fields(body, src) },
                None => Type::Named(format!("{kind} {}", name.unwrap_or_default())),
            }
        }
        "enum_specifier" => {
            let name = field(n, "name").map(|t| text(t, src).to_owned());
            match field(n, "body") {
                Some(body) => Type::Enum { name, items: lower_enumerators(body, src) },
                None => Type::Named(format!("enum {}", name.unwrap_or_default())),
            }
        }
        _ => Type::Other,
    }
}

/// The name a declarator declares and the type it gives it, the specifier's type `base` wrapped by
/// every pointer, array and function declarator on the way in.
pub(crate) fn declarate(base: Type, d: Node<'_>, src: &str) -> (Option<String>, Type) {
    let Some(_level) = Level::enter() else { return (None, Type::Other) };
    match d.kind() {
        "identifier" | "field_identifier" | "type_identifier" | "primitive_type" => (Some(text(d, src).to_owned()), base),
        "pointer_declarator" | "abstract_pointer_declarator" => {
            let ptr = Type::Ptr(Box::new(base));
            match field(d, "declarator") {
                Some(inner) => declarate(ptr, inner, src),
                None => (None, ptr),
            }
        }
        "array_declarator" | "abstract_array_declarator" => {
            let size = field(d, "size").filter(|s| s.kind() != "*").map(|s| Box::new(lower_expr(s, src)));
            let array = Type::Array(Box::new(base), size);
            match field(d, "declarator") {
                Some(inner) => declarate(array, inner, src),
                None => (None, array),
            }
        }
        "function_declarator" | "abstract_function_declarator" => {
            let (variadic, params) = lower_params(field(d, "parameters"), src);
            let func = Type::Func(params, variadic, Box::new(base));
            match field(d, "declarator") {
                Some(inner) => declarate(func, inner, src),
                None => (None, func),
            }
        }
        "parenthesized_declarator" | "attributed_declarator" | "abstract_parenthesized_declarator" => {
            match kids(d).into_iter().find(|k| is_declarator(k.kind())) {
                Some(inner) => declarate(base, inner, src),
                None => (None, base),
            }
        }
        "init_declarator" => match field(d, "declarator") {
            Some(inner) => declarate(base, inner, src),
            None => (None, base),
        },
        _ => (None, base),
    }
}

/// A parameter list: whether it ends in `...`, and each parameter. `(void)` declares none.
pub(crate) fn lower_params(list: Option<Node<'_>>, src: &str) -> (bool, Vec<Param>) {
    let mut variadic = false;
    let mut out = Vec::new();
    for p in list.map(kids).unwrap_or_default() {
        match p.kind() {
            "variadic_parameter" => variadic = true,
            "parameter_declaration" => {
                let base = field(p, "type").map_or(Type::Other, |t| lower_spec(t, src));
                let (name, ty) = match field(p, "declarator") {
                    Some(d) => declarate(base, d, src),
                    None => (None, base),
                };
                out.push(Param { name, ty });
            }
            // A K&R parameter, typed by the declarations after the list.
            "identifier" => out.push(Param { name: Some(text(p, src).to_owned()), ty: Type::Named("int".to_owned()) }),
            _ => {}
        }
    }
    if let [Param { name: None, ty: Type::Named(t) }] = out.as_slice() {
        if t == "void" {
            out.clear();
        }
    }
    (variadic, out)
}

/// The members a `struct` / `union` body declares, those under a preprocessor conditional included.
pub(crate) fn lower_fields(body: Node<'_>, src: &str) -> Vec<FieldDecl> {
    field_items(&kids(body), src)
}

fn field_items(items: &[Node<'_>], src: &str) -> Vec<FieldDecl> {
    let mut out = Vec::new();
    for item in items {
        match item.kind() {
            "field_declaration" => {
                let base = field(*item, "type").map_or(Type::Other, |t| lower_spec(t, src));
                let bits = kids(*item)
                    .into_iter()
                    .find(|k| k.kind() == "bitfield_clause")
                    .and_then(|b| kids(b).into_iter().next())
                    .map(|e| lower_expr(e, src));
                let declarators = fields(*item, "declarator");
                if declarators.is_empty() {
                    out.push(FieldDecl { name: None, ty: base, bits });
                    continue;
                }
                for d in declarators {
                    let (name, ty) = declarate(base.clone(), d, src);
                    out.push(FieldDecl { name, ty, bits: bits.clone() });
                }
            }
            "preproc_if" | "preproc_ifdef" => {
                out.extend(preproc_branches(*item).into_iter().flat_map(|(_, held)| field_items(&held, src)));
            }
            _ => {}
        }
    }
    out
}

fn lower_enumerators(body: Node<'_>, src: &str) -> Vec<(String, Option<Expr>)> {
    kids(body)
        .into_iter()
        .filter(|k| k.kind() == "enumerator")
        .filter_map(|e| Some((text(field(e, "name")?, src).to_owned(), field(e, "value").map(|v| lower_expr(v, src)))))
        .collect()
}

/// The branches of a preprocessor conditional — `#if`, each `#elif`, `#else` — as (condition, the
/// nodes it holds). The condition is written as the directive spells it; `else` for the last branch.
pub(crate) fn preproc_branches(n: Node<'_>) -> Vec<(Node<'_>, Vec<Node<'_>>)> {
    let mut out = Vec::new();
    let mut at = Some(n);
    while let Some(branch) = at {
        let skip: Vec<usize> = ["name", "condition", "alternative"]
            .iter()
            .filter_map(|f| field(branch, f))
            .map(|k| k.id())
            .collect();
        let held: Vec<Node<'_>> = kids(branch).into_iter().filter(|k| !skip.contains(&k.id())).collect();
        out.push((branch, held));
        at = field(branch, "alternative");
    }
    out
}

/// The condition a preprocessor branch tests, as a reader would write it: `ifdef X`, `if X > 1`,
/// `else`.
pub(crate) fn preproc_cond(branch: Node<'_>, src: &str) -> String {
    let directive = match branch.kind() {
        "preproc_ifdef" | "preproc_elifdef" => {
            // `#ifdef` and `#ifndef` share the node; the keyword says which.
            let head = text(branch, src).trim_start().trim_start_matches('#').trim_start();
            let word: String = head.chars().take_while(char::is_ascii_alphabetic).collect();
            return format!("{word} {}", field(branch, "name").map_or("", |n| text(n, src)));
        }
        "preproc_else" => return "else".to_owned(),
        "preproc_elif" => "elif",
        _ => "if",
    };
    format!("{directive} {}", field(branch, "condition").map_or_else(String::new, |c| squashed(c, src)))
}

// ───────────────────────────── statements ─────────────────────────────

pub(crate) fn lower_func(parts: &FnParts<'_>, src: &str) -> Func {
    let (variadic, params) = lower_params(Some(parts.params), src);
    Func { params, variadic, ret: parts.ret.clone(), body: lower_block(parts.body, src) }
}

/// The statements a block holds.
pub(crate) fn lower_block(holder: Node<'_>, src: &str) -> Vec<Stmt> {
    lower_list(&kids(holder), src)
}

/// A run of statement nodes, a macro used as a loop header joined to the block it opens.
fn lower_list(items: &[Node<'_>], src: &str) -> Vec<Stmt> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < items.len() {
        let n = items[i];
        if n.kind() == "expression_statement" && missing_semicolon(n) {
            let call = kids(n).into_iter().next().filter(|c| c.kind() == "call_expression");
            let block = items.get(i + 1).filter(|k| k.kind() == "compound_statement");
            if let (Some(call), Some(block)) = (call, block) {
                out.push(Stmt::MacroLoop { head: lower_expr(call, src), body: lower_block(*block, src) });
                i += 2;
                continue;
            }
        }
        match lower_stmt(n, src) {
            Stmt::Empty => {}
            s => out.push(s),
        }
        i += 1;
    }
    out
}

/// The statements a branch or loop body holds: a block's own, or the one statement written bare —
/// `if (x) return;` and `if (x) { return; }` are one body.
fn body_of(n: Option<Node<'_>>, src: &str) -> Vec<Stmt> {
    match n {
        Some(b) if b.kind() == "compound_statement" => lower_block(b, src),
        Some(s) => match lower_stmt(s, src) {
            Stmt::Empty => Vec::new(),
            other => vec![other],
        },
        None => Vec::new(),
    }
}

/// The expression a `( … )` condition holds.
fn cond_of(n: Option<Node<'_>>, src: &str) -> Expr {
    match n {
        Some(p) if p.kind() == "parenthesized_expression" => {
            kids(p).into_iter().next().map_or(Expr::Unknown(String::new()), |e| lower_expr(e, src))
        }
        Some(e) => lower_expr(e, src),
        None => Expr::Unknown(String::new()),
    }
}

/// An expression written as a statement: an assignment or an increment is the step it takes, any
/// other expression a value computed for its effects.
fn expr_stmt(e: Node<'_>, src: &str) -> Stmt {
    match e.kind() {
        "assignment_expression" => Stmt::Assign {
            op: field(e, "operator").map_or("=", |o| o.kind()).to_owned(),
            lhs: field(e, "left").map_or(Expr::Unknown(String::new()), |l| lower_expr(l, src)),
            rhs: field(e, "right").map_or(Expr::Unknown(String::new()), |r| lower_expr(r, src)),
        },
        "update_expression" => Stmt::IncDec {
            inc: field(e, "operator").is_some_and(|o| o.kind() == "++"),
            target: field(e, "argument").map_or(Expr::Unknown(String::new()), |a| lower_expr(a, src)),
        },
        "gnu_asm_expression" => Stmt::Asm,
        _ => Stmt::Expr(lower_expr(e, src)),
    }
}

/// A local declaration: one spec per declarator; a declaration that declares no object (a `struct`
/// tag, a prototype's worth of type) is a nested type.
fn lower_decl(n: Node<'_>, src: &str) -> Stmt {
    let base = field(n, "type").map_or(Type::Other, |t| lower_spec(t, src));
    let mut specs = Vec::new();
    for d in fields(n, "declarator") {
        let init = (d.kind() == "init_declarator").then(|| field(d, "value")).flatten().map(|v| lower_expr(v, src));
        let (name, ty) = declarate(base.clone(), d, src);
        if matches!(ty, Type::Func(..)) {
            continue; // a prototype inside a body declares no object
        }
        if let Some(name) = name {
            specs.push(Spec { name, ty, init });
        }
    }
    if specs.is_empty() {
        Stmt::NestedType
    } else {
        Stmt::Decl(specs)
    }
}

#[allow(clippy::too_many_lines)] // one arm per statement kind reads better straight-through
pub(crate) fn lower_stmt(n: Node<'_>, src: &str) -> Stmt {
    let Some(_level) = Level::enter() else { return Stmt::Unknown("deep".to_owned()) };
    match n.kind() {
        "compound_statement" => Stmt::Block(lower_block(n, src)),
        "declaration" => lower_decl(n, src),
        "type_definition" | "function_definition" | "linkage_specification" => Stmt::NestedType,
        "expression_statement" => match kids(n).into_iter().next() {
            None => Stmt::Empty,
            Some(e) => expr_stmt(e, src),
        },
        "return_statement" => Stmt::Return(kids(n).into_iter().next().map(|e| lower_expr(e, src))),
        "if_statement" => Stmt::If {
            cond: cond_of(field(n, "condition"), src),
            then: body_of(field(n, "consequence"), src),
            els: field(n, "alternative").and_then(|alt| kids(alt).into_iter().next()).map(|s| {
                Box::new(if s.kind() == "if_statement" { Else::If(lower_stmt(s, src)) } else { Else::Block(body_of(Some(s), src)) })
            }),
        },
        "while_statement" => {
            Stmt::While { cond: cond_of(field(n, "condition"), src), body: body_of(field(n, "body"), src) }
        }
        "do_statement" => Stmt::DoWhile { body: body_of(field(n, "body"), src), cond: cond_of(field(n, "condition"), src) },
        "for_statement" => Stmt::For {
            init: field(n, "initializer").map(|i| {
                Box::new(if i.kind() == "declaration" { lower_decl(i, src) } else { expr_stmt(i, src) })
            }),
            cond: field(n, "condition").map(|c| lower_expr(c, src)),
            update: field(n, "update").map(|u| Box::new(expr_stmt(u, src))),
            body: body_of(field(n, "body"), src),
        },
        "switch_statement" => Stmt::Switch {
            value: cond_of(field(n, "condition"), src),
            cases: field(n, "body")
                .map(kids)
                .unwrap_or_default()
                .into_iter()
                .filter(|c| c.kind() == "case_statement")
                .map(|c| {
                    let value = field(c, "value");
                    let held: Vec<Node<'_>> = kids(c).into_iter().filter(|k| Some(k.id()) != value.map(|v| v.id())).collect();
                    Case { value: value.map(|v| lower_expr(v, src)), body: lower_list(&held, src) }
                })
                .collect(),
        },
        "break_statement" => Stmt::Jump { kind: Jump::Break, label: None },
        "continue_statement" => Stmt::Jump { kind: Jump::Continue, label: None },
        "goto_statement" => Stmt::Jump { kind: Jump::Goto, label: field(n, "label").map(|l| text(l, src).to_owned()) },
        "labeled_statement" => Stmt::Labeled {
            label: field(n, "label").map_or_else(String::new, |l| text(l, src).to_owned()),
            stmt: kids(n)
                .into_iter()
                .find(|k| k.kind() != "statement_identifier")
                .map(|s| Box::new(lower_stmt(s, src))),
        },
        "attributed_statement" => kids(n)
            .into_iter()
            .rfind(|k| k.kind() != "attribute_declaration")
            .map_or(Stmt::Empty, |s| lower_stmt(s, src)),
        "preproc_if" | "preproc_ifdef" => {
            let branches = preproc_branches(n);
            Stmt::PreIf {
                conds: branches.iter().map(|(b, _)| preproc_cond(*b, src)).collect(),
                branches: branches.iter().map(|(_, held)| lower_list(held, src)).collect(),
            }
        }
        "preproc_call" | "preproc_def" | "preproc_function_def" | "preproc_include" => Stmt::Empty,
        "seh_try_statement" => field(n, "body").map_or(Stmt::Empty, |b| Stmt::Block(lower_block(b, src))),
        other => Stmt::Unknown(other.to_owned()),
    }
}

// ───────────────────────────── expressions ─────────────────────────────

fn boxed(n: Option<Node<'_>>, src: &str) -> Box<Expr> {
    Box::new(n.map_or(Expr::Unknown(String::new()), |e| lower_expr(e, src)))
}

/// A number literal: an integer, or a float when it has a fraction or an exponent.
fn number(t: &str) -> Expr {
    let lower = t.to_ascii_lowercase();
    let hex = lower.starts_with("0x");
    let float = lower.contains('.') || (!hex && lower.contains('e')) || (hex && lower.contains('p'));
    Expr::Lit(if float { "Float" } else { "Int" }, t.to_owned())
}

/// A type written for a cast, a `sizeof` or a compound literal.
pub(crate) fn lower_type_descriptor(n: Node<'_>, src: &str) -> Type {
    let base = field(n, "type").map_or(Type::Other, |t| lower_spec(t, src));
    match field(n, "declarator") {
        Some(d) => declarate(base, d, src).1,
        None => base,
    }
}

#[allow(clippy::too_many_lines)] // one arm per expression kind reads better straight-through
pub(crate) fn lower_expr(n: Node<'_>, src: &str) -> Expr {
    let Some(_level) = Level::enter() else { return Expr::Unknown("deep".to_owned()) };
    match n.kind() {
        "identifier" => Expr::Name(text(n, src).to_owned()),
        "number_literal" => number(text(n, src)),
        "char_literal" => Expr::Lit("Char", unquote(text(n, src)).to_owned()),
        "string_literal" => Expr::Lit("Str", unquote(text(n, src)).to_owned()),
        "concatenated_string" => {
            let joined: String = kids(n)
                .into_iter()
                .map(|piece| match piece.kind() {
                    "string_literal" => unquote(text(piece, src)).to_owned(),
                    _ => format!("{{{}}}", text(piece, src)),
                })
                .collect();
            Expr::Lit("Str", joined)
        }
        "true" => Expr::Bool(true),
        "false" => Expr::Bool(false),
        "null" => Expr::Null,
        "binary_expression" => Expr::Binary {
            op: field(n, "operator").map_or("?op", |o| o.kind()).to_owned(),
            l: boxed(field(n, "left"), src),
            r: boxed(field(n, "right"), src),
        },
        "unary_expression" | "pointer_expression" => Expr::Unary {
            op: field(n, "operator").map_or("?un", |o| o.kind()).to_owned(),
            e: boxed(field(n, "argument"), src),
        },
        "update_expression" => Expr::Update {
            inc: field(n, "operator").is_some_and(|o| o.kind() == "++"),
            prefix: n.child(0).is_some_and(|first| !first.is_named()),
            e: boxed(field(n, "argument"), src),
        },
        "assignment_expression" => Expr::Assign {
            op: field(n, "operator").map_or("=", |o| o.kind()).to_owned(),
            l: boxed(field(n, "left"), src),
            r: boxed(field(n, "right"), src),
        },
        "conditional_expression" => Expr::Cond {
            c: boxed(field(n, "condition"), src),
            t: field(n, "consequence").map(|t| Box::new(lower_expr(t, src))),
            f: boxed(field(n, "alternative"), src),
        },
        "comma_expression" => Expr::Comma(boxed(field(n, "left"), src), boxed(field(n, "right"), src)),
        "call_expression" => Expr::Call {
            f: boxed(field(n, "function"), src),
            args: field(n, "arguments")
                .map(kids)
                .unwrap_or_default()
                .into_iter()
                .map(|a| if a.kind() == "compound_statement" { Expr::GnuBlock(lower_block(a, src)) } else { lower_expr(a, src) })
                .collect(),
        },
        "field_expression" => Expr::Field {
            e: boxed(field(n, "argument"), src),
            field: field(n, "field").map_or_else(String::new, |f| text(f, src).to_owned()),
        },
        "subscript_expression" => Expr::Index { e: boxed(field(n, "argument"), src), i: boxed(field(n, "index"), src) },
        "cast_expression" => Expr::Cast {
            ty: field(n, "type").map_or(Type::Other, |t| lower_type_descriptor(t, src)),
            e: boxed(field(n, "value"), src),
        },
        "sizeof_expression" => match (field(n, "type"), field(n, "value")) {
            (Some(t), _) => Expr::SizeofType(lower_type_descriptor(t, src)),
            (None, value) => Expr::SizeofValue(boxed(value, src)),
        },
        "alignof_expression" => Expr::Alignof(field(n, "type").map_or(Type::Other, |t| lower_type_descriptor(t, src))),
        "offsetof_expression" => Expr::Offsetof(
            field(n, "type").map_or(Type::Other, |t| lower_type_descriptor(t, src)),
            field(n, "member").map_or_else(String::new, |m| text(m, src).to_owned()),
        ),
        "compound_literal_expression" => Expr::Compound {
            ty: field(n, "type").map_or(Type::Other, |t| lower_type_descriptor(t, src)),
            elems: field(n, "value").map(|v| lower_elems(v, src)).unwrap_or_default(),
        },
        "initializer_list" => Expr::InitList(lower_elems(n, src)),
        "parenthesized_expression" => match kids(n).into_iter().next() {
            Some(b) if b.kind() == "compound_statement" => Expr::GnuBlock(lower_block(b, src)),
            inner => Expr::Paren(boxed(inner, src)),
        },
        "generic_expression" => Expr::Generic(boxed(kids(n).into_iter().next(), src)),
        "extension_expression" => kids(n).into_iter().next().map_or(Expr::Unknown(String::new()), |e| lower_expr(e, src)),
        "gnu_asm_expression" => Expr::Asm,
        other => Expr::Unknown(other.to_owned()),
    }
}

/// The elements of a brace initializer: plain, a member named (`.x = v`), an element indexed
/// (`[i] = v`). A chain of designators (`.a.b = v`, `[1].x = v`) is named by its whole spelling.
fn lower_elems(list: Node<'_>, src: &str) -> Vec<Elem> {
    kids(list)
        .into_iter()
        .map(|el| {
            if el.kind() != "initializer_pair" {
                return Elem::Plain(lower_expr(el, src));
            }
            let value = field(el, "value").map_or(Expr::Unknown(String::new()), |v| lower_expr(v, src));
            let designators = fields(el, "designator");
            match designators.as_slice() {
                [d] if d.kind() == "subscript_designator" => {
                    Elem::Index(kids(*d).into_iter().next().map_or(Expr::Unknown(String::new()), |i| lower_expr(i, src)), value)
                }
                [d] if d.kind() == "field_identifier" => Elem::Field(text(*d, src).to_owned(), value),
                [d] => Elem::Field(text(*d, src).trim_start_matches('.').to_owned(), value),
                many => Elem::Field(many.iter().map(|d| squashed(*d, src)).collect::<String>().trim_start_matches('.').to_owned(), value),
            }
        })
        .collect()
}
