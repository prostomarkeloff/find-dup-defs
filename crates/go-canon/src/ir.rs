//! An owned syntax tree for Go bodies, lowered once from tree-sitter's.
//!
//! tree-sitter's tree is read-only: a node is a view into a parse that cannot be edited. The normal
//! form ([`crate::normal`]) rewrites a body — flips a branch, lifts a guard, inlines a temporary —
//! so the canonicals are written from this tree instead, which can be. It holds exactly what the
//! canonicals read and nothing else: names and literals as owned text, no positions, no comments.
//!
//! The shapes mirror the grammar's, field for field, so the dump reads one the way it read the
//! other; [`lower_func`] is the only place that knows tree-sitter's names for them.

use tree_sitter::Node;

use crate::canon::Callable;
use crate::tree::{field, fields, has_token, kids, statements, text, unquote};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Type {
    /// A type name as spelled — `int`, `Server`, `http.Handler`.
    Named(String),
    Generic(String, Vec<Type>),
    Ptr(Box<Type>),
    Slice(Box<Type>),
    Array(Box<Type>),
    Map(Box<Type>, Box<Type>),
    /// `""`, `"recv"` (`<-chan`) or `"send"` (`chan<-`).
    Chan(&'static str, Box<Type>),
    Func(Vec<ParamDecl>, Results),
    Struct(Vec<FieldDecl>),
    Iface(Vec<IfaceItem>),
    Approx(Box<Type>),
    Union(Vec<Type>),
    Other,
}

/// One parameter declaration: `a, b int` is one, naming two parameters of one type.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParamDecl {
    pub names: Vec<String>,
    pub ty: Option<Type>,
    pub variadic: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Results {
    None,
    Type(Box<Type>),
    List(Vec<ParamDecl>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FieldDecl {
    /// Empty for an embedded field.
    pub names: Vec<String>,
    pub ty: Option<Type>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum IfaceItem {
    Method { name: String, params: Vec<ParamDecl>, results: Results },
    Embed(Type),
}

/// The left of a `:=`, a `range` that declares, a type switch's alias: a name, `_`, or (in a range
/// that assigns) any expression.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Target {
    Name(String),
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Spec {
    pub names: Vec<String>,
    pub ty: Option<Type>,
    pub values: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Stmt {
    /// `a, b := x, y`.
    Define { lhs: Vec<Target>, rhs: Vec<Expr> },
    /// A local `var` (`konst` false) or `const` declaration, one or more specs.
    Decl { konst: bool, specs: Vec<Spec> },
    NestedType,
    Assign { op: String, lhs: Vec<Expr>, rhs: Vec<Expr> },
    IncDec { inc: bool, target: Expr },
    Send { ch: Expr, value: Expr },
    Expr(Expr),
    Return(Vec<Expr>),
    Go(Expr),
    Defer(Expr),
    /// `break` / `continue` / `goto`, with the label when one is named.
    Jump { kind: Jump, label: Option<String> },
    Fallthrough,
    If { init: Option<Box<Stmt>>, cond: Expr, then: Vec<Stmt>, els: Option<Box<Else>> },
    For { head: ForHead, body: Vec<Stmt> },
    Switch { init: Option<Box<Stmt>>, value: Option<Expr>, cases: Vec<Case> },
    TypeSwitch { init: Option<Box<Stmt>>, alias: Option<Vec<Target>>, value: Expr, cases: Vec<TypeCase> },
    Select { cases: Vec<CommCase> },
    /// `v, ok := <-ch` / `v = <-ch` as a `select` case's communication.
    Recv { define: bool, lhs: Option<Vec<Target>>, rhs: Expr },
    Block(Vec<Stmt>),
    Labeled { label: String, stmt: Box<Stmt> },
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ForHead {
    Loop,
    While(Expr),
    Clause { init: Option<Box<Stmt>>, cond: Option<Expr>, update: Option<Box<Stmt>> },
    /// `for k, v := range xs` (`define`), `for k = range xs`, or `for range xs` (no targets).
    Range { define: bool, lhs: Option<Vec<Target>>, iter: Expr },
}

/// A `case` (`values` set) or the `default` clause (`values` `None`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Case {
    pub values: Option<Vec<Expr>>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TypeCase {
    pub types: Option<Vec<Type>>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CommCase {
    /// `None` for `default`.
    pub comm: Option<Box<Stmt>>,
    pub default: bool,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Expr {
    /// An identifier, `_` included.
    Name(String),
    Bool(bool),
    Nil,
    Iota,
    /// `Int` / `Float` / `Imag` / `Char` / `Str`, with the literal's text (quotes off for the last two).
    Lit(&'static str, String),
    Binary { op: String, l: Box<Expr>, r: Box<Expr> },
    /// Every prefix operator, `&` and `<-` included.
    Unary { op: String, e: Box<Expr> },
    Paren(Box<Expr>),
    Call { f: Box<Expr>, args: Vec<Expr>, targs: Option<Vec<Type>> },
    Selector { operand: Box<Expr>, field: String },
    Index { e: Box<Expr>, i: Box<Expr> },
    Slice { e: Box<Expr>, lo: Option<Box<Expr>>, hi: Option<Box<Expr>>, cap: Option<Box<Expr>> },
    Assert { e: Box<Expr>, ty: Option<Type> },
    Conv { e: Box<Expr>, ty: Option<Type> },
    Inst { ty: Option<Type>, args: Vec<Type> },
    /// A composite literal; `ty` `None` for an elided inner one (`[]Point{{1, 2}}`).
    Composite { ty: Option<Type>, elems: Vec<Elem> },
    FuncLit { params: Vec<ParamDecl>, results: Results, body: Vec<Stmt> },
    Spread(Box<Expr>),
    Type(Type),
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Elem {
    /// `Name: v` — a struct field.
    Field(String, Expr),
    /// `k: v` with any other key — a map's.
    Keyed(Expr, Expr),
    Plain(Expr),
    /// A keyed element whose key could not be read.
    KeyedBlank(Expr),
}

/// A function or method, owned: what every canonical reads.
#[derive(Clone, Debug)]
pub(crate) struct Func {
    pub receiver: Option<String>,
    pub type_params: Vec<String>,
    pub params: Vec<ParamDecl>,
    pub results: Results,
    pub body: Vec<Stmt>,
}

// ───────────────────────────── lowering ─────────────────────────────

pub(crate) fn lower_func(c: &Callable<'_>, src: &str) -> Func {
    Func {
        receiver: c.receiver.map(str::to_owned),
        type_params: c
            .type_params
            .map(kids)
            .unwrap_or_default()
            .into_iter()
            .flat_map(|d| fields(d, "name"))
            .map(|n| text(n, src).to_owned())
            .collect(),
        params: lower_params(c.params, src),
        results: lower_results(c.results, src),
        body: lower_block(c.body, src),
    }
}

/// The statements a block or a clause holds.
pub(crate) fn lower_block(holder: Node<'_>, src: &str) -> Vec<Stmt> {
    statements(holder).into_iter().map(|s| lower_stmt(s, src)).collect()
}

fn lower_params(list: Option<Node<'_>>, src: &str) -> Vec<ParamDecl> {
    list.map(kids)
        .unwrap_or_default()
        .into_iter()
        .map(|decl| ParamDecl {
            names: fields(decl, "name").into_iter().map(|n| text(n, src).to_owned()).collect(),
            ty: field(decl, "type").map(|t| lower_type(t, src)),
            variadic: decl.kind() == "variadic_parameter_declaration",
        })
        .collect()
}

fn lower_results(results: Option<Node<'_>>, src: &str) -> Results {
    match results {
        None => Results::None,
        Some(list) if list.kind() == "parameter_list" => Results::List(lower_params(Some(list), src)),
        Some(ty) => Results::Type(Box::new(lower_type(ty, src))),
    }
}

fn boxed_type(n: Option<Node<'_>>, src: &str) -> Box<Type> {
    Box::new(n.map_or(Type::Other, |t| lower_type(t, src)))
}

pub(crate) fn lower_type(n: Node<'_>, src: &str) -> Type {
    match n.kind() {
        "type_identifier" | "qualified_type" | "identifier" => Type::Named(text(n, src).to_owned()),
        "generic_type" => Type::Generic(
            field(n, "type").map_or_else(String::new, |t| text(t, src).to_owned()),
            field(n, "type_arguments").map(kids).unwrap_or_default().into_iter().map(|a| lower_type(a, src)).collect(),
        ),
        "pointer_type" => Type::Ptr(boxed_type(kids(n).first().copied(), src)),
        "slice_type" => Type::Slice(boxed_type(field(n, "element"), src)),
        "array_type" | "implicit_length_array_type" => Type::Array(boxed_type(field(n, "element"), src)),
        "map_type" => Type::Map(boxed_type(field(n, "key"), src), boxed_type(field(n, "value"), src)),
        "channel_type" => {
            let spelled = text(n, src);
            let dir = if spelled.starts_with("<-") {
                "recv"
            } else if spelled.trim_start_matches("chan").trim_start().starts_with("<-") {
                "send"
            } else {
                ""
            };
            Type::Chan(dir, boxed_type(field(n, "value"), src))
        }
        "function_type" => Type::Func(lower_params(field(n, "parameters"), src), lower_results(field(n, "result"), src)),
        "struct_type" => Type::Struct(lower_fields(n, src)),
        "interface_type" => Type::Iface(lower_iface(n, src)),
        "parenthesized_type" => kids(n).first().map_or(Type::Other, |t| lower_type(*t, src)),
        "negated_type" => Type::Approx(boxed_type(kids(n).first().copied(), src)),
        "type_elem" | "type_constraint" => {
            let mut parts: Vec<Type> = kids(n).into_iter().map(|t| lower_type(t, src)).collect();
            if parts.len() == 1 {
                parts.pop().unwrap_or(Type::Other)
            } else {
                Type::Union(parts)
            }
        }
        _ => Type::Other,
    }
}

pub(crate) fn lower_fields(struct_type: Node<'_>, src: &str) -> Vec<FieldDecl> {
    kids(struct_type)
        .into_iter()
        .flat_map(kids)
        .filter(|k| k.kind() == "field_declaration")
        .map(|decl| FieldDecl {
            names: fields(decl, "name").into_iter().map(|n| text(n, src).to_owned()).collect(),
            ty: field(decl, "type").map(|t| lower_type(t, src)),
        })
        .collect()
}

pub(crate) fn lower_iface(iface: Node<'_>, src: &str) -> Vec<IfaceItem> {
    kids(iface)
        .into_iter()
        .map(|item| {
            if item.kind() == "method_elem" {
                IfaceItem::Method {
                    name: field(item, "name").map_or_else(String::new, |n| text(n, src).to_owned()),
                    params: lower_params(field(item, "parameters"), src),
                    results: lower_results(field(item, "result"), src),
                }
            } else {
                IfaceItem::Embed(lower_type(item, src))
            }
        })
        .collect()
}

fn lower_targets(list: Node<'_>, src: &str) -> Vec<Target> {
    kids(list)
        .into_iter()
        .map(|k| if k.kind() == "identifier" { Target::Name(text(k, src).to_owned()) } else { Target::Expr(lower_expr(k, src)) })
        .collect()
}

fn lower_exprs(list: Node<'_>, src: &str) -> Vec<Expr> {
    kids(list).into_iter().map(|k| lower_expr(k, src)).collect()
}

fn opt_stmt(n: Option<Node<'_>>, src: &str) -> Option<Box<Stmt>> {
    n.map(|s| Box::new(lower_stmt(s, src)))
}

fn specs(decl: Node<'_>, spec_kind: &str, src: &str) -> Vec<Spec> {
    let mut nodes = Vec::new();
    for k in kids(decl) {
        if k.kind() == spec_kind {
            nodes.push(k);
        } else {
            nodes.extend(kids(k).into_iter().filter(|s| s.kind() == spec_kind));
        }
    }
    nodes
        .into_iter()
        .map(|spec| Spec {
            names: fields(spec, "name").into_iter().map(|n| text(n, src).to_owned()).collect(),
            ty: field(spec, "type").map(|t| lower_type(t, src)),
            values: field(spec, "value").map(|v| lower_exprs(v, src)).unwrap_or_default(),
        })
        .collect()
}

#[allow(clippy::too_many_lines)] // one arm per statement kind reads better straight-through
pub(crate) fn lower_stmt(n: Node<'_>, src: &str) -> Stmt {
    match n.kind() {
        "short_var_declaration" => Stmt::Define {
            lhs: field(n, "left").map(|l| lower_targets(l, src)).unwrap_or_default(),
            rhs: field(n, "right").map(|r| lower_exprs(r, src)).unwrap_or_default(),
        },
        "var_declaration" => Stmt::Decl { konst: false, specs: specs(n, "var_spec", src) },
        "const_declaration" => Stmt::Decl { konst: true, specs: specs(n, "const_spec", src) },
        "type_declaration" => Stmt::NestedType,
        "assignment_statement" => Stmt::Assign {
            op: field(n, "operator").map_or("=", |o| o.kind()).to_owned(),
            lhs: field(n, "left").map(|l| lower_exprs(l, src)).unwrap_or_default(),
            rhs: field(n, "right").map(|r| lower_exprs(r, src)).unwrap_or_default(),
        },
        "inc_statement" | "dec_statement" => Stmt::IncDec {
            inc: n.kind() == "inc_statement",
            target: kids(n).first().map_or(Expr::Unknown(String::new()), |e| lower_expr(*e, src)),
        },
        "send_statement" => Stmt::Send {
            ch: field(n, "channel").map_or(Expr::Unknown(String::new()), |c| lower_expr(c, src)),
            value: field(n, "value").map_or(Expr::Unknown(String::new()), |v| lower_expr(v, src)),
        },
        "expression_statement" => Stmt::Expr(kids(n).first().map_or(Expr::Unknown(String::new()), |e| lower_expr(*e, src))),
        "return_statement" => Stmt::Return(kids(n).first().map(|l| lower_exprs(*l, src)).unwrap_or_default()),
        "go_statement" => Stmt::Go(kids(n).first().map_or(Expr::Unknown(String::new()), |e| lower_expr(*e, src))),
        "defer_statement" => Stmt::Defer(kids(n).first().map_or(Expr::Unknown(String::new()), |e| lower_expr(*e, src))),
        "break_statement" | "continue_statement" | "goto_statement" => Stmt::Jump {
            kind: match n.kind() {
                "break_statement" => Jump::Break,
                "continue_statement" => Jump::Continue,
                _ => Jump::Goto,
            },
            label: kids(n).first().map(|l| text(*l, src).to_owned()),
        },
        "fallthrough_statement" => Stmt::Fallthrough,
        "if_statement" => Stmt::If {
            init: opt_stmt(field(n, "initializer"), src),
            cond: field(n, "condition").map_or(Expr::Unknown(String::new()), |c| lower_expr(c, src)),
            then: field(n, "consequence").map(|b| lower_block(b, src)).unwrap_or_default(),
            els: field(n, "alternative").map(|alt| {
                Box::new(if alt.kind() == "if_statement" { Else::If(lower_stmt(alt, src)) } else { Else::Block(lower_block(alt, src)) })
            }),
        },
        "for_statement" => {
            let body_id = field(n, "body").map(|b| b.id());
            let head = kids(n).into_iter().find(|k| Some(k.id()) != body_id);
            let head = match head {
                None => ForHead::Loop,
                Some(h) if h.kind() == "range_clause" => ForHead::Range {
                    define: has_token(h, ":="),
                    lhs: field(h, "left").map(|l| {
                        if has_token(h, ":=") {
                            lower_targets(l, src)
                        } else {
                            lower_exprs(l, src).into_iter().map(Target::Expr).collect()
                        }
                    }),
                    iter: field(h, "right").map_or(Expr::Unknown(String::new()), |r| lower_expr(r, src)),
                },
                Some(h) if h.kind() == "for_clause" => ForHead::Clause {
                    init: opt_stmt(field(h, "initializer"), src),
                    cond: field(h, "condition").map(|c| lower_expr(c, src)),
                    update: opt_stmt(field(h, "update"), src),
                },
                Some(cond) => ForHead::While(lower_expr(cond, src)),
            };
            Stmt::For { head, body: field(n, "body").map(|b| lower_block(b, src)).unwrap_or_default() }
        }
        "expression_switch_statement" => Stmt::Switch {
            init: opt_stmt(field(n, "initializer"), src),
            value: field(n, "value").map(|v| lower_expr(v, src)),
            cases: kids(n)
                .into_iter()
                .filter_map(|c| match c.kind() {
                    "expression_case" => Some(Case {
                        values: Some(field(c, "value").map(|v| lower_exprs(v, src)).unwrap_or_default()),
                        body: lower_block(c, src),
                    }),
                    "default_case" => Some(Case { values: None, body: lower_block(c, src) }),
                    _ => None,
                })
                .collect(),
        },
        "type_switch_statement" => Stmt::TypeSwitch {
            init: opt_stmt(field(n, "initializer"), src),
            alias: field(n, "alias").map(|a| lower_targets(a, src)),
            value: field(n, "value").map_or(Expr::Unknown(String::new()), |v| lower_expr(v, src)),
            cases: kids(n)
                .into_iter()
                .filter_map(|c| match c.kind() {
                    "type_case" => Some(TypeCase {
                        types: Some(fields(c, "type").into_iter().map(|t| lower_type(t, src)).collect()),
                        body: lower_block(c, src),
                    }),
                    "default_case" => Some(TypeCase { types: None, body: lower_block(c, src) }),
                    _ => None,
                })
                .collect(),
        },
        "select_statement" => Stmt::Select {
            cases: kids(n)
                .into_iter()
                .filter_map(|c| match c.kind() {
                    "communication_case" => Some(CommCase {
                        comm: opt_stmt(field(c, "communication"), src),
                        default: false,
                        body: lower_block(c, src),
                    }),
                    "default_case" => Some(CommCase { comm: None, default: true, body: lower_block(c, src) }),
                    _ => None,
                })
                .collect(),
        },
        "receive_statement" => {
            let define = has_token(n, ":=");
            Stmt::Recv {
                define,
                lhs: field(n, "left").map(|l| {
                    if define {
                        lower_targets(l, src)
                    } else {
                        lower_exprs(l, src).into_iter().map(Target::Expr).collect()
                    }
                }),
                rhs: field(n, "right").map_or(Expr::Unknown(String::new()), |r| lower_expr(r, src)),
            }
        }
        "block" => Stmt::Block(lower_block(n, src)),
        "labeled_statement" => Stmt::Labeled {
            label: field(n, "label").map_or_else(String::new, |l| text(l, src).to_owned()),
            stmt: Box::new(
                kids(n).into_iter().find(|k| k.kind() != "label_name").map_or(Stmt::Empty, |s| lower_stmt(s, src)),
            ),
        },
        "empty_statement" => Stmt::Empty,
        other => Stmt::Unknown(other.to_owned()),
    }
}

fn boxed(n: Option<Node<'_>>, src: &str) -> Box<Expr> {
    Box::new(n.map_or(Expr::Unknown(String::new()), |e| lower_expr(e, src)))
}

fn opt_boxed(n: Option<Node<'_>>, src: &str) -> Option<Box<Expr>> {
    n.map(|e| Box::new(lower_expr(e, src)))
}

pub(crate) fn lower_expr(n: Node<'_>, src: &str) -> Expr {
    match n.kind() {
        "identifier" => Expr::Name(text(n, src).to_owned()),
        "true" => Expr::Bool(true),
        "false" => Expr::Bool(false),
        "nil" => Expr::Nil,
        "iota" => Expr::Iota,
        "int_literal" => Expr::Lit("Int", text(n, src).to_owned()),
        "float_literal" => Expr::Lit("Float", text(n, src).to_owned()),
        "imaginary_literal" => Expr::Lit("Imag", text(n, src).to_owned()),
        "rune_literal" => Expr::Lit("Char", unquote(text(n, src)).to_owned()),
        "interpreted_string_literal" | "raw_string_literal" => Expr::Lit("Str", unquote(text(n, src)).to_owned()),
        "binary_expression" => Expr::Binary {
            op: field(n, "operator").map_or("?op", |o| o.kind()).to_owned(),
            l: boxed(field(n, "left"), src),
            r: boxed(field(n, "right"), src),
        },
        "unary_expression" => Expr::Unary {
            op: field(n, "operator").map_or("?un", |o| o.kind()).to_owned(),
            e: boxed(field(n, "operand"), src),
        },
        "parenthesized_expression" => Expr::Paren(boxed(kids(n).first().copied(), src)),
        "call_expression" => Expr::Call {
            f: boxed(field(n, "function"), src),
            args: field(n, "arguments").map(|a| lower_exprs(a, src)).unwrap_or_default(),
            targs: field(n, "type_arguments").map(|t| kids(t).into_iter().map(|a| lower_type(a, src)).collect()),
        },
        "selector_expression" => Expr::Selector {
            operand: boxed(field(n, "operand"), src),
            field: field(n, "field").map_or_else(String::new, |f| text(f, src).to_owned()),
        },
        "index_expression" => Expr::Index { e: boxed(field(n, "operand"), src), i: boxed(field(n, "index"), src) },
        "slice_expression" => Expr::Slice {
            e: boxed(field(n, "operand"), src),
            lo: opt_boxed(field(n, "start"), src),
            hi: opt_boxed(field(n, "end"), src),
            cap: opt_boxed(field(n, "capacity"), src),
        },
        "type_assertion_expression" => {
            Expr::Assert { e: boxed(field(n, "operand"), src), ty: field(n, "type").map(|t| lower_type(t, src)) }
        }
        "type_conversion_expression" => {
            Expr::Conv { e: boxed(field(n, "operand"), src), ty: field(n, "type").map(|t| lower_type(t, src)) }
        }
        "type_instantiation_expression" => Expr::Inst {
            ty: field(n, "type").map(|t| lower_type(t, src)),
            args: kids(n).into_iter().skip(1).map(|a| lower_type(a, src)).collect(),
        },
        "composite_literal" => Expr::Composite {
            ty: field(n, "type").map(|t| lower_type(t, src)),
            elems: field(n, "body").map(|b| lower_elems(b, src)).unwrap_or_default(),
        },
        "literal_value" => Expr::Composite { ty: None, elems: lower_elems(n, src) },
        "func_literal" => Expr::FuncLit {
            params: lower_params(field(n, "parameters"), src),
            results: lower_results(field(n, "result"), src),
            body: field(n, "body").map(|b| lower_block(b, src)).unwrap_or_default(),
        },
        "variadic_argument" => Expr::Spread(boxed(kids(n).first().copied(), src)),
        kind if is_type_kind(kind) => Expr::Type(lower_type(n, src)),
        other => Expr::Unknown(other.to_owned()),
    }
}

fn lower_elems(body: Node<'_>, src: &str) -> Vec<Elem> {
    kids(body)
        .into_iter()
        .map(|el| {
            let value_of = |le: Option<Node<'_>>| le.and_then(|v| kids(v).into_iter().next()).map_or(Expr::Unknown(String::new()), |v| lower_expr(v, src));
            if el.kind() == "keyed_element" {
                let key = field(el, "key");
                let value = value_of(field(el, "value"));
                match key.and_then(|k| kids(k).into_iter().next()) {
                    Some(name) if name.kind() == "identifier" => Elem::Field(text(name, src).to_owned(), value),
                    Some(k) => Elem::Keyed(lower_expr(k, src), value),
                    None => Elem::KeyedBlank(value),
                }
            } else {
                Elem::Plain(value_of(Some(el)))
            }
        })
        .collect()
}

/// Whether a grammar kind is a type — what an argument list may hold in place of an expression
/// (`make([]int, n)`).
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
