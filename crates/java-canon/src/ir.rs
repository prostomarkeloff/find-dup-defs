//! An owned syntax tree for Java bodies, lowered once from tree-sitter's.
//!
//! tree-sitter's tree is read-only: a node is a view into a parse that cannot be edited. The normal
//! form ([`crate::normal`]) rewrites a body — flips a branch, lifts a guard, inlines a temporary —
//! so the canonicals are written from this tree instead, which can be. It holds exactly what the
//! canonicals read and nothing else: names and literals as owned text, no positions, no comments.
//!
//! The shapes mirror the grammar's, field for field, so the dump reads one the way it read the
//! other; [`lower_func`] is the only place that knows tree-sitter's names for them.
//!
//! Java is read without a type checker, so a bare call (`helper()`) is a call of the name `helper`,
//! never resolved to a method on `this`: the closed world cannot say whether it dispatches on an
//! instance. Explicit receivers (`this.x`, `x.m()`) are read as written, `this` as `self`.

use tree_sitter::Node;

use crate::tree::{field, fields, kids, text, unquote};

/// A Java type, owned. Names are as written in the source (`String`, `java.util.List`), resolved
/// through the file's imports by the dump.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Type {
    /// A name: `int`, `String`, `java.util.List`, a type parameter `T`.
    Named(String),
    /// `List<String>` — a name applied to type arguments.
    Generic(String, Vec<Type>),
    /// One array dimension: `int[]` is `Array(Named("int"))`.
    Array(Box<Type>),
    /// `int...` — the variadic spelling, only in a parameter.
    Variadic(Box<Type>),
    /// A wildcard type argument: `?`, `? extends X`, `? super X`.
    Wild(Option<(String, Box<Type>)>),
    /// A multi-catch union: `IOException | SQLException`.
    Union(Vec<Type>),
    /// An intersection in a cast: `(A & B) x`.
    Inter(Vec<Type>),
    /// A type the walk could not shape.
    Other,
}

/// One parameter: Java names every parameter, and `...` marks the last as variadic.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Param {
    pub name: String,
    pub ty: Type,
    pub variadic: bool,
}

/// A function or method, owned: what every canonical reads. Constructors carry `ret` `None` (the
/// language gives them none, and the name slot distinguishes them).
#[derive(Clone, Debug)]
pub(crate) struct Func {
    /// The definition's name as the canon writes it: `Class.method`, or the class's for a constructor.
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Option<Type>,
    /// The names the definition's type parameters bind — its own, not vocabulary: the Type-3
    /// slotting turns them into `_T{n}` but does not report them as type references.
    pub type_params: Vec<String>,
    pub body: Vec<Stmt>,
}

/// The statements a block holds.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Stmt {
    /// One local declaration: `int x = e`, `var x = e` (`ty` `None`).
    Let { ty: Option<Type>, name: String, init: Option<Expr> },
    /// Several declarations of one statement (`int a = 1, b = 2;`), flattened later or kept as a list.
    Decl(Vec<Stmt>),
    /// `=` and the compound forms; `x += e` reads `Bin` from the dump.
    Assign { op: String, lhs: Expr, rhs: Expr },
    /// `x++`, `++x`, `x--`, `--x`.
    Update { inc: bool, prefix: bool, e: Expr },
    /// A bare expression statement.
    Expr(Expr),
    Return(Option<Expr>),
    Throw(Expr),
    Assert(Expr),
    If { cond: Expr, then: Vec<Stmt>, els: Option<Box<Else>> },
    While { cond: Expr, body: Vec<Stmt> },
    DoWhile { body: Vec<Stmt>, cond: Expr },
    /// `for (init; cond; update)` — `init` a declaration or an expression list, either side optional.
    For { init: Option<Box<Stmt>>, cond: Option<Expr>, update: Vec<Expr>, body: Vec<Stmt> },
    /// `for (T x : xs)` — the loop variable binds; its type is not part of the body.
    ForEach { name: String, iter: Expr, body: Vec<Stmt> },
    /// The classic statement `switch`: colon groups, and arrow rules in statement position.
    Switch { value: Expr, cases: Vec<Case> },
    /// `try` (resources empty) and `try`-with-resources.
    Try { resources: Vec<Resource>, body: Vec<Stmt>, catches: Vec<Catch>, finally: Option<Vec<Stmt>> },
    Synchronized { lock: Expr, body: Vec<Stmt> },
    Labeled { label: String, stmt: Box<Stmt> },
    Break(Option<String>),
    Continue(Option<String>),
    /// A bare block in statement position (`{ … }` — a scope, not a switch body).
    Block(Vec<Stmt>),
    /// A local or anonymous class declaration — part of the enclosing body, not a definition of its own.
    NestedItem,
    /// A statement the walk could not shape, named by the grammar's node kind.
    Unknown(String),
}

/// The `else` of an `if`: a block, or the next `if` of the chain.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Else {
    Block(Vec<Stmt>),
    If(Box<Stmt>),
}

/// One `switch` clause: `case 1: case 2: S` is one `Case` holding both values; `default` holds none.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Case {
    pub values: Vec<Expr>,
    pub body: Vec<Stmt>,
}

/// One `catch` clause. `name` is `None` for a catch that binds nothing (`catch (E _)`-less form is
/// not Java, but a multi-catch with no variable parses the same way).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Catch {
    pub types: Vec<Type>,
    pub name: Option<String>,
    pub body: Vec<Stmt>,
}

/// One resource of a try-with-resources.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Resource {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Expr {
    /// A name: a local, a type in expression position, `this` / `super`, a class name.
    Name(String),
    /// A literal in its own spelling: `Num('5')`, `Str('a')`, `Char('x')`.
    Lit(String, String),
    Bool(bool),
    Null,
    Binary { op: String, l: Box<Expr>, r: Box<Expr> },
    Unary { op: String, e: Box<Expr> },
    /// `x++` and the like, in expression position.
    Update { inc: bool, prefix: bool, e: Box<Expr> },
    /// An assignment in expression position (`x = e`, `x += e`).
    #[allow(clippy::enum_variant_names)]
    AssignExpr { op: String, l: Box<Expr>, r: Box<Expr> },
    /// `c ? t : f`.
    Cond { c: Box<Expr>, t: Box<Expr>, f: Box<Expr> },
    /// Parentheses — dropped by the dump, kept by the walkers that need the grouping.
    Paren(Box<Expr>),
    /// A call of a bare name — a static import, a local lambda, or a method the type checker would
    /// resolve; without one, the name is called as written.
    Call { f: Box<Expr>, args: Vec<Expr> },
    /// A method call with an explicit receiver: `x.m(a)`, `this.m(a)`.
    Method { recv: Box<Expr>, name: String, args: Vec<Expr> },
    /// `Foo.m(a)` — the scope is a class name, resolved through the imports.
    StaticCall { scope: String, name: String, args: Vec<Expr> },
    /// `new Foo(a)`. The class name resolves through the imports; array creations are `New` too.
    New { class: String, args: Vec<Expr> },
    /// `super(a)` — constructor delegation.
    Super { args: Vec<Expr> },
    Field { base: Box<Expr>, name: String },
    /// `Foo.BAR` — the scope is a class name.
    StaticField { scope: String, name: String },
    Index { base: Box<Expr>, i: Box<Expr> },
    Cast { ty: Type, e: Box<Expr> },
    /// `x instanceof T` — and the pattern form `x instanceof T t`, whose binding `bind` names.
    InstanceOf { e: Box<Expr>, ty: Type, bind: Option<String> },
    /// `{1, 2}` — an array initializer.
    Array { elems: Vec<Expr> },
    /// A lambda: `(a, b) -> e` or `(a, b) -> { … }`.
    Lambda { params: Vec<Param>, body: LambdaBody },
    /// `Foo::bar` (`target` the class name), `x::bar` (any expression), `Foo::new`.
    MethodRef { target: Box<Expr>, name: String },
    /// `Foo.class`.
    ClassConst { scope: String },
    /// The switch **expression** — `switch (v) { case a -> e; … }` anywhere an expression stands. A
    /// statement-position switch lowers to [`Stmt::Switch`]; this is the value form.
    Switch { value: Box<Expr>, arms: Vec<Arm> },
    /// A block in expression position — an arrow `switch` arm's `{ … }`.
    StmtBlock(Vec<Stmt>),
    /// A `throw` in expression position (an arrow arm's body).
    Throw(Box<Expr>),
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LambdaBody {
    Block(Vec<Stmt>),
    Expr(Box<Expr>),
}

/// One arm of a switch expression: `conds` empty for `default`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Arm {
    pub conds: Vec<Expr>,
    pub value: Box<Expr>,
}

// ───────────────────────────── lowering ─────────────────────────────

/// Lower one method or constructor declaration (a `method_declaration`, `constructor_declaration` or
/// `compact_constructor_declaration` node). `None` for a declaration without a body — an abstract or
/// native method — which is not a definition the passes compare.
pub(crate) fn lower_func(decl: Node<'_>, name: &str, src: &str) -> Option<Func> {
    let body = field(decl, "body")?;
    let params = lower_params(field(decl, "parameters"), src);
    let ret = field(decl, "type").map(|t| lower_type(t, src));
    let type_params = type_param_names(field(decl, "type_parameters"), src);
    Some(Func { name: name.to_owned(), params, ret, type_params, body: lower_block(body, src) })
}

/// The formal parameters of a method or constructor, or a lambda's.
pub(crate) fn lower_params(params: Option<Node<'_>>, src: &str) -> Vec<Param> {
    params.map(kids).unwrap_or_default().into_iter().map(|p| lower_param(p, src)).collect()
}

/// One `formal_parameter` / `spread_parameter`: a name, a type, maybe `...`.
fn lower_param(n: Node<'_>, src: &str) -> Param {
    let name = field(n, "name").map_or_else(|| "", |name| text(name, src)).to_owned();
    let ty = field(n, "type").map_or(Type::Other, |t| lower_type(t, src));
    Param { name, ty, variadic: n.kind() == "spread_parameter" }
}

/// The statements a block or a clause holds — or, for a statement-position single statement (an
/// `if`/`while` body written without braces), that statement as the whole body. The grammar stores
/// both the same way; the walker must not descend into the statement's own children.
pub(crate) fn lower_block(holder: Node<'_>, src: &str) -> Vec<Stmt> {
    if holder.kind() == "block" || holder.kind() == "constructor_body" {
        crate::tree::statements(holder).into_iter().map(|s| lower_stmt(s, src)).collect()
    } else {
        vec![lower_stmt(holder, src)]
    }
}

/// Whether a grammar kind is a declaration a method body may hold in place of a statement — a local
/// class, or a static initializer of an anonymous class.
fn is_nested_decl(kind: &str) -> bool {
    matches!(
        kind,
        "class_declaration"
            | "interface_declaration"
            | "enum_declaration"
            | "record_declaration"
            | "annotation_type_declaration"
            | "static_initializer"
    )
}

#[allow(clippy::too_many_lines)] // one arm per statement kind reads better straight-through
pub(crate) fn lower_stmt(n: Node<'_>, src: &str) -> Stmt {
    match n.kind() {
        "block" => Stmt::Block(lower_block(n, src)),
        "local_variable_declaration" => {
            let mut items = Vec::new();
            let ty = field(n, "type").map(|t| lower_type(t, src)).filter(|t| *t != Type::Named("var".to_owned()));
            for decl in fields(n, "declarator") {
                let name = field(decl, "name").map_or_else(|| "", |name| text(name, src)).to_owned();
                let init = field(decl, "value").map(|v| lower_expr(v, src));
                items.push(Stmt::Let { ty: ty.clone(), name, init });
            }
            if items.len() == 1 { items.remove(0) } else { Stmt::Decl(items) }
        }
        "expression_statement" => Stmt::Expr(lower_expr(kids(n).into_iter().next().unwrap_or(n), src)),
        "return_statement" => Stmt::Return(kids(n).into_iter().next().map(|v| lower_expr(v, src))),
        "throw_statement" => Stmt::Throw(lower_expr(kids(n).into_iter().next().unwrap_or(n), src)),
        "assert_statement" => Stmt::Assert(lower_expr(kids(n).into_iter().next().unwrap_or(n), src)),
        "if_statement" => {
            let cond = field(n, "condition").map_or(Expr::Unknown("if".to_owned()), |c| lower_expr(c, src));
            let then = field(n, "consequence").map_or_else(Vec::new, |t| lower_block(t, src));
            let els = field(n, "alternative").map(|a| {
                if a.kind() == "if_statement" { Box::new(Else::If(Box::new(lower_stmt(a, src)))) } else { Box::new(Else::Block(lower_block(a, src))) }
            });
            Stmt::If { cond, then, els }
        }
        "while_statement" => {
            let cond = field(n, "condition").map_or(Expr::Unknown("while".to_owned()), |c| lower_expr(c, src));
            let body = field(n, "body").map_or_else(Vec::new, |b| lower_block(b, src));
            Stmt::While { cond, body }
        }
        "do_statement" => {
            let cond = field(n, "condition").map_or(Expr::Unknown("do".to_owned()), |c| lower_expr(c, src));
            let body = field(n, "body").map_or_else(Vec::new, |b| lower_block(b, src));
            Stmt::DoWhile { body, cond }
        }
        "for_statement" => {
            let init = field(n, "init").map(|i| {
                Box::new(if i.kind() == "local_variable_declaration" {
                    lower_stmt(i, src)
                } else {
                    Stmt::Expr(expr_list(i, src))
                })
            });
            let cond = field(n, "condition").map(|c| lower_expr(c, src));
            let update = field(n, "update").map_or_else(Vec::new, |u| {
                if u.kind() == "expression" || u.named_child_count() == 0 { vec![lower_expr(u, src)] } else { kids(u).into_iter().map(|e| lower_expr(e, src)).collect() }
            });
            let body = field(n, "body").map_or_else(Vec::new, |b| lower_block(b, src));
            Stmt::For { init, cond, update, body }
        }
        "enhanced_for_statement" => {
            let name = field(n, "name").map_or_else(|| "", |name| text(name, src)).to_owned();
            let iter = field(n, "value").map_or(Expr::Unknown("foreach".to_owned()), |v| lower_expr(v, src));
            let body = field(n, "body").map_or_else(Vec::new, |b| lower_block(b, src));
            Stmt::ForEach { name, iter, body }
        }
        // A statement-position `switch` (the grammar reads both forms as one expression node): colon
        // groups become `Case`s; an arrow rule whose body is an expression or a throw reads as one
        // statement.
        "switch_expression" => Stmt::Switch { value: switch_value(n, src), cases: switch_cases(field(n, "body"), src) },
        "try_statement" | "try_with_resources_statement" => Stmt::Try {
            resources: lower_resources(field(n, "resources"), src),
            body: field(n, "body").map_or_else(Vec::new, |b| lower_block(b, src)),
            catches: kids(n).into_iter().filter(|k| k.kind() == "catch_clause").map(|c| lower_catch(c, src)).collect(),
            finally: kids(n).into_iter().find(|k| k.kind() == "finally_clause").and_then(|f| kids(f).into_iter().next()).map(|b| lower_block(b, src)),
        },
        "synchronized_statement" => {
            let mut children = kids(n).into_iter();
            let lock = children.next().map_or(Expr::Unknown("sync".to_owned()), |l| lower_expr(l, src));
            let body = children.next().map_or_else(Vec::new, |b| lower_block(b, src));
            Stmt::Synchronized { lock, body }
        }
        "explicit_constructor_invocation" => {
            // `super(args)` / `this(args)` — constructor delegation. The grammar names the callee a
            // direct child, not a field.
            let mut children = kids(n).into_iter();
            let callee = children.next().map_or_else(|| "", |c| text(c, src));
            let args = children.next().map_or_else(Vec::new, |a| lower_args(Some(a), src));
            let call = if callee == "super" {
                Expr::Super { args }
            } else {
                Expr::Call { f: Box::new(Expr::Name("this".to_owned())), args }
            };
            Stmt::Expr(call)
        }
        "labeled_statement" => {
            let mut children = kids(n).into_iter();
            let label = children.next().map_or_else(|| "", |l| text(l, src)).to_owned();
            let stmt = children.next().map_or(Stmt::Unknown("labeled".to_owned()), |s| lower_stmt(s, src));
            Stmt::Labeled { label, stmt: Box::new(stmt) }
        }
        "break_statement" => Stmt::Break(kids(n).into_iter().next().map(|l| text(l, src).to_owned())),
        "continue_statement" => Stmt::Continue(kids(n).into_iter().next().map(|l| text(l, src).to_owned())),
        kind if is_nested_decl(kind) => Stmt::NestedItem,
        _ => Stmt::Unknown(n.kind().to_owned()),
    }
}

/// A comma-separated expression list (a `for` initializer that is not a declaration): two or more
/// become `Seq`, one stands alone.
fn expr_list(n: Node<'_>, src: &str) -> Expr {
    let items: Vec<Expr> = kids(n).into_iter().map(|e| lower_expr(e, src)).collect();
    let mut it = items.into_iter();
    let Some(first) = it.next() else { return Expr::Unknown(n.kind().to_owned()) };
    it.fold(first, |acc, e| Expr::Binary { op: ",".to_owned(), l: Box::new(acc), r: Box::new(e) })
}

fn switch_value(n: Node<'_>, src: &str) -> Expr {
    field(n, "condition").map_or(Expr::Unknown("switch".to_owned()), |c| lower_expr(c, src))
}

/// The clauses of a statement-position switch: each colon group one `Case` (labels shared), each
/// arrow rule one single-statement `Case`.
fn switch_cases(body: Option<Node<'_>>, src: &str) -> Vec<Case> {
    let mut out: Vec<Case> = Vec::new();
    // Labels of a clause that carries no body: they fall through into the next clause.
    let mut pending: Vec<Expr> = Vec::new();
    for group in body.map(kids).unwrap_or_default() {
        let mut case = match group.kind() {
            "switch_block_statement_group" => {
                let mut items = kids(group).into_iter().peekable();
                let mut values = Vec::new();
                while items.peek().is_some_and(|k| k.kind() == "switch_label") {
                    let label = items.next().expect("peeked");
                    values.extend(label_value(label, src));
                }
                Case { values, body: items.map(|s| lower_stmt(s, src)).collect() }
            }
            "switch_rule" => {
                let mut items = kids(group).into_iter().peekable();
                let mut values = Vec::new();
                while items.peek().is_some_and(|k| k.kind() == "switch_label") {
                    let label = items.next().expect("peeked");
                    values.extend(label_value(label, src));
                }
                let body = match items.next() {
                    Some(b) if b.kind() == "block" => lower_block(b, src),
                    // The grammar wraps an arrow rule's statement in an `expression_statement` or
                    // `throw_statement` node — the body is the statement inside it.
                    Some(b) if b.kind() == "expression_statement" => {
                        vec![Stmt::Expr(lower_expr(kids(b).into_iter().next().unwrap_or(b), src))]
                    }
                    Some(b) if b.kind() == "throw_statement" => {
                        vec![Stmt::Throw(lower_expr(kids(b).into_iter().next().unwrap_or(b), src))]
                    }
                    Some(b) => vec![Stmt::Expr(lower_expr(b, src))],
                    None => Vec::new(),
                };
                Case { values, body }
            }
            _ => continue,
        };
        if case.body.is_empty() {
            pending.extend(case.values);
            continue;
        }
        let mut values = pending.split_off(0);
        values.extend(case.values);
        case.values = values;
        out.push(case);
    }
    out
}

/// The value a `switch_label` names: `default` none, `case e` the expression.
fn label_value(label: Node<'_>, src: &str) -> Vec<Expr> {
    kids(label).into_iter().next().map_or_else(Vec::new, |v| vec![lower_expr(v, src)])
}

fn lower_catch(c: Node<'_>, src: &str) -> Catch {
    let mut items = kids(c).into_iter();
    let head = items.next();
    let (types, name) = match head.map(|h| h.kind()) {
        Some("catch_formal_parameter") => {
            let h = head.unwrap_or(c);
            // The formal parameter's type is a `catch_type` child — the grammar's multi-catch union —
            // not a `type` field.
            let ty = kids(h).into_iter().find(|k| k.kind() == "catch_type").map_or_else(Vec::new, |t| lower_type_union(t, src));
            let name = field(h, "name").map(|name| text(name, src).to_owned());
            (ty, name)
        }
        Some("catch_type") => {
            let ty = head.map_or_else(Vec::new, |t| lower_type_union(t, src));
            (ty, None)
        }
        _ => (Vec::new(), None),
    };
    let body = items.next().map_or_else(Vec::new, |b| lower_block(b, src));
    Catch { types, name, body }
}

/// A catch's type: one type, or the union of a multi-catch (`IOException | SQLException`).
fn lower_type_union(t: Node<'_>, src: &str) -> Vec<Type> {
    if matches!(t.kind(), "union_type" | "catch_type") { kids(t).into_iter().map(|u| lower_type(u, src)).collect() } else { vec![lower_type(t, src)] }
}

fn lower_resources(spec: Option<Node<'_>>, src: &str) -> Vec<Resource> {
    spec.map(kids).unwrap_or_default().into_iter().map(|r| {
        let name = field(r, "name").map(|name| text(name, src).to_owned());
        let value = field(r, "value").map_or(Expr::Unknown("resource".to_owned()), |v| lower_expr(v, src));
        Resource { name, value }
    }).collect()
}

fn boxed(n: Option<Node<'_>>, src: &str) -> Box<Expr> {
    Box::new(n.map_or(Expr::Unknown(String::new()), |e| lower_expr(e, src)))
}

#[allow(clippy::too_many_lines)] // one arm per expression kind reads better straight-through
pub(crate) fn lower_expr(n: Node<'_>, src: &str) -> Expr {
    match n.kind() {
        "identifier" => Expr::Name(text(n, src).to_owned()),
        "this" => Expr::Name("this".to_owned()),
        "super" => Expr::Name("super".to_owned()),
        "decimal_integer_literal" | "hex_integer_literal" | "octal_integer_literal" | "binary_integer_literal"
        | "decimal_floating_point_literal" | "hex_floating_point_literal" => {
            Expr::Lit("Num".to_owned(), text(n, src).replace('_', ""))
        }
        "string_literal" => {
            let raw = text(n, src);
            let inner = raw.strip_prefix("\"\"\"").and_then(|r| r.strip_suffix("\"\"\"")).unwrap_or_else(|| unquote(raw));
            Expr::Lit("Str".to_owned(), inner.to_owned())
        }
        "character_literal" => Expr::Lit("Char".to_owned(), unquote(text(n, src)).to_owned()),
        "true" => Expr::Bool(true),
        "false" => Expr::Bool(false),
        "null_literal" => Expr::Null,
        "binary_expression" => {
            let op = field(n, "operator").map_or_else(|| "", |o| text(o, src)).to_owned();
            Expr::Binary { op, l: boxed(field(n, "left"), src), r: boxed(field(n, "right"), src) }
        }
        "unary_expression" => {
            let op = field(n, "operator").map_or_else(|| "", |o| text(o, src)).to_owned();
            Expr::Unary { op, e: boxed(field(n, "operand"), src) }
        }
        "update_expression" => {
            let prefix = n.child(0).is_some_and(|c| !c.is_named());
            let inc = text(n, src).contains("++");
            Expr::Update { inc, prefix, e: boxed(kids(n).into_iter().next(), src) }
        }
        "assignment_expression" => {
            let op = field(n, "operator").map_or_else(|| "", |o| text(o, src)).to_owned();
            Expr::AssignExpr { op, l: boxed(field(n, "left"), src), r: boxed(field(n, "right"), src) }
        }
        "ternary_expression" => Expr::Cond {
            c: boxed(field(n, "condition"), src),
            t: boxed(field(n, "consequence"), src),
            f: boxed(field(n, "alternative"), src),
        },
        "parenthesized_expression" => Expr::Paren(boxed(kids(n).into_iter().next(), src)),
        "method_invocation" => {
            let name = field(n, "name").map_or_else(|| "", |name| text(name, src)).to_owned();
            let args = lower_args(field(n, "arguments"), src);
            match field(n, "object") {
                Some(obj) => {
                    if let Some(class) = class_name_of(obj, src) {
                        Expr::StaticCall { scope: class, name, args }
                    } else {
                        Expr::Method { recv: boxed(Some(obj), src), name, args }
                    }
                }
                None => Expr::Call { f: Box::new(Expr::Name(name)), args },
            }
        }
        "object_creation_expression" => {
            let class = field(n, "type").map_or_else(String::new, |t| type_flat_name(t, src));
            // An anonymous class body (`new Handler() { … }`) is part of the expression; its methods
            // are not definitions of the file, so the body is dropped.
            Expr::New { class, args: lower_args(field(n, "arguments"), src) }
        }
        "array_creation_expression" => {
            // `new int[3]`, `new int[]{1, 2}`, `new int[2][3]`: the base type's own dimensions are
            // type, the sizes (and a trailing initializer) are arguments of the creation.
            let base = field(n, "type").map_or_else(|| Type::Other, |t| lower_type(t, src));
            let class = base_name_of(&base);
            let mut args = Vec::new();
            if let Some(dims) = field(n, "dimensions") {
                for d in kids(dims) {
                    if let Some(size) = kids(d).into_iter().next() {
                        args.push(lower_expr(size, src));
                    }
                }
            }
            if let Some(init) = field(n, "value") {
                args.push(lower_expr(init, src));
            }
            Expr::New { class, args }
        }
        "array_initializer" => Expr::Array { elems: kids(n).into_iter().map(|e| lower_expr(e, src)).collect() },
        "array_access" => Expr::Index { base: boxed(field(n, "array"), src), i: boxed(field(n, "index"), src) },
        "cast_expression" => Expr::Cast {
            ty: lower_type_intersection(field(n, "type").unwrap_or(n), src),
            e: boxed(field(n, "value"), src),
        },
        "instanceof_expression" => Expr::InstanceOf {
            e: boxed(field(n, "left"), src),
            ty: field(n, "right").map_or(Type::Other, |t| lower_type(t, src)),
            bind: field(n, "name").map(|name| text(name, src).to_owned()),
        },
        "field_access" => {
            let name = field(n, "field").map_or_else(|| "", |f| text(f, src)).to_owned();
            match field(n, "object") {
                Some(obj) => {
                    if let Some(class) = class_name_of(obj, src) {
                        Expr::StaticField { scope: class, name }
                    } else {
                        Expr::Field { base: boxed(Some(obj), src), name }
                    }
                }
                None => Expr::Field { base: Box::new(Expr::Unknown("field".to_owned())), name },
            }
        }
        "lambda_expression" => Expr::Lambda {
            params: lower_lambda_params(field(n, "parameters"), src),
            body: match field(n, "body") {
                Some(b) if b.kind() == "block" => LambdaBody::Block(lower_block(b, src)),
                Some(b) => LambdaBody::Expr(boxed(Some(b), src)),
                None => LambdaBody::Expr(Box::new(Expr::Unknown("lambda".to_owned()))),
            },
        },
        "method_reference" => {
            let mut items = kids(n).into_iter();
            let target = items.next().map_or(Expr::Unknown("mref".to_owned()), |t| lower_expr(t, src));
            let name = items.next().map_or_else(|| "", |name| text(name, src)).to_owned();
            Expr::MethodRef { target: Box::new(target), name }
        }
        "class_literal" => Expr::ClassConst {
            scope: kids(n).into_iter().next().map_or_else(String::new, |t| type_flat_name(t, src)),
        },
        "switch_expression" => Expr::Switch {
            value: boxed(field(n, "condition"), src),
            arms: switch_arms(field(n, "body"), src),
        },
        _ => Expr::Unknown(n.kind().to_owned()),
    }
}

/// A switch expression's arms: each rule one `Arm` (labels shared), `default` with `conds` empty.
fn switch_arms(body: Option<Node<'_>>, src: &str) -> Vec<Arm> {
    body.map(kids).unwrap_or_default().into_iter().filter_map(|rule| {
        if rule.kind() != "switch_rule" { return None; }
        let mut items = kids(rule).into_iter();
        let mut conds = Vec::new();
        for label in items.by_ref().take_while(|k| k.kind() == "switch_label") {
            conds.extend(label_value(label, src));
        }
        let value = match items.next() {
            Some(b) if b.kind() == "block" => Expr::StmtBlock(lower_block(b, src)),
            Some(b) if b.kind() == "throw_statement" => {
                Expr::Throw(Box::new(kids(b).into_iter().next().map_or(Expr::Unknown("throw".to_owned()), |t| lower_expr(t, src))))
            }
            Some(b) if b.kind() == "expression_statement" => {
                lower_expr(kids(b).into_iter().next().unwrap_or(b), src)
            }
            Some(b) => lower_expr(b, src),
            None => Expr::Unknown("arm".to_owned()),
        };
        Some(Arm { conds, value: Box::new(value) })
    }).collect()
}

fn lower_lambda_params(params: Option<Node<'_>>, src: &str) -> Vec<Param> {
    params.map_or_else(Vec::new, |p| match p.kind() {
        "identifier" => vec![Param { name: text(p, src).to_owned(), ty: Type::Other, variadic: false }],
        "inferred_parameters" => kids(p).into_iter().map(|name| Param { name: text(name, src).to_owned(), ty: Type::Other, variadic: false }).collect(),
        _ => lower_params(Some(p), src),
    })
}

fn lower_args(args: Option<Node<'_>>, src: &str) -> Vec<Expr> {
    args.map(kids).unwrap_or_default().into_iter().map(|a| lower_expr(a, src)).collect()
}

/// A cast's type: an intersection (`(A & B) x`) is two types, any other one.
fn lower_type_intersection(t: Node<'_>, src: &str) -> Type {
    if t.kind() == "intersection_type" {
        Type::Inter(kids(t).into_iter().map(|u| lower_type(u, src)).collect())
    } else {
        lower_type(t, src)
    }
}

pub(crate) fn lower_type(n: Node<'_>, src: &str) -> Type {
    match n.kind() {
        "integral_type" | "floating_point_type" | "boolean_type" | "void_type" | "type_identifier" => {
            Type::Named(text(n, src).to_owned())
        }
        "scoped_type_identifier" => Type::Named(type_flat_name(n, src)),
        "generic_type" => {
            let mut items = kids(n).into_iter();
            let head = items.next().map_or_else(String::new, |t| type_flat_name(t, src));
            let args = items.next().map_or_else(Vec::new, |a| kids(a).into_iter().map(|x| lower_type(x, src)).collect());
            Type::Generic(head, args)
        }
        "wildcard" => {
            // `?`, `? extends X`, `? super X` — the bound keyword is an anonymous token, the type
            // a named child.
            let mut items = kids(n).into_iter();
            let first = items.next();
            let bound = first.and_then(|f| {
                (f.kind() == "extends" || f.kind() == "super").then(|| {
                    (f.kind().to_owned(), Box::new(items.next().map_or(Type::Other, |t| lower_type(t, src))))
                })
            });
            Type::Wild(bound)
        }
        "array_type" => {
            let element = field(n, "element").map_or(Type::Other, |e| lower_type(e, src));
            let dims = field(n, "dimensions").map(kids).map_or(1, |d| d.len()).max(1);
            let mut ty = element;
            for _ in 0..dims {
                ty = Type::Array(Box::new(ty));
            }
            ty
        }
        "union_type" => Type::Union(kids(n).into_iter().map(|u| lower_type(u, src)).collect()),
        "intersection_type" => Type::Inter(kids(n).into_iter().map(|u| lower_type(u, src)).collect()),
        _ => Type::Other,
    }
}

/// A named type's flat dotted spelling: `a.b.C` for `scoped_type_identifier` chains and generic
/// heads alike. Names are as written — the dump resolves them through the imports.
pub(crate) fn type_flat_name(n: Node<'_>, src: &str) -> String {
    match n.kind() {
        "generic_type" => {
            let mut items = kids(n).into_iter();
            items.next().map_or_else(String::new, |t| type_flat_name(t, src))
        }
        "scoped_type_identifier" => {
            let mut items = kids(n).into_iter();
            let mut name = items.next().map_or_else(String::new, |t| type_flat_name(t, src));
            if let Some(rest) = items.next() {
                if !name.is_empty() {
                    name.push('.');
                }
                name.push_str(text(rest, src));
            }
            name
        }
        _ => text(n, src).to_owned(),
    }
}

/// Whether a receiver is a *class name* rather than a value — the Java convention the walk reads
/// without a type checker: a class is spelled `PascalCase` or as a dotted path (`a.b.C`), a variable
/// `camelCase`. `Foo.m()` and `Foo.BAR` are static; `foo.m()` and `foo.bar` dispatch.
fn class_name_of(n: Node<'_>, src: &str) -> Option<String> {
    match n.kind() {
        "identifier" => {
            let name = text(n, src);
            name.chars().next().is_some_and(|c| c.is_ascii_uppercase()).then(|| name.to_owned())
        }
        "scoped_type_identifier" | "generic_type" => Some(type_flat_name(n, src)),
        _ => None,
    }
}

/// The named head of a type — what `new` is written in front of: `int` for `int[]`, `List` for a
/// generic's head. Array dimensions peel off, they are the creation's arguments.
fn base_name_of(ty: &Type) -> String {
    match ty {
        Type::Named(name) | Type::Generic(name, _) => name.clone(),
        Type::Array(inner) => base_name_of(inner),
        _ => String::new(),
    }
}

// ───────────────────────────── class-level declarations ─────────────────────────────

/// What kind of class-like declaration a `ClassDecl` lowers from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClassKind {
    Class,
    Interface,
    Enum,
    Record,
    /// An `@interface` annotation type — an interface the JLS calls it, with element members.
    Annotation,
}

/// A class-like declaration, owned: `class`, `interface`, `enum`, `record`, `@interface`. The
/// [`crate::canon`] class canonical and the schema lens read this; the scan ([`crate::defs`]) walks
/// the members for the defs of their own.
#[derive(Clone, Debug)]
pub(crate) struct ClassDecl {
    pub kind: ClassKind,
    /// The simple name as declared.
    pub name: String,
    /// The number of type parameters (`<T, U>` → 2) — the `gN` tag of the class canonical.
    pub type_params: usize,
    /// The sorted, deduplicated modifier tokens: `public`, `static`, `final`, `abstract`, …
    pub mods: Vec<String>,
    /// The superclass of a `class` — `None` for every other kind (interfaces extend lists, enums
    /// and records extend the language's own types implicitly).
    pub extends: Option<Type>,
    /// The interfaces a `class` / `enum` / `record` implements.
    pub implements: Vec<Type>,
    /// The interfaces an `interface` extends.
    pub extends_list: Vec<Type>,
    /// A record's components — its constructor's parameters and accessors' fields.
    pub components: Vec<Param>,
    pub members: Vec<ClassMember>,
}

/// One member of a class body.
#[derive(Clone, Debug)]
pub(crate) enum ClassMember {
    /// A field (`Field`), a `static final` constant, or an interface constant.
    Field { mods: Vec<String>, ty: Type, name: String, init: Option<Expr>, constant: bool },
    /// A method or constructor. `abstract_` is a declaration without a body (`Func.body` empty) —
    /// an abstract or native method, an annotation element — rendered signature-only.
    Method { mods: Vec<String>, func: Func, abstract_: bool },
    /// A nested class-like declaration.
    Nested(Box<ClassDecl>),
    /// An enum constant: `A`, `A(1, 2)`.
    Variant { name: String, args: Vec<Expr> },
    /// A `static { … }` initializer — unnamed, not part of any canonical.
    StaticInit,
}

/// Lower one class-like declaration (`class_declaration`, `interface_declaration`,
/// `enum_declaration`, `record_declaration`, `annotation_type_declaration`). `None` for a node the
/// grammar recovered around.
pub(crate) fn lower_class(decl: Node<'_>, src: &str) -> Option<ClassDecl> {
    let (kind, body, components, extends_field) = match decl.kind() {
        "class_declaration" => (ClassKind::Class, field(decl, "body"), Vec::new(), field(decl, "superclass")),
        "interface_declaration" => (ClassKind::Interface, field(decl, "body"), Vec::new(), None),
        "enum_declaration" => (ClassKind::Enum, field(decl, "body"), Vec::new(), None),
        "record_declaration" => (
            ClassKind::Record,
            field(decl, "body"),
            // A record's header is its formal parameters — components, held in the `parameters` field.
            lower_params(field(decl, "parameters"), src),
            None,
        ),
        "annotation_type_declaration" => (ClassKind::Annotation, field(decl, "body"), Vec::new(), None),
        _ => return None,
    };
    let name = field(decl, "name").map_or_else(|| "", |name| text(name, src)).to_owned();
    let type_params = field(decl, "type_parameters").map(kids).map_or(0, |p| p.len());
    let mods = modifiers_of(decl);
    let mut extends = extends_field
        .and_then(|s| kids(s).into_iter().next())
        .map(|t| lower_type(t, src));
    let mut implements = Vec::new();
    let mut extends_list = Vec::new();
    if let Some(ifaces) = field(decl, "interfaces") {
        let list = interfaces_list(ifaces, src);
        if kind == ClassKind::Interface {
            extends_list = list;
        } else {
            implements = list;
        }
    } else if let Some(ifaces) = field(decl, "extends_interfaces") {
        extends_list = interfaces_list(ifaces, src);
    }
    // A record's superclass is the language's own `Record`; an enum's `Enum`. Not a fact about the
    // definition.
    if kind == ClassKind::Record {
        extends = None;
    }
    let members = body.map_or_else(Vec::new, |b| lower_members(b, src));
    Some(ClassDecl { kind, name, type_params, mods, extends, implements, extends_list, components, members })
}

/// The interfaces a class-like declaration names, each as a type. The grammar wraps them in a
/// `type_list` in some places and lists them bare in others; both read the same.
fn interfaces_list(n: Node<'_>, src: &str) -> Vec<Type> {
    kids(n)
        .into_iter()
        .flat_map(|k| if k.kind() == "type_list" { kids(k) } else { vec![k] })
        .map(|t| lower_type(t, src))
        .collect()
}

fn lower_members(body: Node<'_>, src: &str) -> Vec<ClassMember> {
    kids(body).into_iter().flat_map(|m| lower_member(m, src)).collect()
}

fn lower_member(m: Node<'_>, src: &str) -> Vec<ClassMember> {
    match m.kind() {
        "field_declaration" | "constant_declaration" => {
            let declared_const = m.kind() == "constant_declaration";
            let ty = field(m, "type").map_or(Type::Other, |t| lower_type(t, src));
            let mods = modifiers_of(m);
            let is_constant = declared_const || (mods.contains(&"static".to_owned()) && mods.contains(&"final".to_owned()));
            fields(m, "declarator")
                .into_iter()
                .map(|decl| {
                    let name = field(decl, "name").map_or_else(|| "", |name| text(name, src)).to_owned();
                    let init = field(decl, "value").map(|v| lower_expr(v, src));
                    ClassMember::Field { mods: mods.clone(), ty: ty.clone(), name, init, constant: is_constant }
                })
                .collect()
        }
        "method_declaration" | "constructor_declaration" | "compact_constructor_declaration" => {
            let name = field(m, "name").map_or_else(|| "", |name| text(name, src)).to_owned();
            let mods = modifiers_of(m);
            let abstract_ = m.kind() == "method_declaration" && field(m, "body").is_none();
            let func = match lower_func(m, &name, src) {
                Some(f) => f,
                // Abstract / native: a signature, not a body.
                None => Func {
                    name,
                    params: lower_params(field(m, "parameters"), src),
                    ret: field(m, "type").map(|t| lower_type(t, src)),
                    type_params: type_param_names(field(m, "type_parameters"), src),
                    body: Vec::new(),
                },
            };
            vec![ClassMember::Method { mods, func, abstract_ }]
        }
        "annotation_type_element_declaration" => {
            let name = field(m, "name").map_or_else(|| "", |name| text(name, src)).to_owned();
            let mods = modifiers_of(m);
            vec![ClassMember::Method {
                mods,
                func: Func { name, params: Vec::new(), ret: field(m, "type").map(|t| lower_type(t, src)), type_params: Vec::new(), body: Vec::new() },
                abstract_: true,
            }]
        }
        "enum_constant" => {
            let name = field(m, "name").map_or_else(|| "", |name| text(name, src)).to_owned();
            let args = field(m, "arguments").map_or_else(Vec::new, |a| lower_args(Some(a), src));
            vec![ClassMember::Variant { name, args }]
        }
        "static_initializer" => vec![ClassMember::StaticInit],
        "class_declaration" | "interface_declaration" | "enum_declaration" | "record_declaration"
        | "annotation_type_declaration" => {
            lower_class(m, src).map_or_else(Vec::new, |d| vec![ClassMember::Nested(Box::new(d))])
        }
        _ => Vec::new(),
    }
}

/// The modifier tokens of a declaration, sorted and deduplicated: `public`, `static`, `final`,
/// `abstract`, `default`, … Annotations are not modifiers.
pub(crate) fn modifiers_of(n: Node<'_>) -> Vec<String> {
    let Some(mods) = kids(n).into_iter().find(|k| k.kind() == "modifiers") else { return Vec::new() };
    let mut out: Vec<String> = Vec::new();
    let mut cursor = mods.walk();
    for child in mods.children(&mut cursor) {
        if !child.is_named() {
            let kind = child.kind().to_owned();
            if !out.contains(&kind) {
                out.push(kind);
            }
        }
    }
    out.sort();
    out
}

/// The annotations attached to a declaration, by name — the `decorators` lens's facts.
pub(crate) fn annotations_of(n: Node<'_>, src: &str) -> Vec<String> {
    let Some(mods) = kids(n).into_iter().find(|k| k.kind() == "modifiers") else { return Vec::new() };
    kids(mods)
        .into_iter()
        .filter(|k| k.kind() == "marker_annotation" || k.kind() == "annotation")
        .filter_map(|a| field(a, "name").map(|t| type_flat_name(t, src)))
        .filter(|name| !name.is_empty())
        .collect()
}

/// The names a declaration's type parameters bind — the types the Type-3 slotting treats as the
/// definition's own rather than vocabulary.
pub(crate) fn type_param_names(type_params: Option<Node<'_>>, src: &str) -> Vec<String> {
    type_params.map(kids).unwrap_or_default().into_iter().filter_map(|p| {
        // A `type_parameter` spells its name as a `type_identifier` child, not a `name` field.
        kids(p).into_iter().find(|k| k.kind() == "type_identifier").map(|name| text(name, src).to_owned())
    }).collect()
}

// ───────────────────────────── bound names ─────────────────────────────

/// Every name a callable binds — parameters, locals, loop variables, catch bindings, lambda
/// parameters, pattern bindings — the set a dump reads a *local* against. A name in it renames to a
/// `_vN` slot; a name outside it is a free name.
pub(crate) fn locals_of(f: &Func) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for p in &f.params {
        out.insert(p.name.clone());
    }
    stmt_names(&f.body, &mut out);
    out
}

fn stmt_names(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Let { name, init, .. } => {
                out.insert(name.clone());
                if let Some(init) = init {
                    expr_names(init, out);
                }
            }
            Stmt::Decl(items) => stmt_names(items, out),
            Stmt::Assign { lhs, rhs, .. } => {
                expr_names(lhs, out);
                expr_names(rhs, out);
            }
            Stmt::Update { e, .. } | Stmt::Expr(e) | Stmt::Throw(e) | Stmt::Assert(e) | Stmt::Return(Some(e)) => expr_names(e, out),
            Stmt::If { cond, then, els } => {
                expr_names(cond, out);
                stmt_names(then, out);
                if let Some(else_box) = els {
                    match else_box.as_ref() {
                        Else::Block(b) => stmt_names(b, out),
                        Else::If(alt) => stmt_names(&[alt.as_ref().clone()], out),
                    }
                }
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                expr_names(cond, out);
                stmt_names(body, out);
            }
            Stmt::For { init, cond, update, body } => {
                if let Some(init) = init {
                    stmt_names(&[init.as_ref().clone()], out);
                }
                if let Some(cond) = cond {
                    expr_names(cond, out);
                }
                for u in update {
                    expr_names(u, out);
                }
                stmt_names(body, out);
            }
            Stmt::ForEach { name, iter, body } => {
                out.insert(name.clone());
                expr_names(iter, out);
                stmt_names(body, out);
            }
            Stmt::Switch { value, cases } => {
                expr_names(value, out);
                for case in cases {
                    for v in &case.values {
                        expr_names(v, out);
                    }
                    stmt_names(&case.body, out);
                }
            }
            Stmt::Try { resources, body, catches, finally } => {
                for r in resources {
                    if let Some(name) = &r.name {
                        out.insert(name.clone());
                    }
                    expr_names(&r.value, out);
                }
                stmt_names(body, out);
                for c in catches {
                    if let Some(name) = &c.name {
                        out.insert(name.clone());
                    }
                    stmt_names(&c.body, out);
                }
                if let Some(fin) = finally {
                    stmt_names(fin, out);
                }
            }
            Stmt::Synchronized { lock, body } => {
                expr_names(lock, out);
                stmt_names(body, out);
            }
            Stmt::Labeled { stmt, .. } => stmt_names(&[stmt.as_ref().clone()], out),
            Stmt::Block(b) => stmt_names(b, out),
            Stmt::Return(None) | Stmt::Break(_) | Stmt::Continue(_) | Stmt::NestedItem | Stmt::Unknown(_) => {}
        }
    }
}

fn expr_names(e: &Expr, out: &mut std::collections::HashSet<String>) {
    match e {
        Expr::Binary { l, r, .. } | Expr::AssignExpr { l, r, .. } => {
            expr_names(l, out);
            expr_names(r, out);
        }
        Expr::Unary { e, .. } | Expr::Update { e, .. } | Expr::Paren(e) | Expr::Cast { e, .. } | Expr::Throw(e) | Expr::InstanceOf { e, .. } => expr_names(e, out),
        Expr::Cond { c, t, f } => {
            expr_names(c, out);
            expr_names(t, out);
            expr_names(f, out);
        }
        Expr::Call { f, args } => {
            expr_names(f, out);
            for a in args {
                expr_names(a, out);
            }
        }
        Expr::Method { recv, args, .. } => {
            expr_names(recv, out);
            for a in args {
                expr_names(a, out);
            }
        }
        Expr::StaticCall { args, .. } | Expr::Super { args } | Expr::New { args, .. } => {
            for a in args {
                expr_names(a, out);
            }
        }
        Expr::Field { base, .. } => expr_names(base, out),
        Expr::Index { base, i } => {
            expr_names(base, out);
            expr_names(i, out);
        }
        Expr::Array { elems } => {
            for el in elems {
                expr_names(el, out);
            }
        }
        Expr::Lambda { params, body } => {
            for p in params {
                out.insert(p.name.clone());
            }
            match body {
                LambdaBody::Block(b) => stmt_names(b, out),
                LambdaBody::Expr(e) => expr_names(e, out),
            }
        }
        Expr::MethodRef { target, .. } => expr_names(target, out),
        Expr::Switch { value, arms } => {
            expr_names(value, out);
            for arm in arms {
                for c in &arm.conds {
                    expr_names(c, out);
                }
                expr_names(&arm.value, out);
            }
        }
        Expr::StmtBlock(b) => stmt_names(b, out),
        Expr::Name(_) | Expr::Lit(..) | Expr::Bool(_) | Expr::Null | Expr::StaticField { .. } | Expr::ClassConst { .. } | Expr::Unknown(_) => {}
    }
}
