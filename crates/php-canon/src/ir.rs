//! An owned syntax tree for PHP bodies and declarations, lowered once from tree-sitter's.
//!
//! tree-sitter's tree is read-only: a node is a view into a parse that cannot be edited. The normal
//! form ([`crate::normal`]) rewrites a body — flips a branch, lifts a guard, inlines a temporary — so
//! the canonicals are written from this tree instead, which can be. It holds exactly what the
//! canonicals read and nothing else: names and literals as owned text, no positions, no comments.
//!
//! What PHP spells two ways is lowered to one shape here, before anything compares it, because the two
//! spellings are one construct to the interpreter:
//!
//! * the **alternative syntax** — `if (…): … endif;`, `foreach (…): … endforeach;` — is the braced one;
//! * `elseif` is `else if`;
//! * `array(…)` is `[…]`, and `list(…) =` is `[…] =`;
//! * the keyword operators `and` / `or` are `&&` / `||` (their precedence is the tree's shape already),
//!   and `<>` is `!=`;
//! * `die` is `exit`.
//!
//! A template stretch between two statements (`?> <p>…</p> <?php`) is output, an `echo` of its text.
//!
//! Lowering is depth-limited ([`dup_defs_core::tree::MAX_DEPTH`]): machine-written PHP nests an array
//! literal or a concatenation thousands of levels deep, and every walk downstream — this one, the
//! canonicals, the engine reading them — recurses. A subtree past the limit is one `Unknown`.

use dup_defs_core::tree::Level;
use tree_sitter::Node;

use crate::tree::{child_of_kind, field, fields, kids, name_of, statements_of, text, var_name, unquote};

/// A declared type: a parameter's, a property's, what a function returns.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Type {
    /// A class or a built-in type, as spelled — `int`, `array`, `Order`, `\App\Models\Order`.
    Named(String),
    /// `?T`.
    Nullable(Box<Type>),
    /// `A|B`.
    Union(Vec<Type>),
    /// `A&B`.
    Inter(Vec<Type>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Param {
    pub name: String,
    pub ty: Option<Type>,
    pub default: Option<Expr>,
    pub by_ref: bool,
    pub variadic: bool,
    /// A constructor parameter that declares a property of the same name — its visibility.
    pub promoted: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Stmt {
    Expr(Expr),
    /// `$x = e`, `$x .= e`, `$x =& e` written as a statement.
    Assign { op: String, lhs: Expr, rhs: Expr },
    /// `$i++;` / `--$i;` — as a statement the two are one.
    IncDec { inc: bool, target: Expr },
    Echo(Vec<Expr>),
    Return(Option<Expr>),
    /// `throw e;` — an expression in PHP 8, the step a body ends with when written as a statement.
    Throw(Expr),
    /// `break` / `continue` with the levels they leave, `goto` with its label.
    Jump { kind: Jump, arg: Option<String> },
    If { cond: Expr, then: Vec<Stmt>, els: Option<Box<Else>> },
    While { cond: Expr, body: Vec<Stmt> },
    DoWhile { body: Vec<Stmt>, cond: Expr },
    For { init: Vec<Expr>, cond: Vec<Expr>, update: Vec<Expr>, body: Vec<Stmt> },
    Foreach { subject: Expr, key: Option<Expr>, value: Expr, body: Vec<Stmt> },
    Switch { value: Expr, cases: Vec<Case> },
    Try { body: Vec<Stmt>, catches: Vec<Catch>, finally: Option<Vec<Stmt>> },
    Block(Vec<Stmt>),
    /// `global $wpdb;` — names the function reads from outside it.
    Global(Vec<String>),
    /// `static $n = 0;` — a local that keeps its value between calls.
    Static(Vec<(String, Option<Expr>)>),
    Unset(Vec<Expr>),
    Label(String),
    /// A function, class or other declaration written inside a body.
    NestedItem,
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
    /// `else if …` / `elseif …` — always a [`Stmt::If`].
    If(Box<Stmt>),
}

/// A `case` (`value` set) or the `default` label (`value` `None`), with the statements up to the next
/// label — PHP's cases fall through, so where one ends is where the next begins.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Case {
    pub value: Option<Expr>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Catch {
    pub types: Vec<String>,
    pub var: Option<String>,
    pub body: Vec<Stmt>,
}

/// A member's name: written, or computed (`$obj->$name`, `$obj->{'a' . $b}`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Member {
    Name(String),
    Dyn(Box<Expr>),
}

/// One argument of a call: named (`f(limit: 3)`) or positional.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Arg {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Elem {
    Plain(Expr),
    /// `key => value`.
    Pair(Expr, Expr),
}

/// One arm of a `match`: the values it matches (`None` for `default`) and its result.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Arm {
    pub conds: Option<Vec<Expr>>,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Expr {
    /// `$name`, without its `$`. `$this` is `Var("this")`.
    Var(String),
    /// `$$name`, `${expr}`.
    DynVar(Box<Expr>),
    /// A constant or function name as written: `PHP_EOL`, `strlen`, `\App\helper`. The scope of a
    /// scoped access (`self`, `static`, `parent`, a class) is one too.
    Name(String),
    /// `Int` / `Float` / `Str`, with the literal's text (quotes off for the last).
    Lit(&'static str, String),
    /// A string with variables or expressions spliced in: its pieces in order, the literal ones `Str`.
    Interp(Vec<Expr>),
    Bool(bool),
    Null,
    /// Every binary operator, `.`, `??` and `instanceof` included.
    Binary { op: String, l: Box<Expr>, r: Box<Expr> },
    /// `!`, `-`, `+`, `~`, and `&` taking a reference.
    Unary { op: String, e: Box<Expr> },
    /// `$i++`, `--$i` inside an expression.
    Update { inc: bool, prefix: bool, e: Box<Expr> },
    /// An assignment used as a value: `while (($row = next()) !== false)`.
    Assign { op: String, l: Box<Expr>, r: Box<Expr> },
    Cond { c: Box<Expr>, t: Option<Box<Expr>>, f: Box<Expr> },
    Comma(Box<Expr>, Box<Expr>),
    Call { f: Box<Expr>, args: Vec<Arg> },
    /// `$recv->name(…)`, `$recv?->name(…)`.
    Method { recv: Box<Expr>, name: Member, args: Vec<Arg>, nullsafe: bool },
    /// `Scope::name(…)` — `self::`, `static::`, `parent::` or a class.
    StaticCall { scope: Box<Expr>, name: Member, args: Vec<Arg> },
    /// `$recv->name`, `$recv?->name`.
    Prop { e: Box<Expr>, name: Member, nullsafe: bool },
    /// `Scope::$name`.
    StaticProp { scope: Box<Expr>, name: String },
    /// `Scope::NAME`, `Scope::class`.
    ClassConst { scope: Box<Expr>, name: String },
    /// `$a[i]`, and `$a[]` appending.
    Index { e: Box<Expr>, i: Option<Box<Expr>> },
    /// `[…]` and `array(…)`.
    Array(Vec<Elem>),
    /// `[$a, $b] = …` and `list($a, $b) = …`.
    List(Vec<Elem>),
    /// `new Class(…)`.
    New { class: Box<Expr>, args: Vec<Arg> },
    /// `new class { … }`.
    AnonClass,
    Closure { params: Vec<Param>, uses: Vec<(String, bool)>, body: Vec<Stmt> },
    Arrow { params: Vec<Param>, body: Box<Expr> },
    Match { subject: Box<Expr>, arms: Vec<Arm> },
    /// `(int) $x` — the type as PHP names it, its aliases folded (`integer` is `int`).
    Cast { ty: String, e: Box<Expr> },
    /// `@expr` — its warnings silenced.
    Silence(Box<Expr>),
    Clone(Box<Expr>),
    Print(Box<Expr>),
    /// `include` / `require` / their `_once` forms.
    Include { kind: &'static str, e: Box<Expr> },
    Yield { key: Option<Box<Expr>>, value: Option<Box<Expr>> },
    YieldFrom(Box<Expr>),
    Throw(Box<Expr>),
    /// `` `cmd` `` — a shell command, its pieces as an interpolated string's.
    Shell(Vec<Expr>),
    /// `...$args` in a call or an array.
    Spread(Box<Expr>),
    Paren(Box<Expr>),
    Unknown(String),
}

/// A function, closure or method, owned: what every canonical reads.
#[derive(Clone, Debug)]
pub(crate) struct Func {
    pub params: Vec<Param>,
    pub ret: Option<Type>,
    pub by_ref: bool,
    pub body: Vec<Stmt>,
}

/// One member of a class-like declaration.
#[derive(Clone, Debug)]
pub(crate) enum ClassMember {
    Const { name: String, value: Expr },
    Prop { modifiers: Vec<String>, ty: Option<Type>, name: String, default: Option<Expr> },
    /// A method; `func` is `None` for an abstract or interface method, which has no body.
    Method { modifiers: Vec<String>, name: String, func: Option<Func>, sig: Func },
    /// `use SomeTrait;` — the methods of another declaration, pasted in.
    TraitUse(Vec<String>),
    /// An enum's `case`.
    Case { name: String, value: Option<Expr> },
}

/// A `class`, `trait`, `enum` or `interface`, owned.
#[derive(Clone, Debug)]
pub(crate) struct ClassDecl {
    /// `class` / `trait` / `enum` / `interface`.
    pub kind: &'static str,
    pub name: String,
    pub modifiers: Vec<String>,
    pub extends: Vec<String>,
    pub implements: Vec<String>,
    /// An enum's backing type (`enum Suit: string`).
    pub backing: Option<String>,
    pub members: Vec<ClassMember>,
}

// ───────────────────────────── declarations ─────────────────────────────

/// The modifiers written on a declaration (`public`, `static`, `abstract`, `final`, `readonly`), in
/// order, lowercased — PHP's keywords are case-insensitive.
pub(crate) fn modifiers(n: Node<'_>, src: &str) -> Vec<String> {
    kids(n)
        .into_iter()
        .filter(|k| k.kind().ends_with("_modifier") && k.kind() != "reference_modifier")
        .map(|k| text(k, src).to_ascii_lowercase())
        .collect()
}

/// The names of the attributes written on a declaration (`#[Route('/x')]` → `Route`).
pub(crate) fn attribute_names(n: Node<'_>, src: &str) -> Vec<String> {
    let Some(list) = field(n, "attributes") else { return Vec::new() };
    kids(list)
        .into_iter()
        .flat_map(kids)
        .filter(|a| a.kind() == "attribute")
        .filter_map(|a| kids(a).into_iter().next().map(|name| name_of(name, src)))
        .collect()
}

pub(crate) fn lower_type(n: Node<'_>, src: &str) -> Type {
    match n.kind() {
        "primitive_type" => Type::Named(text(n, src).to_ascii_lowercase()),
        "bottom_type" => Type::Named("never".to_owned()),
        "optional_type" => Type::Nullable(Box::new(kids(n).into_iter().next().map_or(Type::Named("mixed".to_owned()), |t| lower_type(t, src)))),
        "union_type" | "disjunctive_normal_form_type" => Type::Union(kids(n).into_iter().map(|t| lower_type(t, src)).collect()),
        "intersection_type" => Type::Inter(kids(n).into_iter().map(|t| lower_type(t, src)).collect()),
        "type" => kids(n).into_iter().next().map_or(Type::Named("mixed".to_owned()), |t| lower_type(t, src)),
        _ => Type::Named(name_of(n, src)),
    }
}

/// A parameter list.
pub(crate) fn lower_params(list: Option<Node<'_>>, src: &str) -> Vec<Param> {
    let mut out = Vec::new();
    for p in list.map(kids).unwrap_or_default() {
        if !matches!(p.kind(), "simple_parameter" | "variadic_parameter" | "property_promotion_parameter") {
            continue;
        }
        let name_node = field(p, "name");
        let (name, by_ref_name) = match name_node {
            Some(n) if n.kind() == "by_ref" => (kids(n).into_iter().next().and_then(|v| var_name(v, src)).unwrap_or("").to_owned(), true),
            Some(n) => (var_name(n, src).unwrap_or("").to_owned(), false),
            None => continue,
        };
        let promoted = (p.kind() == "property_promotion_parameter")
            .then(|| fields(p, "visibility").first().map_or_else(|| "public".to_owned(), |v| text(*v, src).to_ascii_lowercase()));
        out.push(Param {
            name,
            ty: field(p, "type").map(|t| lower_type(t, src)),
            default: field(p, "default_value").map(|d| lower_expr(d, src)),
            by_ref: by_ref_name || field(p, "reference_modifier").is_some(),
            variadic: p.kind() == "variadic_parameter",
            promoted,
        });
    }
    out
}

/// A function, method, closure or arrow function's signature and body.
pub(crate) fn lower_func(n: Node<'_>, src: &str) -> Func {
    Func {
        params: lower_params(field(n, "parameters"), src),
        ret: field(n, "return_type").map(|t| lower_type(t, src)),
        by_ref: kids(n).into_iter().any(|k| k.kind() == "reference_modifier") || field(n, "reference_modifier").is_some(),
        body: field(n, "body").map(|b| lower_block(b, src)).unwrap_or_default(),
    }
}

/// The names a `base_clause` / `class_interface_clause` lists.
fn clause_names(n: Option<Node<'_>>, src: &str) -> Vec<String> {
    n.map(kids).unwrap_or_default().into_iter().map(|k| name_of(k, src)).collect()
}

/// A class-like declaration: its header and every member.
pub(crate) fn lower_class(n: Node<'_>, src: &str) -> Option<ClassDecl> {
    let kind = match n.kind() {
        "class_declaration" => "class",
        "trait_declaration" => "trait",
        "enum_declaration" => "enum",
        "interface_declaration" => "interface",
        _ => return None,
    };
    let name = text(field(n, "name")?, src).to_owned();
    let (extends, implements) = if kind == "interface" {
        (clause_names(child_of_kind(n, "base_clause"), src), Vec::new())
    } else {
        (clause_names(child_of_kind(n, "base_clause"), src), clause_names(child_of_kind(n, "class_interface_clause"), src))
    };
    let backing = (kind == "enum").then(|| child_of_kind(n, "primitive_type").map(|t| text(t, src).to_ascii_lowercase())).flatten();
    let members = field(n, "body").map(kids).unwrap_or_default().into_iter().flat_map(|m| lower_member(m, src)).collect();
    Some(ClassDecl { kind, name, modifiers: modifiers(n, src), extends, implements, backing, members })
}

fn lower_member(m: Node<'_>, src: &str) -> Vec<ClassMember> {
    match m.kind() {
        "const_declaration" => kids(m)
            .into_iter()
            .filter(|k| k.kind() == "const_element")
            .filter_map(|e| {
                let parts = kids(e);
                let name = text(*parts.first()?, src).to_owned();
                let value = parts.get(1).map_or(Expr::Unknown(String::new()), |v| lower_expr(*v, src));
                Some(ClassMember::Const { name, value })
            })
            .collect(),
        "property_declaration" => {
            let mods = modifiers(m, src);
            let ty = field(m, "type").map(|t| lower_type(t, src));
            kids(m)
                .into_iter()
                .filter(|k| k.kind() == "property_element")
                .filter_map(|e| {
                    Some(ClassMember::Prop {
                        modifiers: mods.clone(),
                        ty: ty.clone(),
                        name: var_name(field(e, "name")?, src)?.to_owned(),
                        default: field(e, "default_value").map(|d| lower_expr(d, src)),
                    })
                })
                .collect()
        }
        "method_declaration" => {
            let Some(name) = field(m, "name").map(|n| text(n, src).to_owned()) else { return Vec::new() };
            let sig = lower_func(m, src);
            let func = field(m, "body").is_some().then(|| sig.clone());
            vec![ClassMember::Method { modifiers: modifiers(m, src), name, func, sig }]
        }
        "use_declaration" => vec![ClassMember::TraitUse(
            kids(m).into_iter().filter(|k| k.kind() != "use_list").map(|k| name_of(k, src)).collect(),
        )],
        "enum_case" => field(m, "name")
            .map(|name| ClassMember::Case { name: text(name, src).to_owned(), value: field(m, "value").map(|v| lower_expr(v, src)) })
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

// ───────────────────────────── statements ─────────────────────────────

/// The statements a block holds — a braced block, an alternative-syntax block, or a `case`.
pub(crate) fn lower_block(holder: Node<'_>, src: &str) -> Vec<Stmt> {
    lower_list(&statements_of(holder), src)
}

fn lower_list(items: &[Node<'_>], src: &str) -> Vec<Stmt> {
    items
        .iter()
        .filter_map(|n| match lower_stmt(*n, src) {
            Stmt::Empty => None,
            s => Some(s),
        })
        .collect()
}

/// The statements a branch or loop body holds: a block's own, the alternative syntax's run of them, or
/// the one statement written bare — `if ($x) return;` and `if ($x) { return; }` are one body.
fn body_of(nodes: &[Node<'_>], src: &str) -> Vec<Stmt> {
    match nodes {
        [b] if matches!(b.kind(), "compound_statement" | "colon_block") => lower_block(*b, src),
        many => lower_list(many, src),
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

/// A comma list of expressions — `for ($i = 0, $n = count($a); …)`, `echo $a, $b;` — one by one.
fn expr_list(n: Option<Node<'_>>, src: &str) -> Vec<Expr> {
    let mut out = Vec::new();
    let mut at = n;
    while let Some(e) = at {
        if e.kind() == "sequence_expression" {
            let parts = kids(e);
            if let Some(first) = parts.first() {
                out.push(lower_expr(*first, src));
            }
            at = parts.get(1).copied();
        } else {
            out.push(lower_expr(e, src));
            at = None;
        }
    }
    out
}

/// An expression written as a statement: an assignment or an increment is the step it takes, a `throw`
/// the exit it is, any other expression a value computed for its effects.
fn expr_stmt(e: Node<'_>, src: &str) -> Stmt {
    match lower_expr(e, src) {
        Expr::Assign { op, l, r } => Stmt::Assign { op, lhs: *l, rhs: *r },
        Expr::Update { inc, e, .. } => Stmt::IncDec { inc, target: *e },
        Expr::Throw(e) => Stmt::Throw(*e),
        other => Stmt::Expr(other),
    }
}

/// `if (…) … elseif (…) … else …`: the clauses after the first branch, each the `else` of the one
/// before.
fn else_chain(clauses: &[Node<'_>], src: &str) -> Option<Box<Else>> {
    let (first, rest) = clauses.split_first()?;
    let body = fields(*first, "body");
    Some(Box::new(match first.kind() {
        "else_if_clause" => Else::If(Box::new(Stmt::If {
            cond: cond_of(field(*first, "condition"), src),
            then: body_of(&body, src),
            els: else_chain(rest, src),
        })),
        // `else if (…)` written as two words is the same chain.
        _ => match body.as_slice() {
            [only] if only.kind() == "if_statement" => Else::If(Box::new(lower_stmt(*only, src))),
            _ => Else::Block(body_of(&body, src)),
        },
    }))
}

/// A template stretch (`?>…<?php`) as the output it is; nothing for one that holds only whitespace.
fn template(n: Node<'_>, src: &str) -> Stmt {
    let html: String = child_of_text(n, src).split_whitespace().collect::<Vec<_>>().join(" ");
    if html.is_empty() {
        Stmt::Empty
    } else {
        Stmt::Echo(vec![Expr::Lit("Str", html)])
    }
}

fn child_of_text<'s>(n: Node<'_>, src: &'s str) -> &'s str {
    let mut cursor = n.walk();
    let found = n.named_children(&mut cursor).find(|k| k.kind() == "text");
    found.map_or("", |t| text(t, src))
}

#[allow(clippy::too_many_lines)] // one arm per statement kind reads better straight-through
pub(crate) fn lower_stmt(n: Node<'_>, src: &str) -> Stmt {
    let Some(_level) = Level::enter() else { return Stmt::Unknown("deep".to_owned()) };
    match n.kind() {
        "compound_statement" | "colon_block" => Stmt::Block(lower_block(n, src)),
        "expression_statement" => kids(n).into_iter().next().map_or(Stmt::Empty, |e| expr_stmt(e, src)),
        "return_statement" => Stmt::Return(kids(n).into_iter().next().map(|e| lower_expr(e, src))),
        "echo_statement" => Stmt::Echo(expr_list(kids(n).into_iter().next(), src)),
        "text_interpolation" => template(n, src),
        "if_statement" => Stmt::If {
            cond: cond_of(field(n, "condition"), src),
            then: body_of(&fields(n, "body"), src),
            els: else_chain(&fields(n, "alternative"), src),
        },
        "while_statement" => Stmt::While { cond: cond_of(field(n, "condition"), src), body: body_of(&fields(n, "body"), src) },
        "do_statement" => Stmt::DoWhile { body: body_of(&fields(n, "body"), src), cond: cond_of(field(n, "condition"), src) },
        "for_statement" => Stmt::For {
            init: expr_list(field(n, "initialize"), src),
            cond: expr_list(field(n, "condition"), src),
            update: expr_list(field(n, "update"), src),
            body: body_of(&fields(n, "body"), src),
        },
        "foreach_statement" => {
            let body_ids: Vec<usize> = fields(n, "body").iter().map(Node::id).collect();
            let parts: Vec<Node<'_>> = kids(n).into_iter().filter(|k| !body_ids.contains(&k.id())).collect();
            let subject = parts.first().map_or(Expr::Unknown(String::new()), |s| lower_expr(*s, src));
            let (key, value) = match parts.get(1) {
                Some(pair) if pair.kind() == "pair" => {
                    let kv = kids(*pair);
                    (kv.first().map(|k| lower_expr(*k, src)), kv.get(1).map_or(Expr::Unknown(String::new()), |v| lower_expr(*v, src)))
                }
                Some(v) => (None, lower_expr(*v, src)),
                None => (None, Expr::Unknown(String::new())),
            };
            Stmt::Foreach { subject, key, value, body: body_of(&fields(n, "body"), src) }
        }
        "switch_statement" => Stmt::Switch {
            value: cond_of(field(n, "condition"), src),
            cases: field(n, "body")
                .map(kids)
                .unwrap_or_default()
                .into_iter()
                .filter(|c| matches!(c.kind(), "case_statement" | "default_statement"))
                .map(|c| {
                    let value = field(c, "value");
                    let held: Vec<Node<'_>> =
                        statements_of(c).into_iter().filter(|k| Some(k.id()) != value.map(|v| v.id())).collect();
                    Case { value: value.map(|v| lower_expr(v, src)), body: lower_list(&held, src) }
                })
                .collect(),
        },
        "try_statement" => {
            let mut catches = Vec::new();
            let mut finally = None;
            for clause in kids(n) {
                match clause.kind() {
                    "catch_clause" => catches.push(Catch {
                        types: field(clause, "type").map(kids).unwrap_or_default().into_iter().map(|t| name_of(t, src)).collect(),
                        var: field(clause, "name").and_then(|v| var_name(v, src)).map(str::to_owned),
                        body: field(clause, "body").map(|b| lower_block(b, src)).unwrap_or_default(),
                    }),
                    "finally_clause" => finally = Some(field(clause, "body").map(|b| lower_block(b, src)).unwrap_or_default()),
                    _ => {}
                }
            }
            Stmt::Try { body: field(n, "body").map(|b| lower_block(b, src)).unwrap_or_default(), catches, finally }
        }
        "break_statement" => Stmt::Jump { kind: Jump::Break, arg: kids(n).into_iter().next().map(|l| text(l, src).to_owned()) },
        "continue_statement" => Stmt::Jump { kind: Jump::Continue, arg: kids(n).into_iter().next().map(|l| text(l, src).to_owned()) },
        "goto_statement" => Stmt::Jump { kind: Jump::Goto, arg: kids(n).into_iter().next().map(|l| text(l, src).to_owned()) },
        "named_label_statement" => Stmt::Label(kids(n).into_iter().next().map_or_else(String::new, |l| text(l, src).to_owned())),
        "global_declaration" => Stmt::Global(kids(n).into_iter().filter_map(|v| var_name(v, src).map(str::to_owned)).collect()),
        "function_static_declaration" => Stmt::Static(
            kids(n)
                .into_iter()
                .filter(|k| k.kind() == "static_variable_declaration")
                .filter_map(|d| {
                    let name = var_name(field(d, "name")?, src)?.to_owned();
                    Some((name, field(d, "value").map(|v| lower_expr(v, src))))
                })
                .collect(),
        ),
        "unset_statement" => Stmt::Unset(kids(n).into_iter().map(|v| lower_expr(v, src)).collect()),
        "exit_statement" => Stmt::Expr(Expr::Call {
            f: Box::new(Expr::Name("exit".to_owned())),
            args: kids(n).into_iter().map(|e| Arg { name: None, value: lower_expr(e, src) }).collect(),
        }),
        "declare_statement" => {
            let held: Vec<Node<'_>> = kids(n).into_iter().filter(|k| k.kind() != "declare_directive").collect();
            match lower_list(&held, src).as_slice() {
                [] => Stmt::Empty,
                [one] => one.clone(),
                many => Stmt::Block(many.to_vec()),
            }
        }
        "function_definition" | "class_declaration" | "interface_declaration" | "trait_declaration" | "enum_declaration" => {
            Stmt::NestedItem
        }
        "empty_statement" | "namespace_use_declaration" | "const_declaration" | "namespace_definition" | "php_tag"
        | "php_end_tag" | "text" => Stmt::Empty,
        other => Stmt::Unknown(other.to_owned()),
    }
}

// ───────────────────────────── expressions ─────────────────────────────

fn boxed(n: Option<Node<'_>>, src: &str) -> Box<Expr> {
    Box::new(n.map_or(Expr::Unknown(String::new()), |e| lower_expr(e, src)))
}

/// The pieces of an interpolated string, a heredoc, or a shell command: the literal runs joined, every
/// spliced value lowered. A string with nothing spliced in is a plain literal.
fn interpolated(n: Node<'_>, src: &str) -> Expr {
    let mut parts: Vec<Expr> = Vec::new();
    let mut run = String::new();
    let mut pieces = Vec::new();
    for k in kids(n) {
        if k.kind() == "heredoc_body" {
            pieces.extend(kids(k));
        } else {
            pieces.push(k);
        }
    }
    for k in pieces {
        match k.kind() {
            "string_content" | "escape_sequence" | "string_value" | "nowdoc_string" => run.push_str(text(k, src)),
            "heredoc_start" | "heredoc_end" => {}
            _ => {
                if !run.is_empty() {
                    parts.push(Expr::Lit("Str", std::mem::take(&mut run)));
                }
                parts.push(lower_expr(k, src));
            }
        }
    }
    if parts.is_empty() {
        return Expr::Lit("Str", run);
    }
    if !run.is_empty() {
        parts.push(Expr::Lit("Str", run));
    }
    Expr::Interp(parts)
}

/// A member's name: written, or computed.
fn member(n: Option<Node<'_>>, src: &str) -> Member {
    match n {
        Some(m) if m.kind() == "name" => Member::Name(text(m, src).to_owned()),
        Some(m) => Member::Dyn(Box::new(lower_expr(m, src))),
        None => Member::Name(String::new()),
    }
}

/// The scope of `Scope::…`: `self`, `static`, `parent`, a class name, or a value.
fn scope(n: Option<Node<'_>>, src: &str) -> Expr {
    match n {
        Some(s) if s.kind() == "relative_scope" => Expr::Name(text(s, src).to_ascii_lowercase()),
        // A class is named as written: `\Exception` and an `Exception` the namespace declares are two.
        Some(s) if matches!(s.kind(), "name" | "qualified_name" | "relative_name") => Expr::Name(name_of(s, src)),
        Some(s) => lower_expr(s, src),
        None => Expr::Unknown(String::new()),
    }
}

/// A call's arguments, named ones with their name.
fn lower_args(list: Option<Node<'_>>, src: &str) -> Vec<Arg> {
    let mut out = Vec::new();
    for a in list.map(kids).unwrap_or_default() {
        match a.kind() {
            "argument" => {
                let name = field(a, "name");
                let value = kids(a)
                    .into_iter()
                    .rfind(|k| Some(k.id()) != name.map(|n| n.id()) && k.kind() != "reference_modifier")
                    .map_or(Expr::Unknown(String::new()), |v| match v.kind() {
                        // `f(self)` and the cast keywords used as constant names are names.
                        "name" => Expr::Name(text(v, src).to_owned()),
                        _ => lower_expr(v, src),
                    });
                out.push(Arg { name: name.map(|n| text(n, src).to_owned()), value });
            }
            // `strlen(...)` — the first-class callable syntax.
            "variadic_placeholder" => out.push(Arg { name: None, value: Expr::Spread(Box::new(Expr::Unknown("placeholder".to_owned()))) }),
            _ => {}
        }
    }
    out
}

/// The elements of an array literal or a destructuring list.
fn lower_elems(n: Node<'_>, src: &str) -> Vec<Elem> {
    if n.kind() == "list_literal" {
        // A list's `key => target` pairs are not wrapped in a node of their own: the `=>` between two
        // children joins them.
        let mut out: Vec<Elem> = Vec::new();
        let mut pending_key = false;
        let mut cursor = n.walk();
        for child in n.children(&mut cursor) {
            if child.kind() == "=>" {
                pending_key = true;
                continue;
            }
            if !child.is_named() || matches!(child.kind(), "comment" | "text_interpolation") {
                continue;
            }
            let value = lower_expr(child, src);
            match (pending_key, out.pop()) {
                (true, Some(Elem::Plain(key))) => out.push(Elem::Pair(key, value)),
                (_, prev) => {
                    out.extend(prev);
                    out.push(Elem::Plain(value));
                }
            }
            pending_key = false;
        }
        return out;
    }
    kids(n)
        .into_iter()
        .map(|el| {
            let parts = if el.kind() == "array_element_initializer" { kids(el) } else { vec![el] };
            match parts.as_slice() {
                [k, v] => Elem::Pair(lower_expr(*k, src), lower_expr(*v, src)),
                [v] => Elem::Plain(lower_expr(*v, src)),
                _ => Elem::Plain(Expr::Unknown(String::new())),
            }
        })
        .collect()
}

/// A number literal: an integer, or a float when the grammar says so.
fn number(kind: &str, t: &str) -> Expr {
    Expr::Lit(if kind == "float" { "Float" } else { "Int" }, t.to_owned())
}

/// A function or constant name with the `\` that marks a global one taken off: `\strlen` is the
/// `strlen` an unqualified call reaches anyway — PHP falls back to the global function or constant when
/// the namespace has none — and writing it qualified is a speed habit, not another name.
fn global(spelled: String) -> String {
    match spelled.strip_prefix('\\') {
        Some(bare) if !bare.contains('\\') => bare.to_owned(),
        _ => spelled,
    }
}

/// A cast's type as PHP names it, its aliases folded.
fn cast_type(t: &str) -> String {
    match t.to_ascii_lowercase().as_str() {
        "integer" => "int".to_owned(),
        "boolean" => "bool".to_owned(),
        "double" | "real" => "float".to_owned(),
        "binary" => "string".to_owned(),
        other => other.to_owned(),
    }
}

/// A binary operator, its keyword and legacy spellings folded.
fn operator(op: &str) -> String {
    match op.to_ascii_lowercase().as_str() {
        "and" => "&&".to_owned(),
        "or" => "||".to_owned(),
        "<>" => "!=".to_owned(),
        other => other.to_owned(),
    }
}

#[allow(clippy::too_many_lines)] // one arm per expression kind reads better straight-through
pub(crate) fn lower_expr(n: Node<'_>, src: &str) -> Expr {
    let Some(_level) = Level::enter() else { return Expr::Unknown("deep".to_owned()) };
    match n.kind() {
        "variable_name" => Expr::Var(var_name(n, src).unwrap_or("").to_owned()),
        "dynamic_variable_name" => Expr::DynVar(boxed(kids(n).into_iter().next(), src)),
        "name" | "qualified_name" | "relative_name" | "namespace_name" => Expr::Name(global(name_of(n, src))),
        "relative_scope" => Expr::Name(text(n, src).to_ascii_lowercase()),
        "integer" | "float" => number(n.kind(), text(n, src)),
        "string" => Expr::Lit("Str", unquote(text(n, src)).to_owned()),
        "nowdoc" => Expr::Lit(
            "Str",
            field(n, "value").map(kids).unwrap_or_default().into_iter().map(|k| text(k, src)).collect::<String>(),
        ),
        "encapsed_string" | "heredoc" => interpolated(n, src),
        "shell_command_expression" => match interpolated(n, src) {
            Expr::Interp(parts) => Expr::Shell(parts),
            literal => Expr::Shell(vec![literal]),
        },
        "boolean" => Expr::Bool(text(n, src).eq_ignore_ascii_case("true")),
        "null" => Expr::Null,
        "binary_expression" => {
            let op = field(n, "operator").map_or_else(|| "?op".to_owned(), |o| operator(text(o, src)));
            // What `instanceof` tests against is a class, named as written.
            let r = if op == "instanceof" { Box::new(scope(field(n, "right"), src)) } else { boxed(field(n, "right"), src) };
            Expr::Binary { op, l: boxed(field(n, "left"), src), r }
        }
        "unary_op_expression" => Expr::Unary {
            op: field(n, "operator").map_or("?un", |o| o.kind()).to_owned(),
            e: boxed(field(n, "argument").or_else(|| kids(n).into_iter().last()), src),
        },
        "update_expression" => Expr::Update {
            inc: field(n, "operator").is_some_and(|o| o.kind() == "++"),
            prefix: n.child(0).is_some_and(|first| !first.is_named()),
            e: boxed(field(n, "argument"), src),
        },
        "assignment_expression" => Expr::Assign { op: "=".to_owned(), l: boxed(field(n, "left"), src), r: boxed(field(n, "right"), src) },
        "reference_assignment_expression" => {
            Expr::Assign { op: "=&".to_owned(), l: boxed(field(n, "left"), src), r: boxed(field(n, "right"), src) }
        }
        "augmented_assignment_expression" => Expr::Assign {
            op: field(n, "operator").map_or("=", |o| o.kind()).to_owned(),
            l: boxed(field(n, "left"), src),
            r: boxed(field(n, "right"), src),
        },
        "conditional_expression" => Expr::Cond {
            c: boxed(field(n, "condition"), src),
            t: field(n, "body").map(|t| Box::new(lower_expr(t, src))),
            f: boxed(field(n, "alternative"), src),
        },
        "match_expression" => Expr::Match {
            subject: Box::new(cond_of(field(n, "condition"), src)),
            arms: field(n, "body")
                .map(kids)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|arm| {
                    let value = field(arm, "return_expression").map(|v| lower_expr(v, src))?;
                    let conds = match arm.kind() {
                        "match_conditional_expression" => Some(
                            field(arm, "conditional_expressions").map(kids).unwrap_or_default().into_iter().map(|c| lower_expr(c, src)).collect(),
                        ),
                        _ => None,
                    };
                    Some(Arm { conds, value })
                })
                .collect(),
        },
        "function_call_expression" => {
            let f = match field(n, "function") {
                Some(name) if matches!(name.kind(), "name" | "qualified_name" | "relative_name") => {
                    let spelled = global(name_of(name, src));
                    // `die` is `exit`, by another name.
                    Expr::Name(if spelled.eq_ignore_ascii_case("die") { "exit".to_owned() } else { spelled })
                }
                other => *boxed(other, src),
            };
            Expr::Call { f: Box::new(f), args: lower_args(field(n, "arguments"), src) }
        }
        "member_call_expression" | "nullsafe_member_call_expression" => Expr::Method {
            recv: boxed(field(n, "object"), src),
            name: member(field(n, "name"), src),
            args: lower_args(field(n, "arguments"), src),
            nullsafe: n.kind().starts_with("nullsafe"),
        },
        "scoped_call_expression" => Expr::StaticCall {
            scope: Box::new(scope(field(n, "scope"), src)),
            name: member(field(n, "name"), src),
            args: lower_args(field(n, "arguments"), src),
        },
        "member_access_expression" | "nullsafe_member_access_expression" => Expr::Prop {
            e: boxed(field(n, "object"), src),
            name: member(field(n, "name"), src),
            nullsafe: n.kind().starts_with("nullsafe"),
        },
        "scoped_property_access_expression" => Expr::StaticProp {
            scope: Box::new(scope(field(n, "scope"), src)),
            name: field(n, "name").and_then(|v| var_name(v, src)).unwrap_or("").to_owned(),
        },
        "class_constant_access_expression" => {
            let parts = kids(n);
            Expr::ClassConst {
                scope: Box::new(scope(parts.first().copied(), src)),
                name: parts.get(1).map_or_else(String::new, |k| text(*k, src).to_owned()),
            }
        }
        "subscript_expression" => {
            let parts = kids(n);
            Expr::Index { e: boxed(parts.first().copied(), src), i: parts.get(1).map(|i| Box::new(lower_expr(*i, src))) }
        }
        "array_creation_expression" => Expr::Array(lower_elems(n, src)),
        "list_literal" => Expr::List(lower_elems(n, src)),
        "object_creation_expression" => {
            let parts = kids(n);
            match parts.first() {
                Some(c) if c.kind() == "anonymous_class" => Expr::AnonClass,
                Some(c) => Expr::New {
                    class: Box::new(match c.kind() {
                        "name" | "qualified_name" | "relative_name" => Expr::Name(name_of(*c, src)),
                        _ => lower_expr(*c, src),
                    }),
                    args: lower_args(parts.iter().copied().find(|k| k.kind() == "arguments"), src),
                },
                None => Expr::Unknown(String::new()),
            }
        }
        "anonymous_class" => Expr::AnonClass,
        "anonymous_function" => {
            let f = lower_func(n, src);
            let uses = child_of_kind(n, "anonymous_function_use_clause")
                .map(kids)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|u| match u.kind() {
                    "by_ref" => kids(u).into_iter().next().and_then(|v| var_name(v, src)).map(|v| (v.to_owned(), true)),
                    _ => var_name(u, src).map(|v| (v.to_owned(), false)),
                })
                .collect();
            Expr::Closure { params: f.params, uses, body: f.body }
        }
        "arrow_function" => Expr::Arrow { params: lower_params(field(n, "parameters"), src), body: boxed(field(n, "body"), src) },
        "cast_expression" => Expr::Cast {
            ty: field(n, "type").map_or_else(String::new, |t| cast_type(text(t, src))),
            e: boxed(field(n, "value"), src),
        },
        "error_suppression_expression" => Expr::Silence(boxed(kids(n).into_iter().next(), src)),
        "clone_expression" => Expr::Clone(boxed(kids(n).into_iter().next(), src)),
        "print_intrinsic" => Expr::Print(boxed(kids(n).into_iter().next(), src)),
        "include_expression" | "include_once_expression" | "require_expression" | "require_once_expression" => Expr::Include {
            kind: match n.kind() {
                "include_expression" => "include",
                "include_once_expression" => "include_once",
                "require_expression" => "require",
                _ => "require_once",
            },
            e: boxed(kids(n).into_iter().next(), src),
        },
        "yield_expression" => {
            if text(n, src).get(..10).is_some_and(|head| head.eq_ignore_ascii_case("yield from")) {
                return Expr::YieldFrom(boxed(kids(n).into_iter().next(), src));
            }
            match kids(n).into_iter().next() {
                Some(init) if init.kind() == "array_element_initializer" => match kids(init).as_slice() {
                    [k, v] => Expr::Yield { key: Some(Box::new(lower_expr(*k, src))), value: Some(Box::new(lower_expr(*v, src))) },
                    [v] => Expr::Yield { key: None, value: Some(Box::new(lower_expr(*v, src))) },
                    _ => Expr::Yield { key: None, value: None },
                },
                Some(v) => Expr::Yield { key: None, value: Some(Box::new(lower_expr(v, src))) },
                None => Expr::Yield { key: None, value: None },
            }
        }
        "throw_expression" => Expr::Throw(boxed(kids(n).into_iter().next(), src)),
        "parenthesized_expression" => Expr::Paren(boxed(kids(n).into_iter().next(), src)),
        "sequence_expression" => {
            let parts = kids(n);
            Expr::Comma(boxed(parts.first().copied(), src), boxed(parts.get(1).copied(), src))
        }
        "variadic_unpacking" => Expr::Spread(boxed(kids(n).into_iter().next(), src)),
        "by_ref" => Expr::Unary { op: "&".to_owned(), e: boxed(kids(n).into_iter().next(), src) },
        "array_element_initializer" => kids(n).into_iter().last().map_or(Expr::Unknown(String::new()), |v| lower_expr(v, src)),
        other => Expr::Unknown(other.to_owned()),
    }
}
