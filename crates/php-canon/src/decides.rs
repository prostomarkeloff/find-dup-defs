//! What a PHP definition **decides on** — the frontend half of [`Facets::decides`].
//!
//! PHP says which expressions are propositions, and this module only reads them off — by the rules the
//! other frontends read their own, written in the atom grammar all of them share
//! ([`dup_defs_core::decide`]):
//!
//! - every **comparison** — `==`, `===`, `!=`, `!==` as one relation, the four orders (and `<=>`) as
//!   another;
//! - every expression in a position PHP **tests for truth** — the condition of an `if`, a loop, a `?:`,
//!   the operand of `!`, the left of `&&` and `||` (and the right when the whole is tested), what an
//!   `assert` asserts, what `empty` tests;
//! - a `switch`'s cases and a `match`'s arms as the equalities they abbreviate;
//! - the library's tests as the relation they compute: `isset($x)` and `is_null($x)` are `None = x`,
//!   `in_array($x, $xs)` and `array_key_exists($k, $a)` are membership, `strpos($s, $x) !== false` and
//!   `str_contains($s, $x)` are `x in s`, `strcmp($a, $b) === 0` is `a = b`; `$x instanceof C` and
//!   `is_string($x)` are membership in a type, as Python's `isinstance` is.
//!
//! **Absence is one test however spelled.** `$x === null`, `is_null($x)` and `!isset($x)` are
//! `None = x`, the atom Python's `x is None` writes.
//!
//! A term's identity follows the binding the language gives its name: a variable is a hole — every one a
//! function names is its own — and a parameter bound nowhere else is marked as one; a field read off
//! `$this` is a field of the value the method is read off, as the Go frontend reads a receiver's. A
//! function, constant or class is the program's, resolved through the file's namespace and imports
//! ([`crate::names`]); `self::X` is the class the body is in. A constant used as an operand is written as
//! a value ([`decide::value_path`]): what PHP compares is a value, whatever its spelling.
//!
//! [`Facets::decides`]: dup_defs_core::Facets::decides

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::decide::{self, DecisionFacts};

use crate::canon::{each_stmt_part, for_each_child, is_builtin_type, Child, NameKind, Part};
use crate::ir::{Arg, Elem, Expr, Func, Member, Stmt, Type};
use crate::names::{dotted, Names};

/// The functions that compare their first two arguments as `strcmp` does: zero when equal, the sign of
/// the order otherwise.
const COMPARATORS: &[&str] = &["strcmp", "strcasecmp", "strncmp", "strncasecmp", "strnatcmp", "strnatcasecmp", "strcoll", "version_compare"];

/// The functions that find their second argument in their first: not `false` when it is there.
const FINDERS: &[&str] = &["strpos", "stripos", "strrpos", "strripos", "strstr", "stristr", "strrchr", "mb_strpos", "mb_stripos"];

/// The functions that answer whether their second argument holds their first, or their first holds
/// their second (`*_contains`, `str_starts_with`): membership, as the relation they compute.
fn membership(callee: &str) -> Option<bool> {
    Some(match callee {
        // (element, container)
        "in_array" | "array_key_exists" | "key_exists" | "property_exists" | "method_exists" => true,
        // (container, element)
        "str_contains" | "str_starts_with" | "str_ends_with" => false,
        _ => return None,
    })
}

/// `is_string($x)` and kin: the type each tests for, as `instanceof` names one.
fn type_test(callee: &str) -> Option<&'static str> {
    Some(match callee {
        "is_array" => "array",
        "is_string" => "string",
        "is_int" | "is_integer" | "is_long" => "int",
        "is_float" | "is_double" => "float",
        "is_bool" => "bool",
        "is_object" => "object",
        "is_callable" => "callable",
        "is_iterable" => "iterable",
        "is_numeric" => "numeric",
        "is_scalar" => "scalar",
        "is_resource" => "resource",
        "is_countable" => "countable",
        _ => return None,
    })
}

/// A callee's name as the library knows it: lowercased, its leading `\` off.
fn library(f: &Expr) -> Option<String> {
    match f {
        Expr::Name(n) => Some(n.trim_start_matches('\\').to_ascii_lowercase()),
        _ => None,
    }
}

/// What the names of one namespace block stand for, read once per file and shared by its definitions.
pub(crate) struct FileScope<'n> {
    pub(crate) names: &'n Names,
    /// The class the definition is in, fully qualified — what `self::` and `static::` name.
    pub(crate) class: Option<String>,
    /// The class it extends, fully qualified — what `parent::` names.
    pub(crate) parent: Option<String>,
}

impl FileScope<'_> {
    /// The dotted path a name of `kind` stands for.
    fn path_of(&self, kind: NameKind, spelled: &str) -> String {
        match spelled.to_ascii_lowercase().as_str() {
            "self" | "static" => return self.class.as_deref().map_or_else(|| "self".to_owned(), dotted),
            "parent" => return self.parent.as_deref().map_or_else(|| "parent".to_owned(), dotted),
            _ => {}
        }
        dotted(&self.names.resolve(kind, spelled).0)
    }

    /// A definition's identity: what the corpus's calls name it by.
    pub(crate) fn identity(&self, name: &str) -> Arc<str> {
        Arc::from(decide::path(&dotted(&self.names.declared(name))))
    }

    /// A method's identity: its class's path, and the method.
    pub(crate) fn method_identity(&self, class: &str, method: &str) -> Arc<str> {
        Arc::from(decide::path(&format!("{}.{method}", dotted(&self.names.declared(class)))))
    }

    /// The class a declared type names, as a path — through `?T`, the way Python reads `Optional[C]` as
    /// `C`. A built-in type, or a union of classes, is no one class.
    fn type_class(&self, ty: &Type) -> Option<String> {
        match ty {
            Type::Named(n) if !is_builtin_type(n) => Some(decide::path(&self.path_of(NameKind::Class, n))),
            Type::Nullable(inner) => self.type_class(inner),
            _ => None,
        }
    }
}

/// A member chain from its root outwards: the variable or class it starts from, and the members read off
/// it. `$a->b?->c` is the chain `a`, `[b, c]`; parentheses and casts are seen through.
fn chain(mut e: &Expr) -> (Option<&Expr>, Vec<&str>) {
    let mut attrs = Vec::new();
    loop {
        match e {
            Expr::Prop { e: inner, name: Member::Name(n), .. } => {
                attrs.push(n.as_str());
                e = inner;
            }
            Expr::Paren(inner) | Expr::Cast { e: inner, .. } | Expr::Silence(inner) => e = inner,
            _ => break,
        }
    }
    attrs.reverse();
    match e {
        Expr::Var(_) | Expr::Name(_) | Expr::ClassConst { .. } | Expr::StaticProp { .. } => (Some(e), attrs),
        _ => (None, attrs),
    }
}

/// The text of a literal, or `None` for anything that is not one.
fn literal(e: &Expr) -> Option<String> {
    Some(match e {
        Expr::Lit("Int", t) => decide::integer(t),
        Expr::Lit("Float", t) => {
            let clean: String = t.chars().filter(|c| *c != '_').collect();
            clean.parse::<f64>().map_or_else(|_| decide::literal(t), decide::number)
        }
        Expr::Lit(_, t) => decide::literal(t),
        Expr::Bool(b) => String::from(if *b { "True" } else { "False" }),
        Expr::Null => "None".to_owned(),
        Expr::Paren(inner) => return literal(inner),
        Expr::Unary { op, e: inner } if op == "-" => format!("-{}", literal(inner)?),
        _ => return None,
    })
}

fn is_false(e: &Expr) -> bool {
    matches!(strip(e), Expr::Bool(false))
}

fn is_zero(e: &Expr) -> bool {
    literal(strip(e)).is_some_and(|l| l == "0")
}

fn operator(op: &str) -> bool {
    matches!(op, "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" | "**" | ".")
}

/// Parentheses, casts and `@` off: `(int) strcmp($a, $b)` compares as `strcmp($a, $b)`.
fn strip(mut e: &Expr) -> &Expr {
    while let Expr::Paren(inner) | Expr::Cast { e: inner, .. } | Expr::Silence(inner) = e {
        e = inner;
    }
    e
}

/// The positional arguments of a call, when it has no named or spread ones.
fn positional(args: &[Arg]) -> Option<Vec<&Expr>> {
    args.iter().map(|a| (a.name.is_none() && !matches!(a.value, Expr::Spread(_))).then_some(&a.value)).collect()
}

/// Every variable a definition binds, how many times each is assigned, and the single-variable
/// assignments with one value — what a variable assigned once may be read as.
#[derive(Default)]
struct Locals {
    sites: HashMap<String, usize>,
    lets: Vec<(String, Expr)>,
}

impl Locals {
    fn assigned(&mut self, target: &Expr) {
        match target {
            Expr::Var(n) => *self.sites.entry(n.clone()).or_insert(0) += 1,
            Expr::List(elems) => {
                for el in elems {
                    match el {
                        Elem::Plain(t) | Elem::Pair(_, t) => self.assigned(t),
                    }
                }
            }
            Expr::Unary { op, e } if op == "&" => self.assigned(e),
            _ => {}
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Assign { op, lhs, rhs } => {
                if let (Expr::Var(n), "=") = (lhs, op.as_str()) {
                    self.lets.push((n.clone(), rhs.clone()));
                }
                self.assigned(lhs);
                self.expr(rhs);
            }
            Stmt::IncDec { target, .. } => self.assigned(target),
            Stmt::Foreach { subject, key, value, body } => {
                self.expr(subject);
                // A loop variable takes a value on every pass.
                for t in key.iter().chain([value]) {
                    self.assigned(t);
                    self.assigned(t);
                }
                self.stmts(body);
            }
            Stmt::Global(names) => {
                for n in names {
                    *self.sites.entry(n.clone()).or_insert(0) += 2;
                }
            }
            Stmt::Static(vars) => {
                for (n, _) in vars {
                    *self.sites.entry(n.clone()).or_insert(0) += 2;
                }
            }
            Stmt::Try { catches, .. } => {
                for c in catches {
                    if let Some(v) = &c.var {
                        self.assigned(&Expr::Var(v.clone()));
                    }
                }
                self.parts(s);
            }
            _ => self.parts(s),
        }
    }

    fn parts(&mut self, s: &Stmt) {
        each_stmt_part(s, &mut |part| match part {
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Assign { l, .. } | Expr::Update { e: l, .. } => self.assigned(l),
            Expr::Unary { op, e: inner } if op == "&" => self.assigned(inner),
            Expr::Closure { uses, .. } => {
                for (n, by_ref) in uses {
                    if *by_ref {
                        *self.sites.entry(n.clone()).or_insert(0) += 2;
                    }
                }
            }
            _ => {}
        }
        for_each_child(e, &mut |c| match c {
            Child::Expr(x) => self.expr(x),
            Child::Body(b) => self.stmts(b),
        });
    }
}

/// The walk: one definition's propositions, as atoms.
struct Atoms<'s, 'n> {
    scope: &'s FileScope<'n>,
    /// Variables assigned once to a value fixed by the text: read wherever they appear as that value.
    constants: HashMap<String, String>,
    /// Variables assigned once to a call's result: read bare, they are that computation.
    computed: HashMap<String, String>,
    /// Parameters bound nowhere but by the call, read bare as themselves (see [`decide::param`]).
    params: HashSet<String>,
    /// Variables whose class the text names — a parameter's declared type, an object built in place. A
    /// field read off one is a typed field ([`decide::typed_field`]).
    types: HashMap<String, String>,
    out: Vec<Arc<str>>,
}

impl Atoms<'_, '_> {
    fn push(&mut self, atom: Option<Arc<str>>) {
        if let Some(atom) = atom {
            self.out.push(atom);
        }
    }

    /// The identity a chain rooted at `root` reads as. `as_value` writes a constant as the value it must
    /// be where it stands — an operand ([`decide::value_path`]); a callee or a mention keeps the plain
    /// path.
    fn chain_term(&self, root: &Expr, attrs: &[&str], as_value: bool) -> String {
        match root {
            Expr::Var(v) => {
                if let Some(value) = self.constants.get(v) {
                    return if attrs.is_empty() { value.clone() } else { decide::field(attrs) };
                }
                if attrs.is_empty() {
                    if let Some(computation) = self.computed.get(v) {
                        return computation.clone();
                    }
                    if self.params.contains(v) {
                        return decide::param(v);
                    }
                } else if let Some(class) = self.types.get(v) {
                    return decide::typed_chain(class, attrs, |_, _| None);
                }
                decide::field(attrs)
            }
            Expr::Name(n) => {
                let path = dup_defs_core::decide::dotted(&self.scope.path_of(NameKind::Constant, n), attrs);
                if as_value {
                    decide::value_path(&path)
                } else {
                    decide::path(&path)
                }
            }
            Expr::ClassConst { scope, name } => {
                let Expr::Name(class) = scope.as_ref() else { return decide::field(attrs) };
                let class = self.scope.path_of(NameKind::Class, class);
                if name.eq_ignore_ascii_case("class") {
                    // `Order::class` is the class's name: the class itself.
                    return decide::path(&class);
                }
                let path = dup_defs_core::decide::dotted(&format!("{class}.{name}"), attrs);
                if as_value {
                    decide::value_path(&path)
                } else {
                    decide::path(&path)
                }
            }
            Expr::StaticProp { scope, name } => {
                let Expr::Name(class) = scope.as_ref() else { return decide::field(attrs) };
                let class = self.scope.path_of(NameKind::Class, class);
                let mut rest = vec![name.as_str()];
                rest.extend_from_slice(attrs);
                // A static property is state, not a constant: a field of the class.
                decide::typed_chain(&decide::path(&class), &rest, |_, _| None)
            }
            _ => decide::field(attrs),
        }
    }

    /// What a callee is called by: a function's path; `Class::method` the class's path and the method.
    fn callee(&self, e: &Expr) -> Option<String> {
        match e {
            Expr::Call { f, .. } => match f.as_ref() {
                Expr::Name(n) => Some(decide::path(&self.scope.path_of(NameKind::Function, n))),
                _ => None,
            },
            Expr::StaticCall { scope, name: Member::Name(m), .. } => match scope.as_ref() {
                Expr::Name(class) => Some(decide::path(&format!("{}.{m}", self.scope.path_of(NameKind::Class, class)))),
                _ => None,
            },
            // `$this->helper()` calls the method of the class the body is in.
            Expr::Method { recv, name: Member::Name(m), .. } if matches!(recv.as_ref(), Expr::Var(v) if v == "this") => {
                self.scope.class.as_deref().map(|c| decide::path(&format!("{}.{m}", dotted(c))))
            }
            _ => None,
        }
    }

    /// The identity of an operand.
    fn term(&self, e: &Expr) -> String {
        if let Some(lit) = literal(e) {
            return lit;
        }
        match e {
            Expr::Paren(inner) | Expr::Cast { e: inner, .. } | Expr::Silence(inner) => self.term(inner),
            Expr::Var(_) | Expr::Name(_) | Expr::Prop { .. } | Expr::ClassConst { .. } | Expr::StaticProp { .. } => match chain(e) {
                (Some(root), attrs) => self.chain_term(root, &attrs, true),
                (None, attrs) => decide::field(&attrs),
            },
            Expr::Call { args, .. } | Expr::StaticCall { args, .. } | Expr::Method { args, .. } => match self.callee(e) {
                Some(callee) => decide::call(&callee, args.iter().filter_map(|a| self.constant(&a.value)).collect()),
                None => match e {
                    Expr::Method { name: Member::Name(m), .. } => decide::call(&decide::field(&[m.as_str()]), Vec::new()),
                    _ => decide::HOLE.to_owned(),
                },
            },
            Expr::New { class, args } => match class.as_ref() {
                Expr::Name(n) => decide::call(
                    &decide::path(&self.scope.path_of(NameKind::Class, n)),
                    args.iter().filter_map(|a| self.constant(&a.value)).collect(),
                ),
                _ => decide::HOLE.to_owned(),
            },
            Expr::Array(elems) => decide::container(
                elems
                    .iter()
                    .map(|el| match el {
                        Elem::Plain(v) | Elem::Pair(_, v) => self.term(v),
                    })
                    .collect(),
            ),
            Expr::Index { e: base, i } => {
                let key = i.as_deref().and_then(|i| self.constant(i)).unwrap_or_else(|| decide::HOLE.to_owned());
                format!("{}[{key}]", self.term(base))
            }
            Expr::Binary { op, l, r } if operator(op) => {
                let rendered = format!("({} {op} {})", self.term(l), self.term(r));
                if decide::has_identity(&rendered) {
                    rendered
                } else {
                    decide::HOLE.to_owned()
                }
            }
            Expr::Unary { op, e: inner } if op == "-" || op == "~" => {
                let inner = self.term(inner);
                if decide::has_identity(&inner) {
                    format!("{op}{inner}")
                } else {
                    inner
                }
            }
            _ => decide::HOLE.to_owned(),
        }
    }

    /// A value handed to a call that is fixed by the text rather than computed: a literal, a constant, a
    /// variable assigned one.
    fn constant(&self, e: &Expr) -> Option<String> {
        if let Some(lit) = literal(e) {
            return Some(lit);
        }
        match strip(e) {
            Expr::Var(v) => self.constants.get(v).cloned(),
            Expr::Name(_) | Expr::ClassConst { .. } => match chain(strip(e)) {
                (Some(root), attrs) => Some(self.chain_term(root, &attrs, true)),
                (None, _) => None,
            },
            _ => None,
        }
    }

    /// The comparison function a call is, with its two operands.
    fn compared(e: &Expr) -> Option<(&Expr, &Expr)> {
        let Expr::Call { f, args } = strip(e) else { return None };
        let name = library(f)?;
        let args = positional(args)?;
        (COMPARATORS.contains(&name.as_str()) && args.len() >= 2).then(|| (args[0], args[1]))
    }

    /// The search function a call is, with what it searches in and for.
    fn found(e: &Expr) -> Option<(&Expr, &Expr)> {
        let Expr::Call { f, args } = strip(e) else { return None };
        let name = library(f)?;
        let args = positional(args)?;
        (FINDERS.contains(&name.as_str()) && args.len() >= 2).then(|| (args[0], args[1]))
    }

    /// A library call that is itself a proposition: the atoms it computes — `isset($a, $b)` tests each
    /// of its arguments. `None` for any other call.
    fn library_test(&self, e: &Expr) -> Option<Vec<Option<Arc<str>>>> {
        let Expr::Call { f, args } = strip(e) else { return None };
        let name = library(f)?;
        let args = positional(args)?;
        Some(match (name.as_str(), args.as_slice()) {
            ("isset" | "is_null", [_, ..]) => args.iter().map(|a| decide::equal("None", &self.term(a))).collect(),
            ("empty", [x]) => vec![decide::truth(&self.term(x))],
            // `in_array($x, [A, B])` is `$x == A || $x == B` — membership in a literal array is the
            // disjunction of equalities it abbreviates.
            ("in_array", [x, Expr::Array(elems), ..]) if elems.iter().all(|e| matches!(e, Elem::Plain(_))) => {
                let element = self.term(x);
                elems
                    .iter()
                    .map(|e| match e {
                        Elem::Plain(v) | Elem::Pair(_, v) => decide::equal(&element, &self.term(v)),
                    })
                    .collect()
            }
            (callee, [a, b, ..]) if membership(callee).is_some() => {
                let element_first = membership(callee) == Some(true);
                let (element, container) = if element_first { (a, b) } else { (b, a) };
                vec![decide::member(&self.term(element), &self.term(container))]
            }
            (callee, [x]) if type_test(callee).is_some() => {
                vec![decide::member(&self.term(x), &decide::path(type_test(callee).unwrap_or_default()))]
            }
            _ => return None,
        })
    }

    /// An expression in a position PHP tests for truth.
    fn truth(&mut self, e: &Expr) {
        let e = strip(e);
        if let Some(atoms) = self.library_test(e) {
            for a in atoms {
                self.push(a);
            }
            return;
        }
        if let Some((hay, needle)) = Self::found(e) {
            let atom = decide::member(&self.term(needle), &self.term(hay));
            self.push(atom);
            return;
        }
        match e {
            // Structure, not a proposition: its parts are read where the walk reaches them.
            Expr::Binary { op, .. } if matches!(op.as_str(), "==" | "!=" | "===" | "!==" | "<" | "<=" | ">" | ">=" | "<=>" | "instanceof") => {}
            Expr::Unary { op, .. } if op == "!" => {}
            // `&&` and `||` test their left operand and hand the right one back as their value — so the
            // right one is tested exactly when the whole expression is, which is now.
            Expr::Binary { op, r, .. } if op == "&&" || op == "||" || op == "xor" => self.truth(r),
            // An assignment tested is the value assigned: `if ($row = fetch())`.
            Expr::Assign { r, .. } => self.truth(r),
            _ => {
                let atom = decide::truth(&self.term(e));
                self.push(atom);
            }
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => self.truth(cond),
            Stmt::For { cond, .. } => {
                if let Some(last) = cond.last() {
                    self.truth(last);
                }
            }
            Stmt::Switch { value, cases } => {
                // A case is the equality it abbreviates.
                for v in cases.iter().filter_map(|c| c.value.as_ref()) {
                    let atom = decide::equal(&self.term(value), &self.term(v));
                    self.push(atom);
                }
            }
            _ => {}
        }
        each_stmt_part(s, &mut |part| match part {
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Binary { op, l, r } => {
                let (left, right) = (strip(l), strip(r));
                let atom = match op.as_str() {
                    "==" | "!=" | "===" | "!==" => {
                        // `strcmp($a, $b) === 0`, `strpos($s, $x) !== false`: the function's relation.
                        let compared = Self::compared(left).or_else(|| Self::compared(right));
                        let found = Self::found(left).or_else(|| Self::found(right));
                        match (compared, found) {
                            (Some((a, b)), _) if is_zero(left) || is_zero(right) => decide::equal(&self.term(a), &self.term(b)),
                            (_, Some((hay, needle))) if is_false(left) || is_false(right) => {
                                decide::member(&self.term(needle), &self.term(hay))
                            }
                            _ => decide::equal(&self.term(left), &self.term(right)),
                        }
                    }
                    "<" | "<=" | ">" | ">=" | "<=>" => match Self::compared(left).or_else(|| Self::compared(right)) {
                        Some((a, b)) if is_zero(left) || is_zero(right) => decide::order(&self.term(a), &self.term(b)),
                        _ => decide::order(&self.term(left), &self.term(right)),
                    },
                    "instanceof" => match right {
                        Expr::Name(n) => decide::member(&self.term(left), &decide::path(&self.scope.path_of(NameKind::Class, n))),
                        _ => None,
                    },
                    "&&" | "||" | "xor" => {
                        self.truth(left);
                        None
                    }
                    _ => None,
                };
                self.push(atom);
            }
            Expr::Unary { op, e: inner } if op == "!" => self.truth(inner),
            Expr::Cond { c, .. } => self.truth(c),
            Expr::Match { subject, arms } => {
                // An arm is the equalities it lists — `match (true)` lists the conditions themselves.
                let on_true = matches!(strip(subject), Expr::Bool(true));
                for c in arms.iter().filter_map(|a| a.conds.as_ref()).flatten() {
                    if on_true {
                        self.truth(c);
                    } else {
                        let atom = decide::equal(&self.term(subject), &self.term(c));
                        self.push(atom);
                    }
                }
            }
            Expr::Call { f, args } if library(f).is_some_and(|n| n == "assert") => {
                if let Some(a) = args.first() {
                    self.truth(&a.value);
                }
            }
            Expr::Call { .. } => {
                // A library test is a proposition wherever it stands: `$ok = in_array($x, $xs);`.
                if let Some(atoms) = self.library_test(e) {
                    for a in atoms {
                self.push(a);
            }
                }
            }
            _ => {}
        }
        for_each_child(e, &mut |c| match c {
            Child::Expr(x) => self.expr(x),
            Child::Body(b) => self.stmts(b),
        });
    }
}

/// Calls and value reads, by the same identities the atoms are written in.
struct Refs<'w, 's, 'n> {
    walk: &'w Atoms<'s, 'n>,
    calls: Vec<decide::CallSite>,
    mentions: Vec<Arc<str>>,
    /// Properties written other than through a constructor the corpus can name.
    loose: Vec<Arc<str>>,
}

impl Refs<'_, '_, '_> {
    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        if let Stmt::Assign { lhs: Expr::Prop { name: Member::Name(n), .. }, .. } = s {
            self.loose.push(Arc::from(n.as_str()));
        }
        each_stmt_part(s, &mut |part| match part {
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    fn site(&mut self, callee: String, args: &[Arg]) {
        let open = args.iter().any(|a| matches!(a.value, Expr::Spread(_)));
        let args = args.iter().map(|a| (a.name.as_deref().map(Arc::from), Arc::from(self.walk.term(&a.value)))).collect();
        self.calls.push(decide::CallSite { callee: Arc::from(callee), args, open });
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Call { args, .. } | Expr::StaticCall { args, .. } | Expr::Method { args, .. } => {
                if let Some(callee) = self.walk.callee(e) {
                    self.site(callee, args);
                }
            }
            // A constructor call: the class is called with the arguments its constructor takes.
            Expr::New { class, args } => {
                if let Expr::Name(n) = class.as_ref() {
                    let callee = decide::path(&self.walk.scope.path_of(NameKind::Class, n));
                    self.site(callee, args);
                }
            }
            Expr::Assign { l, .. } => {
                if let Expr::Prop { name: Member::Name(n), .. } = l.as_ref() {
                    self.loose.push(Arc::from(n.as_str()));
                }
            }
            // `Order::class`, and a constant read as a value: what is named here is read from elsewhere.
            Expr::ClassConst { scope, name } if name.eq_ignore_ascii_case("class") => {
                if let Expr::Name(n) = scope.as_ref() {
                    self.mentions.push(Arc::from(decide::path(&self.walk.scope.path_of(NameKind::Class, n))));
                }
            }
            _ => {}
        }
        for_each_child(e, &mut |c| match c {
            Child::Expr(x) => self.expr(x),
            Child::Body(b) => self.stmts(b),
        });
    }
}

/// Everything the decisions pass reads off one function or method. `open` — it is reached by calls the
/// corpus cannot show (a method, dispatched through every class that inherits it; a hook callback), so
/// what its parameters hold is not the corpus's to say.
pub(crate) fn decides(f: &Func, scope: &FileScope<'_>, open: bool) -> DecisionFacts {
    let mut locals = Locals::default();
    for p in &f.params {
        locals.assigned(&Expr::Var(p.name.clone()));
        if p.by_ref {
            locals.assigned(&Expr::Var(p.name.clone()));
        }
    }
    locals.stmts(&f.body);
    let order: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
    let untouched: HashSet<String> = order.iter().filter(|name| locals.sites.get(*name) == Some(&1)).cloned().collect();
    let mut types: HashMap<String, String> =
        f.params.iter().filter_map(|p| Some((p.name.clone(), scope.type_class(p.ty.as_ref()?)?))).collect();
    let mut single = locals.lets;
    single.retain(|(name, _)| locals.sites.get(name) == Some(&1));
    for (name, value) in &single {
        if let Expr::New { class, .. } = strip(value) {
            if let Expr::Name(n) = class.as_ref() {
                types.entry(name.clone()).or_insert_with(|| decide::path(&scope.path_of(NameKind::Class, n)));
            }
        }
    }
    let mut walk = Atoms { scope, constants: HashMap::new(), computed: HashMap::new(), params: untouched, types, out: Vec::new() };
    // A binding whose value is itself a constant — possibly through another such binding — is that
    // constant. Resolved to a fixed point: each round can only add, and there are finitely many names.
    loop {
        let found: Vec<(String, String)> = single
            .iter()
            .filter(|(name, _)| !walk.constants.contains_key(name))
            .filter_map(|(name, value)| walk.constant(value).map(|c| (name.clone(), c)))
            .collect();
        if found.is_empty() {
            break;
        }
        walk.constants.extend(found);
    }
    for (name, value) in &single {
        if !walk.constants.contains_key(name) && matches!(strip(value), Expr::Call { .. } | Expr::StaticCall { .. }) {
            let computation = walk.term(value);
            walk.computed.insert(name.clone(), computation);
        }
    }
    walk.stmts(&f.body);
    let mut refs = Refs { walk: &walk, calls: Vec::new(), mentions: Vec::new(), loose: Vec::new() };
    refs.stmts(&f.body);
    let (calls, mut mentions, mut loose) = (refs.calls, refs.mentions, refs.loose);
    loose.sort_unstable();
    loose.dedup();
    mentions.sort_unstable();
    mentions.dedup();
    let mut atoms = walk.out;
    atoms.sort_unstable();
    atoms.dedup();
    let variadic = f.params.iter().any(|p| p.variadic);
    let params = if open || variadic { Vec::new() } else { order.iter().map(|p| Arc::from(p.as_str())).collect() };
    DecisionFacts { atoms, params, calls, mentions, loose }
}

#[cfg(test)]
mod tests {
    use crate::defs::scan_source;
    use dup_defs_core::decide::{canonical, render, resolve_typed_fields};
    use dup_defs_core::ScanOpts;
    use std::sync::Arc;

    /// The atoms the decisions pass reads off the last function in `src`, rendered with nothing
    /// inferred (a typed field is the field of some value).
    fn atoms(src: &str) -> Vec<String> {
        let f: Arc<str> = Arc::from("shop/orders.php");
        let src = format!("<?php\nnamespace Shop;\n{src}\n");
        let d = scan_source(&src, &f, &ScanOpts::default())
            .into_iter()
            .rfind(|d| d.kind.fn_like && d.kind.id != "lenses")
            .expect("a function");
        let mut out: Vec<String> =
            d.facets.decides.iter().map(|a| render(&canonical(&resolve_typed_fields(a, |_, _| None)))).collect();
        out.sort();
        out.dedup();
        out
    }

    #[test]
    fn negation_and_the_spelling_of_a_test_do_not_change_the_decision() {
        let refuse = atoms("function refuse($o) { if ($o->status !== Order::OPEN) { return false; } return ship($o); }");
        let accept = atoms("function accept($o) { return Order::OPEN === $o->status; }");
        let switched = atoms("function label($o) { switch ($o->status) { case Order::OPEN: return 'o'; } return ''; }");
        let matched = atoms("function label($o) { return match ($o->status) { Order::OPEN => 'o', default => '' }; }");
        assert_eq!(refuse, vec!["@Shop.Order.OPEN = _.status"]);
        assert_eq!(accept, refuse);
        assert_eq!(switched, refuse);
        assert_eq!(matched, refuse);
    }

    #[test]
    fn absence_is_one_test_however_spelled() {
        let absent = vec!["None = _.paid_at"];
        assert_eq!(atoms("function f($o) { return $o->paid_at === null; }"), absent);
        assert_eq!(atoms("function f($o) { return is_null($o->paid_at); }"), absent);
        assert_eq!(atoms("function f($o) { if (!isset($o->paid_at)) { return 1; } return 2; }"), absent);
    }

    #[test]
    fn the_library_tests_are_the_relation_they_compute() {
        let compared = atoms("function f($name) { if (strcmp($name, 'admin') === 0) { return 1; } return 0; }");
        assert_eq!(compared, vec!["\"admin\" = $name"]);
        let found = atoms("function f($path) { return strpos($path, '..') !== false; }");
        assert_eq!(found, vec!["\"..\" in $path"]);
        assert_eq!(atoms("function f($path) { return str_contains($path, '..'); }"), found);
        let listed = atoms("function f($role) { return in_array($role, ['admin', 'editor'], true); }");
        let compared = atoms("function f($role) { return $role === 'admin' || 'editor' === $role; }");
        assert_eq!(listed, vec!["\"admin\" = $role", "\"editor\" = $role"], "a literal array is the equalities it lists");
        assert_eq!(listed, compared);
        assert_eq!(atoms("function f($k, $map) { return array_key_exists($k, $map); }"), vec!["$k in $map"]);
        let typed = atoms("function f($x) { return $x instanceof Order || is_string($x); }");
        assert_eq!(typed, vec!["$x in @Shop.Order", "$x in @string"]);
    }

    #[test]
    fn a_field_of_this_is_a_field_and_a_yoda_test_is_the_test() {
        let src = "class Order { const OPEN = 'open'; public function isOpen() { return self::OPEN === $this->status; } }";
        assert_eq!(atoms(src), vec!["@Shop.Order.OPEN = _.status"]);
    }

    #[test]
    fn a_parameter_is_marked_and_a_literal_is_its_value() {
        assert_eq!(atoms("function f($force, $o) { if (!$force) { return $o->x; } return 0; }"), vec!["? $force"]);
        let hex = atoms("function f($o) { return $o->flags == 0x10; }");
        assert_eq!(hex, atoms("function f($o) { return $o->flags == 16; }"));
    }
}
