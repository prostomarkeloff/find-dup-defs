//! What a Java definition **decides on** — the frontend half of [`Facets::decides`].
//!
//! Java says which expressions are propositions, and this module only reads them off — by the rules the
//! other frontends read their own, written in the atom grammar all of them share
//! ([`dup_defs_core::decide`]):
//!
//! - every **comparison** — `==` and `!=` as one relation, the four orders as another;
//! - every expression in a position Java **tests for truth** — the condition of an `if`, a `while`, a
//!   `for`, a `?:`, the operand of `!`, the left of `&&` and `||` (and the right when the whole is
//!   tested), what an `assert` asserts, the argument of Guava's `Preconditions.checkArgument` /
//!   `checkState`;
//! - a `switch`'s cases — statement and expression alike — as the equalities they abbreviate;
//! - the library tests as the relation they compute: `.equals` / `.equalsIgnoreCase` /
//!   `Objects.equals` are equality, `.compareTo(b) == 0` is equality and `.compareTo(b) < 0` (or `>`)
//!   is order, `.contains` / `.containsKey` / `.containsValue` / `.startsWith` / `.endsWith` are
//!   membership, `.indexOf(x) >= 0` and `.indexOf(x) != -1` are membership, `instanceof` is membership
//!   in a type, and `Objects.requireNonNull(x)` / Guava's `checkNotNull(x)` are absence.
//!
//! **Absence is one test however spelled.** `x == null`, `x != null`, `Objects.requireNonNull(x)` and
//! `checkNotNull(x)` are `None = x`, the atom Python's `x is None` and Go's `x == nil` write. The
//! boolean presence tests `.isEmpty`, `.isBlank` and `Optional.isPresent` read the same absence atom
//! (`None = x`): each asks whether the receiver is empty, and the closed world has no dedicated
//! "present" relation — the core grammar's `x is not None` is the closest spelling, and polarity is not
//! part of an atom anyway.
//!
//! A term's identity follows the binding the language gives its name: a name the definition binds is a
//! hole; `this` / `super` is the value the method is read off (a field, as the Go frontend reads a
//! receiver); a name the file neither binds nor imports is its own package's (the class it names), since
//! Java has no other unqualified scope. A class's path is its dotted package-qualified name, so a
//! definition and a `new` of it from another file meet as one path. Java writes constants in
//! `UPPER_SNAKE_CASE` (and enum constants as capitalized members of a capitalized name), which
//! [`decide::names_constant`] already reads — so, unlike Go, a constant needs no [`decide::value_path`]
//! mark: an operand written as a plain path is already read as a value.
//!
//! [`Facets::decides`]: dup_defs_core::Facets::decides
//! [`decide::names_constant`]: dup_defs_core::decide::names_constant

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::decide::{self, dotted, DecisionFacts};

use crate::ir::{Else, Expr, Func, LambdaBody, Param, Stmt, Type};

/// Java's primitive types: names that are no class, so `instanceof` and the typed-field map read none.
const PRIMITIVES: &[&str] = &["int", "long", "short", "byte", "char", "boolean", "float", "double", "void"];

/// What the names of one file stand for, read once per file and shared by its definitions.
pub(crate) struct FileScope {
    /// The file's package, dotted, from its `package` declaration.
    package: String,
    /// What each import the file declares stands for, alias → dotted path.
    imports: HashMap<String, String>,
}

impl FileScope {
    pub(crate) fn new(package: &str, imports: &HashMap<String, Arc<str>>) -> Self {
        Self {
            package: package.to_owned(),
            imports: imports.iter().map(|(k, v)| (k.clone(), v.as_ref().to_owned())).collect(),
        }
    }

    /// A name qualified with the file's package: `Server` → `pkg.Server`.
    fn qualified(&self, name: &str) -> String {
        if self.package.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{name}", self.package)
        }
    }

    /// A definition's identity: what the corpus's calls and reads name it by (`Order.Open` for a
    /// method, `Order` for a class — the caller passes the already-dotted qualified name).
    pub(crate) fn identity(&self, name: &str) -> Arc<str> {
        Arc::from(decide::path(&self.qualified(name)))
    }

    /// The dotted path a name the definition did not bind stands for, `rest` read off it: the package's
    /// own class or member, since Java has no other unqualified scope.
    fn free_path(&self, name: &str, rest: &[&str]) -> String {
        dotted(&self.qualified(name), rest)
    }

    /// The dotted path of a member of an imported package/class, or `None` when `pkg` is not one.
    fn member_path(&self, pkg: &str, member: &str, rest: &[&str]) -> Option<String> {
        let path = self.imports.get(pkg)?;
        let mut all = vec![member];
        all.extend_from_slice(rest);
        Some(dotted(path, &all))
    }

    /// The class a declared type names, as a path — through an array dimension, the way Python reads
    /// `List[C]` as `C`. A primitive is no class.
    fn type_class(&self, ty: &Type) -> Option<String> {
        match ty {
            Type::Named(name) | Type::Generic(name, _) => self.named_class(name),
            Type::Array(inner) => self.type_class(inner),
            _ => None,
        }
    }

    /// A class name resolved to its path, or `None` for a primitive or an empty name.
    fn named_class(&self, name: &str) -> Option<String> {
        if name.is_empty() || PRIMITIVES.contains(&name) {
            return None;
        }
        match name.split_once('.') {
            Some((pkg, member)) => self.member_path(pkg, member, &[]).map(|p| decide::path(&p)),
            None => Some(decide::path(&self.qualified(name))),
        }
    }
}

/// A member chain from its root outwards: the name it starts from, or `None` when it starts at a
/// computed value, and the fields read off it. `Foo.BAR` — a `field_access` whose object is a class
/// name — is the chain `Foo`, `[BAR]`; parentheses and casts are seen through.
fn chain(mut e: &Expr) -> (Option<&str>, Vec<&str>) {
    let mut attrs = Vec::new();
    loop {
        match e {
            Expr::Field { base, name } => {
                attrs.push(name.as_str());
                e = base;
            }
            Expr::StaticField { scope, name } => {
                attrs.push(name.as_str());
                attrs.reverse();
                return (Some(scope.as_str()), attrs);
            }
            Expr::Paren(inner) | Expr::Cast { e: inner, .. } => e = inner,
            _ => break,
        }
    }
    attrs.reverse();
    match e {
        Expr::Name(n) => (Some(n.as_str()), attrs),
        _ => (None, attrs),
    }
}

/// The text of a literal, `None` the absent value, or `None` for anything that is neither.
fn literal(e: &Expr) -> Option<String> {
    Some(match e {
        Expr::Lit(tag, t) if tag == "Num" => {
            // Java spells integers in several bases (`0x1F`, `0b101`, `017`) with digit separators, and
            // a numeric literal carries an optional `l`/`f`/`d` suffix; `decide::integer` normalizes the
            // integer spellings, `decide::number` the floats.
            let clean: String = t.chars().filter(|c| *c != '_').collect();
            let clean = clean.trim_end_matches(['l', 'L', 'f', 'F', 'd', 'D']).to_owned();
            let int = decide::integer(&clean);
            if int.parse::<i128>().is_ok() {
                int
            } else if let Ok(v) = clean.parse::<f64>() {
                decide::number(v)
            } else {
                decide::literal(t)
            }
        }
        Expr::Lit(_, t) => decide::literal(t),
        Expr::Bool(b) => String::from(if *b { "True" } else { "False" }),
        Expr::Null => "None".to_owned(),
        Expr::Paren(inner) => return literal(inner),
        Expr::Unary { op, e: inner } if op == "-" => format!("-{}", literal(inner)?),
        _ => return None,
    })
}

/// Parentheses and casts off: `((int) x)` compares as `x`.
fn strip(mut e: &Expr) -> &Expr {
    while let Expr::Paren(inner) | Expr::Cast { e: inner, .. } = e {
        e = inner;
    }
    e
}

fn is_zero(e: &Expr) -> bool {
    literal(strip(e)).is_some_and(|l| l == "0")
}

fn is_neg_one(e: &Expr) -> bool {
    literal(strip(e)).is_some_and(|l| l == "-1")
}

fn operator(op: &str) -> bool {
    matches!(op, "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" | ">>>")
}

/// The operands of a `compareTo` call, when it is one: `a.compareTo(b)` → `(a, b)`.
fn compared(e: &Expr) -> Option<(&Expr, &Expr)> {
    let Expr::Method { recv, name, args } = strip(e) else { return None };
    if name != "compareTo" || args.len() != 1 {
        return None;
    }
    Some((recv, &args[0]))
}

/// The operands of an `indexOf` call: `a.indexOf(x)` → `(a, x)` — what is searched in, and for.
fn finder(e: &Expr) -> Option<(&Expr, &Expr)> {
    let Expr::Method { recv, name, args } = strip(e) else { return None };
    if name != "indexOf" || args.len() != 1 {
        return None;
    }
    Some((recv, &args[0]))
}

/// Every name a definition binds, how many times each is bound or assigned, the single-name bindings
/// with one value, and the `instanceof` pattern bindings the IR's `locals_of` does not collect.
#[derive(Default)]
struct Locals {
    sites: HashMap<String, usize>,
    lets: Vec<(String, Expr)>,
    patterns: HashSet<String>,
}

impl Locals {
    fn count(&mut self, name: &str) {
        if !name.is_empty() {
            *self.sites.entry(name.to_owned()).or_insert(0) += 1;
        }
    }

    fn params(&mut self, params: &[Param]) {
        for p in params {
            self.count(&p.name);
        }
    }

    fn assigned(&mut self, target: &Expr) {
        if let Expr::Name(n) = target {
            self.count(n);
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let { name, init, .. } => {
                self.count(name);
                if let Some(init) = init {
                    self.lets.push((name.clone(), init.clone()));
                }
            }
            Stmt::ForEach { name, .. } => self.count(name),
            Stmt::Try { resources, catches, .. } => {
                for r in resources {
                    if let Some(name) = &r.name {
                        self.count(name);
                    }
                }
                for c in catches {
                    if let Some(name) = &c.name {
                        self.count(name);
                    }
                }
            }
            _ => {}
        }
        each_part(s, &mut |part| match part {
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::AssignExpr { l, .. } => self.assigned(l),
            Expr::Update { e, .. } => self.assigned(e),
            Expr::InstanceOf { bind: Some(name), .. } => {
                self.patterns.insert(name.clone());
            }
            Expr::Lambda { params, body } => {
                for p in params {
                    self.count(&p.name);
                }
                match body {
                    LambdaBody::Block(b) => self.stmts(b),
                    LambdaBody::Expr(e) => self.expr(e),
                }
                return;
            }
            _ => {}
        }
        for_each_expr_child(e, &mut |c| match c {
            JChild::Expr(x) => self.expr(x),
            JChild::Stmts(b) => self.stmts(b),
        });
    }
}

/// A part of a compound statement: a statement, a list of them, or an expression it evaluates.
enum Part<'s> {
    Stmt(&'s Stmt),
    Stmts(&'s [Stmt]),
    Expr(&'s Expr),
}

/// Every part of `s`, in source order — its initializer, what it tests, its bodies.
fn each_part<'s>(s: &'s Stmt, f: &mut dyn FnMut(Part<'s>)) {
    match s {
        Stmt::Let { init, .. } => {
            if let Some(init) = init {
                f(Part::Expr(init));
            }
        }
        Stmt::Decl(items) => items.iter().for_each(|s| f(Part::Stmt(s))),
        Stmt::Assign { lhs, rhs, .. } => {
            f(Part::Expr(lhs));
            f(Part::Expr(rhs));
        }
        Stmt::Update { e, .. } | Stmt::Expr(e) | Stmt::Throw(e) | Stmt::Assert(e) | Stmt::Return(Some(e)) => f(Part::Expr(e)),
        Stmt::If { cond, then, els } => {
            f(Part::Expr(cond));
            f(Part::Stmts(then));
            match els.as_deref() {
                Some(Else::Block(b)) => f(Part::Stmts(b)),
                Some(Else::If(alt)) => f(Part::Stmt(alt)),
                None => {}
            }
        }
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            f(Part::Expr(cond));
            f(Part::Stmts(body));
        }
        Stmt::For { init, cond, update, body } => {
            if let Some(init) = init {
                f(Part::Stmt(init));
            }
            if let Some(cond) = cond {
                f(Part::Expr(cond));
            }
            for u in update {
                f(Part::Expr(u));
            }
            f(Part::Stmts(body));
        }
        Stmt::ForEach { iter, body, .. } => {
            f(Part::Expr(iter));
            f(Part::Stmts(body));
        }
        Stmt::Switch { value, cases } => {
            f(Part::Expr(value));
            for case in cases {
                for v in &case.values {
                    f(Part::Expr(v));
                }
                f(Part::Stmts(&case.body));
            }
        }
        Stmt::Try { resources, body, catches, finally } => {
            for r in resources {
                f(Part::Expr(&r.value));
            }
            f(Part::Stmts(body));
            for c in catches {
                f(Part::Stmts(&c.body));
            }
            if let Some(fin) = finally {
                f(Part::Stmts(fin));
            }
        }
        Stmt::Synchronized { lock, body } => {
            f(Part::Expr(lock));
            f(Part::Stmts(body));
        }
        Stmt::Labeled { stmt, .. } => f(Part::Stmt(stmt)),
        Stmt::Block(b) => f(Part::Stmts(b)),
        Stmt::Return(None) | Stmt::Break(_) | Stmt::Continue(_) | Stmt::NestedItem | Stmt::Unknown(_) => {}
    }
}

/// A direct child of an expression: another expression, or a statement body it holds.
enum JChild<'e> {
    Expr(&'e Expr),
    Stmts(&'e [Stmt]),
}

/// Every direct child expression or body of `e`.
fn for_each_expr_child<'e>(e: &'e Expr, f: &mut dyn FnMut(JChild<'e>)) {
    match e {
        Expr::Binary { l, r, .. } | Expr::AssignExpr { l, r, .. } => {
            f(JChild::Expr(l));
            f(JChild::Expr(r));
        }
        Expr::Unary { e, .. } | Expr::Update { e, .. } | Expr::Paren(e) | Expr::Cast { e, .. } | Expr::Throw(e) => {
            f(JChild::Expr(e));
        }
        Expr::Cond { c, t, f: alt } => {
            f(JChild::Expr(c));
            f(JChild::Expr(t));
            f(JChild::Expr(alt));
        }
        Expr::Call { f: callee, args } => {
            f(JChild::Expr(callee));
            for a in args {
                f(JChild::Expr(a));
            }
        }
        Expr::Method { recv, args, .. } => {
            f(JChild::Expr(recv));
            for a in args {
                f(JChild::Expr(a));
            }
        }
        Expr::StaticCall { args, .. } | Expr::Super { args } | Expr::New { args, .. } => {
            for a in args {
                f(JChild::Expr(a));
            }
        }
        Expr::Field { base, .. } => f(JChild::Expr(base)),
        Expr::Index { base, i } => {
            f(JChild::Expr(base));
            f(JChild::Expr(i));
        }
        Expr::InstanceOf { e, .. } => f(JChild::Expr(e)),
        Expr::Array { elems } => {
            for el in elems {
                f(JChild::Expr(el));
            }
        }
        Expr::Lambda { body, .. } => match body {
            LambdaBody::Block(b) => f(JChild::Stmts(b)),
            LambdaBody::Expr(e) => f(JChild::Expr(e)),
        },
        Expr::MethodRef { target, .. } => f(JChild::Expr(target)),
        Expr::Switch { value, arms } => {
            f(JChild::Expr(value));
            for arm in arms {
                for c in &arm.conds {
                    f(JChild::Expr(c));
                }
                f(JChild::Expr(&arm.value));
            }
        }
        Expr::StmtBlock(b) => f(JChild::Stmts(b)),
        Expr::Name(_)
        | Expr::Lit(..)
        | Expr::Bool(_)
        | Expr::Null
        | Expr::StaticField { .. }
        | Expr::ClassConst { .. }
        | Expr::Unknown(_) => {}
    }
}

/// The walk: one definition's propositions, as atoms.
struct Atoms<'s> {
    scope: &'s FileScope,
    locals: HashSet<String>,
    /// Locals bound once to a value fixed by the text: read wherever they appear as that value.
    constants: HashMap<String, String>,
    /// Locals bound once to a call's result: read bare, they are that computation.
    computed: HashMap<String, String>,
    /// Parameters bound nowhere but by the call, read bare as themselves (see [`decide::param`]).
    params: HashSet<String>,
    /// Locals whose class the text names — a parameter's declared type, an object built in place. A
    /// field read off one is a typed field ([`decide::typed_field`]).
    types: HashMap<String, String>,
    out: Vec<Arc<str>>,
}

impl Atoms<'_> {
    fn push(&mut self, atom: Option<Arc<str>>) {
        if let Some(atom) = atom {
            self.out.push(atom);
        }
    }

    fn is_local(&self, name: &str) -> bool {
        self.locals.contains(name) || name == "this" || name == "super"
    }

    /// A constant's members: a path grows by them; any other constant has none a text could read, so
    /// what was read off it is a field of a value.
    fn extended(value: &str, rest: &[&str]) -> String {
        if rest.is_empty() {
            value.to_owned()
        } else if value.starts_with('@') || value.starts_with(decide::HOLE) {
            dotted(value, rest)
        } else {
            decide::field(rest)
        }
    }

    /// A chain of fields read off a value of `class`. Java has no record field-type table the closed
    /// world can name, so every hop stays one typed field.
    fn typed(class: &str, rest: &[&str]) -> String {
        decide::typed_chain(class, rest, |_, _| None)
    }

    /// The identity a chain rooted at `root` reads as. `this` / `super` is the value the method is read
    /// off (a field); a bound local is a field; a name the file does not bind is the package's own.
    fn chain_term(&self, root: &str, attrs: &[&str]) -> String {
        if root == "this" || root == "super" {
            return decide::field(attrs);
        }
        if let Some(value) = self.constants.get(root) {
            return Self::extended(value, attrs);
        }
        if attrs.is_empty() {
            if let Some(computation) = self.computed.get(root) {
                return computation.clone();
            }
            if self.params.contains(root) {
                return decide::param(root);
            }
        } else if let Some(class) = self.types.get(root) {
            return Self::typed(class, attrs);
        }
        if self.locals.contains(root) {
            return decide::field(attrs);
        }
        if let Some((member, rest)) = attrs.split_first() {
            if let Some(path) = self.scope.member_path(root, member, rest) {
                return decide::path(&path);
            }
        }
        decide::path(&self.scope.free_path(root, attrs))
    }

    /// What a callee is called by: a free name's plain path, or a class's method one more member of its
    /// chain. `None` for a call on a value — `obj.m()`, `this.m()` — which the closed world cannot name.
    fn callee(&self, e: &Expr) -> Option<String> {
        match e {
            Expr::Call { f, .. } => match chain(f) {
                (Some(root), attrs) if !self.is_local(root) => Some(self.chain_term(root, &attrs)),
                _ => None,
            },
            Expr::Method { recv, name, .. } => {
                let (root, mut attrs) = chain(recv);
                match root {
                    Some(root) if !self.is_local(root) => {
                        attrs.push(name.as_str());
                        Some(self.chain_term(root, &attrs))
                    }
                    _ => None,
                }
            }
            Expr::StaticCall { scope, name, .. } => {
                Some(decide::path(&self.scope.free_path(scope, &[name.as_str()])))
            }
            _ => None,
        }
    }

    /// The positional arguments of a call.
    fn call_args(e: &Expr) -> &[Expr] {
        match e {
            Expr::Call { args, .. }
            | Expr::Method { args, .. }
            | Expr::StaticCall { args, .. }
            | Expr::New { args, .. } => args,
            _ => &[],
        }
    }

    /// The constants among a call's arguments.
    fn constants_of(&self, e: &Expr) -> Vec<String> {
        Self::call_args(e).iter().filter_map(|a| self.constant(a)).collect()
    }

    /// The identity of an operand.
    fn term(&self, e: &Expr) -> String {
        if let Some(lit) = literal(e) {
            return lit;
        }
        match e {
            Expr::Paren(inner) | Expr::Cast { e: inner, .. } => self.term(inner),
            Expr::Name(_) | Expr::Field { .. } | Expr::StaticField { .. } => match chain(e) {
                (Some(root), attrs) => self.chain_term(root, &attrs),
                (None, attrs) => decide::field(&attrs),
            },
            Expr::Call { .. } | Expr::Method { .. } | Expr::StaticCall { .. } => {
                let callee = self.callee(e).unwrap_or_else(|| match e {
                    Expr::Method { name, .. } => decide::field(&[name.as_str()]),
                    _ => decide::field(&[]),
                });
                decide::call(&callee, self.constants_of(e))
            }
            Expr::New { class, args } => {
                let callee = self.scope.named_class(class).unwrap_or_else(|| decide::HOLE.to_owned());
                decide::call(&callee, args.iter().filter_map(|a| self.constant(a)).collect())
            }
            Expr::Array { elems } => decide::container(elems.iter().map(|el| self.term(el)).collect()),
            Expr::Index { base, i } => {
                let key = self.constant(i).unwrap_or_else(|| decide::HOLE.to_owned());
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

    /// A value handed to a call that is fixed by the text rather than computed: a literal, a name the
    /// definition did not bind, or a chain read off one.
    fn constant(&self, e: &Expr) -> Option<String> {
        if let Some(lit) = literal(e) {
            return Some(lit);
        }
        match strip(e) {
            Expr::Name(_) | Expr::Field { .. } | Expr::StaticField { .. } => match chain(strip(e)) {
                (Some(root), attrs) => {
                    if let Some(value) = self.constants.get(root) {
                        return Some(Self::extended(value, &attrs));
                    }
                    if self.is_local(root) {
                        return None;
                    }
                    Some(self.chain_term(root, &attrs))
                }
                (None, _) => None,
            },
            Expr::Unary { op, e: inner } if op == "-" => self.constant(inner).map(|c| format!("-{c}")),
            Expr::Paren(inner) | Expr::Cast { e: inner, .. } => self.constant(inner),
            _ => None,
        }
    }

    /// Whether a receiver names a class (a name the definition did not bind) rather than a value.
    fn is_class_ref(&self, e: &Expr) -> bool {
        matches!(e, Expr::Name(n) if !self.is_local(n))
    }

    /// The atoms a library call spells wherever it stands — equality, membership, absence — or `None`
    /// when it spells none.
    fn proposition(&self, e: &Expr) -> Option<Vec<Option<Arc<str>>>> {
        // A library call spelled through a class name (`Objects.equals`, `Objects.requireNonNull`,
        // Guava's `Preconditions.checkNotNull`) lowers to a `StaticCall`, not a `Method`.
        if let Expr::StaticCall { scope, name, args } = e {
            let arg = |i: usize| args.get(i).map(|a| self.term(a));
            return match (scope.as_str(), name.as_str()) {
                ("Objects", "equals") => Some(vec![decide::equal(&arg(0)?, &arg(1)?)]),
                ("Objects", "requireNonNull") | ("Preconditions", "checkNotNull") => Some(vec![decide::equal("None", &arg(0)?)]),
                _ => None,
            };
        }
        let Expr::Method { recv, name, args } = e else { return None };
        let arg = |i: usize| args.get(i).map(|a| self.term(a));
        Some(match name.as_str() {
            "equals" | "equalsIgnoreCase" => {
                // `a.equals(b)` is `a = b`; the static `Objects.equals(a, b)` is `a = b` too, its
                // receiver a class name rather than a value.
                if self.is_class_ref(recv) {
                    vec![decide::equal(&arg(0)?, &arg(1)?)]
                } else {
                    vec![decide::equal(&self.term(recv), &arg(0)?)]
                }
            }
            "contains" | "containsKey" | "containsValue" | "startsWith" | "endsWith" => {
                vec![decide::member(&arg(0)?, &self.term(recv))]
            }
            "isEmpty" | "isBlank" | "isPresent" => {
                vec![decide::equal("None", &self.term(recv))]
            }
            "requireNonNull" | "checkNotNull" => {
                vec![decide::equal("None", &arg(0)?)]
            }
            _ => return None,
        })
    }

    /// The expression a library call asserts: Guava's `checkArgument` / `checkState` test their first
    /// argument for truth.
    fn asserted(e: &Expr) -> Option<&Expr> {
        match e {
            Expr::Method { name, args, .. } if name == "checkArgument" || name == "checkState" => args.first(),
            Expr::StaticCall { name, args, .. } if name == "checkArgument" || name == "checkState" => args.first(),
            Expr::Call { f, args }
                if matches!(f.as_ref(), Expr::Name(n) if n == "checkArgument" || n == "checkState") =>
            {
                args.first()
            }
            _ => None,
        }
    }

    /// An expression in a position Java tests for truth.
    fn truth(&mut self, e: &Expr) {
        match e {
            // Structure, not a proposition: its parts are read where the walk reaches them.
            Expr::Binary { op, .. } if matches!(op.as_str(), "==" | "!=" | "<" | "<=" | ">" | ">=") => {}
            Expr::Unary { op, .. } if op == "!" => {}
            Expr::Method { .. } | Expr::Call { .. } | Expr::StaticCall { .. } if self.proposition(e).is_some() => {}
            // `&&` and `||` test their left operand and hand the right one back as their value — so the
            // right one is tested exactly when the whole expression is, which is now.
            Expr::Binary { op, r, .. } if op == "&&" || op == "||" => self.truth(r),
            Expr::Paren(inner) => self.truth(inner),
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
            Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } | Stmt::For { cond: Some(cond), .. } => self.truth(cond),
            Stmt::Assert(e) => self.truth(e),
            Stmt::Switch { value, cases } => {
                for case in cases {
                    for v in &case.values {
                        let atom = decide::equal(&self.term(value), &self.term(v));
                        self.push(atom);
                    }
                }
            }
            _ => {}
        }
        each_part(s, &mut |part| match part {
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Binary { op, l, r } => {
                let atom = match op.as_str() {
                    "==" | "!=" => {
                        let cmp = compared(l).or_else(|| compared(r));
                        let fnd = finder(l).or_else(|| finder(r));
                        match (cmp, fnd) {
                            (Some((a, b)), _) if is_zero(l) || is_zero(r) => {
                                decide::equal(&self.term(a), &self.term(b))
                            }
                            (_, Some((hay, needle)))
                                if is_neg_one(l) || is_neg_one(r) || is_zero(l) || is_zero(r) =>
                            {
                                decide::member(&self.term(needle), &self.term(hay))
                            }
                            _ => decide::equal(&self.term(l), &self.term(r)),
                        }
                    }
                    "<" | "<=" | ">" | ">=" => {
                        let cmp = compared(l).or_else(|| compared(r));
                        let fnd = finder(l).or_else(|| finder(r));
                        match (cmp, fnd) {
                            (Some((a, b)), _) if is_zero(l) || is_zero(r) => {
                                decide::order(&self.term(a), &self.term(b))
                            }
                            (_, Some((hay, needle))) if is_zero(l) || is_zero(r) => {
                                decide::member(&self.term(needle), &self.term(hay))
                            }
                            _ => decide::order(&self.term(l), &self.term(r)),
                        }
                    }
                    "&&" | "||" => {
                        self.truth(l);
                        None
                    }
                    _ => None,
                };
                self.push(atom);
            }
            Expr::Unary { op, e: inner } if op == "!" => self.truth(inner),
            Expr::Cond { c, .. } => self.truth(c),
            Expr::InstanceOf { e, ty, .. } => {
                let atom = self.scope.type_class(ty).and_then(|class| decide::member(&self.term(e), &class));
                self.push(atom);
            }
            Expr::Method { .. } | Expr::Call { .. } | Expr::StaticCall { .. } => {
                if let Some(atoms) = self.proposition(e) {
                    for atom in atoms {
                        self.push(atom);
                    }
                }
                if let Some(tested) = Self::asserted(e) {
                    self.truth(tested);
                }
            }
            Expr::Switch { value, arms } => {
                for arm in arms {
                    for c in &arm.conds {
                        let atom = decide::equal(&self.term(value), &self.term(c));
                        self.push(atom);
                    }
                }
            }
            _ => {}
        }
        for_each_expr_child(e, &mut |c| match c {
            JChild::Expr(x) => self.expr(x),
            JChild::Stmts(b) => self.stmts(b),
        });
    }
}

/// Calls and value reads, by the same identities the atoms are written in.
struct Refs<'w, 's> {
    walk: &'w Atoms<'s>,
    calls: Vec<decide::CallSite>,
    mentions: Vec<Arc<str>>,
    /// Fields written other than through a constructor the corpus can name: assigned directly.
    loose: Vec<Arc<str>>,
}

impl Refs<'_, '_> {
    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        if let Stmt::Assign { lhs: Expr::Field { name, .. }, .. } = s {
            self.loose.push(Arc::from(name.as_str()));
        }
        each_part(s, &mut |part| match part {
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    /// A call to a name reached as a path: its arguments by position. Java has no named or spread
    /// arguments, so the call is never open.
    fn site(&mut self, callee: String, args: &[Expr]) {
        let args = args.iter().map(|a| (None, Arc::from(self.walk.term(a)))).collect();
        self.calls.push(decide::CallSite { callee: Arc::from(callee), args, open: false });
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Call { f, args, .. } => {
                match self.walk.callee(e) {
                    Some(callee) if callee.starts_with('@') => self.site(callee, args),
                    _ => self.expr(f),
                }
                for a in args {
                    self.expr(a);
                }
            }
            Expr::Method { recv, args, .. } => {
                match self.walk.callee(e) {
                    Some(callee) if callee.starts_with('@') => self.site(callee, args),
                    _ => self.expr(recv),
                }
                for a in args {
                    self.expr(a);
                }
            }
            Expr::StaticCall { args, .. } => {
                if let Some(callee) = self.walk.callee(e) {
                    if callee.starts_with('@') {
                        self.site(callee, args);
                    }
                }
                for a in args {
                    self.expr(a);
                }
            }
            // A constructor call: the class is called with the arguments its constructor takes.
            Expr::New { class, args } => {
                if let Some(path) = self.walk.scope.named_class(class) {
                    self.site(path, args);
                }
                for a in args {
                    self.expr(a);
                }
            }
            Expr::AssignExpr { l, r, .. } => {
                if let Expr::Field { name, .. } = l.as_ref() {
                    self.loose.push(Arc::from(name.as_str()));
                }
                self.expr(l);
                self.expr(r);
            }
            // A name or chain read as a value: a constant, a class member — what is named here is read
            // from elsewhere.
            Expr::Name(_) | Expr::Field { .. } | Expr::StaticField { .. } => {
                if let (Some(root), attrs) = chain(e) {
                    if !self.walk.is_local(root) {
                        let read = self.walk.chain_term(root, &attrs);
                        if read.starts_with('@') {
                            self.mentions.push(Arc::from(read));
                        }
                    }
                }
            }
            // `Foo.class` reads the class itself.
            Expr::ClassConst { scope } => {
                if let Some(path) = self.walk.scope.named_class(scope) {
                    self.mentions.push(Arc::from(path));
                }
            }
            // `Foo::m` reads the method; `Foo::new` the constructor, named by its class.
            Expr::MethodRef { target, name } => {
                if let Expr::Name(n) = target.as_ref() {
                    if !self.walk.is_local(n) {
                        let mentioned = if name == "new" {
                            self.walk.scope.free_path(n, &[])
                        } else {
                            self.walk.scope.free_path(n, &[name.as_str()])
                        };
                        self.mentions.push(Arc::from(decide::path(&mentioned)));
                    }
                } else {
                    self.expr(target);
                }
            }
            Expr::Lambda { body, .. } => match body {
                LambdaBody::Block(b) => self.stmts(b),
                LambdaBody::Expr(e) => self.expr(e),
            },
            _ => {
                for_each_expr_child(e, &mut |c| match c {
                    JChild::Expr(x) => self.expr(x),
                    JChild::Stmts(b) => self.stmts(b),
                });
            }
        }
    }
}

/// Everything the decisions pass reads off one callable. `open` — the callable is reached by calls the
/// corpus cannot show (a `static main`, the runtime calls it; a method dispatched through every class
/// that inherits it), so what its parameters hold is not the corpus's to say.
pub(crate) fn decides(f: &Func, scope: &FileScope, open: bool) -> DecisionFacts {
    let mut locals = Locals::default();
    locals.params(&f.params);
    locals.stmts(&f.body);

    // Bound names: the IR's `locals_of` covers parameters, locals, loop variables, catch bindings and
    // lambda parameters; add the `instanceof` pattern bindings it does not.
    let mut bound = crate::ir::locals_of(f);
    bound.extend(locals.patterns);
    bound.extend(f.type_params.iter().cloned());

    let order: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
    let variadic: HashSet<&String> = f.params.iter().filter(|p| p.variadic).map(|p| &p.name).collect();
    // A parameter bound nowhere but by the call is read as itself; one assigned again holds no value the
    // caller can fix. A variadic parameter collects every argument from its position on, so no one call
    // site's argument is what it holds.
    let untouched: HashSet<String> = order
        .iter()
        .filter(|name| locals.sites.get(*name) == Some(&1) && !variadic.contains(name))
        .cloned()
        .collect();

    let types: HashMap<String, String> = f
        .params
        .iter()
        .filter_map(|p| Some((p.name.clone(), scope.type_class(&p.ty)?)))
        .collect();

    let mut single = locals.lets;
    single.retain(|(name, _)| locals.sites.get(name) == Some(&1));

    let mut walk = Atoms {
        scope,
        locals: bound,
        constants: HashMap::new(),
        computed: HashMap::new(),
        params: untouched,
        types,
        out: Vec::new(),
    };

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
        if walk.constants.contains_key(name) {
            continue;
        }
        if matches!(value, Expr::Call { .. } | Expr::Method { .. } | Expr::StaticCall { .. }) {
            let computation = walk.term(value);
            walk.computed.insert(name.clone(), computation);
            continue;
        }
        // A value built by `new Foo` is of `Foo`: its fields are typed.
        if let Expr::New { class, .. } = value {
            if let Some(path) = scope.named_class(class) {
                walk.types.entry(name.clone()).or_insert(path);
            }
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

    let params = if open { Vec::new() } else { order.iter().map(|p| Arc::from(p.as_str())).collect() };
    DecisionFacts { atoms, params, calls, mentions, loose }
}

#[cfg(test)]
mod tests {
    use super::{decides, FileScope};
    use crate::ir::lower_func;
    use crate::tree::{field, kids, parse, text};
    use dup_defs_core::decide::{canonical, render, resolve_typed_fields, DecisionFacts};
    use std::collections::HashMap;
    use tree_sitter::Node;

    /// The first method or constructor with a body in `src`.
    fn first_method<'t>(n: Node<'t>, src: &str) -> Option<(Node<'t>, String)> {
        if matches!(
            n.kind(),
            "method_declaration" | "constructor_declaration" | "compact_constructor_declaration"
        ) && field(n, "body").is_some()
        {
            let name = field(n, "name").map_or_else(String::new, |name| text(name, src).to_owned());
            return Some((n, name));
        }
        for child in kids(n) {
            if let Some(found) = first_method(child, src) {
                return Some(found);
            }
        }
        None
    }

    /// What the decisions pass reads off the first method in `src`, as if the file lived in package
    /// `shop.orders` and imported nothing.
    fn facts(src: &str) -> DecisionFacts {
        let tree = parse(src).expect("fixture parses");
        let root = tree.root_node();
        let imports = HashMap::new();
        let scope = FileScope::new("shop.orders", &imports);
        let (decl, name) = first_method(root, src).expect("a method with a body");
        let func = lower_func(decl, &name, src).expect("lowers");
        decides(&func, &scope, false)
    }

    /// The atoms as the engine reads them with nothing inferred (a typed field is the field of some
    /// value), rendered.
    fn atoms(src: &str) -> Vec<String> {
        let mut out: Vec<String> = facts(src)
            .atoms
            .iter()
            .map(|a| render(&canonical(&resolve_typed_fields(a, |_, _| None))))
            .collect();
        out.sort();
        out.dedup();
        out
    }

    #[test]
    fn equality_negation_and_a_switch_spell_one_fact() {
        let refuse = atoms("class Order { boolean refuse(Order o) { if (o.status != Status.OPEN) return false; return true; } }");
        let accept = atoms("class Order { boolean accept(Order o) { return o.status == Status.OPEN; } }");
        let switched = atoms("class Order { String label(Order o) { switch (o.status) { case Status.OPEN: return \"o\"; } return \"\"; } }");
        assert_eq!(refuse, vec!["@shop.orders.Status.OPEN = _.status"]);
        assert_eq!(accept, refuse);
        assert_eq!(switched, refuse);
    }

    #[test]
    fn instanceof_is_membership() {
        let found = atoms("class C { boolean f(Object x) { return x instanceof Order; } }");
        assert_eq!(found, vec!["$x in @shop.orders.Order"]);
    }

    #[test]
    fn equals_and_contains_spell_their_relation() {
        assert_eq!(atoms("class C { boolean f(String a, String b) { return a.equals(b); } }"), vec!["$a = $b"]);
        assert_eq!(
            atoms("class C { boolean f(List<String> xs, String s) { return xs.contains(s); } }"),
            vec!["$s in $xs"]
        );
    }

    #[test]
    fn compare_to_and_index_of_spell_the_relation() {
        assert_eq!(
            atoms("class C { boolean f(String a, String b) { return a.compareTo(b) == 0; } }"),
            vec!["$a = $b"]
        );
        assert_eq!(
            atoms("class C { boolean f(String a, String b) { return a.compareTo(b) < 0; } }"),
            vec!["$a < $b"]
        );
        assert_eq!(
            atoms("class C { boolean f(String s) { return s.indexOf(\"..\") >= 0; } }"),
            vec!["\"..\" in $s"]
        );
    }

    #[test]
    fn absence_is_one_test_however_spelled() {
        // A parameter reads as `$o`: the lexicographic term order the core canonical keeps puts the
        // parameter left of the `None` literal.
        let absent = vec!["$o = None"];
        assert_eq!(atoms("class C { boolean f(Object o) { return o == null; } }"), absent);
        assert_eq!(atoms("class C { Object f(Object o) { return Objects.requireNonNull(o); } }"), absent);
        assert_eq!(atoms("class C { boolean f(String s) { return s.isEmpty(); } }"), vec!["$s = None"]);
    }

    #[test]
    fn a_truth_position_and_an_assertion_are_propositions() {
        let guarded = atoms("class C { void f(boolean force, Order o) { if (!force) { return; } ship(o); } }");
        assert_eq!(guarded, vec!["? $force"]);
        let asserted = atoms("class C { void f(int x) { Preconditions.checkArgument(x > 0); } }");
        assert_eq!(asserted, vec!["$x < 0"]);
    }

    #[test]
    fn a_constructor_exposes_its_parameters_and_the_fields_it_writes() {
        let found = facts("class Order { Order(String name, int qty) { this.name = name; this.qty = qty; } }");
        let params: Vec<&str> = found.params.iter().map(|p| &**p).collect();
        assert_eq!(params, vec!["name", "qty"]);
        let loose: Vec<&str> = found.loose.iter().map(|l| &**l).collect();
        assert_eq!(loose, vec!["name", "qty"]);
    }

    #[test]
    fn a_new_call_names_the_constructor_and_a_class_literal_is_a_mention() {
        let found = facts("class C { Object f() { return new Order(Status.OPEN); } }");
        assert_eq!(found.calls.len(), 1);
        assert_eq!(&*found.calls[0].callee, "@shop.orders.Order");
        let args: Vec<String> = found.calls[0].args.iter().map(|(_, v)| v.to_string()).collect();
        assert_eq!(args, vec!["@shop.orders.Status.OPEN".to_owned()]);

        let mentioned = facts("class C { Class f() { return Order.class; } }");
        assert_eq!(mentioned.mentions.iter().map(|m| &**m).collect::<Vec<_>>(), vec!["@shop.orders.Order"]);
    }

    #[test]
    fn a_method_reference_reads_the_method() {
        let found = facts("class C { Object f() { return Order::open; } }");
        assert_eq!(found.mentions.iter().map(|m| &**m).collect::<Vec<_>>(), vec!["@shop.orders.Order.open"]);
    }

    #[test]
    fn an_open_main_has_no_parameters() {
        let src = "class Main { static void main(String[] args) { go(args); } }";
        let tree = parse(src).expect("parses");
        let root = tree.root_node();
        let imports = HashMap::new();
        let scope = FileScope::new("app", &imports);
        let (decl, name) = first_method(root, src).expect("main");
        let func = lower_func(decl, &name, src).expect("lowers");
        let found = decides(&func, &scope, true);
        assert!(found.params.is_empty(), "{:?}", found.params);
    }
}
