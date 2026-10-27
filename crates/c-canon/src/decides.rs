//! What a C definition **decides on** — the frontend half of [`Facets::decides`].
//!
//! C says which expressions are propositions, and this module only reads them off — by the rules the
//! other frontends read their own, written in the atom grammar all of them share
//! ([`dup_defs_core::decide`]):
//!
//! - every **comparison** — `==` and `!=` as one relation, the four orders as another;
//! - every expression in a position C **tests for truth** — the condition of an `if`, a loop, a `?:`,
//!   the operand of `!`, the left of `&&` and `||` (and the right when the whole is tested), what an
//!   `assert` or a `BUG_ON` asserts;
//! - a `switch`'s cases as the equalities they abbreviate;
//! - the comparison functions as the relation they compute: `strcmp(a, b) == 0` and `!strcmp(a, b)` are
//!   `a = b`, `strcmp(a, b) < 0` is `a < b`, and `strstr(s, x)` tested is `x in s`.
//!
//! **Absence is one test however spelled.** `p == NULL` is `None = p`, the atom Python's `p is None`
//! and Go's `p == nil` write.
//!
//! A term's identity follows the binding the language gives its name: a name the definition binds is
//! a hole; any other name is the program's — C has one namespace for every function and object with
//! external linkage — save a `static` one, which is its file's, so a file-local helper and a call to it
//! meet as one path and two files' helpers of one name do not. A name used as an operand is written as
//! a value ([`decide::value_path`]): what C compares is a value, whatever its spelling.
//!
//! [`Facets::decides`]: dup_defs_core::Facets::decides

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::decide::{self, dotted, DecisionFacts};

use crate::canon::{each_stmt_part, for_each_child, Child, Part};
use crate::ir::{Elem, Expr, FieldDecl, Func, Stmt, Type};

/// The functions that compare their first two arguments as `strcmp` does: zero when equal, the sign
/// of the order otherwise.
const COMPARATORS: &[&str] = &[
    "strcmp", "strncmp", "strcasecmp", "strncasecmp", "memcmp", "wcscmp", "wcsncmp", "bcmp", "stricmp", "strnicmp",
    "_stricmp", "_strnicmp", "strcoll", "g_strcmp0", "av_strcasecmp", "av_strncasecmp", "ngx_strcmp", "ngx_strncmp",
    "ngx_strcasecmp", "ngx_strncasecmp", "ngx_memcmp", "git__strcmp", "git__strcasecmp", "git__strncmp",
    "git__strncasecmp", "curl_strequal", "strcasecompare",
];

/// The functions that find their second argument in their first: non-null when it is there.
const FINDERS: &[&str] = &["strstr", "strchr", "strrchr", "memchr", "strpbrk", "strcasestr", "wcsstr", "wcschr", "memmem"];

/// The functions that hand their first argument back, hinting the branch predictor: transparent.
const HINTS: &[&str] = &["likely", "unlikely", "__builtin_expect", "LIKELY", "UNLIKELY", "av_likely", "av_unlikely", "ngx_likely"];

/// Whether a callee asserts its first argument: `assert`, `av_assert0`, `g_assert`, `BUG_ON`.
fn asserts(callee: &str) -> bool {
    callee.to_ascii_lowercase().contains("assert") || matches!(callee, "BUG_ON" | "WARN_ON" | "WARN_ON_ONCE" | "VERIFY")
}

/// What the names of one file stand for, read once per file and shared by its definitions.
pub(crate) struct FileScope {
    /// The file's own path, dotted — what a `static` name is qualified by.
    module: String,
    /// The functions and objects the file declares `static`.
    statics: HashSet<String>,
    /// The records the file declares, each member with the record it is declared as where that names
    /// one — what a chain of typed fields hops through.
    records: HashMap<String, Vec<(String, Option<String>)>>,
}

impl FileScope {
    pub(crate) fn new(module: String, statics: HashSet<String>, records: &[(String, Vec<FieldDecl>)]) -> Self {
        let mut scope = Self { module, statics, records: HashMap::new() };
        let records: HashMap<String, Vec<(String, Option<String>)>> = records
            .iter()
            .map(|(name, fields)| {
                let members = fields
                    .iter()
                    .filter_map(|f| Some((f.name.clone()?, scope.type_class(&f.ty))))
                    .collect();
                (name.clone(), members)
            })
            .collect();
        scope.records = records;
        scope
    }

    /// The dotted path a name the definition did not bind stands for, `rest` read off it.
    fn free_path(&self, name: &str, rest: &[&str]) -> String {
        if self.statics.contains(name) && !self.module.is_empty() {
            dotted(&format!("{}.{name}", self.module), rest)
        } else {
            dotted(name, rest)
        }
    }

    /// A definition's identity: what the corpus's calls and reads name it by.
    pub(crate) fn identity(&self, name: &str) -> Arc<str> {
        Arc::from(decide::path(&self.free_path(name, &[])))
    }

    /// The members of a record this file declares, by name — an initializer of it sets them.
    pub(crate) fn record(&self, name: &str) -> Option<&Vec<(String, Option<String>)>> {
        self.records.get(name)
    }

    /// The record a declared type names, as a path — through a pointer, the way Python reads
    /// `Optional[C]` as `C`. A scalar type is no record.
    fn type_class(&self, ty: &Type) -> Option<String> {
        match ty {
            Type::Ptr(inner) | Type::Array(inner, _) => self.type_class(inner),
            Type::Named(name) => {
                let bare = name.strip_prefix("struct ").or_else(|| name.strip_prefix("union ")).unwrap_or(name);
                let scalar = bare.split(' ').any(|w| {
                    matches!(w, "void" | "char" | "short" | "int" | "long" | "float" | "double" | "signed" | "unsigned" | "_Bool" | "bool")
                }) || bare.ends_with("_t") && !self.records.contains_key(bare);
                (!scalar && !bare.is_empty()).then(|| decide::path(bare))
            }
            Type::Record { name: Some(n), .. } => Some(decide::path(n)),
            _ => None,
        }
    }

    /// The record a member of a record this file declares is declared as.
    fn field_class(&self, class: &str, field: &str) -> Option<String> {
        let (_, fields) = self.records.iter().find(|(name, _)| decide::path(name) == class)?;
        fields.iter().find(|(f, _)| f == field).and_then(|(_, ty)| ty.clone())
    }
}

/// A member chain from its root outwards: the name it starts from, or `None` when it starts at a
/// computed value, and the members read off it. Parentheses, `&`, `*` and casts are seen through —
/// they change how a value is held or typed, not which value it is.
fn chain(mut e: &Expr) -> (Option<&str>, Vec<&str>) {
    let mut attrs = Vec::new();
    loop {
        match e {
            Expr::Field { e: inner, field } => {
                attrs.push(field.as_str());
                e = inner;
            }
            Expr::Paren(inner) | Expr::Cast { e: inner, .. } => e = inner,
            Expr::Unary { op, e: inner } if op == "&" || op == "*" => e = inner,
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
        Expr::Lit("Int", t) => decide::integer(t),
        Expr::Lit("Float", t) => {
            let clean = t.trim_end_matches(['f', 'F', 'l', 'L']);
            clean.parse::<f64>().map_or_else(|_| decide::literal(t), decide::number)
        }
        Expr::Lit(_, t) => decide::literal(t),
        Expr::Bool(b) => String::from(if *b { "True" } else { "False" }),
        Expr::Null => "None".to_owned(),
        Expr::Paren(inner) | Expr::Cast { e: inner, .. } => return literal(inner),
        _ => return None,
    })
}

fn is_zero(e: &Expr) -> bool {
    literal(e).is_some_and(|l| l == "0" || l == "None" || l == "False")
}

fn operator(op: &str) -> bool {
    matches!(op, "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>")
}

/// The callee's name when it is a bare one.
fn callee_name(f: &Expr) -> Option<&str> {
    match f {
        Expr::Name(n) => Some(n),
        Expr::Paren(inner) => callee_name(inner),
        _ => None,
    }
}

/// Every name a definition binds, how many times each is bound or assigned, and the single-name
/// bindings with one value — what a name bound once may be read as.
#[derive(Default)]
struct Locals {
    bound: HashSet<String>,
    sites: HashMap<String, usize>,
    lets: Vec<(String, Expr)>,
    /// Locals declared as a record: their members are typed.
    typed: Vec<(String, Type)>,
}

impl Locals {
    fn bind(&mut self, name: &str) {
        self.bound.insert(name.to_owned());
        *self.sites.entry(name.to_owned()).or_insert(0) += 1;
    }

    fn assigned(&mut self, target: &Expr) {
        if let Expr::Name(n) = target {
            *self.sites.entry(n.clone()).or_insert(0) += 1;
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Assign { lhs, rhs, .. } => {
                self.assigned(lhs);
                self.expr(lhs);
                self.expr(rhs);
            }
            Stmt::IncDec { target, .. } => {
                self.assigned(target);
                self.expr(target);
            }
            _ => each_stmt_part(s, &mut |part| match part {
                Part::Spec(spec) => {
                    if let Some(init) = &spec.init {
                        self.lets.push((spec.name.clone(), init.clone()));
                        self.expr(init);
                    }
                    self.typed.push((spec.name.clone(), spec.ty.clone()));
                    self.bind(&spec.name);
                }
                Part::Stmt(inner) => self.stmt(inner),
                Part::Stmts(list) => self.stmts(list),
                Part::Expr(e) => self.expr(e),
            }),
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Assign { l, .. } | Expr::Update { e: l, .. } => self.assigned(l),
            Expr::Unary { op, e: inner } if op == "&" => self.assigned(inner),
            _ => {}
        }
        for_each_child(e, &mut |c| match c {
            Child::Expr(x) => self.expr(x),
            Child::Body(b) => self.stmts(b),
        });
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
    /// Locals whose record the text names — a parameter's declared type, a local declared as one. A
    /// member read off one is a typed field ([`decide::typed_field`]).
    types: HashMap<String, String>,
    out: Vec<Arc<str>>,
}

impl Atoms<'_> {
    fn push(&mut self, atom: Option<Arc<str>>) {
        if let Some(atom) = atom {
            self.out.push(atom);
        }
    }

    /// A chain of members read off a value of `class`, hopping through the records the file knows.
    fn typed(&self, class: &str, rest: &[&str]) -> String {
        decide::typed_chain(class, rest, |current, attr| self.scope.field_class(current, attr))
    }

    /// A constant's members: a path grows by them; any other constant has none a text could read, so
    /// what was read off it is a member of a value.
    fn extended(value: &str, rest: &[&str]) -> String {
        if rest.is_empty() {
            value.to_owned()
        } else if value.starts_with('@') || value.starts_with(decide::HOLE) {
            dotted(value, rest)
        } else {
            decide::field(rest)
        }
    }

    /// The identity a chain rooted at `root` reads as. `as_value` writes a file-level name as the
    /// value it must be where it stands — an operand ([`decide::value_path`]); a callee or a mention
    /// keeps the plain path, which is what the definition it names is called by.
    fn chain_term(&self, root: &str, attrs: &[&str], as_value: bool) -> String {
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
            return self.typed(class, attrs);
        }
        if self.locals.contains(root) {
            return decide::field(attrs);
        }
        let path = self.scope.free_path(root, attrs);
        if as_value {
            decide::value_path(&path)
        } else {
            decide::path(&path)
        }
    }

    /// What a callee is called by: a function's plain path; a call through a member is one more member
    /// of its receiver's chain, read the way a member of it is.
    fn callee(&self, f: &Expr) -> String {
        match chain(f) {
            (Some(root), attrs) => self.chain_term(root, &attrs, false),
            (None, attrs) => decide::field(&attrs),
        }
    }

    /// The identity of an operand.
    fn term(&self, e: &Expr) -> String {
        if let Some(lit) = literal(e) {
            return lit;
        }
        match e {
            Expr::Paren(inner) | Expr::Cast { e: inner, .. } => self.term(inner),
            Expr::Unary { op, e: inner } if op == "&" || op == "*" => self.term(inner),
            Expr::Name(_) | Expr::Field { .. } => match chain(e) {
                (Some(root), attrs) => self.chain_term(root, &attrs, true),
                (None, attrs) => decide::field(&attrs),
            },
            Expr::Call { f, args } if callee_name(f).is_some_and(|n| HINTS.contains(&n)) => {
                args.first().map_or_else(|| decide::HOLE.to_owned(), |a| self.term(a))
            }
            Expr::Call { f, args } => decide::call(&self.callee(f), args.iter().filter_map(|a| self.constant(a)).collect()),
            Expr::Compound { ty, elems } => match self.scope.type_class(ty) {
                // A compound literal of a record is its constructor, with the constants it was handed.
                Some(class) => decide::call(&class, elems.iter().filter_map(|el| self.constant(elem_value(el))).collect()),
                None => decide::container(elems.iter().map(|el| self.term(elem_value(el))).collect()),
            },
            Expr::InitList(elems) => decide::container(elems.iter().map(|el| self.term(elem_value(el))).collect()),
            Expr::Index { e: base, i } => {
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
            Expr::SizeofType(_) | Expr::SizeofValue(_) => "sizeof".to_owned(),
            _ => decide::HOLE.to_owned(),
        }
    }

    /// A value handed to a call that is fixed by the text rather than computed: a literal, a name the
    /// definition did not bind, or a list of those.
    fn constant(&self, e: &Expr) -> Option<String> {
        if let Some(lit) = literal(e) {
            return Some(lit);
        }
        match e {
            Expr::Name(_) | Expr::Field { .. } => match chain(e) {
                (Some(root), attrs) => {
                    if let Some(value) = self.constants.get(root) {
                        return Some(Self::extended(value, &attrs));
                    }
                    if self.locals.contains(root) {
                        return None;
                    }
                    Some(self.chain_term(root, &attrs, true))
                }
                (None, _) => None,
            },
            Expr::Unary { op, e: inner } if op == "-" => self.constant(inner).map(|c| format!("-{c}")),
            Expr::Unary { op, e: inner } if op == "&" => self.constant(inner),
            Expr::Paren(inner) | Expr::Cast { e: inner, .. } => self.constant(inner),
            _ => None,
        }
    }

    /// The comparison function a call is, with its two operands.
    fn compared<'e>(&self, e: &'e Expr) -> Option<(&'e Expr, &'e Expr)> {
        let Expr::Call { f, args } = strip(e) else { return None };
        let name = callee_name(f)?;
        if !COMPARATORS.contains(&name) || self.locals.contains(name) {
            return None;
        }
        Some((args.first()?, args.get(1)?))
    }

    /// The search function a call is, with what it searches in and for.
    fn found<'e>(&self, e: &'e Expr) -> Option<(&'e Expr, &'e Expr)> {
        let Expr::Call { f, args } = strip(e) else { return None };
        let name = callee_name(f)?;
        if !FINDERS.contains(&name) || self.locals.contains(name) {
            return None;
        }
        Some((args.first()?, args.get(1)?))
    }

    /// An expression in a position C tests for truth.
    fn truth(&mut self, e: &Expr) {
        let e = strip(e);
        if let Some((a, b)) = self.compared(e) {
            let atom = decide::equal(&self.term(a), &self.term(b));
            self.push(atom);
            return;
        }
        if let Some((hay, needle)) = self.found(e) {
            let atom = decide::member(&self.term(needle), &self.term(hay));
            self.push(atom);
            return;
        }
        match e {
            // Structure, not a proposition: its parts are read where the walk reaches them.
            Expr::Binary { op, .. } if matches!(op.as_str(), "==" | "!=" | "<" | "<=" | ">" | ">=") => {}
            Expr::Unary { op, .. } if op == "!" => {}
            Expr::Call { f, args } if callee_name(f).is_some_and(|n| HINTS.contains(&n)) => {
                if let Some(inner) = args.first() {
                    self.truth(inner);
                }
            }
            // `&&` and `||` test their left operand and hand the right one back as their value — so the
            // right one is tested exactly when the whole expression is, which is now.
            Expr::Binary { op, r, .. } if op == "&&" || op == "||" => self.truth(r),
            // An assignment tested is the value assigned: `if ((p = find(x)))`.
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
            Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } | Stmt::For { cond: Some(cond), .. } => {
                self.truth(cond);
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
            Part::Spec(spec) => {
                if let Some(init) = &spec.init {
                    self.expr(init);
                }
            }
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Binary { op, l, r } => {
                let (left, right) = (strip(l), strip(r));
                let against_zero = is_zero(left) || is_zero(right);
                // `strcmp(a, b) == 0`, `strstr(s, x) != NULL`: the function's relation, not a number's.
                let compared = self.compared(left).or_else(|| self.compared(right));
                let found = self.found(left).or_else(|| self.found(right));
                let special = match (compared, found) {
                    (Some((first, second)), _) if against_zero => Some(match op.as_str() {
                        "==" | "!=" => decide::equal(&self.term(first), &self.term(second)),
                        _ => decide::order(&self.term(first), &self.term(second)),
                    }),
                    (_, Some((hay, needle))) if against_zero && matches!(op.as_str(), "==" | "!=") => {
                        Some(decide::member(&self.term(needle), &self.term(hay)))
                    }
                    _ => None,
                };
                let atom = match (special, op.as_str()) {
                    (Some(atom), _) => atom,
                    (None, "==" | "!=") => decide::equal(&self.term(left), &self.term(right)),
                    (None, "<" | "<=" | ">" | ">=") => decide::order(&self.term(left), &self.term(right)),
                    (None, "&&" | "||") => {
                        self.truth(left);
                        None
                    }
                    _ => None,
                };
                self.push(atom);
            }
            Expr::Unary { op, e: inner } if op == "!" => self.truth(inner),
            Expr::Cond { c, .. } => self.truth(c),
            Expr::Call { f, args } if callee_name(f).is_some_and(|n| asserts(n) && !self.locals.contains(n)) => {
                if let Some(asserted) = args.first() {
                    self.truth(asserted);
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

/// Parentheses and casts off: `(int)(strcmp(a, b))` compares as `strcmp(a, b)`.
fn strip(mut e: &Expr) -> &Expr {
    while let Expr::Paren(inner) | Expr::Cast { e: inner, .. } = e {
        e = inner;
    }
    e
}

fn elem_value(el: &Elem) -> &Expr {
    match el {
        Elem::Plain(v) | Elem::Field(_, v) | Elem::Index(_, v) => v,
    }
}

/// Calls and value reads, by the same identities the atoms are written in.
struct Refs<'w, 's> {
    walk: &'w Atoms<'s>,
    calls: Vec<decide::CallSite>,
    mentions: Vec<Arc<str>>,
    /// Members written other than through an initializer the corpus can name: assigned, or set by an
    /// initializer of a type that is no record.
    loose: Vec<Arc<str>>,
}

impl Refs<'_, '_> {
    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        if let Stmt::Assign { lhs: Expr::Field { field, .. }, .. } = s {
            self.loose.push(Arc::from(field.as_str()));
        }
        each_stmt_part(s, &mut |part| match part {
            Part::Spec(spec) => {
                if let Some(init) = &spec.init {
                    // A record declared with an initializer is its constructor, called with its members.
                    match (init, self.walk.scope.type_class(&spec.ty)) {
                        (Expr::InitList(elems), Some(class)) => self.construct(class, elems),
                        _ => self.expr(init),
                    }
                }
            }
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    /// An initializer of a record: its constructor, called with its members by name — or all of them in
    /// order, which is the other way C lets one be written.
    fn construct(&mut self, class: String, elems: &[Elem]) {
        let args: Option<Vec<_>> = elems
            .iter()
            .map(|el| match el {
                Elem::Field(name, v) => Some((Some(Arc::from(name.as_str())), Arc::from(self.walk.term(v)))),
                Elem::Plain(v) => Some((None, Arc::from(self.walk.term(v)))),
                Elem::Index(..) => None,
            })
            .collect();
        if let Some(args) = args {
            self.calls.push(decide::CallSite { callee: Arc::from(class), args, open: false });
        }
        for el in elems {
            self.expr(elem_value(el));
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Call { f, args } => {
                match callee_name(f) {
                    Some(name) if !self.walk.locals.contains(name) => {
                        let callee = self.walk.callee(f);
                        let args = args.iter().map(|a| (None, Arc::from(self.walk.term(a)))).collect();
                        self.calls.push(decide::CallSite { callee: Arc::from(callee), args, open: false });
                    }
                    _ => self.expr(f),
                }
                for a in args {
                    self.expr(a);
                }
            }
            Expr::Compound { ty, elems } => {
                if let Some(class) = self.walk.scope.type_class(ty) {
                    self.construct(class, elems);
                } else {
                    self.loose.extend(elems.iter().filter_map(|el| match el {
                        Elem::Field(name, _) => Some(Arc::from(name.as_str())),
                        _ => None,
                    }));
                    for el in elems {
                        self.expr(elem_value(el));
                    }
                }
            }
            Expr::Assign { l, r, .. } => {
                if let Expr::Field { field, .. } = l.as_ref() {
                    self.loose.push(Arc::from(field.as_str()));
                }
                self.expr(l);
                self.expr(r);
            }
            Expr::Name(_) | Expr::Field { .. } if chain(e).0.is_some() => {
                let (Some(root), attrs) = chain(e) else { return };
                if !self.walk.locals.contains(root) {
                    let read = self.walk.chain_term(root, &attrs, false);
                    if read.starts_with('@') {
                        self.mentions.push(Arc::from(read));
                    }
                }
            }
            _ => for_each_child(e, &mut |c| match c {
                Child::Expr(x) => self.expr(x),
                Child::Body(b) => self.stmts(b),
            }),
        }
    }
}

/// Everything the decisions pass reads off one function. `open` — the function is reached by calls the
/// corpus cannot show (`main`, run by the system), so what its parameters hold is not the corpus's to
/// say.
pub(crate) fn decides(f: &Func, scope: &FileScope, open: bool) -> DecisionFacts {
    let mut locals = Locals::default();
    for name in f.params.iter().filter_map(|p| p.name.as_deref()) {
        locals.bind(name);
    }
    locals.stmts(&f.body);
    let order: Vec<String> = f.params.iter().map(|p| p.name.clone().unwrap_or_else(|| "_".to_owned())).collect();
    let untouched: HashSet<String> = order.iter().filter(|name| locals.sites.get(*name) == Some(&1)).cloned().collect();
    let mut types: HashMap<String, String> = f
        .params
        .iter()
        .filter_map(|p| Some((p.name.clone()?, scope.type_class(&p.ty)?)))
        .collect();
    for (name, ty) in &locals.typed {
        if let Some(class) = scope.type_class(ty) {
            types.entry(name.clone()).or_insert(class);
        }
    }
    let mut single = locals.lets;
    single.retain(|(name, _)| locals.sites.get(name) == Some(&1));
    let mut walk = Atoms {
        scope,
        locals: locals.bound,
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
        if !walk.constants.contains_key(name) && matches!(strip(value), Expr::Call { .. }) {
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
    let params = if open || f.variadic { Vec::new() } else { order.iter().map(|p| Arc::from(p.as_str())).collect() };
    DecisionFacts { atoms, params, calls, mentions, loose }
}

#[cfg(test)]
mod tests {
    use crate::defs::scan_source;
    use dup_defs_core::decide::{canonical, render, resolve_typed_fields};
    use dup_defs_core::ScanOpts;
    use std::sync::Arc;

    /// The atoms the decisions pass reads off the first function in `src`, rendered with nothing
    /// inferred (a typed field is the field of some value).
    fn atoms(src: &str) -> Vec<String> {
        let f: Arc<str> = Arc::from("shop/orders.c");
        // The function under test is the last one; any before it are what it calls.
        let d = scan_source(src, &f, &ScanOpts::default()).into_iter().rfind(|d| d.kind.id == "functions").expect("a function");
        let mut out: Vec<String> =
            d.facets.decides.iter().map(|a| render(&canonical(&resolve_typed_fields(a, |_, _| None)))).collect();
        out.sort();
        out.dedup();
        out
    }

    #[test]
    fn negation_and_the_spelling_of_a_test_do_not_change_the_decision() {
        let refuse = atoms("int refuse(struct order *o) { if (o->status != STATUS_OPEN) return -1; return 0; }");
        let accept = atoms("int accept(struct order *o) { return o->status == STATUS_OPEN; }");
        let switched = atoms("const char *label(struct order *o) { switch (o->status) { case STATUS_OPEN: return \"o\"; } return \"\"; }");
        assert_eq!(refuse, vec!["@STATUS_OPEN = _.status"]);
        assert_eq!(accept, refuse);
        assert_eq!(switched, refuse);
    }

    #[test]
    fn absence_is_one_test_however_spelled() {
        let absent = vec!["None = _.paid_at"];
        assert_eq!(atoms("int f(struct order *o) { return o->paid_at == NULL; }"), absent);
        assert_eq!(atoms("int f(struct order *o) { return NULL != o->paid_at; }"), absent);
    }

    #[test]
    fn the_comparison_functions_are_the_relation_they_compute() {
        let compared = atoms("int f(const char *name) { if (strcmp(name, \"admin\") == 0) return 1; return 0; }");
        let negated = atoms("int f(const char *name) { if (!strcmp(name, \"admin\")) return 1; return 0; }");
        assert_eq!(compared, vec!["\"admin\" = $name"]);
        assert_eq!(negated, compared);
        let found = atoms("int f(const char *path) { return strstr(path, \"..\") != NULL; }");
        assert_eq!(found, vec!["\"..\" in $path"]);
    }

    #[test]
    fn an_assertion_is_a_truth_position() {
        let asserted = atoms("void f(struct order *o) { assert(o->paid); ship(o); }");
        let tested = atoms("void f(struct order *o) { if (o->paid) ship(o); }");
        assert_eq!(asserted, vec!["? _.paid"]);
        assert_eq!(tested, asserted);
    }

    #[test]
    fn a_static_name_is_its_file_s_and_a_local_constant_is_its_value() {
        let src = "static int limit_of(int k) { return k * 2; }\nint f(struct order *o) { int max = MAX_TOTAL; return o->total < max && limit_of(3); }";
        let found = atoms(src);
        assert!(found.contains(&"@MAX_TOTAL < _.total".to_owned()), "{found:?}");
    }

    #[test]
    fn a_parameter_is_marked_and_a_literal_is_its_value() {
        assert_eq!(atoms("int f(int force, struct order *o) { if (!force) return o->x; return 0; }"), vec!["? $force"]);
        let hex = atoms("int f(struct order *o) { return o->flags == 0x10; }");
        assert_eq!(hex, atoms("int f(struct order *o) { return o->flags == 16u; }"));
    }
}
