//! What a Go definition **decides on** — the frontend half of [`Facets::decides`].
//!
//! Go says which expressions are propositions, and this module only reads them off — by the rules the
//! other frontends read their own, written in the atom grammar all of them share
//! ([`dup_defs_core::decide`]):
//!
//! - every **comparison** — `==` and `!=` as one relation, the four orders as another;
//! - every expression in a position Go **tests for truth** — the condition of an `if` or of a `for`,
//!   the operand of `!`, the left of `&&` and `||` (and the right when the whole is tested), a tagless
//!   `switch`'s cases, a predicate handed to `slices.ContainsFunc` and kin;
//! - a tagged `switch`'s cases as the equalities they abbreviate, and a type switch's as membership
//!   (`x in @io.Reader`) — Rust's `match`, read the same way;
//! - the comma-ok forms as the tests they are: `v, ok := m[k]` is `k in m`, `t, ok := x.(T)` is
//!   `x in T`;
//! - `strings.Contains` / `slices.Contains` as membership, `errors.Is(err, ErrX)` as the equality it
//!   checks through the wrapping, and the `testify` assertions as the relation each asserts.
//!
//! **Absence is one test however spelled.** `x == nil` is `None = x`, the atom Python's `x is None`,
//! Rust's `x.is_none()` and TypeScript's `x === undefined` write.
//!
//! A term's identity follows the binding the language gives its name: a name the definition binds is a
//! hole; the receiver is the value the method is read off; a package member is the import path and the
//! name; a name the file neither binds nor imports is its own package's, since Go has no other
//! unqualified scope but the universe's. A package's path is its import path — the module `go.mod`
//! declares and the directory under it — so a definition and a call to it from another package meet as
//! one path. A package-level name used as an operand is written as a value ([`decide::value_path`]):
//! Go spells its constants like its types, and only the grammar — a comparison takes values — says
//! which one a name is.
//!
//! [`Facets::decides`]: dup_defs_core::Facets::decides

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use dup_defs_core::decide::{self, dotted, DecisionFacts};
use tree_sitter::Node;

use crate::canon::{for_each_child, Child};
use crate::ir::{lower_fields, Elem, Else, Expr, ForHead, Func, ParamDecl, Results, Stmt, Target, Type};
use crate::tree::{field, kids, text};

/// The universe scope: names every Go file sees without declaring or importing them.
const UNIVERSE: &[&str] = &[
    "append", "cap", "clear", "close", "complex", "copy", "delete", "imag", "len", "make", "max", "min", "new",
    "panic", "print", "println", "real", "recover", "bool", "byte", "complex64", "complex128", "error", "float32",
    "float64", "int", "int8", "int16", "int32", "int64", "rune", "string", "uint", "uint8", "uint16", "uint32",
    "uint64", "uintptr", "any", "comparable",
];

/// The functions whose function-typed argument is a predicate: what it returns is tested for truth.
const PREDICATE_FUNCS: &[(&str, &str)] = &[
    ("slices", "ContainsFunc"),
    ("slices", "IndexFunc"),
    ("slices", "DeleteFunc"),
    ("strings", "ContainsFunc"),
    ("strings", "IndexFunc"),
    ("strings", "LastIndexFunc"),
    ("strings", "FieldsFunc"),
    ("bytes", "ContainsFunc"),
    ("bytes", "IndexFunc"),
];

fn normal_segments(p: &Path) -> Vec<String> {
    p.components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str().map(str::to_owned),
            _ => None,
        })
        .collect()
}

thread_local! {
    /// Every directory already probed, with the module it belongs to — the directory holding its
    /// `go.mod` and the module path that declares — or `None` above every module. Each file asks for
    /// its own directory, and a corpus has far fewer directories than files.
    static MODULES: RefCell<HashMap<PathBuf, Option<(PathBuf, String)>>> = RefCell::new(HashMap::new());
}

/// The module a directory belongs to: the nearest `go.mod` at or above it that declares one.
fn module_of(dir: &Path) -> Option<(PathBuf, String)> {
    if let Some(known) = MODULES.with(|m| m.borrow().get(dir).cloned()) {
        return known;
    }
    let declared = std::fs::read_to_string(dir.join("go.mod")).ok().and_then(|gomod| {
        gomod.lines().find_map(|l| l.trim().strip_prefix("module ").map(|m| m.trim().trim_matches('"').to_owned()))
    });
    let found = match declared {
        Some(module) => Some((dir.to_path_buf(), module)),
        None => dir.parent().and_then(module_of),
    };
    MODULES.with(|m| m.borrow_mut().insert(dir.to_path_buf(), found.clone()));
    found
}

/// The import path of the package a file is in, dotted: the module the nearest `go.mod` above it
/// declares, joined with the directory under that. A file outside any module is the directory it is in.
pub(crate) fn package_path(file: &str) -> String {
    let dir = Path::new(file).parent().unwrap_or_else(|| Path::new(""));
    let Some((root, module)) = module_of(dir) else { return normal_segments(dir).join(".") };
    let mut segments: Vec<String> = module.split('/').filter(|s| !s.is_empty()).map(str::to_owned).collect();
    segments.extend(normal_segments(dir.strip_prefix(&root).unwrap_or_else(|_| Path::new(""))));
    segments.join(".")
}

/// What the names of one file stand for, read once per file and shared by its definitions.
pub(crate) struct FileScope {
    /// The file's package, as its dotted import path.
    package: String,
    /// What each package name the file imports stands for, dotted.
    imports: HashMap<String, String>,
    /// The structs the file declares, each field with the class it is declared as where that names
    /// one — what a chain of typed fields hops through.
    records: HashMap<String, Vec<(String, Option<String>)>>,
}

impl FileScope {
    pub(crate) fn new(file: &str, root: Node<'_>, src: &str, imports: &HashMap<String, Arc<str>>, facets: bool) -> Self {
        let mut scope = Self {
            package: package_path(file),
            imports: imports.iter().map(|(k, v)| (k.clone(), v.to_string())).collect(),
            records: HashMap::new(),
        };
        if !facets {
            // The records walk serves only the decisions pass (a record's constructor params); a run
            // that never reads facets skips it.
            return scope;
        }
        let mut records = HashMap::new();
        for decl in kids(root).into_iter().filter(|k| k.kind() == "type_declaration") {
            for spec in kids(decl).into_iter().filter(|k| k.kind() == "type_spec") {
                let (Some(name), Some(ty)) = (field(spec, "name"), field(spec, "type")) else { continue };
                if ty.kind() != "struct_type" {
                    continue;
                }
                let fields = lower_fields(ty, src)
                    .into_iter()
                    .flat_map(|f| {
                        let class = f.ty.as_ref().and_then(|t| scope.type_class(t));
                        let names = if f.names.is_empty() {
                            // An embedded field is named by its type.
                            f.ty.as_ref().and_then(base_name).into_iter().collect()
                        } else {
                            f.names
                        };
                        names.into_iter().map(move |n| (n, class.clone()))
                    })
                    .collect();
                records.insert(text(name, src).to_owned(), fields);
            }
        }
        scope.records = records;
        scope
    }

    fn qualified(&self, name: &str) -> String {
        if self.package.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{name}", self.package)
        }
    }

    /// A definition's identity: what the corpus's calls and reads name it by (`Server.Start` for a
    /// method).
    pub(crate) fn identity(&self, qualified: &str) -> Arc<str> {
        Arc::from(decide::path(&self.qualified(qualified)))
    }

    /// The fields of a struct this file declares, by name — a record's constructor takes them.
    pub(crate) fn record(&self, name: &str) -> Option<&Vec<(String, Option<String>)>> {
        self.records.get(name)
    }

    /// The dotted path a name the definition did not bind stands for, `rest` read off it: the
    /// universe's own names as themselves, any other the package's.
    fn free_path(&self, name: &str, rest: &[&str]) -> String {
        if UNIVERSE.contains(&name) {
            dotted(name, rest)
        } else {
            dotted(&self.qualified(name), rest)
        }
    }

    /// The dotted path of a member of an imported package, or `None` when `pkg` is not one.
    fn member_path(&self, pkg: &str, member: &str, rest: &[&str]) -> Option<String> {
        let path = self.imports.get(pkg)?;
        let mut all = vec![member];
        all.extend_from_slice(rest);
        Some(dotted(path, &all))
    }

    /// The class a declared type names, as a path — through a pointer, the way Python reads
    /// `Optional[C]` as `C`. A universe type is no class.
    fn type_class(&self, ty: &Type) -> Option<String> {
        match ty {
            Type::Ptr(inner) => self.type_class(inner),
            Type::Named(name) | Type::Generic(name, _) => match name.split_once('.') {
                Some((pkg, member)) => self.member_path(pkg, member, &[]).map(|p| decide::path(&p)),
                None if UNIVERSE.contains(&name.as_str()) || name.is_empty() => None,
                None => Some(decide::path(&self.qualified(name))),
            },
            _ => None,
        }
    }

    /// The class a field of a struct this file declares is declared as.
    fn field_class(&self, class: &str, field: &str) -> Option<String> {
        let (_, fields) = self.records.iter().find(|(name, _)| decide::path(&self.qualified(name)) == class)?;
        fields.iter().find(|(f, _)| f == field).and_then(|(_, ty)| ty.clone())
    }
}

/// The name a type is reached by, for an embedded field: `*pkg.Logger` → `Logger`.
fn base_name(ty: &Type) -> Option<String> {
    match ty {
        Type::Ptr(inner) => base_name(inner),
        Type::Named(n) | Type::Generic(n, _) => Some(n.rsplit('.').next().unwrap_or(n).to_owned()),
        _ => None,
    }
}

/// A selector chain from its root outwards: the name it starts from, or `None` when it starts at a
/// computed value, and the fields read off it. Parentheses, `&`, `*` and conversions are seen
/// through — they change how a value is held or typed, not which value it is.
fn chain(mut e: &Expr) -> (Option<&str>, Vec<&str>) {
    let mut attrs = Vec::new();
    loop {
        match e {
            Expr::Selector { operand, field } => {
                attrs.push(field.as_str());
                e = operand;
            }
            Expr::Paren(inner) => e = inner,
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
        Expr::Lit("Float", t) => t.replace('_', "").parse::<f64>().map_or_else(|_| decide::literal(t), decide::number),
        Expr::Lit(_, t) => decide::literal(t),
        Expr::Bool(b) => String::from(if *b { "True" } else { "False" }),
        Expr::Nil => "None".to_owned(),
        Expr::Paren(inner) => return literal(inner),
        _ => return None,
    })
}

/// Whether an import path is one of `testify`'s assertion packages.
fn is_testify(pkg: &str) -> bool {
    pkg.ends_with(".testify.assert") || pkg.ends_with(".testify.require")
}

fn operator(op: &str) -> bool {
    matches!(op, "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" | "&^")
}

/// Every name a definition binds, how many times each is bound or assigned, and the single-name
/// bindings with one value — what a name bound once may be read as.
#[derive(Default)]
struct Locals {
    bound: HashSet<String>,
    sites: HashMap<String, usize>,
    lets: Vec<(String, Expr)>,
}

impl Locals {
    fn bind(&mut self, name: &str) {
        if name != "_" {
            self.bound.insert(name.to_owned());
            *self.sites.entry(name.to_owned()).or_insert(0) += 1;
        }
    }

    fn assigned(&mut self, target: &Expr) {
        if let Expr::Name(n) = target {
            *self.sites.entry(n.clone()).or_insert(0) += 1;
        }
    }

    fn params(&mut self, decls: &[ParamDecl]) {
        for name in decls.iter().flat_map(|d| d.names.iter()) {
            self.bind(name);
        }
    }

    fn targets(&mut self, targets: &[Target], define: bool) {
        for t in targets {
            match t {
                Target::Name(n) if define => self.bind(n),
                Target::Name(n) => self.assigned(&Expr::Name(n.clone())),
                Target::Expr(e) => {
                    self.assigned(e);
                    self.expr(e);
                }
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
            Stmt::Define { lhs, rhs } => {
                if let ([Target::Name(n)], [value]) = (lhs.as_slice(), rhs.as_slice()) {
                    self.lets.push((n.clone(), value.clone()));
                }
                self.targets(lhs, true);
                for e in rhs {
                    self.expr(e);
                }
            }
            Stmt::Decl { specs, .. } => {
                for spec in specs {
                    if let ([n], [value]) = (spec.names.as_slice(), spec.values.as_slice()) {
                        self.lets.push((n.clone(), value.clone()));
                    }
                    spec.names.iter().for_each(|n| self.bind(n));
                    spec.values.iter().for_each(|e| self.expr(e));
                }
            }
            Stmt::Assign { lhs, rhs, .. } => {
                for t in lhs {
                    self.assigned(t);
                    self.expr(t);
                }
                for e in rhs {
                    self.expr(e);
                }
            }
            Stmt::IncDec { target, .. } => {
                self.assigned(target);
                self.expr(target);
            }
            Stmt::Recv { define, lhs, rhs } => {
                if let Some(lhs) = lhs {
                    self.targets(lhs, *define);
                }
                self.expr(rhs);
            }
            Stmt::TypeSwitch { alias, .. } => {
                if let Some(alias) = alias {
                    self.targets(alias, true);
                }
                self.children(s);
            }
            Stmt::For { head: ForHead::Range { define, lhs: Some(lhs), .. }, .. } => {
                self.targets(lhs, *define);
                self.children(s);
            }
            _ => self.children(s),
        }
    }

    /// The statements and expressions directly inside `s`, walked.
    fn children(&mut self, s: &Stmt) {
        each_part(s, &mut |part| match part {
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    fn expr(&mut self, e: &Expr) {
        if let Expr::FuncLit { params, body, .. } = e {
            self.params(params);
            self.stmts(body);
            return;
        }
        for_each_child(e, &mut |c| match c {
            Child::Expr(x) => self.expr(x),
            Child::Body(b) => self.stmts(b),
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
        Stmt::Define { rhs, .. } => rhs.iter().for_each(|e| f(Part::Expr(e))),
        Stmt::Decl { specs, .. } => specs.iter().flat_map(|s| s.values.iter()).for_each(|e| f(Part::Expr(e))),
        Stmt::Assign { lhs, rhs, .. } => lhs.iter().chain(rhs).for_each(|e| f(Part::Expr(e))),
        Stmt::IncDec { target: e, .. } | Stmt::Expr(e) | Stmt::Go(e) | Stmt::Defer(e) => f(Part::Expr(e)),
        Stmt::Send { ch, value } => {
            f(Part::Expr(ch));
            f(Part::Expr(value));
        }
        Stmt::Return(values) => values.iter().for_each(|e| f(Part::Expr(e))),
        Stmt::If { init, cond, then, els } => {
            if let Some(init) = init {
                f(Part::Stmt(init));
            }
            f(Part::Expr(cond));
            f(Part::Stmts(then));
            match els.as_deref() {
                Some(Else::Block(b)) => f(Part::Stmts(b)),
                Some(Else::If(alt)) => f(Part::Stmt(alt)),
                None => {}
            }
        }
        Stmt::For { head, body } => {
            match head {
                ForHead::While(c) => f(Part::Expr(c)),
                ForHead::Range { iter, .. } => f(Part::Expr(iter)),
                ForHead::Clause { init, cond, update } => {
                    if let Some(i) = init {
                        f(Part::Stmt(i));
                    }
                    if let Some(c) = cond {
                        f(Part::Expr(c));
                    }
                    if let Some(u) = update {
                        f(Part::Stmt(u));
                    }
                }
                ForHead::Loop => {}
            }
            f(Part::Stmts(body));
        }
        Stmt::Switch { init, value, cases } => {
            if let Some(init) = init {
                f(Part::Stmt(init));
            }
            if let Some(v) = value {
                f(Part::Expr(v));
            }
            for case in cases {
                case.values.iter().flatten().for_each(|e| f(Part::Expr(e)));
                f(Part::Stmts(&case.body));
            }
        }
        Stmt::TypeSwitch { init, value, cases, .. } => {
            if let Some(init) = init {
                f(Part::Stmt(init));
            }
            f(Part::Expr(value));
            for case in cases {
                f(Part::Stmts(&case.body));
            }
        }
        Stmt::Select { cases } => {
            for case in cases {
                if let Some(comm) = &case.comm {
                    f(Part::Stmt(comm));
                }
                f(Part::Stmts(&case.body));
            }
        }
        Stmt::Recv { rhs, .. } => f(Part::Expr(rhs)),
        Stmt::Block(b) => f(Part::Stmts(b)),
        Stmt::Labeled { stmt, .. } => f(Part::Stmt(stmt)),
        Stmt::NestedType | Stmt::Jump { .. } | Stmt::Fallthrough | Stmt::Empty | Stmt::Unknown(_) => {}
    }
}

/// The walk: one definition's propositions, as atoms.
struct Atoms<'s> {
    scope: &'s FileScope,
    receiver: Option<&'s str>,
    locals: HashSet<String>,
    /// Locals bound once to a value fixed by the text: read wherever they appear as that value.
    constants: HashMap<String, String>,
    /// Locals bound once to a call's result: read bare, they are that computation.
    computed: HashMap<String, String>,
    /// Parameters bound nowhere but by the call, read bare as themselves (see [`decide::param`]).
    params: HashSet<String>,
    /// Locals whose class the text names — a parameter's declared type, a struct literal built in
    /// place. A field read off one is a typed field ([`decide::typed_field`]).
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
        self.locals.contains(name) || self.receiver == Some(name)
    }

    /// A chain of fields read off a value of `class`, hopping through the types the file knows.
    fn typed(&self, class: &str, rest: &[&str]) -> String {
        decide::typed_chain(class, rest, |current, attr| self.scope.field_class(current, attr))
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

    /// The identity a chain rooted at `root` reads as. `as_value` writes a package-level name as the
    /// value it must be where it stands — an operand ([`decide::value_path`]); a callee or a mention
    /// keeps the plain path, which is what the definition it names is called by.
    fn chain_term(&self, root: &str, attrs: &[&str], as_value: bool) -> String {
        let to_path = |dotted: String| if as_value { decide::value_path(&dotted) } else { decide::path(&dotted) };
        if self.receiver == Some(root) {
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
            return self.typed(class, attrs);
        }
        if self.locals.contains(root) {
            return decide::field(attrs);
        }
        if let Some((member, rest)) = attrs.split_first() {
            if let Some(path) = self.scope.member_path(root, member, rest) {
                return to_path(path);
            }
        }
        to_path(self.scope.free_path(root, attrs))
    }

    /// What a callee is called by: a function's plain path; a method is one more member of its
    /// receiver's chain, read the way a field of it is.
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
            Expr::Paren(inner) | Expr::Conv { e: inner, .. } | Expr::Assert { e: inner, .. } => self.term(inner),
            Expr::Unary { op, e: inner } if op == "&" || op == "*" => self.term(inner),
            Expr::Name(_) | Expr::Selector { .. } => match chain(e) {
                (Some(root), attrs) => self.chain_term(root, &attrs, true),
                (None, attrs) => decide::field(&attrs),
            },
            Expr::Call { f, args, .. } => decide::call(&self.callee(f), args.iter().filter_map(|a| self.constant(a)).collect()),
            Expr::Composite { ty, elems } => {
                let values: Vec<&Expr> = elems
                    .iter()
                    .map(|el| match el {
                        Elem::Field(_, v) | Elem::Keyed(_, v) | Elem::Plain(v) | Elem::KeyedBlank(v) => v,
                    })
                    .collect();
                match ty.as_ref().and_then(|t| self.scope.type_class(t)) {
                    // A struct literal is its type's constructor, with the constants it was handed.
                    Some(class) => decide::call(&class, values.iter().filter_map(|v| self.constant(v)).collect()),
                    None => decide::container(values.iter().map(|v| self.term(v)).collect()),
                }
            }
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
            Expr::Unary { op, e: inner } if op == "-" => {
                let inner = self.term(inner);
                if decide::has_identity(&inner) {
                    format!("-{inner}")
                } else {
                    inner
                }
            }
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
            Expr::Name(_) | Expr::Selector { .. } => match chain(e) {
                (Some(root), attrs) => {
                    if let Some(value) = self.constants.get(root) {
                        return Some(Self::extended(value, &attrs));
                    }
                    if self.is_local(root) {
                        return None;
                    }
                    Some(self.chain_term(root, &attrs, true))
                }
                (None, _) => None,
            },
            Expr::Unary { op, e: inner } if op == "-" => self.constant(inner).map(|c| format!("-{c}")),
            Expr::Unary { op, e: inner } if op == "&" => self.constant(inner),
            Expr::Paren(inner) | Expr::Conv { e: inner, .. } => self.constant(inner),
            Expr::Composite { ty, elems } if ty.as_ref().and_then(|t| self.scope.type_class(t)).is_none() => elems
                .iter()
                .map(|el| match el {
                    Elem::Plain(v) => self.constant(v),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .map(decide::container),
            _ => None,
        }
    }

    /// An expression in a position Go tests for truth.
    fn truth(&mut self, e: &Expr) {
        match e {
            // Structure, not a proposition: its parts are read where the walk reaches them.
            Expr::Binary { op, .. } if matches!(op.as_str(), "==" | "!=" | "<" | "<=" | ">" | ">=") => {}
            Expr::Unary { op, .. } if op == "!" => {}
            Expr::Call { f, args, .. } if self.proposition(f, args).is_some() => {}
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

    /// The import path and member of a call to an imported package's function.
    fn package_call<'e>(&self, f: &'e Expr) -> Option<(&str, &'e str)> {
        let Expr::Selector { operand, field } = f else { return None };
        let Expr::Name(pkg) = operand.as_ref() else { return None };
        if self.is_local(pkg) {
            return None;
        }
        Some((self.scope.imports.get(pkg)?.as_str(), field.as_str()))
    }

    /// The atoms a call spells wherever it stands — membership, an error's identity, an assertion —
    /// or `None` when it spells none.
    fn proposition(&self, f: &Expr, args: &[Expr]) -> Option<Vec<Option<Arc<str>>>> {
        let (pkg, member) = self.package_call(f)?;
        let arg = |i: usize| args.get(i).map(|a| self.term(a));
        let atoms = match (pkg, member) {
            ("strings" | "bytes" | "slices", "Contains" | "ContainsRune" | "ContainsAny") => {
                vec![decide::member(&arg(1)?, &arg(0)?)]
            }
            ("errors", "Is") => vec![decide::equal(&arg(0)?, &arg(1)?)],
            (testify, assertion) if is_testify(testify) => {
                match assertion {
                    "Equal" | "NotEqual" | "EqualValues" | "Exactly" | "ErrorIs" | "NotErrorIs" | "Same" | "NotSame" => {
                        vec![decide::equal(&arg(1)?, &arg(2)?)]
                    }
                    "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" => vec![decide::order(&arg(1)?, &arg(2)?)],
                    "Nil" | "NotNil" | "NoError" | "Error" => vec![decide::equal(&arg(1)?, "None")],
                    "Contains" | "NotContains" => vec![decide::member(&arg(2)?, &arg(1)?)],
                    _ => return None,
                }
            }
            _ => return None,
        };
        Some(atoms)
    }

    /// What `assert.True` / `assert.False` assert: an expression in a truth position, read as one — a
    /// comparison or a membership call inside it is that proposition, not a value tested.
    fn asserted<'e>(&self, f: &Expr, args: &'e [Expr]) -> Option<&'e Expr> {
        let (pkg, member) = self.package_call(f)?;
        (is_testify(pkg) && matches!(member, "True" | "False")).then(|| args.get(1)).flatten()
    }

    /// What a predicate function literal returns, each tested for truth.
    fn predicate(&mut self, callback: &Expr) {
        if let Expr::FuncLit { body, .. } = callback {
            for s in body {
                if let Stmt::Return(values) = s {
                    for v in values {
                        self.truth(v);
                    }
                }
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
            Stmt::If { cond, .. }
            | Stmt::For { head: ForHead::While(cond) | ForHead::Clause { cond: Some(cond), .. }, .. } => self.truth(cond),
            Stmt::Switch { value, cases, .. } => {
                for case in cases {
                    for v in case.values.iter().flatten() {
                        // A tagged case is the equality it abbreviates; a tagless one, a condition.
                        let atom = if let Some(tag) = value {
                            decide::equal(&self.term(tag), &self.term(v))
                        } else {
                            self.truth(v);
                            None
                        };
                        self.push(atom);
                    }
                }
            }
            Stmt::TypeSwitch { value, cases, .. } => {
                let subject = self.term(value);
                for types in cases.iter().filter_map(|c| c.types.as_ref()) {
                    for ty in types {
                        let atom = match ty {
                            Type::Named(n) if n == "nil" => decide::equal(&subject, "None"),
                            other => self.class_of(other).and_then(|class| decide::member(&subject, &class)),
                        };
                        self.push(atom);
                    }
                }
            }
            Stmt::Define { lhs, rhs } if lhs.len() == 2 => self.comma_ok(rhs),
            Stmt::Assign { op, lhs, rhs } if op == "=" && lhs.len() == 2 => self.comma_ok(rhs),
            _ => {}
        }
        each_part(s, &mut |part| match part {
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    /// The comma-ok forms test what they read: `v, ok := m[k]` is `k in m`, `t, ok := x.(T)` is
    /// `x in T`.
    fn comma_ok(&mut self, rhs: &[Expr]) {
        let atom = match rhs {
            [Expr::Index { e, i }] => decide::member(&self.term(i), &self.term(e)),
            [Expr::Assert { e, ty: Some(ty) }] => self.class_of(ty).and_then(|class| decide::member(&self.term(e), &class)),
            _ => None,
        };
        self.push(atom);
    }

    /// The class a type names for a membership test: a declared class, or a universe type by name.
    fn class_of(&self, ty: &Type) -> Option<String> {
        self.scope.type_class(ty).or_else(|| match ty {
            Type::Named(n) => Some(decide::path(n)),
            _ => None,
        })
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Binary { op, l, r } => {
                let atom = match op.as_str() {
                    "==" | "!=" => decide::equal(&self.term(l), &self.term(r)),
                    "<" | "<=" | ">" | ">=" => decide::order(&self.term(l), &self.term(r)),
                    "&&" | "||" => {
                        self.truth(l);
                        None
                    }
                    _ => None,
                };
                self.push(atom);
            }
            Expr::Unary { op, e: inner } if op == "!" => self.truth(inner),
            Expr::Call { f, args, .. } => {
                if let Some(atoms) = self.proposition(f, args) {
                    for atom in atoms {
                        self.push(atom);
                    }
                }
                if let Some(tested) = self.asserted(f, args) {
                    self.truth(tested);
                }
                if let Some(pair) = self.package_call(f) {
                    if PREDICATE_FUNCS.iter().any(|(p, m)| (*p, *m) == pair) {
                        if let Some(callback) = args.last() {
                            self.predicate(callback);
                        }
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
struct Refs<'w, 's> {
    walk: &'w Atoms<'s>,
    calls: Vec<decide::CallSite>,
    mentions: Vec<Arc<str>>,
    /// Fields written other than through a constructor the corpus can name: assigned, or set by a
    /// literal of a type that is no path.
    loose: Vec<Arc<str>>,
}

impl Refs<'_, '_> {
    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        if let Stmt::Assign { lhs, .. } = s {
            for target in lhs {
                if let Expr::Selector { field, .. } = target {
                    self.loose.push(Arc::from(field.as_str()));
                }
            }
        }
        each_part(s, &mut |part| match part {
            Part::Stmt(inner) => self.stmt(inner),
            Part::Stmts(list) => self.stmts(list),
            Part::Expr(e) => self.expr(e),
        });
    }

    /// Whether a callee names a function by path: a name the definition did not bind, or a member of
    /// an imported package — not a method read off a value.
    fn named_callee(&self, f: &Expr) -> bool {
        match f {
            Expr::Name(n) => !self.walk.is_local(n),
            Expr::Selector { .. } => self.walk.package_call(f).is_some(),
            Expr::Paren(inner) => self.named_callee(inner),
            _ => false,
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Call { f, args, .. } => {
                if self.named_callee(f) {
                    let callee = self.walk.callee(f);
                    if callee.starts_with('@') {
                        let open = args.iter().any(|a| matches!(a, Expr::Spread(_)));
                        let args = args.iter().map(|a| (None, Arc::from(self.walk.term(a)))).collect();
                        self.calls.push(decide::CallSite { callee: Arc::from(callee), args, open });
                    }
                } else {
                    self.expr(f);
                }
                for a in args {
                    self.expr(a);
                }
            }
            // A struct literal is its type's constructor, called with its fields by name — or all of
            // them in order, which is the other way Go lets one be written.
            Expr::Composite { ty, elems } => {
                match ty.as_ref().and_then(|t| self.walk.scope.type_class(t)) {
                    Some(class) => {
                        let args: Option<Vec<_>> = elems
                            .iter()
                            .map(|el| match el {
                                Elem::Field(name, v) => Some((Some(Arc::from(name.as_str())), Arc::from(self.walk.term(v)))),
                                Elem::Plain(v) => Some((None, Arc::from(self.walk.term(v)))),
                                Elem::Keyed(..) | Elem::KeyedBlank(_) => None,
                            })
                            .collect();
                        if let Some(args) = args {
                            self.calls.push(decide::CallSite { callee: Arc::from(class), args, open: false });
                        }
                    }
                    None => self.loose.extend(elems.iter().filter_map(|el| match el {
                        Elem::Field(name, _) => Some(Arc::from(name.as_str())),
                        _ => None,
                    })),
                }
                for el in elems {
                    match el {
                        Elem::Field(_, v) | Elem::Plain(v) | Elem::KeyedBlank(v) => self.expr(v),
                        Elem::Keyed(k, v) => {
                            self.expr(k);
                            self.expr(v);
                        }
                    }
                }
            }
            Expr::Name(_) | Expr::Selector { .. } if chain(e).0.is_some() => {
                let (Some(root), attrs) = chain(e) else { return };
                if !self.walk.is_local(root) {
                    let read = self.walk.chain_term(root, &attrs, false);
                    if read.starts_with('@') {
                        self.mentions.push(Arc::from(read));
                    }
                }
            }
            Expr::FuncLit { body, .. } => self.stmts(body),
            _ => for_each_child(e, &mut |c| match c {
                Child::Expr(x) => self.expr(x),
                Child::Body(b) => self.stmts(b),
            }),
        }
    }
}

/// Everything the decisions pass reads off one callable. `open` — the callable is reached by calls the
/// corpus cannot show (a method, dispatched through whatever interface it satisfies; a function the
/// runtime calls), so what its parameters hold is not the corpus's to say.
pub(crate) fn decides(f: &Func, scope: &FileScope, open: bool) -> DecisionFacts {
    let mut locals = Locals::default();
    locals.params(&f.params);
    if let Results::List(list) = &f.results {
        locals.params(list);
    }
    locals.stmts(&f.body);
    let mut bound = locals.bound;
    // A type parameter is a type the caller picks, no more a fact than a value the caller passes; the
    // blank identifier is a value thrown away.
    bound.extend(f.type_params.iter().cloned());
    bound.insert("_".to_owned());
    let order: Vec<String> = f
        .params
        .iter()
        .flat_map(|d| if d.names.is_empty() { vec!["_".to_owned()] } else { d.names.clone() })
        .collect();
    // A variadic parameter collects every argument from its position on: no one call site's argument
    // is what it holds.
    let variadic: HashSet<&String> = f.params.iter().filter(|d| d.variadic).flat_map(|d| d.names.iter()).collect();
    let untouched: HashSet<String> = order
        .iter()
        .filter(|name| locals.sites.get(*name) == Some(&1) && !variadic.contains(name))
        .cloned()
        .collect();
    let types: HashMap<String, String> = f
        .params
        .iter()
        .filter_map(|d| Some((d, scope.type_class(d.ty.as_ref()?)?)))
        .flat_map(|(d, class)| d.names.iter().map(move |n| (n.clone(), class.clone())))
        .collect();
    let mut single = locals.lets;
    single.retain(|(name, _)| locals.sites.get(name) == Some(&1));
    let mut walk = Atoms {
        scope,
        receiver: f.receiver.as_deref(),
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
        if matches!(value, Expr::Call { .. }) {
            let computation = walk.term(value);
            walk.computed.insert(name.clone(), computation);
            continue;
        }
        // A value built from a struct literal — or its address — is of that struct: its fields are typed.
        let built = match value {
            Expr::Unary { op, e } if op == "&" => e.as_ref(),
            other => other,
        };
        if let Expr::Composite { ty: Some(ty), .. } = built {
            if let Some(class) = scope.type_class(ty) {
                walk.types.entry(name.clone()).or_insert(class);
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
    use super::{decides, package_path, FileScope};
    use crate::canon::file_imports;
    use crate::defs::callable_of;
    use crate::ir::lower_func;
    use crate::tree::{kids, parse};
    use dup_defs_core::decide::{canonical, render, resolve_typed_fields, DecisionFacts};

    /// What the decisions pass reads off the first function or method in `src`, as if the file lived
    /// in package `shop/orders`.
    fn facts(src: &str) -> DecisionFacts {
        let tree = parse(src).expect("fixture parses");
        let root = tree.root_node();
        let imports = file_imports(root, src);
        let scope = FileScope::new("shop/orders/orders.go", root, src, &imports, true);
        let decl = kids(root)
            .into_iter()
            .find(|k| matches!(k.kind(), "function_declaration" | "method_declaration"))
            .expect("fixture has a function");
        let c = callable_of(decl, src).expect("with a body");
        decides(&lower_func(&c, src), &scope, false)
    }

    /// The atoms as the engine reads them with nothing inferred (a typed field is the field of some
    /// value), rendered.
    fn atoms(src: &str) -> Vec<String> {
        let mut out: Vec<String> =
            facts(src).atoms.iter().map(|a| render(&canonical(&resolve_typed_fields(a, |_, _| None)))).collect();
        out.sort();
        out.dedup();
        out
    }

    #[test]
    fn a_package_is_the_module_go_mod_declares_and_the_directory_under_it() {
        let root = std::env::temp_dir().join(format!("go-canon-gomod-{}", std::process::id()));
        std::fs::create_dir_all(root.join("core/store")).expect("temp dir");
        std::fs::write(root.join("go.mod"), "module example.com/shop\n\ngo 1.22\n").expect("go.mod");
        let file = root.join("core/store/orders.go");
        assert_eq!(package_path(&file.to_string_lossy()), "example.com.shop.core.store");
        assert_eq!(package_path(&root.join("main.go").to_string_lossy()), "example.com.shop");
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(package_path("shop/orders/orders.go"), "shop.orders", "no module: the directory");
    }

    #[test]
    fn negation_and_the_spelling_of_a_test_do_not_change_the_decision() {
        let import = "package orders\nimport \"shop/model\"\n";
        let refuse = atoms(&format!(
            "{import}func refuse(o *Order) error {{ if o.Status != model.StatusOpen {{ return ErrClosed }}; return nil }}\n"
        ));
        let accept = atoms(&format!("{import}func accept(o *Order) bool {{ return o.Status == model.StatusOpen }}\n"));
        let switched = atoms(&format!(
            "{import}func label(o *Order) string {{ switch o.Status {{ case model.StatusOpen: return \"o\" }}; return \"\" }}\n"
        ));
        assert_eq!(refuse, vec!["@shop.model.StatusOpen = _.Status"]);
        assert_eq!(accept, refuse);
        assert_eq!(switched, refuse);
    }

    #[test]
    fn a_package_level_operand_is_a_value_whatever_its_casing() {
        // `StatusOpen` is spelled like a type; as the operand of a comparison it can only be a value,
        // so the field compared with it is a fact, not a filter.
        let found = facts("package orders\nfunc open(o *Order) bool { return o.Status == StatusOpen }\n");
        assert!(found.atoms.iter().all(|a| dup_defs_core::decide::is_fact(a)), "{:?}", found.atoms);
        assert_eq!(found.atoms.len(), 1);
    }

    #[test]
    fn absence_and_an_error_are_one_test_each_however_spelled() {
        let absent = vec!["None = _.PaidAt"];
        assert_eq!(atoms("package orders\nfunc f(o *Order) bool { return o.PaidAt == nil }\n"), absent);
        assert_eq!(atoms("package orders\nfunc f(o *Order) bool { return nil != o.PaidAt }\n"), absent);
        let wrapped = atoms(
            "package orders\nimport \"errors\"\nfunc f(o *Order) bool { err := o.Charge(); return errors.Is(err, ErrDeclined) }\n",
        );
        let compared =
            atoms("package orders\nfunc f(o *Order) bool { err := o.Charge(); return err == ErrDeclined }\n");
        assert_eq!(wrapped, vec!["@shop.orders.ErrDeclined = _.Charge()"]);
        assert_eq!(compared, wrapped);
    }

    #[test]
    fn membership_spellings_meet() {
        let listed = atoms("package orders\nimport \"slices\"\nfunc f(o *Order) bool { return slices.Contains(allowed, o.Kind) }\n");
        let indexed = atoms("package orders\nfunc f(o *Order) bool { _, ok := allowed[o.Kind]; return ok }\n");
        assert_eq!(listed, vec!["_.Kind in @shop.orders.allowed"]);
        assert_eq!(indexed, listed);
        assert_eq!(
            atoms("package orders\nimport \"strings\"\nfunc f(name string) bool { return strings.Contains(name, \"..\") }\n"),
            vec!["\"..\" in $name"]
        );
    }

    #[test]
    fn a_type_switch_and_an_assertion_test_membership_and_nil_absence() {
        let switched = atoms(
            "package orders\nimport \"fmt\"\nfunc f(x any) string { switch v := x.(type) { case fmt.Stringer: return v.String(); case nil: return \"\" }; return \"?\" }\n",
        );
        assert_eq!(switched, vec!["$x = None", "$x in @fmt.Stringer"]);
        let asserted =
            atoms("package orders\nimport \"fmt\"\nfunc f(x any) bool { _, ok := x.(fmt.Stringer); return ok }\n");
        assert_eq!(asserted, vec!["$x in @fmt.Stringer"]);
    }

    #[test]
    fn the_receiver_is_the_value_the_method_is_read_off() {
        let found = atoms("package orders\nfunc (o *Order) Open() bool { return o.State == StateOpen }\n");
        assert_eq!(found, vec!["@shop.orders.StateOpen = _.State"]);
    }

    #[test]
    fn a_predicate_and_an_assertion_are_truth_positions() {
        let filtered = atoms(
            "package orders\nimport \"slices\"\nfunc f(xs []*Order) bool { return slices.ContainsFunc(xs, func(o *Order) bool { return o.Paid }) }\n",
        );
        let tested = atoms("package orders\nfunc f(xs []*Order) { for _, o := range xs { if o.Paid { ship(o) } } }\n");
        assert_eq!(filtered, vec!["? _.Paid"]);
        assert_eq!(tested, filtered);
        let asserted = atoms(
            "package orders\nimport \"github.com/stretchr/testify/assert\"\nfunc TestOpen(t *testing.T) { o := load(); assert.Equal(t, StatusOpen, o.Status) }\n",
        );
        let checked = atoms(
            "package orders\nfunc TestOpen(t *testing.T) { o := load(); if o.Status != StatusOpen { t.Fatal(\"closed\") } }\n",
        );
        assert_eq!(asserted, vec!["@shop.orders.StatusOpen = _.Status"]);
        assert_eq!(checked, asserted);
        let contained = atoms(
            "package orders\nimport (\n\"strings\"\n\"github.com/stretchr/testify/assert\"\n)\nfunc TestName(t *testing.T) { assert.True(t, strings.Contains(load().Name, \"x\")) }\n",
        );
        assert_eq!(contained, vec!["\"x\" in _.Name"], "the assertion's proposition, not a truth of the call");
    }

    #[test]
    fn a_local_bound_to_a_constant_is_that_constant_and_a_parameter_is_marked() {
        let direct = atoms("package orders\nfunc f(o *Order) bool { return o.Total < MaxTotal }\n");
        let through = atoms("package orders\nfunc f(o *Order) bool { limit := MaxTotal; return o.Total < limit }\n");
        assert_eq!(direct, vec!["@shop.orders.MaxTotal < _.Total"]);
        assert_eq!(through, direct);
        assert_eq!(atoms("package orders\nfunc f(force bool, o *Order) int { if !force { return o.X }; return 0 }\n"), vec![
            "? $force"
        ]);
        let hex = atoms("package orders\nfunc f(o *Order) bool { return o.Flags == 0x10 }\n");
        assert_eq!(hex, atoms("package orders\nfunc f(o *Order) bool { return o.Flags == 16 }\n"));
    }

    #[test]
    fn calls_and_struct_literals_are_the_call_sites_parameters_are_filled_from() {
        let found = facts(
            "package orders\nimport \"shop/model\"\nfunc f(xs []int, opts ...Option) { charge(model.StatusOpen, 3); _ = &Order{Status: StatusOpen, Total: 0}; _ = Line{1, \"a\"} }\n",
        );
        // Each call as `callee(key=arg, …)`, a positional argument keyless.
        let calls: Vec<String> = found
            .calls
            .iter()
            .map(|c| {
                let args: Vec<String> =
                    c.args.iter().map(|(k, v)| k.as_ref().map_or_else(|| v.to_string(), |k| format!("{k}={v}"))).collect();
                format!("{}({})", c.callee, args.join(", "))
            })
            .collect();
        assert_eq!(calls, vec![
            "@shop.orders.charge(@=shop.model.StatusOpen, 3)",
            "@shop.orders.Order(Status=@=shop.orders.StatusOpen, Total=0)",
            "@shop.orders.Line(1, \"a\")",
        ]);
        let params: Vec<&str> = found.params.iter().map(|p| &**p).collect();
        assert_eq!(params, vec!["xs", "opts"]);
        let mentions: Vec<&str> = found.mentions.iter().map(|m| &**m).collect();
        assert_eq!(mentions, vec!["@shop.model.StatusOpen", "@shop.orders.StatusOpen"]);
    }

    #[test]
    fn a_variadic_parameter_is_no_one_arguments() {
        let found = atoms("package orders\nfunc f(names ...string) bool { return names == nil }\n");
        assert_eq!(found, vec!["None = _"], "a hole, never filled from a call site");
    }
}
