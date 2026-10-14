//! What a Python definition **decides on** — the frontend half of [`Facets::decides`].
//!
//! Python says which expressions are propositions, and this module only reads them off:
//!
//! - every **comparison**, wherever it stands — a comparison is Boolean by grammar, and one written
//!   as a query filter or a returned value tests the same fact as one written under an `if`;
//! - every expression in a position the language **tests for truth** — the test of an `if`, `elif`,
//!   `while`, `assert` and conditional expression, a comprehension's `if`, a `match` guard, and the
//!   operands of `and`, `or` and `not`;
//! - every **`match` pattern**, which is a comparison the grammar spells as a shape;
//! - `isinstance` and `issubclass`, which the language defines as membership in a type.
//!
//! A term's identity follows the binding the language gives the name it starts from: a name the
//! definition binds — a parameter, a local, a loop or comprehension variable, a capture — is a hole;
//! an imported name is the path it was imported from; a name the module binds itself is a local path;
//! anything else is a builtin. Nothing is evaluated and nothing is guessed: a value the definition
//! computed is a hole, and what computed it is the callee's identity.
//!
//! [`Facets::decides`]: dup_defs_core::Facets::decides

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::decide;
use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{self as ast, BoolOp, CmpOp, Expr, Parameters, Pattern, Singleton, Stmt, UnaryOp};

use crate::canon::{repr_number, Import};

/// What the names of one file stand for, read once per file and shared by its definitions.
pub(crate) struct FileScope {
    /// Local name → the path it was imported from, as written (a relative import keeps its dots).
    imports: HashMap<String, String>,
    /// The file's own module path, dotted, as its location spells it.
    module: String,
    /// The package a relative import is resolved against: the module itself for a package's
    /// `__init__`, its parent otherwise — which is the language's rule.
    package: String,
    /// Names the module binds at top level other than by importing.
    defined: HashSet<String>,
    /// The module's record classes — what each one's constructor takes, in order, with the class each
    /// field is annotated as, where the annotation names one.
    records: HashMap<String, Vec<(String, Option<String>)>>,
}

impl FileScope {
    /// The file's own module path — what a definition's identity is local to.
    pub(crate) fn module(&self) -> &str {
        &self.module
    }

    pub(crate) fn new(file: &str, imports: &[Import], body: &[Stmt]) -> Self {
        let trimmed = file.strip_prefix("./").unwrap_or(file);
        let trimmed = trimmed.strip_suffix(".py").unwrap_or(trimmed);
        let (stem, is_package) = match trimmed.strip_suffix("/__init__") {
            Some(package) => (package, true),
            None => (trimmed, false),
        };
        let module: String = stem.split(['/', '\\']).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(".");
        let package = if is_package {
            module.clone()
        } else {
            module.rsplit_once('.').map_or_else(String::new, |(parent, _)| parent.to_owned())
        };
        let mut defined = HashSet::new();
        module_level_names(body, &mut defined);
        let mut scope = Self {
            imports: imports.iter().map(|i| (i.local.clone(), i.path.clone())).collect(),
            module,
            package,
            defined,
            records: HashMap::new(),
        };
        let records = body
            .iter()
            .filter_map(|stmt| match stmt {
                Stmt::ClassDef(c) => scope.record_fields(c).map(|fields| (c.name.id.as_str().to_owned(), fields)),
                _ => None,
            })
            .collect();
        scope.records = records;
        scope
    }

    /// The fields a class's constructor takes, when the class is a record: no base to inherit fields or
    /// a constructor from, no method that builds or reshapes the instance (`__init__`, `__new__`,
    /// `__post_init__`), and at least one annotated field. Those are the classes whose instances hold
    /// exactly what their constructor calls passed — a dataclass, in practice, though nothing here asks
    /// what decorates it. `ClassVar` fields are not constructor parameters and are left out.
    fn record_fields(&self, class: &ast::StmtClassDef) -> Option<Vec<(String, Option<String>)>> {
        if class.arguments.as_ref().is_some_and(|a| !a.args.is_empty() || !a.keywords.is_empty()) {
            return None;
        }
        let mut fields = Vec::new();
        for stmt in &class.body {
            match stmt {
                Stmt::FunctionDef(f) if matches!(f.name.id.as_str(), "__init__" | "__new__" | "__post_init__") => {
                    return None;
                }
                Stmt::AnnAssign(a) => {
                    let Expr::Name(name) = a.target.as_ref() else { continue };
                    if self.annotation_names(&a.annotation, "ClassVar") {
                        continue;
                    }
                    fields.push((name.id.as_str().to_owned(), self.annotated_class(&a.annotation)));
                }
                _ => {}
            }
        }
        (!fields.is_empty()).then_some(fields)
    }

    /// True when the annotation is `typing.<name>[…]` — read through the import table, not by spelling.
    fn annotation_names(&self, annotation: &Expr, name: &str) -> bool {
        let Expr::Subscript(s) = annotation else { return false };
        match chain(&s.value) {
            (Some(root), attrs) => self.free(root, &attrs) == decide::path(&format!("typing.{name}")),
            _ => false,
        }
    }

    /// The one class an annotation names — `C`, `C | None`, `Optional[C]`, `Union[C, None]`, or the
    /// forward reference `"C"` — or `None` when it names none or several.
    pub(crate) fn annotated_class(&self, annotation: &Expr) -> Option<String> {
        match annotation {
            Expr::Name(_) | Expr::Attribute(_) => match chain(annotation) {
                (Some(root), attrs) => Some(self.free(root, &attrs)),
                _ => None,
            },
            Expr::StringLiteral(s) => {
                let text = s.value.to_str();
                text.chars().all(|c| c.is_alphanumeric() || c == '_').then(|| self.free(text, &[]))
            }
            Expr::BinOp(b) if matches!(b.op, ast::Operator::BitOr) => {
                let (left, right) = (&*b.left, &*b.right);
                match (matches!(left, Expr::NoneLiteral(_)), matches!(right, Expr::NoneLiteral(_))) {
                    (false, true) => self.annotated_class(left),
                    (true, false) => self.annotated_class(right),
                    _ => None,
                }
            }
            Expr::Subscript(s) if self.annotation_names(annotation, "Optional") => self.annotated_class(&s.slice),
            Expr::Subscript(s) if self.annotation_names(annotation, "Union") => match &*s.slice {
                Expr::Tuple(t) => {
                    let named: Vec<&Expr> = t.elts.iter().filter(|e| !matches!(e, Expr::NoneLiteral(_))).collect();
                    match named.as_slice() {
                        [only] => self.annotated_class(only),
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// The fields of a record this module defines, by the class's term.
    pub(crate) fn record(&self, class_term: &str) -> Option<&Vec<(String, Option<String>)>> {
        self.records.iter().find(|(name, _)| decide::local(&self.module, name) == class_term).map(|(_, fields)| fields)
    }

    /// The class a field of a record this module defines is annotated as.
    fn field_class(&self, class_term: &str, field: &str) -> Option<String> {
        self.record(class_term)?.iter().find(|(name, _)| name == field).and_then(|(_, class)| class.clone())
    }

    /// The term a free name stands for, with `rest` (attributes after it) appended.
    fn free(&self, name: &str, rest: &[&str]) -> String {
        let tail = |head: &str| {
            let mut out = head.to_owned();
            for attr in rest {
                out.push('.');
                out.push_str(attr);
            }
            out
        };
        if let Some(path) = self.imports.get(name) {
            let level = path.chars().take_while(|&c| c == '.').count();
            if level == 0 {
                return decide::path(&tail(path));
            }
            // `from ..pkg import x`: the dots climb from the package, then the rest descends. What the
            // tree calls the result is not known to this file, so it is a local path.
            let mut base: Vec<&str> = self.package.split('.').filter(|s| !s.is_empty()).collect();
            for _ in 1..level {
                base.pop();
            }
            let written: Vec<&str> = path[level..].split('.').filter(|s| !s.is_empty()).collect();
            let Some((member, module_tail)) = written.split_last() else {
                return decide::path(&tail(name));
            };
            base.extend(module_tail);
            return decide::local(&base.join("."), &tail(member));
        }
        if self.defined.contains(name) {
            return decide::local(&self.module, &tail(name));
        }
        decide::path(&tail(name))
    }
}

/// Names a module binds at top level by defining or assigning them — including under a module-level
/// `if`/`try`/`with`, which run at import. Imports are kept apart, in [`FileScope::imports`].
fn module_level_names(body: &[Stmt], out: &mut HashSet<String>) {
    for stmt in body {
        match stmt {
            Stmt::FunctionDef(f) => {
                out.insert(f.name.id.as_str().to_owned());
            }
            Stmt::ClassDef(c) => {
                out.insert(c.name.id.as_str().to_owned());
            }
            Stmt::Assign(a) => a.targets.iter().for_each(|t| target_names(t, out)),
            Stmt::AnnAssign(a) => target_names(&a.target, out),
            Stmt::AugAssign(a) => target_names(&a.target, out),
            Stmt::TypeAlias(a) => target_names(&a.name, out),
            Stmt::If(node) => {
                module_level_names(&node.body, out);
                for clause in &node.elif_else_clauses {
                    module_level_names(&clause.body, out);
                }
            }
            Stmt::Try(node) => {
                module_level_names(&node.body, out);
                module_level_names(&node.orelse, out);
                module_level_names(&node.finalbody, out);
                for handler in &node.handlers {
                    let ast::ExceptHandler::ExceptHandler(h) = handler;
                    module_level_names(&h.body, out);
                }
            }
            Stmt::With(node) => module_level_names(&node.body, out),
            _ => {}
        }
    }
}

fn target_names(expr: &Expr, out: &mut HashSet<String>) {
    match expr {
        Expr::Name(name) => {
            out.insert(name.id.as_str().to_owned());
        }
        Expr::Starred(starred) => target_names(&starred.value, out),
        Expr::Tuple(tuple) => tuple.elts.iter().for_each(|e| target_names(e, out)),
        Expr::List(list) => list.elts.iter().for_each(|e| target_names(e, out)),
        _ => {}
    }
}

/// Every name a definition binds, at any depth: what the language makes local to it. `global` and
/// `nonlocal` take a name back out — it then refers to the enclosing scope, as written.
#[derive(Default)]
struct Locals<'a> {
    bound: HashSet<String>,
    outer: HashSet<String>,
    /// How many places bind each name — any binding at all, at any depth.
    sites: HashMap<String, usize>,
    /// The value of each plain `name = value` (or `name: T = value`), as written.
    assigned: HashMap<String, &'a Expr>,
}

/// What a definition binds: the names that are holes, and among them the ones bound exactly once by a
/// plain assignment, with the value they were bound to — a single-assignment name *is* that value
/// wherever it is read, since there is no other place it could have got one from.
struct Bindings<'a> {
    bound: HashSet<String>,
    single: Vec<(String, &'a Expr)>,
    /// The definition's own parameters, in the order positional arguments fill them.
    params: Vec<String>,
    /// Those bound nowhere but by the call: what they hold is what every caller hands them.
    untouched: HashSet<String>,
}

impl<'a> Locals<'a> {
    fn of(def: &'a Stmt) -> Bindings<'a> {
        let mut locals = Self::default();
        let mut params = Vec::new();
        if let Stmt::FunctionDef(f) = def {
            let p = &f.parameters;
            params = p.posonlyargs.iter().chain(&p.args).chain(&p.kwonlyargs).map(|x| x.parameter.name.id.as_str().to_owned()).collect();
            locals.params(&f.parameters);
            locals.visit_body(&f.body);
        }
        let Self { mut bound, outer, sites, assigned } = locals;
        bound.retain(|name| !outer.contains(name));
        let mut single: Vec<(String, &Expr)> = assigned
            .into_iter()
            .filter(|(name, _)| sites.get(name) == Some(&1) && bound.contains(name))
            .collect();
        single.sort_by(|a, b| a.0.cmp(&b.0));
        let untouched = params.iter().filter(|name| sites.get(*name) == Some(&1) && bound.contains(*name)).cloned().collect();
        Bindings { bound, single, params, untouched }
    }

    fn bind_target(&mut self, target: &Expr) {
        let mut names = HashSet::new();
        target_names(target, &mut names);
        for name in names {
            *self.sites.entry(name.clone()).or_insert(0) += 1;
            self.bound.insert(name);
        }
    }

    fn bind_name(&mut self, name: &str) {
        *self.sites.entry(name.to_owned()).or_insert(0) += 1;
        self.bound.insert(name.to_owned());
    }

    fn params(&mut self, params: &Parameters) {
        for p in params.posonlyargs.iter().chain(&params.args).chain(&params.kwonlyargs) {
            self.bind_name(p.parameter.name.id.as_str());
        }
        for p in params.vararg.iter().chain(params.kwarg.iter()) {
            self.bind_name(p.name.id.as_str());
        }
    }
}

impl<'a> Visitor<'a> for Locals<'a> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::FunctionDef(f) => {
                self.bind_name(f.name.id.as_str());
                self.params(&f.parameters);
            }
            Stmt::ClassDef(c) => {
                self.bind_name(c.name.id.as_str());
            }
            Stmt::Assign(a) => {
                a.targets.iter().for_each(|t| self.bind_target(t));
                if let [Expr::Name(name)] = a.targets.as_slice() {
                    self.assigned.insert(name.id.as_str().to_owned(), &a.value);
                }
            }
            Stmt::AnnAssign(a) => {
                self.bind_target(&a.target);
                if let (Expr::Name(name), Some(value)) = (a.target.as_ref(), &a.value) {
                    self.assigned.insert(name.id.as_str().to_owned(), value);
                }
            }
            Stmt::AugAssign(a) => self.bind_target(&a.target),
            Stmt::For(f) => self.bind_target(&f.target),
            Stmt::With(w) => {
                for item in &w.items {
                    if let Some(vars) = &item.optional_vars {
                        self.bind_target(vars);
                    }
                }
            }
            Stmt::Global(g) => g.names.iter().for_each(|n| {
                self.outer.insert(n.id.as_str().to_owned());
            }),
            Stmt::Nonlocal(n) => n.names.iter().for_each(|n| {
                self.outer.insert(n.id.as_str().to_owned());
            }),
            _ => {}
        }
        visitor::walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::Named(named) => self.bind_target(&named.target),
            Expr::Lambda(lambda) => {
                if let Some(params) = &lambda.parameters {
                    self.params(params);
                }
            }
            _ => {}
        }
        visitor::walk_expr(self, expr);
    }

    fn visit_comprehension(&mut self, comprehension: &'a ast::Comprehension) {
        self.bind_target(&comprehension.target);
        visitor::walk_comprehension(self, comprehension);
    }

    fn visit_except_handler(&mut self, handler: &'a ast::ExceptHandler) {
        let ast::ExceptHandler::ExceptHandler(h) = handler;
        if let Some(name) = &h.name {
            self.bind_name(name.id.as_str());
        }
        visitor::walk_except_handler(self, handler);
    }

    fn visit_pattern(&mut self, pattern: &'a Pattern) {
        match pattern {
            Pattern::MatchAs(p) => {
                if let Some(name) = &p.name {
                    self.bind_name(name.id.as_str());
                }
            }
            Pattern::MatchStar(p) => {
                if let Some(name) = &p.name {
                    self.bind_name(name.id.as_str());
                }
            }
            Pattern::MatchMapping(p) => {
                if let Some(rest) = &p.rest {
                    self.bind_name(rest.id.as_str());
                }
            }
            _ => {}
        }
        visitor::walk_pattern(self, pattern);
    }
}

/// An attribute chain from its root outwards, or `None` when it does not start at a name.
pub(crate) fn chain(mut expr: &Expr) -> (Option<&str>, Vec<&str>) {
    let mut attrs = Vec::new();
    while let Expr::Attribute(a) = expr {
        attrs.push(a.attr.id.as_str());
        expr = &a.value;
    }
    attrs.reverse();
    match expr {
        Expr::Name(n) => (Some(n.id.as_str()), attrs),
        _ => (None, attrs),
    }
}

/// The text of a literal, or `None` for anything that is not one.
fn literal(expr: &Expr) -> Option<String> {
    Some(match expr {
        Expr::StringLiteral(s) => decide::literal(s.value.to_str()),
        Expr::BytesLiteral(b) => {
            let bytes: Vec<u8> = b.value.bytes().collect();
            format!("b{}", decide::literal(&String::from_utf8_lossy(&bytes)))
        }
        Expr::NumberLiteral(n) => repr_number(&n.value),
        Expr::BooleanLiteral(b) => if b.value { "True" } else { "False" }.to_owned(),
        Expr::NoneLiteral(_) => "None".to_owned(),
        Expr::EllipsisLiteral(_) => "...".to_owned(),
        _ => return None,
    })
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
    /// Locals whose class the text names — an annotated parameter, a single assignment of a call or of
    /// a field whose class is known. A field read off one is a typed field ([`decide::typed_field`]).
    types: HashMap<String, String>,
    out: Vec<Arc<str>>,
}

impl Atoms<'_> {
    fn push(&mut self, atom: Option<Arc<str>>) {
        if let Some(atom) = atom {
            self.out.push(atom);
        }
    }

    fn named(&self, name: &str, rest: &[&str]) -> String {
        if let Some(value) = self.constants.get(name) {
            return Self::extended(value, rest);
        }
        // Read bare, a local holding a call's result is that computation and a parameter is whatever
        // the callers hand it. Read through an attribute, the attribute is the fact and the value it
        // was read off stays a hole — the same rule every receiver gets.
        if rest.is_empty() {
            if let Some(computation) = self.computed.get(name) {
                return computation.clone();
            }
            if self.params.contains(name) {
                return decide::param(name);
            }
        } else if let Some(class) = self.types.get(name) {
            return self.typed(class, rest);
        }
        if self.locals.contains(name) {
            decide::field(rest)
        } else {
            self.scope.free(name, rest)
        }
    }

    /// A chain of fields read off a value of `class`: each hop whose class the module knows moves to
    /// it, and the last known hop is the typed field, with whatever was read past it.
    fn typed(&self, class: &str, rest: &[&str]) -> String {
        let mut current = class.to_owned();
        for (i, attr) in rest.iter().enumerate() {
            if i + 1 == rest.len() {
                return decide::typed_field(&current, attr, &[]);
            }
            match self.scope.field_class(&current, attr) {
                Some(next) => current = next,
                None => return decide::typed_field(&current, attr, &rest[i + 1..]),
            }
        }
        decide::field(rest)
    }

    /// A constant's attributes: a path grows by them; any other constant has none a text could read,
    /// so what was read off it is a field of a value.
    fn extended(value: &str, rest: &[&str]) -> String {
        if rest.is_empty() {
            value.to_owned()
        } else if value.starts_with('@') {
            format!("{value}.{}", rest.join("."))
        } else {
            decide::field(rest)
        }
    }

    /// The identity of an operand.
    fn term(&self, expr: &Expr) -> String {
        match expr {
            Expr::Name(_) | Expr::Attribute(_) => match chain(expr) {
                (Some(root), attrs) => self.named(root, &attrs),
                // A chain rooted at a computed value: the value is a hole, the attributes are not.
                (None, attrs) => decide::field(&attrs),
            },
            Expr::Call(call) => {
                let callee = self.term(&call.func);
                let mut constants: Vec<String> =
                    call.arguments.args.iter().filter_map(|a| self.constant(a)).collect();
                for kw in &call.arguments.keywords {
                    if let (Some(arg), Some(value)) = (&kw.arg, self.constant(&kw.value)) {
                        constants.push(format!("{}={value}", arg.id.as_str()));
                    }
                }
                decide::call(&callee, constants)
            }
            Expr::Await(a) => self.term(&a.value),
            Expr::Starred(s) => self.term(&s.value),
            Expr::Subscript(s) => {
                let key = self.constant(&s.slice).unwrap_or_else(|| decide::HOLE.to_owned());
                format!("{}[{key}]", self.term(&s.value))
            }
            Expr::Set(_) | Expr::Tuple(_) | Expr::List(_) => {
                let elts: &[Expr] = match expr {
                    Expr::Set(s) => &s.elts,
                    Expr::Tuple(t) => &t.elts,
                    Expr::List(l) => &l.elts,
                    _ => &[],
                };
                decide::container(elts.iter().map(|e| self.term(e)).collect())
            }
            Expr::BinOp(b) => {
                let rendered = format!("({} {} {})", self.term(&b.left), b.op.as_str(), self.term(&b.right));
                if decide::has_identity(&rendered) {
                    rendered
                } else {
                    decide::HOLE.to_owned()
                }
            }
            Expr::UnaryOp(u) if !matches!(u.op, UnaryOp::Not) => {
                let inner = self.term(&u.operand);
                if decide::has_identity(&inner) {
                    format!("{}{inner}", u.op.as_str())
                } else {
                    inner
                }
            }
            _ => literal(expr).unwrap_or_else(|| decide::HOLE.to_owned()),
        }
    }

    /// A value handed to a call that is fixed by the text rather than computed: a literal, a name
    /// the definition did not bind, or a container of those.
    fn constant(&self, expr: &Expr) -> Option<String> {
        if let Some(lit) = literal(expr) {
            return Some(lit);
        }
        match expr {
            Expr::Name(_) | Expr::Attribute(_) => match chain(expr) {
                (Some(root), attrs) if self.constants.contains_key(root) => {
                    Some(Self::extended(&self.constants[root], &attrs))
                }
                (Some(root), attrs) if !self.locals.contains(root) => Some(self.scope.free(root, &attrs)),
                _ => None,
            },
            Expr::UnaryOp(u) if !matches!(u.op, UnaryOp::Not) => {
                self.constant(&u.operand).map(|inner| format!("{}{inner}", u.op.as_str()))
            }
            Expr::Set(_) | Expr::Tuple(_) | Expr::List(_) => {
                let elts: &[Expr] = match expr {
                    Expr::Set(s) => &s.elts,
                    Expr::Tuple(t) => &t.elts,
                    Expr::List(l) => &l.elts,
                    _ => &[],
                };
                elts.iter().map(|e| self.constant(e)).collect::<Option<Vec<_>>>().map(decide::container)
            }
            _ => None,
        }
    }

    /// `isinstance(x, T)` / `issubclass(x, T)` written as the membership the language defines it as.
    fn instance_test(&self, call: &ast::ExprCall) -> Option<Arc<str>> {
        let callee = self.term(&call.func);
        let builtin = |name: &str| callee == decide::path(name);
        if !(builtin("isinstance") || builtin("issubclass")) || !call.arguments.keywords.is_empty() {
            return None;
        }
        match &*call.arguments.args {
            [value, kind] => decide::member(&self.term(value), &self.term(kind)),
            _ => None,
        }
    }

    /// The argument of `bool(x)`, which the language defines as testing `x` for truth.
    fn truth_call<'e>(&self, call: &'e ast::ExprCall) -> Option<&'e Expr> {
        match (&*call.arguments.args, call.arguments.keywords.is_empty()) {
            ([only], true) if self.term(&call.func) == decide::path("bool") => Some(only),
            _ => None,
        }
    }

    /// An expression in a position the language tests for truth.
    fn truth(&mut self, expr: &Expr) {
        match expr {
            // Structure, not a proposition: its parts are read where the walk reaches it.
            Expr::Compare(_) => {}
            // `and`/`or` test every operand but the last and hand the last back as their value — so
            // the last is tested exactly when the whole expression is, which is now.
            Expr::BoolOp(b) => {
                if let Some(last) = b.values.last() {
                    self.truth(last);
                }
            }
            Expr::UnaryOp(u) if matches!(u.op, UnaryOp::Not) => {}
            Expr::Named(named) => self.truth(&named.value),
            Expr::Call(call) => {
                // Both are propositions wherever they stand, and the walk reads them where it
                // reaches the call: membership in a type, and the truth of the argument.
                if self.instance_test(call).is_some() || self.truth_call(call).is_some() {
                    return;
                }
                let atom = decide::truth(&self.term(expr));
                self.push(atom);
            }
            _ => {
                let atom = decide::truth(&self.term(expr));
                self.push(atom);
            }
        }
    }

    fn pattern(&mut self, subject: &str, pattern: &Pattern) {
        match pattern {
            Pattern::MatchValue(p) => {
                let atom = decide::equal(subject, &self.term(&p.value));
                self.push(atom);
            }
            Pattern::MatchSingleton(p) => {
                let value = match p.value {
                    Singleton::None => "None",
                    Singleton::True => "True",
                    Singleton::False => "False",
                };
                let atom = decide::equal(subject, value);
                self.push(atom);
            }
            Pattern::MatchClass(p) => {
                let atom = decide::member(subject, &self.term(&p.cls));
                self.push(atom);
                for inner in &p.arguments.patterns {
                    self.pattern(decide::HOLE, inner);
                }
                for kw in &p.arguments.keywords {
                    let attr = kw.attr.id.as_str();
                    let field = if subject.starts_with(decide::HOLE) {
                        format!("{subject}.{attr}")
                    } else {
                        decide::field(&[attr])
                    };
                    self.pattern(&field, &kw.pattern);
                }
            }
            Pattern::MatchOr(p) => p.patterns.iter().for_each(|alt| self.pattern(subject, alt)),
            Pattern::MatchAs(p) => {
                if let Some(inner) = &p.pattern {
                    self.pattern(subject, inner);
                }
            }
            Pattern::MatchMapping(p) => {
                for (key, value) in p.keys.iter().zip(&p.patterns) {
                    let key = self.constant(key).unwrap_or_else(|| decide::HOLE.to_owned());
                    self.pattern(&format!("{subject}[{key}]"), value);
                }
            }
            Pattern::MatchSequence(p) => p.patterns.iter().for_each(|inner| self.pattern(decide::HOLE, inner)),
            Pattern::MatchStar(_) => {}
        }
    }
}

impl<'a> Visitor<'a> for Atoms<'_> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::If(node) => {
                self.truth(&node.test);
                for clause in &node.elif_else_clauses {
                    if let Some(test) = &clause.test {
                        self.truth(test);
                    }
                }
            }
            Stmt::While(node) => self.truth(&node.test),
            Stmt::Assert(node) => self.truth(&node.test),
            Stmt::Match(node) => {
                let subject = self.term(&node.subject);
                for case in &node.cases {
                    self.pattern(&subject, &case.pattern);
                    if let Some(guard) = &case.guard {
                        self.truth(guard);
                    }
                }
            }
            _ => {}
        }
        visitor::walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::Compare(c) => {
                let mut left: &Expr = &c.left;
                for (op, right) in c.ops.iter().zip(c.comparators.iter()) {
                    let (l, r) = (self.term(left), self.term(right));
                    let atom = match op {
                        CmpOp::Eq | CmpOp::NotEq | CmpOp::Is | CmpOp::IsNot => decide::equal(&l, &r),
                        CmpOp::Lt | CmpOp::LtE | CmpOp::Gt | CmpOp::GtE => decide::order(&l, &r),
                        CmpOp::In | CmpOp::NotIn => match right {
                            // `x in (a, b)` is `x == a or x == b` — the language's own definition of
                            // membership in a literal sequence or set.
                            Expr::Tuple(_) | Expr::List(_) | Expr::Set(_) => {
                                let elts: &[Expr] = match right {
                                    Expr::Tuple(t) => &t.elts,
                                    Expr::List(x) => &x.elts,
                                    Expr::Set(x) => &x.elts,
                                    _ => &[],
                                };
                                for elt in elts {
                                    let atom = decide::equal(&l, &self.term(elt));
                                    self.push(atom);
                                }
                                None
                            }
                            _ => decide::member(&l, &r),
                        },
                    };
                    self.push(atom);
                    left = right;
                }
            }
            Expr::BoolOp(b) => {
                debug_assert!(matches!(b.op, BoolOp::And | BoolOp::Or));
                // Every operand but the last is tested for truth; the last is the expression's value,
                // tested only where the expression itself is (see `truth`). `x or 0` is a default.
                for value in &b.values[..b.values.len().saturating_sub(1)] {
                    self.truth(value);
                }
            }
            Expr::UnaryOp(u) if matches!(u.op, UnaryOp::Not) => self.truth(&u.operand),
            Expr::If(i) => self.truth(&i.test),
            Expr::Call(call) => {
                let atom = self.instance_test(call);
                self.push(atom);
                if let Some(tested) = self.truth_call(call) {
                    self.truth(tested);
                }
            }
            _ => {}
        }
        visitor::walk_expr(self, expr);
    }

    fn visit_comprehension(&mut self, comprehension: &'a ast::Comprehension) {
        for test in &comprehension.ifs {
            self.truth(test);
        }
        visitor::walk_comprehension(self, comprehension);
    }
}

/// What one definition tells the decisions pass: the atoms it decides on (sorted, deduplicated), the
/// parameters calls fill, the calls it makes and the paths it reads as values.
#[derive(Default)]
pub(crate) struct DecisionFacts {
    pub(crate) atoms: Vec<Arc<str>>,
    pub(crate) params: Vec<Arc<str>>,
    pub(crate) calls: Vec<decide::CallSite>,
    pub(crate) mentions: Vec<Arc<str>>,
    pub(crate) loose: Vec<Arc<str>>,
}

/// Calls and value reads, by the same identities the atoms are written in.
struct Refs<'w, 's> {
    walk: &'w Atoms<'s>,
    calls: Vec<decide::CallSite>,
    mentions: Vec<Arc<str>>,
    /// Field names written other than through a constructor the corpus can name: keywords to a callee
    /// that is not a path, and attributes assigned.
    loose: Vec<Arc<str>>,
}

impl<'a> Visitor<'a> for Refs<'_, '_> {
    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::Call(call) => {
                let callee = self.walk.term(&call.func);
                if callee.starts_with('@') && matches!(&*call.func, Expr::Name(_) | Expr::Attribute(_)) {
                    let mut site = decide::CallSite { callee: Arc::from(callee), args: Vec::new(), open: false };
                    for arg in &*call.arguments.args {
                        if matches!(arg, Expr::Starred(_)) {
                            site.open = true;
                        }
                        site.args.push((None, Arc::from(self.walk.term(arg))));
                    }
                    for kw in &*call.arguments.keywords {
                        match &kw.arg {
                            Some(name) => site.args.push((Some(Arc::from(name.id.as_str())), Arc::from(self.walk.term(&kw.value)))),
                            None => site.open = true,
                        }
                    }
                    self.calls.push(site);
                } else {
                    for kw in &*call.arguments.keywords {
                        if let Some(name) = &kw.arg {
                            self.loose.push(Arc::from(name.id.as_str()));
                        }
                    }
                    self.visit_expr(&call.func);
                }
                for arg in &*call.arguments.args {
                    self.visit_expr(arg);
                }
                for kw in &*call.arguments.keywords {
                    self.visit_expr(&kw.value);
                }
            }
            Expr::Name(_) | Expr::Attribute(_) => {
                if let (Some(_), _) = chain(expr) {
                    let read = self.walk.term(expr);
                    if read.starts_with('@') {
                        self.mentions.push(Arc::from(read));
                    }
                } else {
                    visitor::walk_expr(self, expr);
                }
            }
            _ => visitor::walk_expr(self, expr),
        }
    }

    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        let targets: Vec<&Expr> = match stmt {
            Stmt::Assign(a) => a.targets.iter().collect(),
            Stmt::AugAssign(a) => vec![&*a.target],
            Stmt::AnnAssign(a) => vec![&*a.target],
            _ => Vec::new(),
        };
        for target in targets {
            if let Expr::Attribute(a) = target {
                self.loose.push(Arc::from(a.attr.id.as_str()));
            }
        }
        visitor::walk_stmt(self, stmt);
    }

    /// An annotation names a type; it does not hand a value anywhere.
    fn visit_annotation(&mut self, _: &'a Expr) {}
}

/// Everything the decisions pass reads off one definition. Empty for anything that is not a
/// function — a class body declares, it does not decide.
pub(crate) fn decides(def: &Stmt, scope: &FileScope) -> DecisionFacts {
    let Stmt::FunctionDef(f) = def else { return DecisionFacts::default() };
    let Bindings { bound, single, params, untouched } = Locals::of(def);
    let p = &f.parameters;
    let types: HashMap<String, String> = p
        .posonlyargs
        .iter()
        .chain(&p.args)
        .chain(&p.kwonlyargs)
        .filter_map(|x| {
            let annotation = x.parameter.annotation.as_ref()?;
            Some((x.parameter.name.id.as_str().to_owned(), scope.annotated_class(annotation)?))
        })
        .collect();
    let mut walk = Atoms {
        scope,
        locals: bound,
        constants: HashMap::new(),
        computed: HashMap::new(),
        params: untouched,
        types,
        out: Vec::new(),
    };
    // A single-assignment local whose value is itself a constant — possibly through another such
    // local — is that constant. Resolved to a fixed point: each round can only add, and there are
    // finitely many names.
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
        let call = match value {
            Expr::Await(a) => &*a.value,
            other => other,
        };
        if let Expr::Call(c) = call {
            if !walk.constants.contains_key(name) {
                let computation = walk.term(call);
                walk.computed.insert(name.clone(), computation);
                // A value built by calling a class is of that class: its fields are typed.
                let callee = walk.term(&c.func);
                if callee.starts_with('@') && !walk.types.contains_key(name) {
                    walk.types.insert(name.clone(), callee);
                }
            }
        }
    }
    // A local bound once to a field whose class the module knows is of that class.
    for (name, value) in &single {
        if walk.types.contains_key(name) {
            continue;
        }
        if let (Some(root), attrs) = chain(value) {
            if let (Some(class), false) = (walk.types.get(root).cloned(), attrs.is_empty()) {
                let mut current = Some(class);
                for attr in &attrs {
                    current = current.and_then(|c| scope.field_class(&c, attr));
                }
                if let Some(class) = current {
                    walk.types.insert(name.clone(), class);
                }
            }
        }
    }
    walk.visit_body(&f.body);
    let mut refs = Refs { walk: &walk, calls: Vec::new(), mentions: Vec::new(), loose: Vec::new() };
    refs.visit_body(&f.body);
    let (calls, mut mentions, mut loose) = (refs.calls, refs.mentions, refs.loose);
    loose.sort_unstable();
    loose.dedup();
    mentions.sort_unstable();
    mentions.dedup();
    let mut atoms = walk.out;
    atoms.sort_unstable();
    atoms.dedup();
    // A decorated definition is called by whatever it was registered with, not only by the calls the
    // corpus shows: what its parameters hold is not the corpus's to say.
    let params = if f.decorator_list.is_empty() { params.iter().map(|p| Arc::from(p.as_str())).collect() } else { Vec::new() };
    DecisionFacts { atoms, params, calls, mentions, loose }
}

#[cfg(test)]
mod tests {
    use super::{decides, FileScope};
    use crate::canon::module_imports;
    use std::sync::Arc;
    use dup_defs_core::decide::render;
    use ruff_python_parser::parse_module;

    /// The atoms of the first function in `src`, rendered, as if the file lived at `file`.
    fn atoms_at(file: &str, src: &str) -> Vec<String> {
        let module = parse_module(src).expect("fixture parses").into_syntax();
        let scope = FileScope::new(file, &module_imports(src), &module.body);
        let def = module
            .body
            .iter()
            .find(|s| matches!(s, ruff_python_ast::Stmt::FunctionDef(_)))
            .expect("fixture has a function");
        decides(def, &scope).atoms.iter().map(|a| render(a)).collect()
    }

    fn atoms(src: &str) -> Vec<String> {
        atoms_at("app/orders.py", src)
    }

    #[test]
    fn negation_and_spelling_do_not_change_what_is_decided() {
        let refuse = atoms(
            "from app.models import Status\n\
             def refuse(order):\n    if order.status != Status.OPEN:\n        raise ValueError\n",
        );
        let accept = atoms(
            "from app.models import Status\n\
             def accept(order):\n    return order.status == Status.OPEN\n",
        );
        assert_eq!(refuse, vec!["@app.models.Status.OPEN = _.status"]);
        assert_eq!(refuse, accept);
    }

    #[test]
    fn a_filter_and_a_guard_over_one_fact_meet() {
        let guard = atoms("def guard(item):\n    if not item.visible:\n        return None\n    return item\n");
        let filter = atoms("def visible(items):\n    return [i for i in items if i.visible]\n");
        assert_eq!(guard, vec!["? _.visible"]);
        assert_eq!(guard, filter);
    }

    #[test]
    fn a_bound_name_is_a_hole_and_a_free_one_is_its_path() {
        let found = atoms(
            "import app.config as conf\nLIMIT = 3\n\
             def f(order, cap):\n    total = order.total\n    return conf.flag or total < cap or order.count < LIMIT\n",
        );
        // `cap` is a parameter: the frontend marks it, and the engine decides what it holds.
        assert_eq!(found, vec!["$cap < _", "@~app.orders~LIMIT < _.count", "? @app.config.flag"]);
    }

    #[test]
    fn a_call_keeps_its_callee_and_its_constants_not_its_computed_arguments() {
        let found = atoms(
            "from app import rules\n\
             def f(order, now):\n    if rules.covered(order, at=now, kind='trial'):\n        return 1\n",
        );
        assert_eq!(found, vec!["? @app.rules.covered(kind=\"trial\")"]);
    }

    #[test]
    fn isinstance_is_membership_in_a_type_and_bool_is_truth() {
        let found = atoms(
            "from app.models import Paid\n\
             def f(x):\n    return isinstance(x, Paid) and bool(x.ready)\n",
        );
        assert_eq!(found, vec!["? _.ready", "$x in @app.models.Paid"]);
    }

    #[test]
    fn match_patterns_are_comparisons() {
        let found = atoms(
            "from app.models import Status, Paid\n\
             def f(order):\n    match order.status:\n        case Status.OPEN | Status.HELD:\n            return 1\n        case None:\n            return 0\n",
        );
        assert_eq!(
            found,
            vec!["@app.models.Status.HELD = _.status", "@app.models.Status.OPEN = _.status", "None = _.status"]
        );
    }

    #[test]
    fn a_relative_import_resolves_against_the_package_as_a_local_path() {
        let found = atoms_at(
            "src/app/billing/check.py",
            "from ..models import Status\ndef f(o):\n    return o.status == Status.OPEN\n",
        );
        assert_eq!(found, vec!["@~src.app.models~Status.OPEN = _.status"]);
    }

    #[test]
    fn the_last_operand_of_and_or_is_a_value_unless_the_whole_is_tested() {
        // `order.total or 0` tests `order.total` and returns the default untested.
        assert_eq!(atoms("def f(order):\n    return order.total or 0\n"), vec!["? _.total"]);
        // Under an `if`, the last operand's truth decides the branch too.
        assert_eq!(
            atoms("def f(order):\n    if order.paid or order.free:\n        return 1\n"),
            vec!["? _.free", "? _.paid"]
        );
    }

    #[test]
    fn a_local_assigned_once_to_a_constant_is_that_constant() {
        let direct = atoms("from app import cfg\ndef f(t):\n    return t.monthly or cfg.flags.extended\n");
        let through = atoms(
            "from app import cfg\ndef f(t):\n    flags = cfg.flags\n    extended = flags.extended\n    return t.monthly or extended\n",
        );
        assert_eq!(direct, through);
        // Bound twice, it is a value the text does not fix: a hole.
        let twice = atoms("from app import cfg\ndef f(t):\n    on = cfg.a\n    if t.x:\n        on = cfg.b\n    return on or t.y\n");
        assert_eq!(twice, vec!["? _.x"]);
    }

    #[test]
    fn membership_in_a_literal_container_is_a_disjunction_of_equalities() {
        let written_in = atoms("from app.models import S\ndef f(o):\n    return o.state not in (S.A, S.B)\n");
        let written_or = atoms("from app.models import S\ndef f(o):\n    return o.state == S.A or o.state == S.B\n");
        assert_eq!(written_in, written_or);
    }

    #[test]
    fn a_bare_parameter_is_marked_and_a_bare_computed_local_is_its_call() {
        assert_eq!(atoms("def f(flag, t):\n    if not flag:\n        return t.x\n"), vec!["? $flag"]);
        // Reassigned, a parameter no longer holds what the callers gave it.
        assert_eq!(atoms("def f(flag):\n    flag = not flag\n    return 1 if flag else 2\n"), Vec::<String>::new());
        assert_eq!(
            atoms("from app import rules\ndef f(o):\n    covered = rules.covered(o)\n    if covered:\n        return 1\n"),
            vec!["? @app.rules.covered()"]
        );
    }

    #[test]
    fn calls_and_value_reads_are_reported_by_identity() {
        let src = "from app import rules, hooks\ndef f(o, now):\n    hooks.register(rules.check)\n    return rules.limit(o, at=now, kind='a')\n";
        let module = parse_module(src).expect("parse").into_syntax();
        let scope = FileScope::new("app/orders.py", &module_imports(src), &module.body);
        let facts = decides(&module.body[1], &scope);
        let callees: Vec<&str> = facts.calls.iter().map(|c| &*c.callee).collect();
        assert_eq!(callees, vec!["@app.hooks.register", "@app.rules.limit"]);
        assert_eq!(facts.mentions.iter().map(|m| &**m).collect::<Vec<_>>(), vec!["@app.rules.check"]);
        let limit = &facts.calls[1];
        assert_eq!(limit.args[1], (Some(Arc::from("at")), Arc::from("$now")));
        assert_eq!(facts.params.iter().map(|p| &**p).collect::<Vec<_>>(), vec!["o", "now"]);
    }

    #[test]
    fn a_field_read_off_a_record_is_typed_through_the_records_own_annotations() {
        let src = "from dataclasses import dataclass\n\
                   @dataclass(frozen=True)\nclass Live:\n    tariff: str\n    mode: int\n\
                   @dataclass(frozen=True)\nclass Facts:\n    live: Live | None\n    count: int\n\
                   def f(facts: Facts, code):\n    live = facts.live\n    return code == live.tariff or facts.live.mode > 1\n";
        let found = atoms_at("app/rules.py", src);
        assert_eq!(found, vec!["%@~app.rules~Live#mode < 1", "$code = %@~app.rules~Live#tariff"]);
    }

    #[test]
    fn a_class_with_a_base_or_its_own_constructor_is_not_a_record() {
        let src = "class Base:\n    a: int\nclass Child(Base):\n    b: int\nclass Built:\n    c: int\n    def __init__(self):\n        self.c = 1\n";
        let module = parse_module(src).expect("parse").into_syntax();
        let scope = FileScope::new("app/m.py", &module_imports(src), &module.body);
        assert!(scope.record("@~app.m~Base").is_some());
        assert!(scope.record("@~app.m~Child").is_none());
        assert!(scope.record("@~app.m~Built").is_none());
    }

    #[test]
    fn global_takes_a_name_back_to_the_module() {
        let found = atoms("COUNT = 0\ndef f():\n    global COUNT\n    COUNT = 1\n    return COUNT > 0\n");
        assert_eq!(found, vec!["0 < @~app.orders~COUNT"]);
    }
}
