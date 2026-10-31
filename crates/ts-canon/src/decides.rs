//! What a TypeScript definition **decides on** — the frontend half of [`Facets::decides`].
//!
//! TypeScript says which expressions are propositions, and this module only reads them off — by the
//! same rules the Python frontend reads its own, written in the atom grammar both share
//! ([`dup_defs_core::decide`]):
//!
//! - every **comparison**, wherever it stands — `===` and `==` and their negations as one relation,
//!   the four orders as another — and `instanceof`, `in` and `typeof x === 'string'`, which the
//!   language defines as membership;
//! - every expression in a position the language **tests for truth** — the test of an `if`, a loop,
//!   a conditional, the operand of `!`, every operand of `&&` and `||` but the last (and the last when
//!   the whole is tested), the predicate a `filter` / `find` / `some` / `every` is handed — but not the
//!   left of `??`, which the language has for defaults, nor the bound of a counting `for`, which is
//!   Python's `for i in range(n)` spelled out;
//! - every **`case`** of a `switch`, which compares its subject with the case's value;
//! - `Array.isArray`, `Boolean(x)`, and `.includes` / `.has`, the spellings of membership and truth a
//!   TypeScript writer reaches for where a Python one writes `isinstance`, `bool` and `in`.
//!
//! A term's identity follows the binding the language gives the name it starts from: a name the
//! definition binds is a hole; an imported name is the path it was imported from — a relative import
//! a local path, named by the module it resolves to; a name the module binds itself is a local path;
//! anything else is a global. `null` and `undefined` are both the absent value, written `None` as the
//! Python frontend writes its own.
//!
//! [`Facets::decides`]: dup_defs_core::Facets::decides

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::decide::{self, DecisionFacts};
use oxc_ast::ast::{
    self, Argument, BindingPattern, ChainElement, Declaration, Expression,
    FormalParameters, SimpleAssignmentTarget, Statement,
};
use oxc_ast_visit::{walk, Visit};
use oxc_syntax::operator::{BinaryOperator, LogicalOperator, UnaryOperator};

/// What an imported name stands for.
enum Binding {
    /// A package or an absolute module path: `@a.b.c`.
    Path(String),
    /// A module in this tree, reached by a relative specifier: the module and the name inside it.
    Local(String, String),
}

/// What the names of one file stand for, read once per file and shared by its definitions.
pub(crate) struct FileScope {
    imports: HashMap<String, Binding>,
    /// The file's own module path, dotted, as its location spells it.
    module: String,
    /// Names the module binds at top level other than by importing.
    defined: HashSet<String>,
}

/// A path's segments as a module: a known extension off the last (`./util.js` names `util.ts`), and
/// an `index` file standing for its directory, as the language's resolution has it.
fn module_of(mut segments: Vec<String>) -> String {
    if let Some(last) = segments.last_mut() {
        for ext in [".d.ts", ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"] {
            if let Some(stem) = last.strip_suffix(ext) {
                *last = stem.to_owned();
                break;
            }
        }
    }
    if segments.last().is_some_and(|s| s == "index") {
        segments.pop();
    }
    segments.retain(|s| !s.is_empty());
    segments.join(".")
}

impl FileScope {
    /// A scope with no imports and no defined names — for a run that never reads facets.
    pub(crate) fn empty(file: &str) -> Self {
        let trimmed = file.strip_prefix("./").unwrap_or(file);
        let module = module_of(trimmed.split(['/', '\\']).map(str::to_owned).collect());
        Self { imports: HashMap::new(), module, defined: HashSet::new() }
    }

    pub(crate) fn new(file: &str, program: &ast::Program<'_>) -> Self {
        let trimmed = file.strip_prefix("./").unwrap_or(file);
        let module = module_of(trimmed.split(['/', '\\']).map(str::to_owned).collect());
        let mut imports = HashMap::new();
        let mut defined = HashSet::new();
        for stmt in &program.body {
            match stmt {
                Statement::ImportDeclaration(decl) => {
                    let spec = decl.source.value.as_str();
                    let relative = spec.starts_with('.');
                    let target = if relative {
                        module_of(crate::canon::resolve_specifier(spec, trimmed))
                    } else {
                        spec.trim_start_matches('@').replace('/', ".")
                    };
                    let Some(specifiers) = &decl.specifiers else { continue };
                    for specifier in specifiers {
                        let (local, member) = match specifier {
                            ast::ImportDeclarationSpecifier::ImportSpecifier(s) => {
                                (s.local.name.to_string(), Some(s.imported.name().to_string()))
                            }
                            // A default import is named, by convention, as what the module exports.
                            ast::ImportDeclarationSpecifier::ImportDefaultSpecifier(s) => {
                                let name = s.local.name.to_string();
                                (name.clone(), relative.then_some(name))
                            }
                            ast::ImportDeclarationSpecifier::ImportNamespaceSpecifier(s) => (s.local.name.to_string(), None),
                        };
                        let binding = match (relative, member) {
                            (true, member) => Binding::Local(target.clone(), member.unwrap_or_default()),
                            (false, Some(member)) => Binding::Path(format!("{target}.{member}")),
                            (false, None) => Binding::Path(target.clone()),
                        };
                        imports.insert(local, binding);
                    }
                }
                other => declared_names(other, &mut defined),
            }
        }
        Self { imports, module, defined }
    }

    /// The file's own module path — what a definition's identity is local to.
    pub(crate) fn module(&self) -> &str {
        &self.module
    }

    /// The term a free name stands for, with `rest` (members after it) appended.
    fn free(&self, name: &str, rest: &[&str]) -> String {
        let tail = |head: &str| {
            let mut out = head.to_owned();
            for attr in rest {
                if !out.is_empty() {
                    out.push('.');
                }
                out.push_str(attr);
            }
            out
        };
        match self.imports.get(name) {
            Some(Binding::Path(path)) => decide::path(&tail(path)),
            Some(Binding::Local(module, member)) => decide::local(module, &tail(member)),
            None if self.defined.contains(name) => decide::local(&self.module, &tail(name)),
            None => decide::path(&tail(name)),
        }
    }
}

/// Names a module statement binds at top level, `export` seen through.
fn declared_names(stmt: &Statement<'_>, out: &mut HashSet<String>) {
    match stmt {
        Statement::ExportNamedDeclaration(e) => {
            if let Some(decl) = &e.declaration {
                declaration_names(decl, out);
            }
        }
        Statement::ExportDefaultDeclaration(e) => match &e.declaration {
            ast::ExportDefaultDeclarationKind::FunctionDeclaration(f) => {
                if let Some(id) = &f.id {
                    out.insert(id.name.to_string());
                }
            }
            ast::ExportDefaultDeclarationKind::ClassDeclaration(c) => {
                if let Some(id) = &c.id {
                    out.insert(id.name.to_string());
                }
            }
            _ => {}
        },
        other => {
            if let Some(decl) = other.as_declaration() {
                declaration_names(decl, out);
            }
        }
    }
}

fn declaration_names(decl: &Declaration<'_>, out: &mut HashSet<String>) {
    match decl {
        Declaration::VariableDeclaration(v) => {
            for d in &v.declarations {
                if let BindingPattern::BindingIdentifier(id) = &d.id {
                    out.insert(id.name.to_string());
                }
            }
        }
        Declaration::FunctionDeclaration(f) => {
            if let Some(id) = &f.id {
                out.insert(id.name.to_string());
            }
        }
        Declaration::ClassDeclaration(c) => {
            if let Some(id) = &c.id {
                out.insert(id.name.to_string());
            }
        }
        Declaration::TSEnumDeclaration(e) => {
            out.insert(e.id.name.to_string());
        }
        _ => {}
    }
}

/// Every name a definition binds, at any depth, and how many times each is bound or assigned.
#[derive(Default)]
struct Locals {
    bound: HashSet<String>,
    sites: HashMap<String, usize>,
}

impl Locals {
    fn mark(&mut self, name: &str) {
        *self.sites.entry(name.to_owned()).or_insert(0) += 1;
    }
}

impl<'a> Visit<'a> for Locals {
    fn visit_binding_identifier(&mut self, it: &ast::BindingIdentifier<'a>) {
        self.bound.insert(it.name.to_string());
        self.mark(it.name.as_str());
    }

    fn visit_assignment_expression(&mut self, it: &ast::AssignmentExpression<'a>) {
        if let Some(SimpleAssignmentTarget::AssignmentTargetIdentifier(id)) = it.left.as_simple_assignment_target() {
            self.mark(id.name.as_str());
        }
        walk::walk_assignment_expression(self, it);
    }

    fn visit_update_expression(&mut self, it: &ast::UpdateExpression<'a>) {
        if let SimpleAssignmentTarget::AssignmentTargetIdentifier(id) = &it.argument {
            self.mark(id.name.as_str());
        }
        walk::walk_update_expression(self, it);
    }
}

/// Every `const` declarator in `stmts` bound to a plain name, at any depth but inside nested
/// functions — a `const` is bound once by the language, so its name *is* its value wherever read.
fn consts<'x, 'b>(stmts: &'x [Statement<'b>], out: &mut Vec<(&'x str, &'x Expression<'b>)>) {
    for stmt in stmts {
        match stmt {
            Statement::VariableDeclaration(v) if v.kind == ast::VariableDeclarationKind::Const => {
                for d in &v.declarations {
                    if let (BindingPattern::BindingIdentifier(id), Some(init)) = (&d.id, &d.init) {
                        out.push((id.name.as_str(), init));
                    }
                }
            }
            Statement::BlockStatement(b) => consts(&b.body, out),
            Statement::IfStatement(i) => {
                consts(std::slice::from_ref(&i.consequent), out);
                if let Some(alt) = &i.alternate {
                    consts(std::slice::from_ref(alt), out);
                }
            }
            Statement::ForStatement(f) => consts(std::slice::from_ref(&f.body), out),
            Statement::ForOfStatement(f) => consts(std::slice::from_ref(&f.body), out),
            Statement::ForInStatement(f) => consts(std::slice::from_ref(&f.body), out),
            Statement::WhileStatement(w) => consts(std::slice::from_ref(&w.body), out),
            Statement::DoWhileStatement(w) => consts(std::slice::from_ref(&w.body), out),
            Statement::TryStatement(t) => {
                consts(&t.block.body, out);
                if let Some(h) = &t.handler {
                    consts(&h.body.body, out);
                }
                if let Some(f) = &t.finalizer {
                    consts(&f.body, out);
                }
            }
            Statement::SwitchStatement(s) => s.cases.iter().for_each(|c| consts(&c.consequent, out)),
            Statement::LabeledStatement(l) => consts(std::slice::from_ref(&l.body), out),
            _ => {}
        }
    }
}

/// A member chain from its root outwards: the root's name (`this` for the receiver), or `None` when
/// the chain starts at a computed value.
fn chain<'x>(mut expr: &'x Expression<'_>) -> (Option<&'x str>, Vec<&'x str>) {
    let mut attrs = Vec::new();
    loop {
        match expr {
            Expression::StaticMemberExpression(m) => {
                attrs.push(m.property.name.as_str());
                expr = &m.object;
            }
            Expression::PrivateFieldExpression(m) => {
                attrs.push(m.field.name.as_str());
                expr = &m.object;
            }
            Expression::ChainExpression(c) => match c.expression.as_member_expression() {
                Some(ast::MemberExpression::StaticMemberExpression(m)) => {
                    attrs.push(m.property.name.as_str());
                    expr = &m.object;
                }
                _ => break,
            },
            Expression::ParenthesizedExpression(p) => expr = &p.expression,
            Expression::TSNonNullExpression(e) => expr = &e.expression,
            _ => break,
        }
    }
    attrs.reverse();
    match expr {
        Expression::Identifier(id) => (Some(id.name.as_str()), attrs),
        Expression::ThisExpression(_) => (Some("this"), attrs),
        _ => (None, attrs),
    }
}

/// The text of a literal, or `None` for anything that is not one.
fn literal(expr: &Expression<'_>) -> Option<String> {
    Some(match expr {
        Expression::StringLiteral(s) => decide::literal(s.value.as_str()),
        Expression::NumericLiteral(n) => decide::number(n.value),
        Expression::BooleanLiteral(b) => if b.value { "True" } else { "False" }.to_owned(),
        Expression::NullLiteral(_) => "None".to_owned(),
        Expression::Identifier(id) if id.name == "undefined" => "None".to_owned(),
        Expression::TemplateLiteral(t) if t.expressions.is_empty() => {
            let text = t.quasis.first().and_then(|q| q.value.cooked.as_ref()).map_or("", |c| c.as_str());
            decide::literal(text)
        }
        Expression::BigIntLiteral(b) => b.raw.map_or_else(String::new, |r| r.to_string()),
        _ => return None,
    })
}

/// The methods a predicate is handed to: the callback's returned value is tested for truth, as a
/// comprehension's `if` is.
const PREDICATE_METHODS: &[&str] = &["filter", "find", "findIndex", "findLast", "findLastIndex", "some", "every"];

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
    out: Vec<Arc<str>>,
}

impl Atoms<'_> {
    fn push(&mut self, atom: Option<Arc<str>>) {
        if let Some(atom) = atom {
            self.out.push(atom);
        }
    }

    fn named(&self, name: &str, rest: &[&str]) -> String {
        if name == "this" {
            return decide::field(rest);
        }
        if let Some(value) = self.constants.get(name) {
            return Self::extended(value, rest);
        }
        if rest.is_empty() {
            if let Some(computation) = self.computed.get(name) {
                return computation.clone();
            }
            if self.params.contains(name) {
                return decide::param(name);
            }
        }
        if self.locals.contains(name) {
            decide::field(rest)
        } else {
            self.scope.free(name, rest)
        }
    }

    /// A constant's members: a path grows by them; any other constant has none a text could read,
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

    fn call_term(&self, callee: &Expression<'_>, arguments: &[Argument<'_>]) -> String {
        let callee = self.term(callee);
        let constants: Vec<String> =
            arguments.iter().filter_map(|a| a.as_expression().and_then(|e| self.constant(e))).collect();
        decide::call(&callee, constants)
    }

    /// The identity of an operand.
    fn term(&self, expr: &Expression<'_>) -> String {
        if let Some(lit) = literal(expr) {
            return lit;
        }
        match expr {
            Expression::Identifier(_)
            | Expression::ThisExpression(_)
            | Expression::StaticMemberExpression(_)
            | Expression::PrivateFieldExpression(_) => match chain(expr) {
                (Some(root), attrs) => self.named(root, &attrs),
                (None, attrs) => decide::field(&attrs),
            },
            Expression::ChainExpression(c) => match &c.expression {
                ChainElement::CallExpression(call) => self.call_term(&call.callee, &call.arguments),
                _ => match chain(expr) {
                    (Some(root), attrs) => self.named(root, &attrs),
                    (None, attrs) => decide::field(&attrs),
                },
            },
            Expression::ComputedMemberExpression(m) => {
                let key = self.constant(&m.expression).unwrap_or_else(|| decide::HOLE.to_owned());
                format!("{}[{key}]", self.term(&m.object))
            }
            Expression::CallExpression(c) => self.call_term(&c.callee, &c.arguments),
            Expression::NewExpression(n) => self.call_term(&n.callee, &n.arguments),
            Expression::AwaitExpression(a) => self.term(&a.argument),
            Expression::ParenthesizedExpression(p) => self.term(&p.expression),
            Expression::TSAsExpression(e) => self.term(&e.expression),
            Expression::TSSatisfiesExpression(e) => self.term(&e.expression),
            Expression::TSNonNullExpression(e) => self.term(&e.expression),
            Expression::TSTypeAssertion(e) => self.term(&e.expression),
            Expression::ArrayExpression(a) => decide::container(
                a.elements.iter().filter_map(|e| e.as_expression().map(|e| self.term(e))).collect(),
            ),
            Expression::BinaryExpression(b) if !is_comparison(b.operator) => {
                let rendered = format!("({} {} {})", self.term(&b.left), b.operator.as_str(), self.term(&b.right));
                if decide::has_identity(&rendered) {
                    rendered
                } else {
                    decide::HOLE.to_owned()
                }
            }
            Expression::UnaryExpression(u) if matches!(u.operator, UnaryOperator::UnaryNegation | UnaryOperator::UnaryPlus | UnaryOperator::BitwiseNot) => {
                let inner = self.term(&u.argument);
                if decide::has_identity(&inner) {
                    format!("{}{inner}", u.operator.as_str())
                } else {
                    inner
                }
            }
            _ => decide::HOLE.to_owned(),
        }
    }

    /// A value handed to a call that is fixed by the text rather than computed: a literal, a name the
    /// definition did not bind, or an array of those.
    fn constant(&self, expr: &Expression<'_>) -> Option<String> {
        if let Some(lit) = literal(expr) {
            return Some(lit);
        }
        match expr {
            Expression::Identifier(_) | Expression::StaticMemberExpression(_) => match chain(expr) {
                (Some(root), attrs) if self.constants.contains_key(root) => {
                    Some(Self::extended(&self.constants[root], &attrs))
                }
                (Some(root), attrs) if root != "this" && !self.locals.contains(root) => Some(self.scope.free(root, &attrs)),
                _ => None,
            },
            Expression::UnaryExpression(u) if matches!(u.operator, UnaryOperator::UnaryNegation | UnaryOperator::UnaryPlus) => {
                self.constant(&u.argument).map(|inner| format!("{}{inner}", u.operator.as_str()))
            }
            Expression::ArrayExpression(a) => a
                .elements
                .iter()
                .map(|e| e.as_expression().and_then(|e| self.constant(e)))
                .collect::<Option<Vec<_>>>()
                .map(decide::container),
            Expression::ParenthesizedExpression(p) => self.constant(&p.expression),
            Expression::TSAsExpression(e) => self.constant(&e.expression),
            _ => None,
        }
    }

    /// The calls that are propositions wherever they stand — `Array.isArray`, `Boolean`, `.includes`
    /// / `.has` — read as the membership or truth they spell. True when the call was one of them.
    fn proposition_call(&mut self, c: &ast::CallExpression<'_>) -> bool {
        let [only] = c.arguments.as_slice() else { return false };
        let Some(arg) = only.as_expression() else { return false };
        match chain(&c.callee) {
            (Some("Array"), attrs) if attrs == ["isArray"] && !self.locals.contains("Array") => {
                let atom = decide::member(&self.term(arg), &decide::path("Array"));
                self.push(atom);
                true
            }
            (Some("Boolean"), attrs) if attrs.is_empty() && !self.locals.contains("Boolean") => {
                self.truth(arg);
                true
            }
            _ => match &c.callee {
                Expression::StaticMemberExpression(m) if matches!(m.property.name.as_str(), "includes" | "has") => {
                    let element = self.term(arg);
                    // `[a, b].includes(x)` is `x === a || x === b` — membership in a literal array is
                    // the disjunction of equalities it abbreviates.
                    if let Expression::ArrayExpression(a) = &m.object {
                        for e in &a.elements {
                            if let Some(e) = e.as_expression() {
                                let atom = decide::equal(&element, &self.term(e));
                                self.push(atom);
                            }
                        }
                    } else {
                        let atom = decide::member(&element, &self.term(&m.object));
                        self.push(atom);
                    }
                    true
                }
                _ => false,
            },
        }
    }

    /// An expression in a position the language tests for truth.
    fn truth(&mut self, expr: &Expression<'_>) {
        match expr {
            // Structure, not a proposition: its parts are read where the walk reaches it.
            Expression::BinaryExpression(b) if is_comparison(b.operator) => {}
            Expression::UnaryExpression(u) if u.operator == UnaryOperator::LogicalNot => {}
            // `&&` and `||` test their left operand and hand the right one back as their value — so the
            // right one is tested exactly when the whole expression is, which is now.
            Expression::LogicalExpression(l) => self.truth(&l.right),
            Expression::ParenthesizedExpression(p) => self.truth(&p.expression),
            Expression::AssignmentExpression(a) => self.truth(&a.right),
            Expression::CallExpression(c) if self.is_proposition_call(c) => {}
            Expression::ChainExpression(c)
                if matches!(&c.expression, ChainElement::CallExpression(call) if self.is_proposition_call(call)) => {}
            _ => {
                let atom = decide::truth(&self.term(expr));
                self.push(atom);
            }
        }
    }

    /// Whether [`Atoms::proposition_call`] reads this call, without reading it.
    fn is_proposition_call(&self, c: &ast::CallExpression<'_>) -> bool {
        let [only] = c.arguments.as_slice() else { return false };
        if only.as_expression().is_none() {
            return false;
        }
        match chain(&c.callee) {
            (Some("Array"), attrs) if attrs == ["isArray"] && !self.locals.contains("Array") => true,
            (Some("Boolean"), attrs) if attrs.is_empty() && !self.locals.contains("Boolean") => true,
            _ => matches!(&c.callee, Expression::StaticMemberExpression(m) if matches!(m.property.name.as_str(), "includes" | "has")),
        }
    }

    /// What a predicate callback returns, each tested for truth.
    fn predicate(&mut self, callback: &Expression<'_>) {
        match callback {
            Expression::ArrowFunctionExpression(a) if a.expression => {
                if let Some(Statement::ExpressionStatement(e)) = a.body.statements.first() {
                    self.truth(&e.expression);
                }
            }
            Expression::ArrowFunctionExpression(a) => {
                let mut returned = Vec::new();
                returns(&a.body.statements, &mut returned);
                for e in returned {
                    self.truth(e);
                }
            }
            Expression::FunctionExpression(f) => {
                if let Some(body) = &f.body {
                    let mut returned = Vec::new();
                    returns(&body.statements, &mut returned);
                    for e in returned {
                        self.truth(e);
                    }
                }
            }
            _ => {}
        }
    }
}

/// `typeof x === 'string'`: the operand, and the type the language names by that string.
fn typeof_test<'x, 'b>(b: &'x ast::BinaryExpression<'b>) -> Option<(&'x Expression<'b>, &'x str)> {
    match (&b.left, &b.right) {
        (Expression::UnaryExpression(u), Expression::StringLiteral(s))
        | (Expression::StringLiteral(s), Expression::UnaryExpression(u))
            if u.operator == UnaryOperator::Typeof =>
        {
            Some((&u.argument, s.value.as_str()))
        }
        _ => None,
    }
}

/// True for a counting loop: its test orders a name its update steps.
fn counts(it: &ast::ForStatement<'_>) -> bool {
    let (Some(Expression::BinaryExpression(test)), Some(update)) = (&it.test, &it.update) else { return false };
    is_comparison(test.operator)
        && !matches!(test.operator, BinaryOperator::In | BinaryOperator::Instanceof)
        && [&test.left, &test.right]
            .into_iter()
            .any(|side| matches!(side, Expression::Identifier(id) if steps(update, id.name.as_str())))
}

/// True when `update` assigns `name` — `i++`, `i -= 2`, `i = next(i)`, or one of a comma's parts.
fn steps(update: &Expression<'_>, name: &str) -> bool {
    match update {
        Expression::UpdateExpression(u) => {
            matches!(&u.argument, SimpleAssignmentTarget::AssignmentTargetIdentifier(id) if id.name == name)
        }
        Expression::AssignmentExpression(a) => matches!(
            a.left.as_simple_assignment_target(),
            Some(SimpleAssignmentTarget::AssignmentTargetIdentifier(id)) if id.name == name
        ),
        Expression::SequenceExpression(s) => s.expressions.iter().any(|e| steps(e, name)),
        _ => false,
    }
}

/// The values a body returns, not counting what nested functions return.
fn returns<'x, 'b>(stmts: &'x [Statement<'b>], out: &mut Vec<&'x Expression<'b>>) {
    for stmt in stmts {
        match stmt {
            Statement::ReturnStatement(r) => out.extend(r.argument.as_ref()),
            Statement::BlockStatement(b) => returns(&b.body, out),
            Statement::IfStatement(i) => {
                returns(std::slice::from_ref(&i.consequent), out);
                if let Some(alt) = &i.alternate {
                    returns(std::slice::from_ref(alt), out);
                }
            }
            _ => {}
        }
    }
}

fn is_comparison(op: BinaryOperator) -> bool {
    matches!(
        op,
        BinaryOperator::Equality
            | BinaryOperator::Inequality
            | BinaryOperator::StrictEquality
            | BinaryOperator::StrictInequality
            | BinaryOperator::LessThan
            | BinaryOperator::LessEqualThan
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterEqualThan
            | BinaryOperator::In
            | BinaryOperator::Instanceof
    )
}

impl<'a> Visit<'a> for Atoms<'_> {
    fn visit_binary_expression(&mut self, it: &ast::BinaryExpression<'a>) {
        let atom = if let Some((operand, kind)) = typeof_test(it) {
            // Membership of the operand in the type the string names.
            decide::member(&self.term(operand), &decide::path(kind))
        } else {
            match it.operator {
                BinaryOperator::Equality
                | BinaryOperator::Inequality
                | BinaryOperator::StrictEquality
                | BinaryOperator::StrictInequality => decide::equal(&self.term(&it.left), &self.term(&it.right)),
                BinaryOperator::LessThan
                | BinaryOperator::LessEqualThan
                | BinaryOperator::GreaterThan
                | BinaryOperator::GreaterEqualThan => decide::order(&self.term(&it.left), &self.term(&it.right)),
                // `x instanceof T` is membership in a type; `k in o`, membership of a key.
                BinaryOperator::In | BinaryOperator::Instanceof => {
                    decide::member(&self.term(&it.left), &self.term(&it.right))
                }
                _ => None,
            }
        };
        self.push(atom);
        walk::walk_binary_expression(self, it);
    }

    fn visit_logical_expression(&mut self, it: &ast::LogicalExpression<'a>) {
        // The left operand of `&&` / `||` is tested; the right is the value (see `truth`). `a ?? b` is
        // not a test: the language has it for exactly one thing, a value in place of a missing one,
        // which is what Python's `x or default` says only by abusing a truth test. Read as a fact,
        // every `options.limit ?? 10` in a tree joined every other one, and pairs of functions that
        // merely defaulted the same fields filled the report.
        if matches!(it.operator, LogicalOperator::And | LogicalOperator::Or) {
            self.truth(&it.left);
        }
        walk::walk_logical_expression(self, it);
    }

    fn visit_unary_expression(&mut self, it: &ast::UnaryExpression<'a>) {
        if it.operator == UnaryOperator::LogicalNot {
            self.truth(&it.argument);
        }
        walk::walk_unary_expression(self, it);
    }

    fn visit_conditional_expression(&mut self, it: &ast::ConditionalExpression<'a>) {
        self.truth(&it.test);
        walk::walk_conditional_expression(self, it);
    }

    fn visit_if_statement(&mut self, it: &ast::IfStatement<'a>) {
        self.truth(&it.test);
        walk::walk_if_statement(self, it);
    }

    fn visit_while_statement(&mut self, it: &ast::WhileStatement<'a>) {
        self.truth(&it.test);
        walk::walk_while_statement(self, it);
    }

    fn visit_do_while_statement(&mut self, it: &ast::DoWhileStatement<'a>) {
        self.truth(&it.test);
        walk::walk_do_while_statement(self, it);
    }

    fn visit_for_statement(&mut self, it: &ast::ForStatement<'a>) {
        if counts(it) {
            // `for (let i = 0; i < n; i++)` is how the language writes Python's `for i in range(n)`:
            // the test bounds a counter, and iterating is not deciding.
            if let Some(init) = &it.init {
                self.visit_for_statement_init(init);
            }
            if let Some(update) = &it.update {
                self.visit_expression(update);
            }
            self.visit_statement(&it.body);
            return;
        }
        if let Some(test) = &it.test {
            self.truth(test);
        }
        walk::walk_for_statement(self, it);
    }

    fn visit_switch_statement(&mut self, it: &ast::SwitchStatement<'a>) {
        // `switch (true)` is a ladder of conditions; any other subject is compared with each case.
        let ladder = matches!(it.discriminant, Expression::BooleanLiteral(ref b) if b.value);
        let subject = self.term(&it.discriminant);
        for case in &it.cases {
            if let Some(test) = &case.test {
                if ladder {
                    self.truth(test);
                } else {
                    let atom = decide::equal(&subject, &self.term(test));
                    self.push(atom);
                }
            }
        }
        walk::walk_switch_statement(self, it);
    }

    fn visit_call_expression(&mut self, it: &ast::CallExpression<'a>) {
        let _ = self.proposition_call(it);
        if let Expression::StaticMemberExpression(m) = &it.callee {
            if PREDICATE_METHODS.contains(&m.property.name.as_str()) {
                if let Some(callback) = it.arguments.first().and_then(Argument::as_expression) {
                    self.predicate(callback);
                }
            }
        }
        walk::walk_call_expression(self, it);
    }
}

/// Calls and value reads, by the same identities the atoms are written in.
struct Refs<'w, 's> {
    walk: &'w Atoms<'s>,
    calls: Vec<decide::CallSite>,
    mentions: Vec<Arc<str>>,
    /// Members assigned (`x.f = …`): any of them may write a field past its constructor.
    loose: Vec<Arc<str>>,
}

impl Refs<'_, '_> {
    fn call<'a>(&mut self, callee: &Expression<'a>, arguments: &[Argument<'a>]) {
        let named = matches!(chain(callee), (Some(root), _) if root != "this");
        let term = self.walk.term(callee);
        if named && term.starts_with('@') {
            let mut site = decide::CallSite { callee: Arc::from(term), args: Vec::new(), open: false };
            for arg in arguments {
                match arg {
                    Argument::SpreadElement(_) => site.open = true,
                    other => {
                        if let Some(e) = other.as_expression() {
                            site.args.push((None, Arc::from(self.walk.term(e))));
                        }
                    }
                }
            }
            self.calls.push(site);
        } else {
            self.visit_expression(callee);
        }
        for arg in arguments {
            match arg {
                Argument::SpreadElement(s) => self.visit_expression(&s.argument),
                other => {
                    if let Some(e) = other.as_expression() {
                        self.visit_expression(e);
                    }
                }
            }
        }
    }
}

impl<'a> Visit<'a> for Refs<'_, '_> {
    fn visit_expression(&mut self, it: &Expression<'a>) {
        match it {
            Expression::CallExpression(c) => self.call(&c.callee, &c.arguments),
            Expression::NewExpression(n) => self.call(&n.callee, &n.arguments),
            Expression::Identifier(_) | Expression::StaticMemberExpression(_) if chain(it).0.is_some() => {
                let read = self.walk.term(it);
                if read.starts_with('@') {
                    self.mentions.push(Arc::from(read));
                }
            }
            _ => walk::walk_expression(self, it),
        }
    }

    fn visit_assignment_expression(&mut self, it: &ast::AssignmentExpression<'a>) {
        if let Some(ast::SimpleAssignmentTarget::StaticMemberExpression(m)) = it.left.as_simple_assignment_target() {
            self.loose.push(Arc::from(m.property.name.as_str()));
        }
        walk::walk_assignment_expression(self, it);
    }

    /// An annotation names a type; it does not hand a value anywhere.
    fn visit_ts_type_annotation(&mut self, _: &ast::TSTypeAnnotation<'a>) {}
}

/// Everything the decisions pass reads off one callable: its parameters and body. `decorated` — a
/// decorated method is called by whatever it was registered with, not only by the calls the corpus
/// shows, so what its parameters hold is not the corpus's to say.
pub(crate) fn decides(params: &FormalParameters<'_>, body: &[Statement<'_>], scope: &FileScope, decorated: bool) -> DecisionFacts {
    let mut locals = Locals::default();
    locals.visit_formal_parameters(params);
    for stmt in body {
        locals.visit_statement(stmt);
    }
    let order: Vec<String> = params
        .items
        .iter()
        .filter_map(|p| match &p.pattern {
            BindingPattern::BindingIdentifier(id) => Some(id.name.to_string()),
            _ => None,
        })
        .collect();
    let untouched: HashSet<String> = order.iter().filter(|name| locals.sites.get(*name) == Some(&1)).cloned().collect();
    let mut single = Vec::new();
    consts(body, &mut single);
    single.retain(|(name, _)| locals.sites.get(*name) == Some(&1));
    let mut walk = Atoms {
        scope,
        locals: locals.bound,
        constants: HashMap::new(),
        computed: HashMap::new(),
        params: untouched,
        out: Vec::new(),
    };
    // A `const` whose value is itself a constant — possibly through another such `const` — is that
    // constant. Resolved to a fixed point: each round can only add, and there are finitely many names.
    loop {
        let found: Vec<(String, String)> = single
            .iter()
            .filter(|(name, _)| !walk.constants.contains_key(*name))
            .filter_map(|(name, value)| walk.constant(value).map(|c| ((*name).to_owned(), c)))
            .collect();
        if found.is_empty() {
            break;
        }
        walk.constants.extend(found);
    }
    for (name, value) in &single {
        let call = match value {
            Expression::AwaitExpression(a) => &a.argument,
            other => *other,
        };
        if matches!(call, Expression::CallExpression(_) | Expression::NewExpression(_)) && !walk.constants.contains_key(*name) {
            let computation = walk.term(call);
            walk.computed.insert((*name).to_owned(), computation);
        }
    }
    for stmt in body {
        walk.visit_statement(stmt);
    }
    let mut refs = Refs { walk: &walk, calls: Vec::new(), mentions: Vec::new(), loose: Vec::new() };
    for stmt in body {
        refs.visit_statement(stmt);
    }
    let (calls, mut mentions, mut loose) = (refs.calls, refs.mentions, refs.loose);
    loose.sort_unstable();
    loose.dedup();
    mentions.sort_unstable();
    mentions.dedup();
    let mut atoms = walk.out;
    atoms.sort_unstable();
    atoms.dedup();
    let params = if decorated { Vec::new() } else { order.iter().map(|p| Arc::from(p.as_str())).collect() };
    DecisionFacts { atoms, params, calls, mentions, loose }
}

#[cfg(test)]
mod tests {
    use super::{decides, FileScope};
    use dup_defs_core::decide::render;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::{Declaration, Statement};
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    /// The atoms of the first function in `src`, rendered, as if the file lived at `file`.
    fn atoms_at(file: &str, src: &str) -> Vec<String> {
        let allocator = Allocator::default();
        let ret = Parser::new(&allocator, src, SourceType::ts()).parse();
        let scope = FileScope::new(file, &ret.program);
        let f = ret
            .program
            .body
            .iter()
            .find_map(|s| match s {
                Statement::FunctionDeclaration(f) => Some(f),
                Statement::ExportNamedDeclaration(e) => match &e.declaration {
                    Some(Declaration::FunctionDeclaration(f)) => Some(f),
                    _ => None,
                },
                _ => None,
            })
            .expect("fixture has a function");
        let body = f.body.as_ref().expect("a body");
        decides(&f.params, &body.statements, &scope, false).atoms.iter().map(|a| render(a)).collect()
    }

    fn atoms(src: &str) -> Vec<String> {
        atoms_at("app/orders.ts", src)
    }

    #[test]
    fn negation_and_spelling_do_not_change_what_is_decided() {
        let refuse = atoms(
            "import { Status } from 'app/models';\nfunction refuse(order: Order) { if (order.status !== Status.Open) { throw new Error('x'); } }\n",
        );
        let accept = atoms("import { Status } from 'app/models';\nfunction accept(order: Order) { return order.status === Status.Open; }\n");
        assert_eq!(refuse, vec!["@app.models.Status.Open = _.status"]);
        assert_eq!(refuse, accept);
    }

    #[test]
    fn a_filter_and_a_guard_over_one_fact_meet() {
        let guard = atoms("function guard(item: Item) { if (!item.visible) { return null; } return item; }\n");
        let filter = atoms("function visible(items: Item[]) { return items.filter((i) => i.visible); }\n");
        assert_eq!(guard, vec!["? _.visible"]);
        assert_eq!(guard, filter);
    }

    #[test]
    fn a_relative_import_is_a_local_path_named_by_the_module_it_resolves_to() {
        let found = atoms_at(
            "src/app/billing/check.ts",
            "import { Status } from '../models';\nexport function f(o: Order) { return o.status === Status.Open; }\n",
        );
        assert_eq!(found, vec!["@~src.app.models~Status.Open = _.status"]);
    }

    #[test]
    fn membership_spellings_meet() {
        let written_in = atoms("import { S } from 'app/models';\nfunction f(o: O) { return [S.A, S.B].includes(o.state); }\n");
        let written_or = atoms("import { S } from 'app/models';\nfunction f(o: O) { return o.state === S.A || o.state === S.B; }\n");
        assert_eq!(written_in, written_or);
        let typed = atoms("function f(x: unknown) { return typeof x === 'string' || Array.isArray(x) || x instanceof Date; }\n");
        assert_eq!(typed, vec!["$x in @Array", "$x in @Date", "$x in @string"]);
    }

    #[test]
    fn a_switch_compares_its_subject_with_each_case() {
        let found = atoms(
            "import { Status } from 'app/models';\nfunction f(o: O) { switch (o.status) { case Status.Open: return 1; case Status.Held: return 2; default: return 0; } }\n",
        );
        assert_eq!(found, vec!["@app.models.Status.Held = _.status", "@app.models.Status.Open = _.status"]);
    }

    #[test]
    fn null_and_undefined_are_one_absence_and_a_default_is_not_a_test() {
        assert_eq!(atoms("function f(o: O) { return o.at === undefined; }\n"), atoms("function f(o: O) { return o.at == null; }\n"));
        assert_eq!(atoms("function f(o: O) { return o.limit ?? 10; }\n"), Vec::<String>::new());
    }

    #[test]
    fn a_const_bound_to_a_constant_is_that_constant_and_a_parameter_is_marked() {
        let direct = atoms("import { cfg } from 'app/cfg';\nfunction f(t: T) { return t.monthly || cfg.flags.extended; }\n");
        let through = atoms(
            "import { cfg } from 'app/cfg';\nfunction f(t: T) { const flags = cfg.flags; const extended = flags.extended; return t.monthly || extended; }\n",
        );
        assert_eq!(direct, through);
        assert_eq!(atoms("function f(flag: boolean, t: T) { if (!flag) { return t.x; } }\n"), vec!["? $flag"]);
        // Reassigned, a parameter no longer holds what the callers gave it.
        assert_eq!(atoms("function f(flag: boolean) { flag = !flag; return flag ? 1 : 2; }\n"), Vec::<String>::new());
    }

    #[test]
    fn a_counting_loop_decides_nothing_and_a_searching_one_does() {
        let counting = atoms("function f(xs: X[]) { for (let i = 0, n = xs.length; i < n; i++) { if (xs[i].done) { return i; } } }\n");
        assert_eq!(counting, vec!["? _.done"]);
        let searching = atoms("function f(n: N) { for (let c = n; c.kind !== 'root'; c = c.parent) { if (c.done) { return c; } } }\n");
        assert_eq!(searching, vec!["\"root\" = _.kind", "? _.done"]);
    }

    #[test]
    fn membership_through_an_optional_chain_is_read_once() {
        let optional = atoms("function f(t: T, s: S) { return t?.items.includes(s); }\n");
        let plain = atoms("function f(t: T, s: S) { return t.items.includes(s); }\n");
        assert_eq!(optional, plain);
        assert_eq!(plain, vec!["$s in _.items"]);
    }
}
