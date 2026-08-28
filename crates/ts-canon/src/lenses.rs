//! **Lenses for TypeScript** — the oxc walk, and only the walk.
//!
//! The vocabulary, the stitching, the corpus scoring and the record itself live in
//! [`dup_defs_core::lens`]. What is here is how TypeScript answers each of the ten questions.
//!
//! One is left silent rather than approximated: **`use`**, a definition's call sites, needs a walk
//! of the whole tree that this frontend does not yet make.
//! [`dup_defs_core::lens::merge_use_facts`] finds nothing to merge, and the corpus IDF weighs a
//! lens that never speaks at zero without anyone declaring it absent.
//!
//! `resources` is the one worth naming, because the obvious answer is wrong. TypeScript's scoped
//! acquisition is `using` / `await using` (TC39 explicit resource management), not `try`/`finally` —
//! a `finally` is a *cleanup path*, which is a control-flow fact and is projected as one. Reading
//! every `finally` as a held resource would fill the lens with the language's most common idiom and
//! say nothing.

use std::collections::{BTreeSet, HashSet};

use dup_defs_core::lens::{Lens, LensFacts};
use oxc_ast::ast::{
    self, BindingPattern, Class, ClassElement, Expression, FormalParameters, Statement,
    TSInterfaceDeclaration, TSType,
};

/// Facts gathered in one walk of a body, sliced up per lens afterwards.
#[derive(Default)]
struct Walked {
    effects: Vec<String>,
    control: Vec<String>,
    failures: BTreeSet<String>,
    resources: Vec<String>,
}

struct Walk<'a> {
    /// Names the module itself introduced — erased, exactly as on the widest rung of the ladder, so
    /// a lens is not held apart by the very identities it exists to see past.
    bound: &'a HashSet<String>,
    depth: usize,
    facts: Walked,
}

impl<'a> Walk<'a> {
    fn run(body: &[Statement<'_>], bound: &'a HashSet<String>) -> Walked {
        let mut walk = Walk { bound, depth: 0, facts: Walked::default() };
        walk.stmts(body);
        walk.facts
    }

    fn tag(&mut self, tag: &str) {
        self.facts.control.push(dup_defs_core::lens::control_tag(self.depth, tag));
    }

    fn nested(&mut self, f: impl FnOnce(&mut Self)) {
        self.depth += 1;
        f(self);
        self.depth -= 1;
    }

    /// The name a call names, when it names something the module did not introduce. A bare `foo(…)`
    /// yields `foo`; `obj.method(…)` yields `.method` — the receiver is whatever it is, the *method*
    /// is the grammar.
    fn callee(&self, expr: &Expression<'_>) -> Option<String> {
        match expr {
            Expression::Identifier(id) => {
                let name = id.name.as_str();
                (!self.bound.contains(name)).then(|| name.to_owned())
            }
            Expression::StaticMemberExpression(member) => {
                let name = member.property.name.as_str();
                (!self.bound.contains(name)).then(|| format!(".{name}"))
            }
            _ => None,
        }
    }

    fn stmts(&mut self, stmts: &[Statement<'_>]) {
        for stmt in stmts {
            self.stmt(stmt);
        }
    }

    /// A statement's body, braced or not — braces are punctuation, not structure.
    fn body(&mut self, stmt: &Statement<'_>) {
        match stmt {
            Statement::BlockStatement(block) => self.stmts(&block.body),
            other => self.stmt(other),
        }
    }

    #[allow(clippy::too_many_lines)] // one match over the statement variants reads better whole
    fn stmt(&mut self, stmt: &Statement<'_>) {
        match stmt {
            Statement::BlockStatement(block) => self.nested(|w| w.stmts(&block.body)),
            Statement::IfStatement(node) => {
                self.tag("if");
                self.expr(&node.test);
                self.nested(|w| {
                    w.body(&node.consequent);
                    if let Some(alt) = &node.alternate {
                        w.body(alt);
                    }
                });
            }
            Statement::ForStatement(node) => {
                self.tag("for");
                self.nested(|w| w.body(&node.body));
            }
            Statement::ForInStatement(node) => {
                self.tag("forin");
                self.expr(&node.right);
                self.nested(|w| w.body(&node.body));
            }
            Statement::ForOfStatement(node) => {
                self.tag(if node.r#await { "forawaitof" } else { "forof" });
                self.expr(&node.right);
                self.nested(|w| w.body(&node.body));
            }
            Statement::WhileStatement(node) => {
                self.tag("while");
                self.expr(&node.test);
                self.nested(|w| w.body(&node.body));
            }
            Statement::DoWhileStatement(node) => {
                self.tag("dowhile");
                self.expr(&node.test);
                self.nested(|w| w.body(&node.body));
            }
            Statement::SwitchStatement(node) => {
                self.tag("switch");
                self.expr(&node.discriminant);
                self.nested(|w| {
                    for case in &node.cases {
                        w.stmts(&case.consequent);
                    }
                });
            }
            Statement::TryStatement(node) => {
                self.tag("try");
                self.nested(|w| w.stmts(&node.block.body));
                if let Some(handler) = &node.handler {
                    self.tag("catch");
                    // What a handler binds is what this body knows can go wrong. An annotated
                    // parameter names the type; a bare one names only that something was caught.
                    if let Some(param) = &handler.param {
                        let annotation = param.type_annotation.as_ref().map(|a| &a.type_annotation);
                        for name in type_names(annotation, self.bound) {
                            self.facts.failures.insert(name);
                        }
                    }
                    self.nested(|w| w.stmts(&handler.body.body));
                }
                if let Some(finalizer) = &node.finalizer {
                    self.tag("finally");
                    self.nested(|w| w.stmts(&finalizer.body));
                }
            }
            Statement::ThrowStatement(node) => {
                self.tag("throw");
                // `throw new SomeError(…)` — the error this body raises.
                if let Expression::NewExpression(new) = &node.argument {
                    if let Some(name) = self.callee(&new.callee) {
                        self.facts.failures.insert(name);
                    }
                }
                self.expr(&node.argument);
            }
            Statement::ReturnStatement(node) => {
                self.tag("return");
                if let Some(arg) = &node.argument {
                    self.expr(arg);
                }
            }
            Statement::LabeledStatement(node) => self.stmt(&node.body),
            Statement::VariableDeclaration(node) => {
                // 🔴 `using` / `await using` is TypeScript's scoped acquisition — the direct
                // counterpart of Python's `with`. `try`/`finally` is a cleanup *path*, projected as
                // control flow; reading it as a held resource would fill this lens with the
                // language's commonest idiom and say nothing.
                let is_using = matches!(
                    node.kind,
                    ast::VariableDeclarationKind::Using | ast::VariableDeclarationKind::AwaitUsing
                );
                for decl in &node.declarations {
                    if let Some(init) = &decl.init {
                        if is_using {
                            if let Some(name) = self.acquisition(init) {
                                self.facts.resources.push(name);
                            }
                        }
                        self.expr(init);
                    }
                }
            }
            Statement::ExpressionStatement(node) => self.expr(&node.expression),
            _ => {}
        }
    }

    /// The thing a `using` declaration acquires — the call that produced it.
    fn acquisition(&self, expr: &Expression<'_>) -> Option<String> {
        match expr {
            Expression::CallExpression(call) => self.callee(&call.callee),
            Expression::AwaitExpression(node) => self.acquisition(&node.argument),
            Expression::NewExpression(new) => self.callee(&new.callee),
            _ => None,
        }
    }

    fn expr(&mut self, expr: &Expression<'_>) {
        match expr {
            Expression::CallExpression(call) => {
                if let Some(name) = self.callee(&call.callee) {
                    self.facts.effects.push(name);
                }
                self.expr(&call.callee);
                for arg in &call.arguments {
                    if let Some(inner) = arg.as_expression() {
                        self.expr(inner);
                    }
                }
            }
            Expression::NewExpression(new) => {
                if let Some(name) = self.callee(&new.callee) {
                    self.facts.effects.push(name);
                }
            }
            Expression::AwaitExpression(node) => {
                self.tag("await");
                self.expr(&node.argument);
            }
            Expression::StaticMemberExpression(member) => self.expr(&member.object),
            Expression::BinaryExpression(node) => {
                self.expr(&node.left);
                self.expr(&node.right);
            }
            Expression::LogicalExpression(node) => {
                self.expr(&node.left);
                self.expr(&node.right);
            }
            Expression::ConditionalExpression(node) => {
                self.tag("ternary");
                self.expr(&node.test);
                self.expr(&node.consequent);
                self.expr(&node.alternate);
            }
            Expression::AssignmentExpression(node) => self.expr(&node.right),
            Expression::TSNonNullExpression(node) => self.expr(&node.expression),
            Expression::TSAsExpression(node) => self.expr(&node.expression),
            _ => {}
        }
    }
}

/// The names a type annotation mentions that the module did not introduce, so `Promise<Foo>` and
/// `Promise<Bar>` agree on `Promise` and say nothing else.
fn type_names(ty: Option<&TSType<'_>>, bound: &HashSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    collect_type_names(ty, bound, &mut out);
    out
}

fn collect_type_names(ty: Option<&TSType<'_>>, bound: &HashSet<String>, out: &mut Vec<String>) {
    let Some(ty) = ty else { return };
    match ty {
        TSType::TSTypeReference(node) => {
            let name = match &node.type_name {
                ast::TSTypeName::IdentifierReference(id) => id.name.to_string(),
                ast::TSTypeName::QualifiedName(q) => q.right.name.to_string(),
                ast::TSTypeName::ThisExpression(_) => "this".to_owned(),
            };
            out.push(if bound.contains(&name) { "_".to_owned() } else { name });
            if let Some(args) = &node.type_arguments {
                for arg in &args.params {
                    collect_type_names(Some(arg), bound, out);
                }
            }
        }
        TSType::TSUnionType(node) => {
            for member in &node.types {
                collect_type_names(Some(member), bound, out);
            }
        }
        TSType::TSArrayType(node) => {
            out.push("[]".to_owned());
            collect_type_names(Some(&node.element_type), bound, out);
        }
        TSType::TSStringKeyword(_) => out.push("string".to_owned()),
        TSType::TSNumberKeyword(_) => out.push("number".to_owned()),
        TSType::TSBooleanKeyword(_) => out.push("boolean".to_owned()),
        TSType::TSVoidKeyword(_) => out.push("void".to_owned()),
        TSType::TSNullKeyword(_) => out.push("null".to_owned()),
        TSType::TSUndefinedKeyword(_) => out.push("undefined".to_owned()),
        TSType::TSAnyKeyword(_) => out.push("any".to_owned()),
        TSType::TSUnknownKeyword(_) => out.push("unknown".to_owned()),
        _ => out.push("?".to_owned()),
    }
}

/// What contract the callable offers: its arity shape and the names in its annotations.
fn signature_facts(
    params: &FormalParameters<'_>,
    return_type: Option<&TSType<'_>>,
    is_async: bool,
    bound: &HashSet<String>,
) -> Vec<String> {
    let mut out = vec![format!("arity={}", params.items.len())];
    if params.rest.is_some() {
        out.push("rest".to_owned());
    }
    if is_async {
        out.push("async".to_owned());
    }
    for param in &params.items {
        if param.optional {
            out.push("optional".to_owned());
        }
        let annotation = param.type_annotation.as_ref().map(|a| &a.type_annotation);
        out.extend(type_names(annotation, bound).into_iter().map(|n| format!("param:{n}")));
    }
    out.extend(type_names(return_type, bound).into_iter().map(|n| format!("ret:{n}")));
    out
}

/// What role it plays: the decorator names on the definition.
fn decorator_facts(decorators: &[ast::Decorator<'_>]) -> Vec<String> {
    let mut out: Vec<String> = decorators
        .iter()
        .filter_map(|d| match &d.expression {
            Expression::Identifier(id) => Some(id.name.to_string()),
            Expression::CallExpression(call) => match &call.callee {
                Expression::Identifier(id) => Some(id.name.to_string()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The shape a class declares: one fact per property, sorted.
///
/// A declaration is a **set** — the same class with its properties reordered is the same class —
/// where a body's statements are a sequence.
fn class_schema(class: &Class<'_>, bound: &HashSet<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for element in &class.body.body {
        if let ClassElement::PropertyDefinition(prop) = element {
            let annotation = prop.type_annotation.as_ref().map(|a| &a.type_annotation);
            let mut fact = type_names(annotation, bound).join(" ");
            if prop.r#static {
                fact.push_str(" static");
            }
            if prop.readonly {
                fact.push_str(" readonly");
            }
            if prop.optional {
                fact.push_str(" optional");
            }
            out.push(fact);
        }
    }
    out.sort();
    out
}

/// The shape an interface declares: one fact per member, sorted.
fn interface_schema(node: &TSInterfaceDeclaration<'_>, bound: &HashSet<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for member in &node.body.body {
        match member {
            ast::TSSignature::TSPropertySignature(prop) => {
                let annotation = prop.type_annotation.as_ref().map(|a| &a.type_annotation);
                let mut fact = type_names(annotation, bound).join(" ");
                if prop.optional {
                    fact.push_str(" optional");
                }
                if prop.readonly {
                    fact.push_str(" readonly");
                }
                out.push(fact);
            }
            ast::TSSignature::TSMethodSignature(method) => {
                let annotation = method.return_type.as_ref().map(|a| &a.type_annotation);
                out.push(format!("method {}", type_names(annotation, bound).join(" ")));
            }
            _ => {}
        }
    }
    out.sort();
    out
}

/// Facts for a callable: everything the ten lenses can say about a function.
pub(crate) fn callable_facts(
    params: &FormalParameters<'_>,
    body: &[Statement<'_>],
    return_type: Option<&TSType<'_>>,
    decorators: &[ast::Decorator<'_>],
    is_async: bool,
    bound: &HashSet<String>,
    scope_canonical: String,
) -> LensFacts {
    let walked = Walk::run(body, bound);
    let mut facts = LensFacts::new();
    // A *set*: what it depends on, order irrelevant.
    facts.extend(Lens::Outgoing, walked.effects.iter().cloned().collect::<BTreeSet<_>>());
    // A *sequence*: the protocol it drives, order load-bearing.
    facts.extend(Lens::Effects, walked.effects.clone());
    facts.extend(Lens::Control, walked.control.clone());
    facts.extend(Lens::Failures, walked.failures.iter().cloned());
    facts.extend(Lens::Resources, walked.resources.clone());
    facts.extend(Lens::Signature, signature_facts(params, return_type, is_async, bound));
    facts.extend(Lens::Decorators, decorator_facts(decorators));
    // One fact rather than one per statement: what it asserts is all-or-nothing.
    facts.push(Lens::Scope, scope_canonical);
    facts
}

/// Facts for a class: what it declares, and under what decorators.
pub(crate) fn class_facts(class: &Class<'_>, bound: &HashSet<String>) -> LensFacts {
    let mut facts = LensFacts::new();
    facts.extend(Lens::Schema, class_schema(class, bound));
    facts.extend(Lens::Decorators, decorator_facts(&class.decorators));
    facts
}

/// Facts for an interface: the shape it declares.
pub(crate) fn interface_facts(node: &TSInterfaceDeclaration<'_>, bound: &HashSet<String>) -> LensFacts {
    let mut facts = LensFacts::new();
    facts.extend(Lens::Schema, interface_schema(node, bound));
    facts
}

/// Names the module itself introduced — so a lens sees past this module's own identities.
#[must_use]
pub fn module_bound_names(prog: &ast::Program<'_>) -> HashSet<String> {
    let mut out = HashSet::new();
    for stmt in &prog.body {
        collect_declared(stmt, &mut out);
    }
    out
}

fn collect_declared(stmt: &Statement<'_>, out: &mut HashSet<String>) {
    match stmt {
        Statement::FunctionDeclaration(node) => {
            if let Some(id) = &node.id {
                out.insert(id.name.to_string());
            }
        }
        Statement::ClassDeclaration(node) => {
            if let Some(id) = &node.id {
                out.insert(id.name.to_string());
            }
        }
        Statement::TSTypeAliasDeclaration(node) => {
            out.insert(node.id.name.to_string());
        }
        Statement::TSInterfaceDeclaration(node) => {
            out.insert(node.id.name.to_string());
        }
        Statement::TSEnumDeclaration(node) => {
            out.insert(node.id.name.to_string());
        }
        Statement::VariableDeclaration(node) => {
            for decl in &node.declarations {
                if let BindingPattern::BindingIdentifier(id) = &decl.id {
                    out.insert(id.name.to_string());
                }
            }
        }
        Statement::ExportNamedDeclaration(node) => {
            if let Some(decl) = &node.declaration {
                collect_declared_from(decl, out);
            }
        }
        Statement::ExportDefaultDeclaration(node) => {
            if let ast::ExportDefaultDeclarationKind::FunctionDeclaration(f) = &node.declaration {
                if let Some(id) = &f.id {
                    out.insert(id.name.to_string());
                }
            }
        }
        _ => {}
    }
}

fn collect_declared_from(decl: &ast::Declaration<'_>, out: &mut HashSet<String>) {
    match decl {
        ast::Declaration::FunctionDeclaration(node) => {
            if let Some(id) = &node.id {
                out.insert(id.name.to_string());
            }
        }
        ast::Declaration::ClassDeclaration(node) => {
            if let Some(id) = &node.id {
                out.insert(id.name.to_string());
            }
        }
        ast::Declaration::TSTypeAliasDeclaration(node) => {
            out.insert(node.id.name.to_string());
        }
        ast::Declaration::TSInterfaceDeclaration(node) => {
            out.insert(node.id.name.to_string());
        }
        ast::Declaration::VariableDeclaration(node) => {
            for inner in &node.declarations {
                if let BindingPattern::BindingIdentifier(id) = &inner.id {
                    out.insert(id.name.to_string());
                }
            }
        }
        _ => {}
    }
}
