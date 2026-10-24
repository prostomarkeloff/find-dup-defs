//! **Lenses for Go** — the tree-sitter walk, and only the walk.
//!
//! The vocabulary, the stitching, the corpus scoring and the record itself live in
//! [`dup_defs_core::lens`]; the `use` lens's sites are [`crate::canon::collect_sites`]. What is here
//! is how Go answers the other questions.
//!
//! One of them Go cannot answer, and it is left silent rather than approximated: **`decorators`**.
//! Go attaches nothing to a declaration that changes what it does — no decorator, no attribute — and
//! a lens that never speaks weighs nothing in the corpus IDF without anyone declaring it absent.
//!
//! Two have a Go answer that does not look like the other languages':
//!
//! * **`failures`** — Go fails by returning an error, not by raising one. What a body fails *with*
//!   is the error it constructs (`errors.New`, `fmt.Errorf`, a wrapping call), the sentinel it
//!   returns or tests for (`ErrNotFound`, `errors.Is(err, sql.ErrNoRows)`), and `panic`. What it does
//!   with a failure it was handed is `if err != nil { return …, err }` — Go's spelling of Rust's `?`,
//!   and recorded as the same fact.
//! * **`resources`** — what a body holds open is what it releases on the way out: Go's scope guard is
//!   `defer`. The fact is what was *acquired* — `f, err := os.Open(p); defer f.Close()` holds
//!   `os.Open`, `ctx, cancel := context.WithTimeout(…); defer cancel()` holds `context.WithTimeout` —
//!   and, when the released value was not bound here (`defer s.mu.Unlock()`), the release itself.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::lens::{control_tag, Lens, LensFacts};
use tree_sitter::Node;

use crate::canon::Callable;
use crate::tree::{field, fields, kids, statements, text, unquote};

/// Facts gathered in one walk of a body, sliced up per lens afterwards.
#[derive(Default)]
struct Walked {
    /// Every external callee, in call order — the `effects` lens; deduped, it is `outgoing`.
    effects: Vec<String>,
    /// Control-flow tags carrying their nesting.
    control: Vec<String>,
    /// What the body fails with.
    failures: BTreeSet<String>,
    /// What the body holds open for the length of the function.
    resources: Vec<String>,
}

struct Walk<'a> {
    src: &'a str,
    imports: &'a HashMap<String, Arc<str>>,
    /// Names the file itself introduced — erased, exactly as on the widest rung of the ladder, so a
    /// lens is not held apart by the very identities it exists to see past.
    file_names: &'a HashSet<String>,
    /// Names the callable binds: calling one is calling what it was handed, not a dependency.
    locals: &'a HashSet<String>,
    receiver: Option<&'a str>,
    /// What each local was bound to by a call — `f, err := os.Open(p)` binds `f` to `os.Open` — so a
    /// `defer f.Close()` can say what was held.
    bound_by: HashMap<String, String>,
    depth: usize,
    facts: Walked,
}

impl<'a> Walk<'a> {
    fn text(&self, n: Node<'_>) -> &'a str {
        text(n, self.src)
    }

    fn is_local(&self, name: &str) -> bool {
        self.locals.contains(name) || self.receiver == Some(name)
    }

    fn is_package(&self, name: &str) -> bool {
        self.imports.contains_key(name) && !self.is_local(name)
    }

    fn tag(&mut self, tag: &str) {
        self.facts.control.push(control_tag(self.depth, tag));
    }

    fn nested(&mut self, f: impl FnOnce(&mut Self)) {
        self.depth += 1;
        f(self);
        self.depth -= 1;
    }

    /// The name a call names, when it names something the file did not introduce: a bare `f(…)`
    /// yields `f`, a package member `pkg.F`, and `x.m(…)` yields `.m` — the receiver is whatever it is,
    /// the *method* is the grammar. A call through a local is a call of what it was handed: none.
    fn callee(&self, call: Node<'_>) -> Option<String> {
        let f = field(call, "function")?;
        match f.kind() {
            "identifier" => {
                let name = self.text(f);
                (!self.file_names.contains(name) && !self.is_local(name)).then(|| name.to_owned())
            }
            "selector_expression" => {
                let operand = field(f, "operand")?;
                let name = self.text(field(f, "field")?);
                if operand.kind() == "identifier" && self.is_package(self.text(operand)) {
                    return Some(format!("{}.{name}", self.text(operand)));
                }
                (!self.file_names.contains(name)).then(|| format!(".{name}"))
            }
            _ => None,
        }
    }

    fn walk(&mut self, n: Node<'_>) {
        match n.kind() {
            "call_expression" => {
                if let Some(callee) = self.callee(n) {
                    self.failure_of_call(n, &callee);
                    self.facts.effects.push(callee);
                }
            }
            "if_statement" => {
                if let Some(init) = field(n, "initializer") {
                    self.walk(init);
                }
                if let Some(cond) = field(n, "condition") {
                    self.walk(cond);
                }
                if self.propagates(n) {
                    self.facts.failures.insert("?".to_owned());
                }
                self.tag("if");
                self.nested(|w| {
                    if let Some(then) = field(n, "consequence") {
                        w.walk(then);
                    }
                    if let Some(alt) = field(n, "alternative") {
                        w.walk(alt);
                    }
                });
                return;
            }
            "for_statement" => {
                let head = kids(n).into_iter().find(|k| Some(k.id()) != field(n, "body").map(|b| b.id()));
                self.tag(match head.map(|h| h.kind()) {
                    None => "loop",
                    Some("range_clause" | "for_clause") => "for",
                    Some(_) => "while",
                });
                if let Some(head) = head {
                    self.walk(head);
                }
                self.nested(|w| {
                    if let Some(body) = field(n, "body") {
                        w.walk(body);
                    }
                });
                return;
            }
            "expression_switch_statement" | "type_switch_statement" | "select_statement" => {
                for part in ["initializer", "value"] {
                    if let Some(p) = field(n, part) {
                        self.walk(p);
                    }
                }
                self.tag(match n.kind() {
                    "type_switch_statement" => "typeswitch",
                    "select_statement" => "select",
                    _ => "switch",
                });
                self.nested(|w| {
                    for case in kids(n).into_iter().filter(|k| k.kind().ends_with("_case")) {
                        w.walk(case);
                    }
                });
                return;
            }
            "go_statement" => self.tag("go"),
            "defer_statement" => {
                self.tag("defer");
                self.release(n);
            }
            "return_statement" => {
                self.tag("return");
                for value in kids(n).first().map(|l| kids(*l)).unwrap_or_default() {
                    self.sentinel(value);
                }
            }
            "short_var_declaration" => self.note_bindings(n),
            // `err == ErrX` — the failure a body tells apart from the rest.
            "binary_expression" if field(n, "operator").is_some_and(|o| matches!(o.kind(), "==" | "!=")) => {
                for side in ["left", "right"] {
                    if let Some(s) = field(n, side) {
                        self.sentinel(s);
                    }
                }
            }
            _ => {}
        }
        for k in kids(n) {
            self.walk(k);
        }
    }

    /// The errors a call constructs, raises, or tests for.
    fn failure_of_call(&mut self, call: Node<'_>, callee: &str) {
        let member = callee.rsplit('.').next().unwrap_or(callee);
        let builds = matches!(callee, "panic" | "errors.New" | "errors.Join" | "fmt.Errorf")
            || (callee.contains('.')
                && !callee.starts_with('.')
                && matches!(member, "Errorf" | "Wrap" | "Wrapf" | "WithMessage" | "WithMessagef" | "WithStack"));
        if builds {
            self.facts.failures.insert(callee.to_owned());
        }
        // `errors.Is(err, ErrX)` — the failure a body tells apart from the rest.
        if callee == "errors.Is" {
            if let Some(target) = field(call, "arguments").and_then(|a| kids(a).into_iter().nth(1)) {
                self.sentinel(target);
            }
        }
    }

    /// A sentinel error named by the convention every Go package keeps: `ErrX`, or `pkg.ErrX`.
    fn sentinel(&mut self, n: Node<'_>) {
        let name = match n.kind() {
            "identifier" => self.text(n),
            "selector_expression" => field(n, "field").map_or("", |f| self.text(f)),
            _ => return,
        };
        let is_sentinel = name.strip_prefix("Err").is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_uppercase()));
        if is_sentinel {
            let spelled = self.text(n).to_owned();
            self.facts.failures.insert(spelled);
        }
    }

    /// `if err != nil { … return …err… }` — the failure handed back to the caller.
    fn propagates(&self, n: Node<'_>) -> bool {
        let Some(cond) = field(n, "condition").filter(|c| c.kind() == "binary_expression") else { return false };
        if field(cond, "operator").is_none_or(|o| o.kind() != "!=") {
            return false;
        }
        let (Some(left), Some(right)) = (field(cond, "left"), field(cond, "right")) else { return false };
        let checked = match (left.kind(), right.kind()) {
            ("identifier", "nil") => self.text(left),
            ("nil", "identifier") => self.text(right),
            _ => return false,
        };
        field(n, "consequence").is_some_and(|then| {
            statements(then)
                .into_iter()
                .filter(|s| s.kind() == "return_statement")
                .any(|ret| mentions(ret, checked, self.src))
        })
    }

    /// What a `defer` releases: the call that acquired the value it releases, or the release itself.
    fn release(&mut self, n: Node<'_>) {
        let Some(call) = kids(n).into_iter().next().filter(|c| c.kind() == "call_expression") else { return };
        let Some(f) = field(call, "function") else { return };
        let held = match f.kind() {
            "identifier" => {
                let name = self.text(f);
                self.bound_by.get(name).cloned().or_else(|| (!self.is_local(name)).then(|| name.to_owned()))
            }
            "selector_expression" => {
                let operand = field(f, "operand").filter(|o| o.kind() == "identifier").map(|o| self.text(o));
                operand.and_then(|o| self.bound_by.get(o).cloned()).or_else(|| self.callee(call))
            }
            _ => None,
        };
        if let Some(held) = held {
            self.facts.resources.push(held);
        }
    }

    /// `a, b := f(…)` — remember that `a` and `b` came from `f`.
    fn note_bindings(&mut self, n: Node<'_>) {
        let values = field(n, "right").map(kids).unwrap_or_default();
        let [call] = values.as_slice() else { return };
        if call.kind() != "call_expression" {
            return;
        }
        let Some(callee) = self.callee(*call) else { return };
        for name in field(n, "left").map(kids).unwrap_or_default() {
            if name.kind() == "identifier" && self.text(name) != "_" {
                self.bound_by.insert(self.text(name).to_owned(), callee.clone());
            }
        }
    }
}

/// Whether an identifier spelled `name` occurs anywhere under `n`.
fn mentions(n: Node<'_>, name: &str, src: &str) -> bool {
    (n.kind() == "identifier" && text(n, src) == name) || kids(n).into_iter().any(|k| mentions(k, name, src))
}

/// Type names a type mentions that the file did not introduce, so `map[string]*User` and
/// `map[string]*Order` agree on `map` and on `string` and say nothing else.
fn type_shape(n: Node<'_>, src: &str, file_names: &HashSet<String>, out: &mut Vec<String>) {
    match n.kind() {
        "type_identifier" => {
            let name = text(n, src);
            out.push(if file_names.contains(name) { "_".to_owned() } else { name.to_owned() });
        }
        "qualified_type" => out.push(text(n, src).to_owned()),
        "pointer_type" | "parenthesized_type" | "negated_type" | "type_elem" | "type_constraint" => {
            for k in kids(n) {
                type_shape(k, src, file_names, out);
            }
        }
        "slice_type" | "array_type" | "implicit_length_array_type" => {
            out.push("[]".to_owned());
            if let Some(e) = field(n, "element") {
                type_shape(e, src, file_names, out);
            }
        }
        "map_type" => {
            out.push("map".to_owned());
            for part in ["key", "value"] {
                if let Some(p) = field(n, part) {
                    type_shape(p, src, file_names, out);
                }
            }
        }
        "channel_type" => {
            out.push("chan".to_owned());
            if let Some(v) = field(n, "value") {
                type_shape(v, src, file_names, out);
            }
        }
        "generic_type" => {
            if let Some(t) = field(n, "type") {
                type_shape(t, src, file_names, out);
            }
            for arg in field(n, "type_arguments").map(kids).unwrap_or_default() {
                type_shape(arg, src, file_names, out);
            }
        }
        "function_type" => out.push("func".to_owned()),
        "interface_type" => out.push("interface".to_owned()),
        "struct_type" => out.push("struct".to_owned()),
        _ => out.push("?".to_owned()),
    }
}

/// What contract the callable offers: its arity shape and the names in its types.
fn signature_facts(c: &Callable<'_>, src: &str, file_names: &HashSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    let params = c.params.map(kids).unwrap_or_default();
    let arity: usize = params.iter().map(|d| fields(*d, "name").len().max(1)).sum();
    out.push(format!("arity={arity}"));
    if c.method {
        out.push("method".to_owned());
    }
    if params.iter().any(|d| d.kind() == "variadic_parameter_declaration") {
        out.push("variadic".to_owned());
    }
    for decl in &params {
        let Some(ty) = field(*decl, "type") else { continue };
        let mut names = Vec::new();
        type_shape(ty, src, file_names, &mut names);
        for _ in 0..fields(*decl, "name").len().max(1) {
            out.extend(names.iter().map(|n| format!("param:{n}")));
        }
    }
    let results: Vec<Node<'_>> = match c.results {
        Some(list) if list.kind() == "parameter_list" => kids(list).into_iter().filter_map(|d| field(d, "type")).collect(),
        Some(ty) => vec![ty],
        None => Vec::new(),
    };
    for ty in results {
        let mut names = Vec::new();
        type_shape(ty, src, file_names, &mut names);
        out.extend(names.into_iter().map(|n| format!("ret:{n}")));
    }
    out
}

/// Facts for a callable: everything the lenses can say about a function or method.
pub(crate) fn callable_facts(
    c: &Callable<'_>,
    src: &str,
    imports: &HashMap<String, Arc<str>>,
    locals: &HashSet<String>,
    file_names: &HashSet<String>,
    scope_canonical: String,
) -> LensFacts {
    let mut walk = Walk {
        src,
        imports,
        file_names,
        locals,
        receiver: c.receiver,
        bound_by: HashMap::new(),
        depth: 0,
        facts: Walked::default(),
    };
    walk.walk(c.body);
    let walked = walk.facts;
    let mut facts = LensFacts::new();
    // A *set*: what it depends on, order irrelevant.
    facts.extend(Lens::Outgoing, walked.effects.iter().cloned().collect::<BTreeSet<_>>());
    // A *sequence*: the protocol it drives, order load-bearing.
    facts.extend(Lens::Effects, walked.effects);
    facts.extend(Lens::Control, walked.control);
    facts.extend(Lens::Failures, walked.failures);
    facts.extend(Lens::Resources, walked.resources);
    facts.extend(Lens::Signature, signature_facts(c, src, file_names));
    // One fact rather than one per statement: what it asserts is all-or-nothing, and as one rare
    // string it carries the IDF weight that claim deserves.
    facts.push(Lens::Scope, scope_canonical);
    facts
}

/// The shape a struct declares: one fact per field — its type's names, and the keys of its tag
/// (`json`, `db`, `yaml`), which are the field's options — sorted, because a declaration is a set.
pub(crate) fn struct_facts(struct_type: Node<'_>, src: &str, file_names: &HashSet<String>) -> LensFacts {
    let mut shapes = Vec::new();
    for list in kids(struct_type) {
        for decl in kids(list).into_iter().filter(|k| k.kind() == "field_declaration") {
            let mut type_names = Vec::new();
            if let Some(ty) = field(decl, "type") {
                type_shape(ty, src, file_names, &mut type_names);
            }
            let mut shape = type_names.join(" ");
            if let Some(tag) = field(decl, "tag") {
                let keys: Vec<&str> =
                    unquote(text(tag, src)).split_whitespace().filter_map(|kv| kv.split_once(':').map(|(k, _)| k)).collect();
                if !keys.is_empty() {
                    shape = format!("{shape} {}", keys.join(" "));
                }
            }
            let named = fields(decl, "name").len();
            if named == 0 {
                shapes.push(format!("embed {shape}"));
            }
            for _ in 0..named {
                shapes.push(shape.clone());
            }
        }
    }
    shapes.sort();
    let mut facts = LensFacts::new();
    facts.extend(Lens::Schema, shapes);
    facts
}

/// Names the file itself introduced — its functions, types, constants and variables — so a lens
/// sees past the identities of this file.
#[must_use]
pub(crate) fn file_bound_names(root: Node<'_>, src: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for item in kids(root) {
        match item.kind() {
            "function_declaration" => {
                if let Some(name) = field(item, "name") {
                    out.insert(text(name, src).to_owned());
                }
            }
            "type_declaration" | "const_declaration" | "var_declaration" => {
                let mut specs = Vec::new();
                for k in kids(item) {
                    if k.kind().ends_with("_spec") || k.kind() == "type_alias" {
                        specs.push(k);
                    } else {
                        specs.extend(kids(k));
                    }
                }
                for spec in specs {
                    for name in fields(spec, "name") {
                        out.insert(text(name, src).to_owned());
                    }
                }
            }
            _ => {}
        }
    }
    out.remove("_");
    out
}
