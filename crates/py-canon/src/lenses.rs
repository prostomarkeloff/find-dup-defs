//! **Lenses** — the perspective axis, the one that is *not* a rung of the erasure ladder.
//!
//! The ladder (locals → own name → module names → everything) always canonicalizes the same text:
//! the definition's body. A lens canonicalizes a *different projection* of the same definition, so
//! two lenses are not a wider and a narrower version of one another — measured on an application
//! corpus, the use-profile lens overlaps the body passes by under 10%.
//!
//! Every lens here answers one question about a definition and throws the rest away:
//!
//! | lens | question | keeps |
//! |---|---|---|
//! | `outgoing`  | what does it depend on?      | the *set* of external callees |
//! | `effects`   | what protocol does it drive? | the *sequence* of external callees |
//! | `control`   | how does it branch?          | if/for/while/try/with/return/raise skeleton |
//! | `failures`  | how does it fail?            | raised and caught exception types |
//! | `resources` | what does it hold open?      | `with` / `async with` context expressions |
//! | `signature` | what contract does it offer? | arity shape + annotation names |
//! | `decorators`| what role does it play?      | the decorator names |
//! | `schema`    | what shape does it declare?  | column types and their options, as a *set* |
//! | `scope`     | what does its body do?       | the body with every name its module introduced erased |
//! | `use`       | how is it handled?           | the statements elsewhere that mention it |
//!
//! Names the *module itself* introduced are erased before any lens runs, exactly as on rung 3 —
//! otherwise a lens is held apart by the very identities it is meant to see past. What survives is
//! the grammar of talking to things the module did not define, projected one way per lens.
//!
//! The lenses are **stitched into one record**, not reported as separate kinds. Each contributes
//! its facts under its own prefix (`control:if`, `outgoing:.commit`), and the Type-3 pass's
//! IDF-weighted cosine over those lines *is* the vote: agreement through several lenses raises the
//! score, agreement through one barely moves it, and a fact the whole corpus shares is weighted to
//! nothing without anyone having to declare it noise. A cross-name exact match means every lens
//! agreed at once. Asking for the `lenses` kind (`--kinds lenses`) seats all of them: a lens is a
//! weight on one scale rather than a separate question, and the corpus IDF already silences the
//! ones a given tree has nothing to say through.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use dup_defs_core::{Analysis, CanonDialect, Def};
use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{self as ast, Expr, Stmt};

use crate::frontend::LENSES;

/// A projection with fewer facts than this carries no shape — two definitions that each call one
/// external thing match trivially. The lens counterpart of the Type-3 `SHINGLE_LINES` floor.
const MIN_FACTS: usize = 3;

/// The perspectives. Each is a `(Def → Vec<String>)` projection plus the kind it is reported under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lens {
    Outgoing,
    Effects,
    Control,
    Failures,
    Resources,
    Signature,
    Decorators,
    Schema,
    /// The body itself, with every name the module introduced erased — the widest rung of the
    /// erasure ladder, seated here as a perspective among the rest.
    Scope,
    /// The definition's *use sites*: the statements elsewhere in the tree that mention its name.
    /// Unlike every other lens this cannot be computed from the definition alone, so its facts are
    /// merged in by [`merge_use_facts`] once the whole tree has been walked.
    Use,
}

impl Lens {
    /// The prefix this lens stamps on its facts, so the stitched record stays attributable and two
    /// lenses can never accidentally agree on the same string.
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Lens::Outgoing => "outgoing",
            Lens::Effects => "effects",
            Lens::Control => "control",
            Lens::Failures => "failures",
            Lens::Resources => "resources",
            Lens::Signature => "signature",
            Lens::Decorators => "decorators",
            Lens::Schema => "schema",
            Lens::Scope => "scope",
            Lens::Use => "use",
        }
    }

    /// Every lens, in report order.
    pub(crate) fn all() -> [Lens; 10] {
        [
            Lens::Outgoing,
            Lens::Effects,
            Lens::Control,
            Lens::Failures,
            Lens::Resources,
            Lens::Signature,
            Lens::Decorators,
            Lens::Schema,
            Lens::Scope,
            Lens::Use,
        ]
    }
}

// ── Fact collection ────────────────────────────────────────────────────────

/// The name a call expression names, when it names something the module did not introduce. A bare
/// `foo(…)` yields `foo`; `obj.method(…)` yields `.method` — the receiver is whatever it is, the
/// *method* is the grammar. A call on a module-local name (its own model class, a sibling helper)
/// yields nothing: that is an identity, and identities are what the lens erases.
fn callee(expr: &Expr, bound: &HashSet<String>) -> Option<String> {
    match expr {
        Expr::Name(n) => {
            let id = n.id.as_str();
            (!bound.contains(id)).then(|| id.to_owned())
        }
        Expr::Attribute(a) => {
            let attr = a.attr.id.as_str();
            (!bound.contains(attr)).then(|| format!(".{attr}"))
        }
        _ => None,
    }
}

/// A type annotation reduced to the names it mentions that the module did not introduce, so
/// `Optional[JsonCache]` and `Optional[ThumbEntry]` agree on `Optional` and say nothing else.
fn annotation_shape(expr: &Expr, bound: &HashSet<String>, out: &mut Vec<String>) {
    match expr {
        Expr::Name(n) => {
            let id = n.id.as_str();
            out.push(if bound.contains(id) { "_".to_owned() } else { id.to_owned() });
        }
        Expr::Attribute(a) => out.push(format!(".{}", a.attr.id.as_str())),
        Expr::Subscript(sub) => {
            annotation_shape(&sub.value, bound, out);
            annotation_shape(&sub.slice, bound, out);
        }
        Expr::BinOp(b) => {
            annotation_shape(&b.left, bound, out);
            annotation_shape(&b.right, bound, out);
        }
        Expr::Tuple(t) => t.elts.iter().for_each(|e| annotation_shape(e, bound, out)),
        Expr::StringLiteral(_) => out.push("str-anno".to_owned()),
        Expr::NoneLiteral(_) => out.push("None".to_owned()),
        _ => out.push("?".to_owned()),
    }
}

/// Facts gathered in one walk of a definition, sliced up per lens afterwards.
#[derive(Default)]
struct Facts {
    /// Every external callee, in call order (the `effects` lens).
    effects: Vec<String>,
    /// Control-flow tags with their nesting depth (the `control` lens).
    control: Vec<String>,
    /// Raised and caught exception types (the `failures` lens).
    failures: BTreeSet<String>,
    /// Context expressions of `with` blocks (the `resources` lens).
    resources: Vec<String>,
}

struct FactWalk<'a> {
    bound: &'a HashSet<String>,
    depth: usize,
    facts: Facts,
}

impl<'a> FactWalk<'a> {
    /// Walk the *body* of a definition. The top node is skipped so a function's own `def` line does
    /// not become a control-flow fact about itself.
    fn run(stmt: &Stmt, bound: &'a HashSet<String>) -> Facts {
        let mut w = FactWalk { bound, depth: 0, facts: Facts::default() };
        match stmt {
            Stmt::FunctionDef(f) => f.body.iter().for_each(|s| w.visit_stmt(s)),
            Stmt::ClassDef(c) => c.body.iter().for_each(|s| w.visit_stmt(s)),
            _ => w.visit_stmt(stmt),
        }
        w.facts
    }

    fn tag(&mut self, tag: &str) {
        self.facts.control.push(format!("{}{tag}", "+".repeat(self.depth.min(4))));
    }

    fn nested(&mut self, f: impl FnOnce(&mut Self)) {
        self.depth += 1;
        f(self);
        self.depth -= 1;
    }
}

impl<'a> Visitor<'a> for FactWalk<'_> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::If(_) => self.tag("if"),
            Stmt::For(_) => self.tag("for"),
            Stmt::While(_) => self.tag("while"),
            Stmt::Try(_) => self.tag("try"),
            Stmt::With(w) => {
                self.tag(if w.is_async { "awith" } else { "with" });
                for item in &w.items {
                    if let Expr::Call(c) = &item.context_expr {
                        if let Some(name) = callee(&c.func, self.bound) {
                            self.facts.resources.push(name);
                        }
                    } else if let Some(name) = callee(&item.context_expr, self.bound) {
                        self.facts.resources.push(name);
                    }
                }
            }
            Stmt::Return(_) => self.tag("return"),
            Stmt::Break(_) => self.tag("break"),
            Stmt::Continue(_) => self.tag("continue"),
            Stmt::Raise(r) => {
                self.tag("raise");
                if let Some(exc) = &r.exc {
                    let target = match exc.as_ref() {
                        Expr::Call(c) => c.func.as_ref(),
                        other => other,
                    };
                    if let Some(name) = callee(target, self.bound) {
                        self.facts.failures.insert(format!("raise {name}"));
                    }
                }
            }
            _ => {}
        }
        // Compound statements nest; simple ones do not.
        if matches!(
            stmt,
            Stmt::If(_) | Stmt::For(_) | Stmt::While(_) | Stmt::Try(_) | Stmt::With(_)
        ) {
            self.nested(|w| visitor::walk_stmt(w, stmt));
        } else {
            visitor::walk_stmt(self, stmt);
        }
    }

    fn visit_except_handler(&mut self, handler: &'a ast::ExceptHandler) {
        let ast::ExceptHandler::ExceptHandler(h) = handler;
        if let Some(t) = &h.type_ {
            let mut names = Vec::new();
            annotation_shape(t, self.bound, &mut names);
            for n in names {
                self.facts.failures.insert(format!("except {n}"));
            }
        }
        visitor::walk_except_handler(self, handler);
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        if let Expr::Call(c) = expr {
            if let Some(name) = callee(&c.func, self.bound) {
                self.facts.effects.push(name);
            }
        }
        visitor::walk_expr(self, expr);
    }
}

// ── Projections ────────────────────────────────────────────────────────────

/// The signature's shape: how many parameters of each flavour, each annotation reduced to the names
/// the module did not introduce, plus the return annotation. Parameter *names* are dropped — a
/// contract is its types and arity, not its spelling.
fn signature_facts(stmt: &Stmt, bound: &HashSet<String>) -> Vec<String> {
    let Stmt::FunctionDef(f) = stmt else { return Vec::new() };
    let p = &f.parameters;
    let mut out = vec![format!("arity {}", p.posonlyargs.len() + p.args.len() + p.kwonlyargs.len())];
    // Only what is *present* becomes a fact. Emitting `posonly 0` / `kwonly 0` / `star 0` /
    // `async 0` / `default 0` for every ordinary function gave this lens seven or more facts on
    // nothing at all, and measured against a corpus that made it dominate two thirds of all
    // findings — outweighing lenses that had actually observed something. A shape says what it
    // has, not what it lacks.
    if !p.posonlyargs.is_empty() {
        out.push(format!("posonly {}", p.posonlyargs.len()));
    }
    if !p.kwonlyargs.is_empty() {
        out.push(format!("kwonly {}", p.kwonlyargs.len()));
    }
    if p.vararg.is_some() {
        out.push("vararg".to_owned());
    }
    if p.kwarg.is_some() {
        out.push("kwarg".to_owned());
    }
    if f.is_async {
        out.push("async".to_owned());
    }
    let mut defaults = 0usize;
    for param in p.posonlyargs.iter().chain(p.args.iter()).chain(p.kwonlyargs.iter()) {
        if param.default.is_some() {
            defaults += 1;
        }
        // An unannotated parameter carries no contract, so it contributes nothing rather than a
        // placeholder every unannotated parameter in the tree would share.
        if let Some(a) = &param.parameter.annotation {
            let mut names = Vec::new();
            annotation_shape(a, bound, &mut names);
            out.push(format!("arg {}", names.join("|")));
        }
    }
    if defaults > 0 {
        out.push(format!("defaults {defaults}"));
    }
    if let Some(r) = &f.returns {
        let mut ret = Vec::new();
        annotation_shape(r, bound, &mut ret);
        out.push(format!("ret {}", ret.join("|")));
    }
    out
}

/// The decorators a definition wears, in source order, reduced to callee names.
fn decorator_facts(stmt: &Stmt, bound: &HashSet<String>) -> Vec<String> {
    let decorators = match stmt {
        Stmt::FunctionDef(f) => &f.decorator_list,
        Stmt::ClassDef(c) => &c.decorator_list,
        _ => return Vec::new(),
    };
    decorators
        .iter()
        .filter_map(|d| {
            let target = match &d.expression {
                Expr::Call(c) => c.func.as_ref(),
                other => other,
            };
            callee(target, bound)
        })
        .collect()
}

/// A declared field: its type plus the options set on it, as one fact. Two things make a schema
/// declaration unlike a function body, and both are handled here rather than by the shared canon:
///
/// * **Order is not meaning.** The sequence of `Column(...)` lines is arbitrary — the same table
///   declared with its columns in another order is the same table — so the facts are sorted and
///   compared as a set, where a function's statements are a sequence.
/// * **Its literals are identities.** `__tablename__ = "person_summary"`, an index's name, a
///   `ForeignKey("persons.person_id")` target — these name *this* table, exactly as an identifier
///   would, so they are dropped. The literal `ForeignKey` / `Index` *call* survives: that a column
///   references something is shape, which table it references is identity.
///
/// The column's **type is kept verbatim** even when the module imported it. Types are the grammar a
/// schema is written in — the counterpart of the method name in `session.get(…)` — so the rung-3
/// erasure that applies everywhere else would erase the very thing worth comparing.
fn field_fact(name: &str, value: &Expr) -> Option<String> {
    let Expr::Call(call) = value else { return None };
    let ctor = match call.func.as_ref() {
        Expr::Name(n) => n.id.as_str(),
        Expr::Attribute(a) => a.attr.id.as_str(),
        _ => return None,
    };
    if name.starts_with("__") {
        return None; // `__tablename__` / `__table_args__` — identity and placement, not shape
    }
    let mut parts: Vec<String> = Vec::new();
    for arg in &call.arguments.args {
        match arg {
            // `mapped_column(BigInteger, ...)` / `Column(DateTime(timezone=False), ...)`
            Expr::Name(n) => parts.push(n.id.as_str().to_owned()),
            Expr::Call(inner) => match inner.func.as_ref() {
                // A nested constructor: keep its name, drop its arguments — `ForeignKey("persons.id")`
                // says "references something", and which something is this table's identity.
                Expr::Name(n) => parts.push(n.id.as_str().to_owned()),
                Expr::Attribute(a) => parts.push(a.attr.id.as_str().to_owned()),
                _ => {}
            },
            Expr::Attribute(a) => parts.push(a.attr.id.as_str().to_owned()),
            _ => {}
        }
    }
    for kw in &call.arguments.keywords {
        if let Some(arg) = &kw.arg {
            parts.push(arg.id.as_str().to_owned());
        }
    }
    parts.sort_unstable();
    parts.dedup();
    Some(format!("{ctor} {}", parts.join(" ")))
}

/// The shape a class declares: one fact per field, sorted. Empty for anything that declares no
/// fields, which is every function and most classes.
fn schema_facts(stmt: &Stmt) -> Vec<String> {
    let Stmt::ClassDef(class) = stmt else { return Vec::new() };
    let mut out: Vec<String> = Vec::new();
    for member in &class.body {
        let (name, value) = match member {
            Stmt::Assign(a) => match (a.targets.first(), &a.value) {
                (Some(Expr::Name(n)), v) => (n.id.as_str(), v.as_ref()),
                _ => continue,
            },
            Stmt::AnnAssign(a) => match (a.target.as_ref(), &a.value) {
                (Expr::Name(n), Some(v)) => (n.id.as_str(), v.as_ref()),
                _ => continue,
            },
            _ => continue,
        };
        if let Some(fact) = field_fact(name, value) {
            out.push(fact);
        }
    }
    // A declaration is a set: the same table with its columns reordered is the same table.
    out.sort();
    out
}

/// Project one definition through one lens. `scope_lines` is the module-scope rendering of the
/// body, computed by the caller (it needs the shared canonicalizer); `Use` yields nothing here and
/// is filled in later by [`merge_use_facts`].
fn project(
    lens: Lens,
    stmt: &Stmt,
    bound: &HashSet<String>,
    facts: &Facts,
    scope_lines: &[String],
) -> Vec<String> {
    match lens {
        // A *set*: what it depends on, order irrelevant.
        Lens::Outgoing => facts.effects.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect(),
        // A *sequence*: the protocol it drives, order load-bearing.
        Lens::Effects => facts.effects.clone(),
        Lens::Control => facts.control.clone(),
        Lens::Failures => facts.failures.iter().cloned().collect(),
        Lens::Resources => facts.resources.clone(),
        Lens::Signature => signature_facts(stmt, bound),
        Lens::Decorators => decorator_facts(stmt, bound),
        Lens::Schema => schema_facts(stmt),
        Lens::Scope => scope_lines.to_vec(),
        Lens::Use => Vec::new(),
    }
}

// ── Entry point ────────────────────────────────────────────────────────────

/// Facts from *every* enabled lens, each prefixed with its lens tag. Fewer than [`MIN_FACTS`]
/// facts overall ⇒ nothing to say about this definition.
fn stitched_facts(
    lenses: &[Lens],
    stmt: &Stmt,
    bound: &HashSet<String>,
    facts: &Facts,
    scope_lines: &[String],
) -> Vec<String> {
    let mut out = Vec::new();
    for &lens in lenses {
        let tag = lens.tag();
        out.extend(
            project(lens, stmt, bound, facts, scope_lines).into_iter().map(|f| format!("{tag}:{f}")),
        );
    }
    out
}

/// One [`Def`] per definition, carrying every enabled lens's view of it. `bound` is the module's own
/// bound-name set (rung 3), so every lens sees past the identities the module introduced.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lens_def(
    lenses: &[Lens],
    stmt: &Stmt,
    name: &str,
    file: &Arc<str>,
    line: usize,
    col: usize,
    loc: usize,
    bound: &HashSet<String>,
    scope_lines: &[String],
    out: &mut Vec<Def>,
) {
    if lenses.is_empty() {
        return;
    }
    let facts = FactWalk::run(stmt, bound);
    let stitched = stitched_facts(lenses, stmt, bound, &facts, scope_lines);
    if stitched.len() < MIN_FACTS {
        return;
    }
    let canonical = stitched.join(" ");
    out.push(Def {
        lang: "py",
        kind: &LENSES,
        name: name.to_owned(),
        file: Arc::clone(file),
        line,
        col,
        loc,
        args: stitched.len(),
        text_orig: stitched.join("\n"),
        cluster_canonical: Some(canonical.clone()),
        analysis: Some(Analysis {
            xname_canonical: canonical,
            size: stitched.len(),
            type3_lines: stitched,
            canon_dialect: CanonDialect::CPythonAst,
        }),
        // Filled in by `score_lens_defs` once the whole corpus is known.
        thickness: None,
    });
}

/// Fold each definition's use sites into its lens record. These are the only facts that cannot be
/// computed from the definition alone — they live in every *other* file — so they are merged here,
/// once the tree has been walked, rather than during the per-file scan. They arrive already
/// prefixed by channel (`use:` / `usectx:` / `cooc:`), which is why nothing is tagged here.
pub(crate) fn merge_use_facts(defs: &mut [Def], mut facts: std::collections::HashMap<String, Vec<String>>) {
    for d in defs.iter_mut() {
        if d.kind.id != "lenses" {
            continue;
        }
        let Some(extra) = facts.remove(&d.name) else { continue };
        let Some(a) = d.analysis.as_mut() else { continue };
        a.type3_lines.extend(extra);
        a.type3_lines.sort();
        a.size = a.type3_lines.len();
        let canonical = a.type3_lines.join(" ");
        a.xname_canonical.clone_from(&canonical);
        d.text_orig = a.type3_lines.join("\n");
        d.args = a.type3_lines.len();
        d.cluster_canonical = Some(canonical);
    }
}

/// Saturation constant for a stitched record's information mass — what counts as "a substantial
/// record". Calibrated so a handful of distinctive facts clears half scale on the trees this was
/// measured against.
const LENS_MASS_K: f64 = 10.0;

/// Score every lens record against the corpus, once scanning is done.
///
/// A fact's weight is its IDF over the corpus of lens records, so what counts as signal is decided
/// by the tree rather than by a list: `control:return` is in nearly every record and weighs nothing,
/// while `outgoing:.skip_locked` is rare and weighs a lot. The count of *lenses* that actually spoke
/// rides alongside, because a record whose facts all come from one lens is one opinion, not a
/// consensus — the same reason the ladder's `n` is evidence rather than volume.
pub(crate) fn score_lens_defs(defs: &mut [Def]) {
    let mass_k = LENS_MASS_K;
    let mut df: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut corpus = 0usize;
    for d in defs.iter() {
        if d.kind.id != "lenses" {
            continue;
        }
        corpus += 1;
        if let Some(a) = &d.analysis {
            for fact in a.type3_lines.iter().map(String::as_str).collect::<BTreeSet<_>>() {
                *df.entry(fact).or_insert(0) += 1;
            }
        }
    }
    if corpus == 0 {
        return;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = corpus as f64;
    let scores: Vec<Option<f64>> = defs
        .iter()
        .map(|d| {
            if d.kind.id != "lenses" {
                return None;
            }
            let a = d.analysis.as_ref()?;
            let mut mass = 0.0f64;
            let mut lenses_heard: BTreeSet<&str> = BTreeSet::new();
            for fact in a.type3_lines.iter().map(String::as_str).collect::<BTreeSet<_>>() {
                #[allow(clippy::cast_precision_loss)]
                let idf = (n / df.get(fact).copied().unwrap_or(1).max(1) as f64).ln();
                mass += idf.max(0.0);
                if let Some((tag, _)) = fact.split_once(':') {
                    lenses_heard.insert(tag);
                }
            }
            let mass_score = 1.0 - (-mass / mass_k).exp();
            #[allow(clippy::cast_precision_loss)]
            let breadth = 1.0 - (-(lenses_heard.len() as f64) / 2.0).exp();
            Some(0.7 * mass_score + 0.3 * breadth)
        })
        .collect();
    for (d, score) in defs.iter_mut().zip(scores) {
        if let Some(s) = score {
            d.thickness = Some(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Lens;
    use crate::canon::module_bound_names;
    use ruff_python_parser::parse_module;

    /// The lens's view of the first definition that has one, tags included. Goes through
    /// `stitched_facts` rather than `lens_def` so a projection below the emission floor is still
    /// observable — the floor is a property of the record, not of the lens.
    fn project_first(src: &str, lens: Lens) -> Vec<String> {
        let module = parse_module(src).expect("parse").into_syntax();
        let bound = module_bound_names(&module.body);
        for stmt in &module.body {
            let facts = super::FactWalk::run(stmt, &bound);
            let projected = super::stitched_facts(&[lens], stmt, &bound, &facts, &[]);
            if !projected.is_empty() {
                return projected;
            }
        }
        Vec::new()
    }

    #[test]
    fn outgoing_keeps_external_callees_and_drops_module_local_ones() {
        let src = concat!(
            "from x import helper\n",
            "class Model: pass\n",
            "def f(session):\n",
            "    row = session.get(Model, 1)\n",
            "    helper(row)\n",
            "    session.flush()\n",
            "    return session.commit()\n",
        );
        let got = project_first(src, Lens::Outgoing);
        // `.get` / `.commit` are grammar; `Model` and `helper` are identities this module introduced.
        assert!(
            got.contains(&"outgoing:.get".to_owned()) && got.contains(&"outgoing:.commit".to_owned()),
            "{got:?}"
        );
        assert!(!got.iter().any(|f| f.contains("helper") || f.contains("Model")), "{got:?}");
    }

    #[test]
    fn control_records_branching_with_nesting_and_ignores_calls() {
        let src = "def f(xs):\n    for x in xs:\n        if x:\n            return x\n    raise ValueError\n";
        let got = project_first(src, Lens::Control);
        assert_eq!(
            got,
            vec!["control:for", "control:+if", "control:++return", "control:raise"],
            "{got:?}"
        );
    }

    /// The signature lens reports what a contract *has*, never what it lacks: an unannotated
    /// function says almost nothing, while annotations and unusual parameter kinds are facts.
    #[test]
    fn signature_reports_only_what_is_present() {
        let bare = project_first("def f(a, b, c):\n    return a\n", Lens::Signature);
        assert_eq!(bare, vec!["signature:arity 3"], "an unannotated signature is one fact, not eight");

        let rich = project_first(
            "async def f(a: int, *rest, b: str = 'x') -> bool:\n    return b\n",
            Lens::Signature,
        );
        for want in ["kwonly 1", "vararg", "async", "arg int", "arg str", "defaults 1", "ret bool"] {
            let tagged = format!("signature:{want}");
            assert!(rich.contains(&tagged), "missing {tagged:?} in {rich:?}");
        }
        assert!(!rich.iter().any(|f| f.ends_with(" 0")), "no zero-valued facts: {rich:?}");
    }

    /// The schema lens's load-bearing claim: two tables declaring the same shape agree even when
    /// their names, their table names, their column names and their column *order* all differ —
    /// and a table that declares one extra column does not.
    #[test]
    fn schema_lens_compares_declarations_as_an_unordered_set_of_shapes() {
        let same = |a: &str, b: &str| project_first(a, Lens::Schema) == project_first(b, Lens::Schema);
        let attempts = concat!(
            "class AuthLoginAttempt(Base):\n",
            "    __tablename__ = 'auth_login_attempts'\n",
            "    email: Mapped[str] = mapped_column(Text, primary_key=True)\n",
            "    tried_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)\n",
        );
        // Same shape: other class name, other table name, other column names, columns reordered.
        let limits = concat!(
            "class RateLimit(Base):\n",
            "    __tablename__ = 'public_rate_limit_attempts'\n",
            "    seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)\n",
            "    source_hash: Mapped[str] = mapped_column(Text, primary_key=True)\n",
        );
        assert!(same(attempts, limits), "{:?} vs {:?}", project_first(attempts, Lens::Schema), project_first(limits, Lens::Schema));

        // One extra declared column is a different shape — the lens is not blind, just name-blind.
        let scoped = concat!(
            "class ScopedAttempt(Base):\n",
            "    __tablename__ = 'scoped_attempts'\n",
            "    email: Mapped[str] = mapped_column(Text, primary_key=True)\n",
            "    channel_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)\n",
            "    tried_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)\n",
        );
        assert!(!same(attempts, scoped));
    }

    /// The load-bearing claim for the whole axis: two orthogonal implementations of one shape agree
    /// through a lens even though they share no identifier at all.
    #[test]
    fn orthogonal_implementations_agree_through_the_control_lens() {
        let a = "from .clock import now\nclass A: pass\ndef get(s, k, ttl):\n    row = s.get(A, k)\n    if row is None:\n        return None\n    if now() - row.at > ttl:\n        return None\n    return row.blob\n";
        let b = "from .timing import moment\nclass B: pass\ndef fetch(db, i, age):\n    e = db.get(B, i)\n    if e is None:\n        return None\n    if moment() - e.stamp > age:\n        return None\n    return e.payload\n";
        let (fa, fb) = (project_first(a, Lens::Control), project_first(b, Lens::Control));
        // Guard against agreeing vacuously: an empty projection equals an empty projection.
        assert!(fa.len() >= super::MIN_FACTS, "the control lens projected nothing: {fa:?}");
        assert_eq!(fa, fb, "the control lens disagreed");
    }
}

