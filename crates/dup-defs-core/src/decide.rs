//! **Decision atoms** — the one spelling every frontend writes [`Facets::decides`] in.
//!
//! A definition decides by testing propositions. What a proposition *is* belongs to the language —
//! which expressions compare, which positions are truth-tested — so each frontend finds them. What a
//! proposition is *written as* belongs here, because the engine compares atoms from every language
//! as plain strings, and two frontends that spell `x == 1` differently would never meet.
//!
//! ## The grammar
//!
//! An atom is a relation over one or two **terms**, joined by [`SEP`]:
//!
//! | relation | from | operands |
//! |---|---|---|
//! | `=` | equality and identity, and their negations | unordered |
//! | `<` | the four order comparisons | unordered |
//! | `in` | membership and its negation | element, container |
//! | `?` | an expression tested for truth on its own | one |
//!
//! **Polarity is not part of an atom.** A decision's *support* — the set of facts it depends on —
//! is the same for a test and for its negation, and two places that make one decision are free to
//! write it either way round: one refuses when `x != a`, the other accepts when `x == a`. For the same
//! reason the order relations fold into one: `a < b` and `b <= a` are each other's negation, and
//! `a > b` is `b < a` written backwards.
//!
//! A term is an identity or a [`HOLE`]:
//!
//! - **hole** — a value the definition computed or was handed: a local, a parameter. Its name is the
//!   writer's choice, not a fact, exactly as the body passes' alpha-renaming has it;
//! - **field** — an attribute reached from a hole, `_.a.b`: which value it was is the writer's, which
//!   attribute was tested is the decision's;
//! - **path** — a name the module did not bind itself, resolved to the dotted path it stands for
//!   (`@a.b.C.X`); a **local path** is one the module defined (`@~module~C.X`), written with the
//!   module's own file path because only the corpus knows what the rest of the tree calls it — see
//!   [`rewrite_locals`];
//! - **literal** — its text, quoted by [`literal`];
//! - **call** — the callee's identity and the constants handed to it, `callee(c1, c2)`: what was
//!   computed is a fact, the computed value is not.
//!
//! [`Facets::decides`]: crate::Facets::decides

use std::fmt::Write as _;
use std::sync::Arc;

/// Separates an atom's relation from its terms. A control character, so no identifier can contain it
/// and [`literal`] escapes it out of any text — splitting on it is exact.
pub const SEP: char = '\u{1f}';

/// The term a value takes when it has no identity of its own.
pub const HOLE: &str = "_";

/// Marks a path the module defined itself; see [`local`].
const LOCAL: &str = "@~";

/// Marks a definition's own parameter, read bare; see [`param`].
const PARAM: char = '$';

/// Marks a field read off a value whose class the text names; see [`typed_field`].
const TYPED: char = '%';

/// Separates a typed field's class from the field.
const TYPED_SPLIT: char = '#';

/// Separates a local path's module from the name inside it. `~` cannot occur in an identifier.
const LOCAL_SPLIT: char = '~';

/// A field reached from a hole: `_.a.b`.
#[must_use]
pub fn field(attrs: &[&str]) -> String {
    let mut out = String::from(HOLE);
    for attr in attrs {
        out.push('.');
        out.push_str(attr);
    }
    out
}

/// A definition's own parameter, read bare: a hole the frontend cannot fill, but the corpus might.
///
/// What a parameter holds is decided by the call sites, which live in other definitions and other
/// files. When every one of them hands it the same identity the parameter *is* that identity — the
/// engine resolves it ([`resolve_params`]); otherwise it becomes the [`HOLE`] it was all along.
#[must_use]
pub fn param(name: &str) -> String {
    format!("{PARAM}{name}")
}

/// A field read off a value whose class the text names (an annotation, a constructor call), with the
/// attributes read past it.
///
/// A record's field is a parameter of its constructor, and the engine resolves it the way it resolves
/// a parameter — to what every construction the corpus shows passes for it, under the closed-world
/// reading ([`resolve_typed_fields`]). Unresolved it falls back to the receiver-erased field `_.f`,
/// the spelling every other field read has.
#[must_use]
pub fn typed_field(class: &str, field: &str, rest: &[&str]) -> String {
    let mut out = format!("{TYPED}{class}{TYPED_SPLIT}{field}");
    for attr in rest {
        out.push('.');
        out.push_str(attr);
    }
    out
}

/// Replace every typed field in `text` by what `resolve(class, field)` says it holds, or by the
/// receiver-erased `_.field` when it says nothing. Quoted literals are skipped.
#[must_use]
pub fn resolve_typed_fields(text: &str, resolve: impl Fn(&str, &str) -> Option<String>) -> String {
    if !text.contains(TYPED) {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if c == '"' {
            let quoted = &rest[1..];
            let end = closing_quote(quoted);
            out.push('"');
            out.push_str(&quoted[..end]);
            rest = &quoted[end..];
        } else if c == TYPED {
            let body = &rest[1..];
            let Some(split) = body.find(TYPED_SPLIT) else {
                out.push(c);
                rest = body;
                continue;
            };
            let class = &body[..split];
            let after = &body[split + 1..];
            let end = after.find(|ch: char| !(ch.is_alphanumeric() || ch == '_')).unwrap_or(after.len());
            let field_name = &after[..end];
            out.push_str(&resolve(class, field_name).unwrap_or_else(|| field(&[field_name])));
            rest = &after[end..];
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// The classes a text reads typed fields of, with the field.
pub fn typed_fields_in(text: &str) -> impl Iterator<Item = (&str, &str)> {
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(['"', TYPED]) {
        let tail = &rest[at..];
        if let Some(quoted) = tail.strip_prefix('"') {
            rest = &quoted[closing_quote(quoted)..];
            continue;
        }
        let body = &tail[1..];
        let Some(split) = body.find(TYPED_SPLIT) else { break };
        let after = &body[split + 1..];
        let end = after.find(|ch: char| !(ch.is_alphanumeric() || ch == '_')).unwrap_or(after.len());
        found.push((&body[..split], &after[..end]));
        rest = &after[end..];
    }
    found.into_iter()
}

/// One call a definition makes to a name it reaches as a path: the callee as written and each argument
/// as the term it is — positionally (`None`) or by keyword. `open` when the arguments cannot be mapped
/// onto parameters at all (`*args`, `**kwargs`), which leaves every parameter of the callee unknown.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CallSite {
    pub callee: Arc<str>,
    pub args: Vec<(Option<Arc<str>>, Arc<str>)>,
    pub open: bool,
}

/// A name resolved to the dotted path it stands for.
#[must_use]
pub fn path(dotted: &str) -> String {
    format!("@{dotted}")
}

/// A name the module defined itself: `module` is the module's own dotted file path, `name` the name
/// and any attributes after it.
///
/// Written apart from [`path`] because the module's file path is not what the rest of the tree calls
/// it: a file at `src/pkg/m.py` is imported as `pkg.m`. Which prefix the tree drops is not known to one
/// file, and guessing it is the kind of rule this grammar refuses — [`rewrite_locals`] lets the engine
/// learn it from the imports the corpus actually wrote.
#[must_use]
pub fn local(module: &str, name: &str) -> String {
    format!("{LOCAL}{module}{LOCAL_SPLIT}{name}")
}

/// A literal, by its text, in one quoting every frontend shares: double quotes, with `"`, `\` and
/// every control character escaped — which keeps [`SEP`] out of it.
#[must_use]
pub fn literal(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{{{:x}}}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A call: its callee's identity and the constants among its arguments, in a fixed order — the order
/// arguments are written in is not a fact about what was computed when they are keywords, and
/// positional constants are rare enough that sorting them costs nothing a reader would miss.
#[must_use]
pub fn call(callee: &str, mut constants: Vec<String>) -> String {
    constants.sort_unstable();
    constants.dedup();
    format!("{callee}({})", constants.join(", "))
}

/// The members of a container literal, as one term: a set, whichever bracket wrote it, since every
/// relation that reads one (membership) is blind to order and repetition.
#[must_use]
pub fn container(mut members: Vec<String>) -> String {
    members.sort_unstable();
    members.dedup();
    format!("{{{}}}", members.join(", "))
}

/// True when the term carries an identity anywhere in it. A hole, and anything built only of holes
/// (`(_ + _)`, `{_, _}`), says nothing two definitions could share.
#[must_use]
pub fn has_identity(term: &str) -> bool {
    term.split(['(', ')', '{', '}', ',', ' '])
        .any(|piece| !piece.is_empty() && piece != HOLE && !is_operator(piece))
}

/// Replace every parameter token in `text` by what `resolve` says the parameter holds, or by the
/// [`HOLE`] when it says nothing. Quoted literals are skipped.
#[must_use]
pub fn resolve_params(text: &str, resolve: impl Fn(&str) -> Option<String>) -> String {
    if !text.contains(PARAM) {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if c == '"' {
            let quoted = &rest[1..];
            let end = closing_quote(quoted);
            out.push('"');
            out.push_str(&quoted[..end]);
            rest = &quoted[end..];
        } else if c == PARAM {
            let name_end = rest[1..].find(|ch: char| !(ch.is_alphanumeric() || ch == '_')).map_or(rest.len(), |i| i + 1);
            let name = &rest[1..name_end];
            out.push_str(&resolve(name).unwrap_or_else(|| HOLE.to_owned()));
            rest = &rest[name_end..];
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// An atom brought back to canonical form after its terms were rewritten: the operands of a symmetric
/// relation in order again. Filling a parameter changes a term's spelling, and with it which side of
/// `=` it sorts to — without this, one fact would be written two ways depending on whether it was
/// reached through a parameter.
#[must_use]
pub fn canonical(atom: &str) -> String {
    let (rel, terms) = parts(atom);
    match (rel, terms.as_slice()) {
        ("=" | "<", [a, b]) if a > b => format!("{rel}{SEP}{b}{SEP}{a}"),
        _ => atom.to_owned(),
    }
}

/// The parameters a text refers to.
pub fn params_in(text: &str) -> impl Iterator<Item = &str> {
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(['"', PARAM]) {
        let tail = &rest[at..];
        if let Some(quoted) = tail.strip_prefix('"') {
            rest = &quoted[closing_quote(quoted)..];
        } else {
            let body = &tail[1..];
            let end = body.find(|ch: char| !(ch.is_alphanumeric() || ch == '_')).unwrap_or(body.len());
            found.push(&body[..end]);
            rest = &body[end..];
        }
    }
    found.into_iter()
}

fn is_operator(piece: &str) -> bool {
    !piece.is_empty() && piece.chars().all(|c| "+-*/%@&|^<>=!~".contains(c))
}

fn atom(rel: &str, terms: &[&str]) -> Option<Arc<str>> {
    if !terms.iter().any(|t| has_identity(t)) {
        return None;
    }
    let mut out = String::from(rel);
    for term in terms {
        out.push(SEP);
        out.push_str(term);
    }
    Some(Arc::from(out))
}

fn sorted<'a>(a: &'a str, b: &'a str) -> [&'a str; 2] {
    if a <= b {
        [a, b]
    } else {
        [b, a]
    }
}

/// Equality or identity between two terms, or the negation of either.
#[must_use]
pub fn equal(a: &str, b: &str) -> Option<Arc<str>> {
    atom("=", &sorted(a, b))
}

/// Any of the four order comparisons between two terms.
#[must_use]
pub fn order(a: &str, b: &str) -> Option<Arc<str>> {
    atom("<", &sorted(a, b))
}

/// Membership of `element` in `container`, or its negation.
#[must_use]
pub fn member(element: &str, container: &str) -> Option<Arc<str>> {
    atom("in", &[element, container])
}

/// A term tested for truth on its own.
#[must_use]
pub fn truth(term: &str) -> Option<Arc<str>> {
    atom("?", &[term])
}

/// An atom's relation and terms.
#[must_use]
pub fn parts(atom: &str) -> (&str, Vec<&str>) {
    let mut it = atom.split(SEP);
    let rel = it.next().unwrap_or("");
    (rel, it.collect())
}

/// The atom for a human: `a = b`, `a < b`, `a in b`, `? a`.
#[must_use]
pub fn render(atom: &str) -> String {
    let (rel, terms) = parts(atom);
    match terms.as_slice() {
        [one] => format!("{rel} {one}"),
        [a, b] => format!("{a} {rel} {b}"),
        _ => atom.replace(SEP, " "),
    }
}

/// Rewrite every local path in `atom` by `resolve(module)`, which names the module the way the rest
/// of the tree imports it. Quoted literals are skipped, so a string that merely *contains* the marker
/// is left alone.
#[must_use]
pub fn rewrite_locals(atom: &str, resolve: impl Fn(&str) -> String) -> Arc<str> {
    if !atom.contains(LOCAL) {
        return Arc::from(atom);
    }
    let mut out = String::with_capacity(atom.len());
    let mut rest = atom;
    while !rest.is_empty() {
        if let Some(quoted) = rest.strip_prefix('"') {
            let end = closing_quote(quoted);
            out.push('"');
            out.push_str(&quoted[..end]);
            rest = &quoted[end..];
            continue;
        }
        if let Some(after) = rest.strip_prefix(LOCAL) {
            if let Some(split) = after.find(LOCAL_SPLIT) {
                let module = &after[..split];
                out.push('@');
                let named = resolve(module);
                if !named.is_empty() {
                    out.push_str(&named);
                    out.push('.');
                }
                rest = &after[split + LOCAL_SPLIT.len_utf8()..];
                continue;
            }
        }
        let c = rest.chars().next().unwrap_or(' ');
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    Arc::from(out)
}

/// Every local module an atom names.
pub fn local_modules(atom: &str) -> impl Iterator<Item = &str> {
    let mut found = Vec::new();
    let mut rest = atom;
    while let Some(at) = rest.find(['"', '@']) {
        let tail = &rest[at..];
        if let Some(quoted) = tail.strip_prefix('"') {
            rest = &quoted[closing_quote(quoted)..];
        } else if let Some(after) = tail.strip_prefix(LOCAL) {
            match after.find(LOCAL_SPLIT) {
                Some(split) => {
                    found.push(&after[..split]);
                    rest = &after[split..];
                }
                None => break,
            }
        } else {
            rest = &tail[1..];
        }
    }
    found.into_iter()
}

/// Index just past the closing quote of a [`literal`] whose opening quote was already consumed.
fn closing_quote(quoted: &str) -> usize {
    let mut escaped = false;
    for (i, c) in quoted.char_indices() {
        match c {
            '\\' if !escaped => escaped = true,
            '"' if !escaped => return i + 1,
            _ => escaped = false,
        }
    }
    quoted.len()
}

#[cfg(test)]
mod tests {
    use super::{
        call, container, equal, field, has_identity, literal, local, local_modules, member, order, parts, path,
        render, rewrite_locals, truth, HOLE,
    };

    #[test]
    fn negation_and_operand_order_do_not_change_the_atom() {
        // `x.status == Status.A` and `Status.A != x.status` test one fact.
        let status = field(&["status"]);
        let a = path("app.Status.A");
        assert_eq!(equal(&status, &a), equal(&a, &status));
    }

    #[test]
    fn the_four_order_comparisons_are_one_relation() {
        let total = field(&["total"]);
        let limit = path("app.LIMIT");
        assert_eq!(order(&total, &limit), order(&limit, &total));
    }

    #[test]
    fn membership_keeps_its_direction() {
        let code = field(&["code"]);
        let allowed = field(&["allowed"]);
        assert_ne!(member(&code, &allowed), member(&allowed, &code));
    }

    #[test]
    fn an_atom_made_only_of_holes_says_nothing() {
        assert_eq!(equal(HOLE, HOLE), None);
        assert_eq!(truth("(_ + _)"), None);
        assert!(truth(&field(&["ready"])).is_some());
        assert!(!has_identity("{_, _}"));
        assert!(has_identity(&call("@app.check", Vec::new())));
    }

    #[test]
    fn a_literal_cannot_smuggle_the_separator_or_a_closing_quote() {
        let tricky = literal("a\u{1f}b\"c");
        assert!(!tricky.contains('\u{1f}'));
        assert_eq!(tricky, "\"a\\u{1f}b\\\"c\"");
    }

    #[test]
    fn calls_and_containers_do_not_depend_on_written_order() {
        assert_eq!(
            call("@app.f", vec![literal("b"), literal("a")]),
            call("@app.f", vec![literal("a"), literal("b")])
        );
        assert_eq!(container(vec![path("x.B"), path("x.A")]), container(vec![path("x.A"), path("x.B")]));
    }

    #[test]
    fn local_paths_are_rewritten_to_what_the_tree_calls_the_module() {
        let atom = equal(&field(&["state"]), &local("src.app.models", "State.OPEN")).unwrap_or_default();
        let named = rewrite_locals(&atom, |module| module.trim_start_matches("src.").to_owned());
        assert_eq!(render(&named), "@app.models.State.OPEN = _.state");
        assert_eq!(local_modules(&atom).collect::<Vec<_>>(), vec!["src.app.models"]);
    }

    #[test]
    fn a_parameter_resolves_to_what_it_holds_or_to_a_hole() {
        use super::{param, params_in, resolve_params};
        let atom = truth(&param("flag")).unwrap_or_default();
        let named = resolve_params(&atom, |p| (p == "flag").then(|| path("app.cfg.on")));
        assert_eq!(render(&named), "? @app.cfg.on");
        let unknown = resolve_params(&atom, |_| None);
        assert!(!has_identity(parts(&unknown).1[0]));
        assert_eq!(params_in(&equal(&param("a"), &literal("$b")).unwrap_or_default()).collect::<Vec<_>>(), vec!["a"]);
    }

    #[test]
    fn a_typed_field_resolves_or_falls_back_to_the_erased_field() {
        use super::{resolve_typed_fields, typed_field, typed_fields_in};
        let atom = equal(&typed_field("@app.Snap", "tariff", &[]), HOLE).unwrap_or_default();
        assert_eq!(typed_fields_in(&atom).collect::<Vec<_>>(), vec![("@app.Snap", "tariff")]);
        let filled = resolve_typed_fields(&atom, |c, f| (c == "@app.Snap" && f == "tariff").then(|| field(&["subtype"])));
        assert_eq!(render(&filled), "_.subtype = _");
        let erased = resolve_typed_fields(&typed_field("@app.Snap", "live", &["mode"]), |_, _| None);
        assert_eq!(erased, "_.live.mode");
    }

    #[test]
    fn a_rewritten_symmetric_atom_is_put_back_in_order() {
        use super::{canonical, param, resolve_params};
        let through_param = equal(&param("source"), &path("app.Source.TG")).unwrap_or_default();
        let written = equal(HOLE, &path("app.Source.TG")).unwrap_or_default();
        assert_eq!(canonical(&resolve_params(&through_param, |_| None)), *written);
    }

    #[test]
    fn a_marker_inside_a_literal_is_not_a_path() {
        let atom = equal(&field(&["note"]), &literal("@~not~a.path")).unwrap_or_default();
        assert_eq!(&*rewrite_locals(&atom, |_| "x".to_owned()), &*atom);
        assert_eq!(local_modules(&atom).count(), 0);
    }
}
