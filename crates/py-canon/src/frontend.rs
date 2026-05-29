//! The [`Frontend`] implementation for Python.
//!
//! Lowers the proven [`find_module_defs`](crate::find_module_defs) extraction
//! ([`module_defs_from`](crate::defs::module_defs_from)) to the engine's [`Def`] contract,
//! computing each definition's canonical strings up front (during the single file parse the
//! caller drives), so the engine never dispatches canonicalization back into this crate.
//!
//! Python declares five kinds; `interfaces` is TypeScript-only. The canon is computed from a
//! def's *canon-input* text (`ModuleDef::text` — receiver-stripped for methods, identical to
//! `text_orig` otherwise) via the existing [`ast_canonical`] / [`analyze_functions`] entry
//! points, so the output is byte-identical to the pre-trait engine, which called those same two
//! functions. (A node-based, re-parse-free fast path is a later optimization.)

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Analysis, Def, Frontend, KindSpec};
use rayon::prelude::*;

use crate::canon::{analyze_functions, ast_canonical};
use crate::defs::module_defs_from;

// Section bases reproduce the engine's historical ordering: constants(0), functions(1→1/2/3 via
// pass offset), methods(4→4/5/6), classes(7), type-aliases(9). `interfaces`(8) is TS-only.
/// `def foo(...)` — top-level functions.
pub static FUNCTIONS: KindSpec =
    KindSpec { id: "functions", label: "FUNCTION", noun_plural: "functions", section: 1, body: true, fn_like: true };
/// Class methods, qualified `Class.method`.
pub static METHODS: KindSpec =
    KindSpec { id: "methods", label: "METHOD", noun_plural: "methods", section: 4, body: true, fn_like: true };
/// `class Foo: ...`.
pub static CLASSES: KindSpec =
    KindSpec { id: "classes", label: "CLASS", noun_plural: "classes", section: 7, body: true, fn_like: false };
/// Module-level `UPPER_CASE = ...` constants.
pub static CONSTANTS: KindSpec =
    KindSpec { id: "constants", label: "CONSTANT", noun_plural: "constants", section: 0, body: false, fn_like: false };
/// PEP 695 `type X = ...` aliases (note the space in `noun_plural`, distinct from the `id`).
pub static TYPE_ALIASES: KindSpec =
    KindSpec { id: "type-aliases", label: "TYPE_ALIAS", noun_plural: "type aliases", section: 9, body: false, fn_like: false };

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &CONSTANTS, &TYPE_ALIASES];

/// Map the extraction's legacy kind string to its `&'static KindSpec`. Internal to the frontend
/// — the engine never does this; it reads `KindSpec` fields directly.
fn kind_spec(id: &str) -> &'static KindSpec {
    match id {
        "functions" => &FUNCTIONS,
        "methods" => &METHODS,
        "classes" => &CLASSES,
        "constants" => &CONSTANTS,
        "type-aliases" => &TYPE_ALIASES,
        other => unreachable!("py-canon emitted unknown kind {other:?}"),
    }
}

/// Precompute the canonical strings from a def's canon-input text. Mirrors the pre-trait engine
/// dispatch exactly: `ast_canonical` for body kinds, `analyze_functions` for callables.
/// Non-body kinds (constants / type-aliases) carry no canonical — the engine clusters them on
/// `text_orig` directly.
fn canon_for(spec: &KindSpec, canon_text: &str) -> (Option<String>, Option<Analysis>) {
    let cluster_canonical = spec.body.then(|| ast_canonical(canon_text));
    let analysis = if spec.fn_like {
        analyze_functions(&[canon_text.to_owned()])
            .into_iter()
            .next()
            .flatten()
            .map(|(_cc, xname, lines, size)| Analysis { xname_canonical: xname, type3_lines: lines, size })
    } else {
        None
    };
    (cluster_canonical, analysis)
}

fn scan_source(source: &str, file: &Arc<str>) -> Vec<Def> {
    // `module_defs_from`'s own `file` field is discarded — we attach the shared `Arc` below.
    module_defs_from(source, file)
        .into_iter()
        .map(|md| {
            let kind = kind_spec(&md.kind);
            let (cluster_canonical, analysis) = canon_for(kind, &md.text);
            Def {
                lang: "py",
                kind,
                name: md.name,
                file: Arc::clone(file),
                line: md.line,
                col: md.col,
                loc: md.loc,
                args: md.args,
                text_orig: md.text_orig,
                cluster_canonical,
                analysis,
            }
        })
        .collect()
}

/// Python frontend over Ruff's parser.
pub struct Python;

impl Frontend for Python {
    fn lang(&self) -> &'static str {
        "py"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["py"]
    }
    fn kinds(&self) -> &'static [&'static KindSpec] {
        KINDS
    }
    fn scan(&self, files: &[Arc<str>]) -> Vec<Def> {
        files
            .par_iter()
            .flat_map(|f| fs::read_to_string(&**f).map_or_else(|_| Vec::new(), |src| scan_source(&src, f)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{scan_source, FUNCTIONS, METHODS};
    use std::sync::Arc;

    #[test]
    fn scan_lowers_kinds_and_precomputes_canon() {
        let src = "MAX = 5\n\n\ndef compute(x, y):\n    total = x + y\n    return total * 2\n\n\nclass C:\n    def fetch(self, k):\n        return self.store[k] + 1\n";
        let file: Arc<str> = Arc::from("t.py");
        let defs = scan_source(src, &file);

        let func = defs.iter().find(|d| d.name == "compute").expect("compute fn");
        assert_eq!(func.kind.id, FUNCTIONS.id);
        assert!(func.cluster_canonical.is_some(), "body kind has cluster canonical");
        assert!(func.analysis.is_some(), "fn_like kind has analysis");
        assert_eq!(func.lang, "py");

        let method = defs.iter().find(|d| d.name == "C.fetch").expect("method");
        assert_eq!(method.kind.id, METHODS.id);
        // `self` counted in args (user-visible), stripped from the canon input.
        assert_eq!(method.args, 2);
        assert!(method.analysis.is_some());

        let konst = defs.iter().find(|d| d.name == "MAX").expect("constant");
        assert!(!konst.kind.body, "constant is a raw-text kind");
        assert!(konst.cluster_canonical.is_none(), "non-body kind has no canonical");
        assert!(konst.analysis.is_none());
    }
}
