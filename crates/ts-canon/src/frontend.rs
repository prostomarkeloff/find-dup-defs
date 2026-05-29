//! The [`Frontend`] implementation for TypeScript.
//!
//! Lowers the existing [`find_module_defs`](crate::find_module_defs) extraction
//! ([`module_defs_from`](crate::defs::module_defs_from)) to the engine's [`Def`] contract,
//! computing each definition's canonical strings up front via the same [`ast_canonical`] /
//! [`analyze_functions`] entry points the pre-trait engine called — so output is byte-identical.
//!
//! TypeScript declares six kinds: it adds `interfaces` to the shared five. A method or
//! interface that does not analyze (TS methods are not standalone-parseable statements) yields
//! `analysis: None`, exactly as before. (A node-based, re-parse-free fast path is a later
//! optimization.)

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Analysis, Def, Frontend, KindSpec};
use rayon::prelude::*;

use crate::canon::{analyze_functions, ast_canonical};
use crate::defs::module_defs_from;

/// `function foo(...)` / `const foo = (...) => {}`.
pub static FUNCTIONS: KindSpec =
    KindSpec { id: "functions", label: "FUNCTION", noun_plural: "functions", section: 1, body: true, fn_like: true };
/// Class methods, qualified `Class.method`.
pub static METHODS: KindSpec =
    KindSpec { id: "methods", label: "METHOD", noun_plural: "methods", section: 4, body: true, fn_like: true };
/// `class Foo { ... }`.
pub static CLASSES: KindSpec =
    KindSpec { id: "classes", label: "CLASS", noun_plural: "classes", section: 7, body: true, fn_like: false };
/// `interface X { ... }` — first-class kind so directives can target it independently.
pub static INTERFACES: KindSpec =
    KindSpec { id: "interfaces", label: "INTERFACE", noun_plural: "interfaces", section: 8, body: true, fn_like: false };
/// Module-level `const NAME = ...` with an `UPPER_SNAKE` name and a non-function initializer.
pub static CONSTANTS: KindSpec =
    KindSpec { id: "constants", label: "CONSTANT", noun_plural: "constants", section: 0, body: false, fn_like: false };
/// `type X = ...` (note the space in `noun_plural`, distinct from the hyphenated `id`).
pub static TYPE_ALIASES: KindSpec =
    KindSpec { id: "type-aliases", label: "TYPE_ALIAS", noun_plural: "type aliases", section: 9, body: false, fn_like: false };

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &INTERFACES, &CONSTANTS, &TYPE_ALIASES];

fn kind_spec(id: &str) -> &'static KindSpec {
    match id {
        "functions" => &FUNCTIONS,
        "methods" => &METHODS,
        "classes" => &CLASSES,
        "interfaces" => &INTERFACES,
        "constants" => &CONSTANTS,
        "type-aliases" => &TYPE_ALIASES,
        other => unreachable!("ts-canon emitted unknown kind {other:?}"),
    }
}

/// Precompute the canonical strings from a def's text. TypeScript has no receiver strip, so the
/// canon input equals `text_orig`. Mirrors the pre-trait engine dispatch exactly.
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
    module_defs_from(source, file)
        .into_iter()
        .map(|md| {
            let kind = kind_spec(&md.kind);
            let (cluster_canonical, analysis) = canon_for(kind, &md.text);
            Def {
                lang: "ts",
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

/// TypeScript frontend over the oxc parser.
pub struct TypeScript;

impl Frontend for TypeScript {
    fn lang(&self) -> &'static str {
        "ts"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["ts", "tsx", "mts", "cts"]
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
    use super::{scan_source, FUNCTIONS, INTERFACES};
    use std::sync::Arc;

    #[test]
    fn scan_lowers_kinds_and_precomputes_canon() {
        let src = "export const MAX_RETRIES = 5;\n\nexport interface Repo {\n    get(id: number): number;\n}\n\nexport function compute(x: number, y: number): number {\n    const total = x + y;\n    return total * 2;\n}\n";
        let file: Arc<str> = Arc::from("t.ts");
        let defs = scan_source(src, &file);

        let func = defs.iter().find(|d| d.name == "compute").expect("compute fn");
        assert_eq!(func.kind.id, FUNCTIONS.id);
        assert_eq!(func.lang, "ts");
        assert!(func.cluster_canonical.is_some());
        assert!(func.analysis.is_some());

        let iface = defs.iter().find(|d| d.name == "Repo").expect("interface");
        assert_eq!(iface.kind.id, INTERFACES.id);
        assert!(iface.kind.body, "interface is a body kind");
        assert!(!iface.kind.fn_like, "interface is not callable");
        assert!(iface.cluster_canonical.is_some());
        assert!(iface.analysis.is_none());

        let konst = defs.iter().find(|d| d.name == "MAX_RETRIES").expect("constant");
        assert!(!konst.kind.body);
        assert!(konst.cluster_canonical.is_none());
    }
}
