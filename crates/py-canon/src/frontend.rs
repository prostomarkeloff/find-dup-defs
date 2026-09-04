//! The [`Frontend`] implementation for Python — kind declarations + the file-reading driver.
//!
//! The actual scan (single Ruff parse per file → [`Def`]s with canon precomputed off the AST
//! nodes) lives in [`crate::defs::scan_source`]; this module owns the [`KindSpec`] vocabulary
//! and the `Python` registry entry. Python declares five kinds; `interfaces` is TypeScript-only.

use std::fs;
use std::sync::Arc;

use dup_defs_core::{Def, Frontend, KindSpec, ScanOpts};
use rayon::prelude::*;

use crate::defs::scan_source;

// The `KindSpec` vocabulary is shared across frontends — re-exported from `find-dup-defs-canon` so callers
// (`crate::frontend::METHODS`, …) are unchanged. Python declares five kinds; `interfaces` is TS-only.
pub use dup_defs_core::kinds::{
    CLASSES, CONSTANTS, FUNCTIONS, LENSES, METHODS, TYPE_ALIASES,
};

static KINDS: &[&KindSpec] = &[&FUNCTIONS, &METHODS, &CLASSES, &CONSTANTS, &TYPE_ALIASES];

/// The kind list with the opt-in `lenses` kind appended. It is never in the default set: it costs a
/// second walk of every file plus a per-definition projection, and it answers a question most runs
/// do not ask. Requested by name (`--kinds lenses`), so the choice is a CLI argument rather than a
/// side channel, and the default run stays byte-identical.
static KINDS_WITH_LENSES: &[&KindSpec] =
    &[&FUNCTIONS, &METHODS, &CLASSES, &CONSTANTS, &TYPE_ALIASES, &LENSES];

// Which lenses vote, and the record they stitch into, are the shared module's — re-exported here so
// the frontend's own call sites read the same as they did.
pub(crate) use dup_defs_core::lens::enabled_lenses;

/// Map the extraction's kind string to its `&'static KindSpec`. Internal to the frontend — the
/// engine never does this; it reads `KindSpec` fields directly. Delegates to the shared vocabulary
/// rather than re-listing it: a second copy of the same id→spec table drifts the moment a kind is
/// added to one and not the other.
pub(crate) fn kind_spec(id: &str) -> &'static KindSpec {
    dup_defs_core::kind_spec(id)
        .unwrap_or_else(|| unreachable!("py-canon emitted unknown kind {id:?}"))
}

/// A content digest that only narrows — the bytes decide — so it is the cheapest word-at-a-time
/// multiply-and-rotate rather than a keyed hash: over a quarter-gigabyte tree the default hasher
/// was a measurable share of the scan.
fn digest(bytes: &[u8]) -> u64 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let mut h: u64 = bytes.len() as u64;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]]);
        h = (h.rotate_left(5) ^ word).wrapping_mul(K);
    }
    for &b in chunks.remainder() {
        h = (h.rotate_left(5) ^ u64::from(b)).wrapping_mul(K);
    }
    h
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
    fn kinds(&self, opts: &ScanOpts) -> &'static [&'static KindSpec] {
        if opts.wants("lenses") {
            KINDS_WITH_LENSES
        } else {
            KINDS
        }
    }
    /// The `use` lens counts a name's mentions across the whole set, so in lens mode this frontend
    /// takes every file — and deduplicates by content itself, see [`Python::scan`].
    fn scans_across_files(&self, opts: &ScanOpts) -> bool {
        opts.wants("lenses")
    }
    fn scan(&self, files: &[Arc<str>], opts: &ScanOpts) -> Vec<Def> {
        if !opts.wants("lenses") {
            return files
                .par_iter()
                .flat_map(|f| {
                    fs::read_to_string(&**f).map_or_else(|_| Vec::new(), |src| scan_source(&src, f, opts))
                })
                .collect();
        }
        // 🔴 In lens mode the engine hands over EVERY file, twins included: the `use` lens counts
        // how many places mention a name across the whole set, which is a fact about the tree and
        // not about any one file, so the engine cannot thin the list for it. That does not mean
        // every file has to be parsed. A twin is the same bytes again, and the same bytes produce
        // the same definitions and the same use sites; so the content is read once, parsed once,
        // its definitions replayed onto each twin's path, and its use sites counted once per copy.
        // On a real monorepo that is one parse in six.
        let sources: Vec<Option<String>> = files.par_iter().map(|f| fs::read_to_string(&**f).ok()).collect();
        let digests: Vec<u64> = sources
            .par_iter()
            .map(|src| digest(src.as_deref().unwrap_or("").as_bytes()))
            .collect();
        // The digest narrows it; the bytes decide it. Which copy is parsed does not vary between
        // runs: the first in the (sorted) list. A file that cannot be read is its own group, and the
        // scan skips it as it always did.
        let mut by_digest: std::collections::HashMap<u64, Vec<usize>> = std::collections::HashMap::new();
        let mut parse: Vec<usize> = Vec::new();
        let mut copies_of: Vec<Vec<usize>> = vec![Vec::new(); files.len()];
        for (i, src) in sources.iter().enumerate() {
            let Some(content) = src else {
                parse.push(i);
                continue;
            };
            let seen = by_digest.entry(digests[i]).or_default();
            if let Some(&source) = seen.iter().find(|&&r| sources[r].as_deref() == Some(content.as_str())) {
                copies_of[source].push(i);
            } else {
                seen.push(i);
                parse.push(i);
            }
        }
        let parsed: Vec<Vec<Def>> = parse
            .par_iter()
            .map(|&i| sources[i].as_ref().map_or_else(Vec::new, |src| scan_source(src, &files[i], opts)))
            .collect();
        // Definitions in file order for the parsed files, then each twin's copies. The order matters
        // to exactly one consumer — `merge_use_facts` attaches a name's use facts to the FIRST
        // definition bearing it — and a twin's source always precedes the twin, so the first bearer
        // is a parsed file's definition in either order.
        let copies: Vec<Def> = parse
            .par_iter()
            .zip(&parsed)
            .flat_map_iter(|(&i, found)| {
                copies_of[i].iter().flat_map(move |&twin| {
                    found.iter().map(move |def| {
                        let mut copy = def.clone();
                        copy.file = Arc::clone(&files[twin]);
                        copy
                    })
                })
            })
            .collect();
        let mut defs: Vec<Def> = parsed.into_iter().flatten().collect();
        defs.extend(copies);
        // Phase two: the body scan above supplies the anchors (every top-level name it found), so
        // the profile pass needs no resolver — only a second walk of the same files. The `use` lens
        // sees a definition from outside, so its facts exist only once every file has been read;
        // scoring is corpus-relative for the same reason.
        let distinct: Vec<(&str, usize)> =
            parse.iter().filter_map(|&i| sources[i].as_deref().map(|src| (src, 1 + copies_of[i].len()))).collect();
        let facts = crate::uses::use_facts(&distinct, &defs);
        dup_defs_core::lens::merge_use_facts(&mut defs, facts);
        dup_defs_core::lens::score_lens_defs(&mut defs);
        defs
    }
}

#[cfg(test)]
mod tests {
    use super::Python;
    use dup_defs_core::{Frontend, ScanOpts};

    #[test]
    fn registry_metadata() {
        let py = Python;
        assert_eq!(py.lang(), "py");
        assert_eq!(py.extensions(), &["py"]);
        // Default run: the opt-in profile kind is absent, so no empty section is advertised.
        assert_eq!(py.kinds(&ScanOpts::default()).len(), 5);
        assert!(py.kinds(&ScanOpts::default()).iter().all(|k| k.id != "interfaces"), "interfaces is TS-only");
    }
}
