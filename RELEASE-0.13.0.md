## Converge 5× faster, lens runs 2×, every byte the same

**Converge went from 3.9 s to 0.8 s** on a 371k-definition tree and from 1.25 s to 0.37 s on a
mixed standard-library tree. **`--kinds lenses` went from 16.2 s to 7.7 s** on the same tree. Ten
output modes were diffed against a 0.12.0 binary on both trees — the 278 MB uncapped converge
ranking included — and nothing moved.

## The finding

0.12.0's profile of converge said "nothing dominates": suffix automata, matching, tokenizing, all
a few percent each. That was the profile *by function*. The profile *by thread* said the main
thread held 16% of the whole run's CPU, three times any worker — and the pass was 12% of CPU but
55% of wall. It was not compute-bound. It was **glue, on one thread**: a `&str`-keyed statement
index, a set deduplicating 7.9 million seeds, a hash map of 1.85 million pairs, a string hash and
a `ln` for every rarity looked up, and 385 000 findings built and sorted so that the cap could keep
a hundred.

## What converge does now

- **Bodies, not definitions.** 219k definitions spell 31k bodies; every table derived from text
  is built once per body. A definition is two ids and a subject set.
- **Ids everywhere.** Statement keys and shapes interned in lexicographic order (the seed walk
  and the shape sums are string orders), names and lines by hash; a rarity is an array read.
- **Sort, then decide per pair.** Seeds packed as one word and one index, sorted in parallel, the
  dedup-and-longest-run rule applied inside each pair's run in the original order.
- **One anti-unification per distinct fork**, keyed on the gap lines, over tokens already lexed;
  only the term is kept, and the names a printed fork parts by are recomputed for the survivors.
- **A shortlist before any finding exists**: rank by thickness, keep what can still be in the
  head of each kind, build those, sort on the full key, cap.

## What the lens run does now

- **Twins are counted, not walked.** The `use` lens counts a name's mentions across the whole
  file set, so the engine deliberately hands the Python frontend every file. The frontend now
  parses one file per distinct content (28 763 → 4 796), replays the definitions onto each copy,
  and gives the use-site walk each distinct source with its multiplicity — a site is a multiset
  entry. A new defaulted `Frontend::scans_across_files` lets the engine keep deduplicating for
  the other frontends and re-take their corpus-relative score over the replayed set.
- **79% of the characters never needed a suffix automaton.** The exact clusterer builds one per
  string before it filters a single pair. Its own two upper bounds — length and character
  multiset — are now applied first, and a string with no possible partner is left out; it was a
  singleton either way. Name-gated clustering of lens records: 4.0 s → 1.5 s.
- Smaller, all exact: parallel lens scoring; parallel interning and document frequency in the
  type-3 join, with the line lists borrowed instead of copied; the `scope` lens computes only the
  one rendering it reads; module-scope renaming consults two sets instead of cloning their union
  per function; the report resolves real paths per directory rather than per file.

## Where the floor is

After the prefilter, the isolated name-gated pass over lens records is entirely inside
`difflib-fast`: automata 30%, matching 35%, per-cluster minimum 25%. Nothing more comes out
exactly without changing that crate.

## Versions

| crate | | why |
|---|---|---|
| `find-dup-defs` | 0.12.0 → **0.13.0** | `type3::type3_clusters` now takes borrowed line lists; breaking for a library consumer, invisible to a CLI user |
| `dup-defs-core` | 0.2.1 → **0.2.2** | `Frontend::scans_across_files` (defaulted); `score_lens_defs` runs on the pool |
| `py-canon` | 0.4.0 → **0.4.1** | content deduplication and use-site counting in lens mode |

`rs-canon` and `ts-canon` are untouched and not republished — 0.2.0 remains current for both.

## Install

```console
uv tool install find-dup-defs        # or: pipx install find-dup-defs
cargo install find-dup-defs
```

Prebuilt binaries for linux (gnu/musl, x86_64/aarch64), macOS (arm64/x86_64) and Windows are
attached below.
