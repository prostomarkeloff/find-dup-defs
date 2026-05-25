<div align="center">

# find-dup-defs

**Find the copy-pasted code your linter can't — across the whole repo, even after a rename.**

[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/find-dup-defs.svg)](https://crates.io/crates/find-dup-defs)
[![exact difflib](https://img.shields.io/badge/similarity-byte--for--byte%20difflib-blue.svg)](https://crates.io/crates/difflib-fast)

Duplicate & near-duplicate top-level definitions — functions, classes, constants, `type` aliases —
clustered by the **exact `difflib` similarity ratio**. Renamed copies included. **9,500 files in under
8 seconds.**

</div>

---

Copy-paste hides from most linters: they look *within* a file, and they fold the moment you rename a
variable. `find-dup-defs` reads the **whole codebase**, reduces every top-level definition to a
structural canonical, and clusters the ones that are really the same — including renamed and lightly
edited copies.

```console
$ find-dup-defs mypy --kinds functions

[ERROR] functions 'is_generic/is_generic_instance' (cross-name) — 2 definitions:
    mypy/stats.py:475:1
    mypy/types_utils.py:120:1

[WARNING] functions '_profile_type_check/perform_type_check' (type-3) — 2 definitions, min similarity 0.829:
    misc/log_trace_check.py:26:1
    misc/profile_check.py:48:1

scanned 437 files, 2201 top-level defs → 46 clusters (13 ERROR, 33 WARNING)
```

`(cross-name)` = renamed-identical · `(type-3)` = renamed near-copy · no tag = same-name body cluster.
Exit code is non-zero when any **ERROR** cluster is found — drop it straight into CI.

---

## Install

```bash
cargo install find-dup-defs
```

…or grab a prebuilt binary for your platform from the
[Releases page](https://github.com/prostomarkeloff/find-dup-defs/releases/latest) — no Rust needed.

```bash
find-dup-defs path/to/project --kinds functions,classes   # what real copy-paste usually lives in
find-dup-defs path/to/project --errors-only --json        # gate + machine-readable
```

---

## What it finds — three passes

Every `.py` file is parsed **once** (Ruff parser, modern syntax — PEP 695 / 701), yielding each
top-level definition and its canonical forms. Three complementary passes then run over them:

1. **name-gated** — same-`(kind, name)` defs clustered by exact Ratcliff–Obershelp similarity on the
   `ast.dump`-shape canonical (names kept, docstrings stripped). ERROR when a cluster's min pairwise
   ratio ≥ `--error-threshold`, else WARNING.
2. **cross-name** — *renamed copy-paste*: functions reduced to an **alpha-renamed** canonical (own name
   + locals → positional placeholders) and bucketed; ≥2 distinct names across ≥2 files is a duplicate
   the name gate is blind to. ERROR when the canonical is substantial.
3. **Type-3 (`ECScan`)** — *renamed near-copies*: IDF-weighted cosine over each function's
   name-agnostic lines (rare-shingle candidates → cosine verify → union-find), catching edited renamed
   copies the exact pass misses.

Pass 1's similarity is **byte-for-byte** `difflib.SequenceMatcher(autojunk=False).ratio()` via
[`difflib-fast`](https://crates.io/crates/difflib-fast); clustering is single-linkage. Passes 2–3 are
the renamed duplication the same-name gate can't see — the part that actually matters on real code.

---

## Speed

End-to-end (walk + parse + canonicalize + cluster), cold, Apple M3 Pro:

| repo | files | top-level defs | duplicate clusters | time |
|---|---|---|---|---|
| mypy | 437 | 3,926 | 45 | **0.07 s** |
| django | 2,910 | 10,971 | 215 | **0.48 s** |
| sympy | 1,589 | 20,368 | 522 | **0.34 s** |
| transformers | 4,403 | 23,757 | 2,327 | **1.6 s** |
| Home Assistant | 9,498 | 62,101 | 2,600 | **7.7 s** |

---

## Built as a workspace

Two crates, each useful on its own:

| crate | role |
|---|---|
| [`find-dup-defs`](crates/find-dup-defs) | the CLI — the three passes, human + JSON output, CI exit code |
| [`py-canon`](crates/py-canon) | the **Python frontend** — Ruff parse → `ast.dump`-shape canonical + def scan |

The name is **language-agnostic on purpose**: Python is the current frontend; other languages can be
added later as further frontend crates feeding the same CLI. The similarity engine is the exact
Ratcliff–Obershelp library [`difflib-fast`](https://github.com/prostomarkeloff/difflib-fast).

---

<div align="center">

**Copy-paste has nowhere to hide.**

Made with ⚡ by [@prostomarkeloff](https://github.com/prostomarkeloff)

</div>
