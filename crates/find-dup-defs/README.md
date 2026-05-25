# find-dup-defs

Find **duplicate / near-duplicate top-level definitions** across a Python codebase — functions,
classes, `UPPER_CASE` constants, and `type` aliases that have been copy-pasted (and maybe lightly
edited) across files.

```
find-dup-defs path/to/project [--threshold 0.8] [--kinds functions,classes] [--min-size 2] [--json]
```

## How it works

Every `.py` file is parsed **once** ([`py-canon`](../py-canon), Ruff parser, PEP 695 / 701), yielding
each top-level definition plus its canonical forms. Three complementary passes then run:

1. **name-gated** — same-`(kind, name)` functions/classes are clustered by exact
   Ratcliff–Obershelp body similarity (≥ `--threshold`) on the `ast.dump`-shape canonical (names
   preserved, docstrings stripped); a cluster gates **ERROR** when its min pairwise ratio ≥
   `--error-threshold`, else **WARNING**. Same-named constants / type-aliases are flagged by name
   alone (ERROR).
2. **cross-name** — *renamed copy-paste*: functions reduced to an **alpha-renamed** canonical (own
   name + locals → positional placeholders) and bucketed; a bucket with **≥2 distinct names across ≥2
   files** is a duplicate the name-gated pass can't see. ERROR when the canonical is substantial
   (≥ 20 AST nodes), else WARNING.
3. **Type-3 (`ECScan`)** — *renamed near-copies*: IDF-weighted cosine over each function's
   name-agnostic lines (rare-shingle candidate generation + cosine verify + union-find), catching
   edited renamed copies the exact cross-name pass misses. ERROR at min-cosine ≥ 0.9, else WARNING.
   (`--type3-theta` sets the detection floor.)

Similarity in pass 1 is **byte-for-byte** `difflib.SequenceMatcher` ratio (via
[`difflib-fast`](https://crates.io/crates/difflib-fast)); clustering is single-linkage. Passes 2–3 catch the *renamed*
duplication the same-name gate is blind to. Exit code is non-zero if any ERROR cluster is found
(CI-friendly). Use `--no-cross-name` / `--no-type3` to disable, `--errors-only` to gate only.

## Example

```
$ find-dup-defs mypy --kinds functions

[ERROR] functions 'is_generic/is_generic_instance' (cross-name) — 2 definitions:
    mypy/stats.py:475:1
    mypy/types_utils.py:120:1

[WARNING] functions '_profile_type_check/perform_type_check' (type-3) — 2 definitions, min similarity 0.829:
    misc/log_trace_check.py:26:1
    misc/profile_check.py:48:1

scanned 437 files, 2201 top-level defs → 46 clusters (13 ERROR, 33 WARNING)
```

`(cross-name)` = renamed-identical, `(type-3)` = renamed near-copy; no tag = same-name body cluster.

## Speed

End-to-end (walk + parse + canonicalize + cluster), cold, Apple M3 Pro:

| repo | files | top-level defs | duplicate clusters | time |
|---|---|---|---|---|
| mypy | 437 | 3,926 | 45 | 0.07 s |
| django | 2,910 | 10,971 | 215 | 0.48 s |
| sympy | 1,589 | 20,368 | 522 | 0.34 s |
| transformers | 4,403 | 23,757 | 2,327 | 1.6 s |
| Home Assistant | 9,498 | 62,101 | 2,600 | 7.7 s |

(at `--threshold 0.8`, all kinds. Lower thresholds / `--kinds functions,classes` change what surfaces;
constants like `T = TypeVar("T")` cluster trivially, so `--kinds functions,classes` is usually what you
want for real copy-paste.)

## License

MIT.
