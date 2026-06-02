<div align="center">

# find-dup-defs

**Find the copy-pasted code your linter can't — rank it by how worth refactoring it is — and surface the recurring shapes that should become one helper.**

[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/find-dup-defs.svg)](https://crates.io/crates/find-dup-defs)
[![exact difflib](https://img.shields.io/badge/similarity-byte--for--byte%20difflib-blue.svg)](https://crates.io/crates/difflib-fast)

A duplicate-definition detector for **Python, TypeScript, and Rust**. It clusters duplicate &
near-duplicate top-level definitions — functions, **methods**, classes, constants, `type` aliases,
TS interfaces / Rust traits — by structural AST canonicalization, grades each cluster
**ERROR / WARNING / INFO** by a normalized *Thickness* score, and **suggests your project's noise
filters for you**. Opt into **patternology** and it also finds *collapsible* duplication: the
recurring structure that should become one parameterized helper.

**One engine, three single-parse frontends** (Ruff · oxc · syn). **2–12× faster than PMD CPD / jscpd.**

</div>

---

## Why

[GitClear's 2025 report](https://www.gitclear.com/ai_assistant_code_quality_2025_research) (211M LOC):
**copy-pasted lines grew 8.3% → 12.3%** of all changes 2021→2024, while *refactored* lines **dropped
from 25% to under 10%** — for the first time, copy/paste exceeded reuse. AI assistants don't know your
project's `_helper.py`; they emit the copy. `find-dup-defs` is the gate that catches it, and the
calibration that tells you which copies are actually worth a PR.

---

## Install

```bash
cargo install find-dup-defs
# …or a prebuilt binary from the Releases page.
```

---

## Quickstart

```bash
# 1. Calibrate — histogram of refactor-worthiness + ready-to-paste noise filters
find-dup-defs ./src --calibrate

# 2. Gate CI on the actionable tail only
find-dup-defs ./src --error-thickness 0.5 -D @find-dup-defs.directives --errors-only

# 3. (opt-in) Surface helper-extraction candidates
find-dup-defs ./src --patternology
```

`--calibrate` is the intended first step: it never gates, it just reports.

```console
$ find-dup-defs ./src --calibrate
=== thickness calibration (ERROR): 76 clusters analyzed ===
  T [0.2, 0.3)  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 25
  T [0.3, 0.4)  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 27
  T [0.4, 0.5)  ▇▇▇▇▇▇▇▇▇ 8
  …
suggested thresholds (p50/p75/p90):
  balanced   --error-thickness 0.34  →  21 ERROR remain  (median dup: 14 loc, 2 args)

=== inferred directives (auto-detected noise patterns) ===
  → -D 'de-escalate:*@*/{test,tests,__tests__}/*=test parametrize/fixture candidates'
    rationale: 21 clusters live entirely in test paths
    affects: 21 total (10 ERROR, 11 WARNING, 0 INFO)
```

---

## What it detects

Two complementary things, from one parse per file:

### 1 · Duplicate definitions — the gate

Three passes find *duplicated definitions* and grade them by severity:

| Pass | Catches | How |
|---|---|---|
| **name-gated** | same-named copies | same `(kind, name)` defs clustered by exact Ratcliff–Obershelp similarity on the alpha-renamed canonical |
| **cross-name** | renamed copy-paste | alpha-renamed canonical bucketed; ≥2 distinct names across ≥2 sites |
| **Type-3** | edited renamed copies | IDF-weighted cosine over name-agnostic lines — catches what the exact pass misses |

Each cluster is graded **ERROR / WARNING / INFO** by its *Thickness* (see below) and is a directive
target you can suppress / de-escalate / annotate.

### 2 · Patternology — collapsible duplication (opt-in, `--patternology`)

The passes above find duplicate *functions*. Patternology finds the **recurring structure that should
become one helper** — and surfaces a motif *only if* it actually collapses into a clean,
reflection-free helper (see [Patternology](#patternology) for the mechanism). Advisory only — never an
ERROR gate; it's a refactor map, not a CI failure.

```console
$ find-dup-defs ./crates --only rs --patternology     # the tool on its own code
--- helper candidates in functions (patternology — collapsible duplication) ---
DUPLICATE FUNCTION [WARNING]: analyze_impl_fn/analyze_item_fn  [ast sim 1.00, n=2, loc=3, args=1]
  # helper: fn _fn(_v0: &?) -> AnalyzedFn { analyze(&_v0.sig.ident.to_string(), &_v0.sig, &_v0.block) }
  #         (1 param); collapses 2 sites, ~3 loc saved
```

The proposed body, parameter count, and LOC saved ride along on the finding (the real output is one
line; wrapped here). A Python run renders `def …:` pseudo-source the same way.

---

## Languages

| Frontend | Parser | Duplicate passes | Patternology |
|---|---|:---:|:---:|
| **Python** (`.py`) | Ruff (PEP 695 / 701) | ✅ | ✅ |
| **TypeScript** (`.ts` `.tsx` `.mts` `.cts`) | oxc (TS 5.x / JSX / decorators) | ✅ | — *(no dialect yet)* |
| **Rust** (`.rs`) | syn (full item grammar) | ✅ | ✅ |

`--only py,ts,rs` scopes a run to specific frontends. Each is a single parse per file; method
receivers (`self` / `cls` / `&self`) are stripped so a method matches an equivalent free function.

**Built-in noise filtering** at extraction (these never form phantom clusters):
- *Python / TS* — `@overload` / `@abstractmethod` / Protocol stubs, `raise NotImplementedError`,
  `return False / None / 0` dispatch overrides, `@property` + setter/deleter.
- *Rust* — one-line `write!`/`writeln!` `Display`/`Debug` impls, `matches!` predicates, `todo!`/`panic!`
  stubs; `#[cfg(...)]`-gated same-name siblings collapse to one logical item.

---

## Severity & Thickness

```
ERROR  ←→  WARNING  ←→  INFO
  gate       review     hidden by default (JSON-only; --show-info to display)
```

Three knobs move clusters between tiers:

- `--error-thickness X` — ERROR → WARNING when T < X
- `--warning-thickness X` — WARNING → INFO when T < X
- `--escalate-thickness X` — anything with T ≥ X is forced to ERROR

**Thickness** is a normalized [0, 1] "get-me-refactored" score — the single number you sort by:

```
T = 0.7 · sat(volume, 30) + 0.1 · sat(args, 5) + 0.2 · sim       sat(x, k) = 1 − exp(−x/k)
volume = (n_members − 1) · loc        # lines you'd actually delete (dominant signal)
```

Wide signatures read as architecturally chunkier; higher-similarity dups score higher. Sort by T →
biggest refactor wins first.

---

## Calibration & directives

`find-dup-defs` is meant to tune itself, then be gated by an explicit, committed config — never by
hidden heuristics.

### `--calibrate`

Prints a thickness histogram, three percentile-anchored threshold suggestions (`permissive` /
`balanced` / `strict`, each with a concrete code sample at the cut), and **inferred directives** —
ready-to-paste `-D` strings for the noise patterns it found in *your* tree:

| Detected pattern | Suggested directive |
|---|---|
| ≥3 clusters entirely in test paths | `de-escalate:*@*/{test,tests,__tests__,fixtures,integration,e2e}/*` |
| ≥3 in `.test.*` / `.spec.*` files | `de-escalate:*@*.{test,spec}.*` |
| ≥5 in i18n / locale / translation dirs | `suppress:*@*/{locale,locales,i18n,translations}/*` |
| ≥3 touching `*_pb2*` / `*_grpc*` | `suppress:*@*_pb2*` |
| ≥3 under `*/migrations/*` | `suppress:*@*migrations/*` |
| ≥5 in `.d.ts` / `*.stories.*` | `suppress:*@*.d.ts` · `de-escalate:*@*.stories.*` |
| vendored snapshot roots (`/util/vs/`, `/vendor/`, …) | `suppress:*@*<prefix>*` (auto-derived, marker-gated) |
| `(kind,name)` group > 256 members (entry-point names) | `settings:max-name-group=256` |
| **patternology candidates present** | **`settings:pattern-min-thickness=<p75>`** — drops the thin 2-site tail |

Globs support `{a,b,c}` alternation, so one paste covers a whole convention family. The vendored
detector is marker-gated — same-name files across dirs *without* a recognized vendored marker are
treated as real cross-layer duplication, not auto-suppressed.

### Directive language ([`directiva`](https://crates.io/crates/directiva))

```
ACTION : [<KIND>] NAME [@PATH] [=NOTE]
```

| Action | Effect |
|---|---|
| `suppress` | drop the finding entirely |
| `de-escalate` / `escalate` | one tier down / up |
| `note` | annotate, no severity change |
| `set` | pipeline config (`set:max-name-group=256`, `set:gpu=on`, `set:pattern-min-thickness=0.5`) |

```bash
# Intentional, per-repo:
-D 'de-escalate:<methods>Plugin.get_*_hook=intentional plugin no-op API'
-D 'suppress:<functions>spawn@*lib-rt/*=bootstrap copy, cannot import'
-D 'escalate:<methods>Lock.*@*/storage/*=must share impl before v1.0'

# Keep them in a committed file and point CI at it (one per line; `#` comments; `@-` reads stdin):
-D @find-dup-defs.directives
```

Nothing is filtered until you paste a directive — calibration *suggests*, you *decide*.

---

## Patternology

> Opt-in via `--patternology`. Advisory: WARNING for a tight family, INFO otherwise — **never ERROR.**

A duplicate-function pass asks "are these two functions the same?". Patternology asks "does this
recurring *shape* collapse into one parameterized helper?" — and surfaces a motif only when the answer
is yes. It folds the instances by **Plotkin anti-unification** (least general generalization, made
robust to arity divergence) into a template with holes `?` at the variation points, then keeps it only
if the holes are **bindable expression parameters** — not leaky statement divergences and not
name-identity *selectors* (a varying method/kwarg name would need reflection, so `obj.?()` is rejected,
not surfaced as a "helper"). Pure-structure coincidences (`? = ?; ? = ?`) are dropped by a
shared-anchor floor.

Two granularities:

- **whole-function** — families that share a shape, clustered by structural tf·idf cosine with a
  **greedy maximal-clique cover** (no single-linkage blob), collapsed into one helper.
- **sub-block** — a recurring *statement-window* idiom **embedded** inside otherwise-different
  functions, found by **support** (how many functions contain it), not pairwise similarity — the case
  whole-function cosine structurally cannot reach. E.g. a fetch-one idiom shared by seven unrelated
  repository methods → `? = await _v0.execute(?); return ?.scalar_one_or_none()` (3 params).

**Language-agnostic engine.** The mechanism lives behind a `Dialect` trait; adding a language is a
trait impl (slot classification + a pseudo-source renderer), the engine core is untouched. Ships with
`PyDialect` (CPython `ast.dump`) and `RustDialect` (`rs-canon`). A run **partitions defs by language**
and folds each group with its own dialect — Python and Rust functions never anti-unify against each
other.

Each finding carries the proposed helper body (rendered as readable pseudo-source), its parameter
count, an estimated LOC saved, and a **stable signature key** (holes `?`, atoms verbatim): the same
idiom in different files/packages yields the same key, so an external loop
(`for pkg in …: find-dup-defs --patternology --json pkg`) + a glue script grouping on the signature
gives ecosystem-wide **codometry**.

Knobs: `--pattern-theta` (whole-fn cosine floor, default 0.85), `--pattern-support` (sub-block support
floor, default 3), and `-D settings:pattern-min-thickness=<F>` (drop candidates below a thickness floor
— calibrated by `--calibrate`).

---

## Performance

`hyperfine --warmup 1 --runs 3`, macOS arm64, vs [jscpd@4](https://github.com/kucherenko/jscpd) and
[PMD CPD 7.24](https://pmd.github.io/) (both Python-mode, same tree):

| Repo (Python files) | find-dup-defs | PMD CPD | jscpd |
|---|---|---|---|
| `pip` (633) | **0.18 s** | 0.87 s (4.9×) | 3.21 s (18.2×) |
| `mypy` (155) | **0.18 s** | 0.81 s (4.6×) | 1.47 s (8.4×) |
| `sympy` (1 589) | **1.22 s** | 4.29 s (3.5×) | 15.18 s (12.4×) |
| `django` (2 910) | **1.01 s** | 2.08 s (2.1×) | 9.67 s (9.6×) |

It does **more** semantic work (alpha-renamed canonical, IDF cosine, severity grading, calibration)
and is still **3–12× faster** — Rust + rayon, single-parse frontends, no JVM/Node tax. Throughput on
`django` (426K SLOC): **~422K SLOC/s** vs PMD ~205K, jscpd ~44K.

<details>
<summary><b>GPU acceleration (optional, macOS / Metal)</b></summary>

`difflib-fast` can offload the name-gated Ratcliff–Obershelp clustering to the Apple-Silicon GPU via
its stateful `Rationer` handle — wired in but **off by default** and gated twice: build with
`--features gpu`, enable at runtime with `-D 'settings:gpu=on'` (`on` / `gpu+cpu` / `gpu` / `off`).
Only large all-ASCII same-name groups (≥ ~300 members) route to Metal; everything else stays on CPU,
and output is **byte-for-byte identical** in every mode.

**Honest read — it rarely helps end-to-end.** The GPU only accelerates clustering of a *single* large
group (`difflib-fast`'s own bench: 1.1–1.4× there), while the tool's real shape is *many* mostly-small
groups (the 0.6–0.99× case). On `rustc/tests/ui` (20 425 files, with `functions:main ×12 678`):
`gpu=off` 33.97 s vs `gpu=on` 33.62 s — a tie. Keep CPU for everyday runs.
</details>

---

## Benchmarks — real repos

**10 production TypeScript repos** (vscode, the TS compiler, vue, angular, svelte, nest, astro,
prisma, next.js, excalidraw; ≈6M SLOC). `--calibrate` + auto-inferred directives + balanced thickness
cut raw ERROR count by **94% on average**:

| Repo | LOC | Raw ERROR | After | %cut | Top remaining cluster |
|---|---:|---:|---:|---:|---|
| microsoft/vscode | 3.1M | 5428 | 174 | **97%** | `registerCLIChatCommands` 771 LOC |
| microsoft/TypeScript | 265k | 1840 | 9 | **100%** | `NavigationBarItem` interface |
| vercel/next.js | 756k | 489 | 26 | **95%** | `defaultLoader` 115 LOC |
| angular/angular | 1.0M | 627 | 54 | **91%** | `conditionalCreate/conditionalBranchCreate` |
| prisma/prisma | 222k | 322 | 68 | **79%** | `fieldToColumnType` 95 LOC × 3 adapters |

**28 large Python repos** (≈8M SLOC). Auto-applied directives cut raw ERROR by **67% on average**:

| Repo | Raw ERROR | After | %cut | Top remaining cluster |
|---|---:|---:|---:|---|
| home-assistant/core | 4475 | 850 | 81% | `ConfigFlow.async_step_*` (n=178) |
| apache/airflow | 2203 | 337 | 84% | `CloudComposerGetEnvironmentOperator` (n=18) |
| django/django | 559 | 71 | 87% | `TupleGreaterThan.get_fallback_sql` (n=4) |
| scipy/scipy | 492 | 140 | 71% | `dct/dst/idct/idst` (n=4) |
| pandas-dev/pandas | 406 | 78 | 80% | `read_csv/read_table` (n=2) |

Concrete wins this surfaced: `pip` Version `__lt__…__gt__` ×6 → one `_compare` helper (−130 lines);
`scipy` `dct/dst/idct/idst` ×4 → a factory (−330 lines); `scikit-learn`
`BaseSGD{Classifier,Regressor}._fit` — a textbook sibling-estimator dupe.

The top remaining clusters are PR candidates a human reviewer would also flag, with the noise
(vendored snapshots, test fixtures, `.d.ts`, Storybook) automatically removed.

---

## AI-agent integration

```bash
# 1. Calibrate → JSON
find-dup-defs ./repo --calibrate --json > calib.json

# 2. Full scan with the agent's chosen tuning + inferred directives
find-dup-defs ./repo \
  --error-thickness <calib> \
  $(jq -r '.inferred_directives[].directive | "-D \"" + . + "\""' calib.json) \
  --errors-only --json > findings.json

# 3. Each finding ships everything to write a PR — no FS roundtrips:
#    groups[].snippet (full source of one member) · members[] (every file:line)
#    · thickness (priority) · notes[] (directive annotations)
```

---

## Architecture

Six crates, layered so the engine never depends on a frontend and the contract crate stays pure:

```
              dup-defs-core            ← the contract: Def / KindSpec / Analysis / CanonDialect /
                  ▲                       the Frontend trait / LineMap.  No deps.
        ┌─────────┴─────────┐
   canon-core          find-dup-defs   ← canon-core: shared frontend helpers (alpha-rename, the
        ▲                  (engine+CLI)   KindSpec vocabulary, count_loc, AnalyzedFn).
   ┌────┼────┐               │           find-dup-defs: the 3 passes + patternology + severity +
 py-   rs-   ts-canon ───────┘           directives + calibration + reports.
 canon canon            (engine depends on the contract + each frontend, NOT on canon-core)
```

| crate | role |
|---|---|
| [`find-dup-defs`](crates/find-dup-defs) | engine + CLI; frontend-agnostic, clusters a `Vec<Def>` and never names a language |
| [`dup-defs-core`](crates/dup-defs-core) | the engine↔frontend **contract** (`Def` / `KindSpec` / `Analysis` / `Frontend` / `LineMap`) |
| [`canon-core`](crates/canon-core) | **shared frontend helpers** between the contract and the frontends — no duplication across the three |
| [`py-canon`](crates/py-canon) · [`ts-canon`](crates/ts-canon) · [`rs-canon`](crates/rs-canon) | the Python / TypeScript / Rust frontends (Ruff · oxc · syn) |

Adding a language = one more `<lang>-canon` frontend implementing `Frontend` (and, for patternology, a
`Dialect` impl) — no engine changes. The similarity engine is the exact Ratcliff–Obershelp + simjoin
port [`difflib-fast`](https://github.com/prostomarkeloff/difflib-fast). Dogfooded on its own source to
**0 ERROR** (`find-dup-defs crates -D @find-dup-defs.directives`).

---

## CLI reference

```
USAGE:  find-dup-defs [OPTIONS] <PATHS>...

LANGUAGES
  --only <CODES>            Restrict to frontends (py,ts,rs). Default: all found in PATHS.

SEVERITY (thickness ladder)
  --error-thickness <F>     Demote ERROR → WARNING if T < F   (default 0.0 = off)
  --warning-thickness <F>   Demote WARNING → INFO  if T < F   (default 0.0 = off)
  --escalate-thickness <F>  Promote anything → ERROR if T ≥ F (default 0.0 = off)

SIMILARITY
  -t, --threshold <F>       Name-gated cluster floor (default 0.5)
  -e, --error-threshold <F> Name-gated ERROR floor   (default 0.85)
  --type3-theta <F>         Type-3 cosine floor       (default 0.7)

PATTERNOLOGY (opt-in · Python + Rust · advisory, never ERROR)
  --patternology            Surface collapsible-duplication helper candidates
  --pattern-theta <F>       Whole-fn structural cosine floor (default 0.85)
  --pattern-support <N>     Sub-block idiom support floor     (default 3)
                            (drop the thin tail with -D settings:pattern-min-thickness=<F>)

FILTERS
  -D, --directive <S>       ACTION:[<KIND>]NAME[@PATH][=NOTE], repeatable. ACTION ∈
                            suppress / de-escalate / escalate / note / set:KEY=VALUE.
                            `@PATH` reads a directive file (`#` comments; `@-` = stdin).
                            Globs: * ? {a,b} [a-z] (+ \ escapes).
  --kinds <K,…>             functions,methods,classes,interfaces,constants,type-aliases
  --min-size <N>            Only clusters with ≥ N members (default 2)
  --max-name-group <N>      Skip name-gated clustering for (kind,name) groups > N
  --errors-only             Filter output to ERROR
  --show-info               Include INFO in the human report

MODES
  --calibrate               Histogram + threshold suggestions + inferred directives
  --json                    Machine-readable output
  --no-cross-name / --no-type3   Skip pass 2 / pass 3
```

---

## Limitations

- **Python / TypeScript / Rust** today; patternology covers **Python + Rust** (a TS dialect is a
  follow-up). New languages are a `<lang>-canon` sibling — PRs welcome.
- Rust patternology is **initial**: `rs-canon` splices statement bodies as node children rather than
  lists, so long-body alignment is prefix-only, and macro internals are opaque.
- **Type-4** (semantic equivalence, different syntax → same logic) — out of scope.
- **Token-level** sub-expression duplication — out of scope; pair with jscpd / PMD CPD if you need it.
- Calibration is heuristic — the thickness formula constants were tuned on the benchmark corpora above;
  your codebase may want different.

---

<div align="center">

**Copy-paste has nowhere to hide.**

Made with ⚡ by [@prostomarkeloff](https://github.com/prostomarkeloff)

</div>
