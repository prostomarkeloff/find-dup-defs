<div align="center">

# find-dup-defs

**Find the copy-pasted code your linter can't — and tell you which copies to refactor first.**

[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/find-dup-defs.svg)](https://crates.io/crates/find-dup-defs)
[![exact difflib](https://img.shields.io/badge/similarity-byte--for--byte%20difflib-blue.svg)](https://crates.io/crates/difflib-fast)

Duplicate & near-duplicate definitions — functions, **methods**, classes, constants, `type` aliases,
**TypeScript interfaces / Rust traits** — clustered by structural AST canonicalization, ranked by a
normalized **Thickness** score, graded ERROR / WARNING / INFO, with **auto-suggested project-specific
noise filters** out of the box.

**Python, TypeScript, and Rust today** — `--only py,ts,rs` to scope per language. One engine, three
single-parse native frontends (Ruff for Python, oxc for TypeScript, syn for Rust).

**2-12× faster than PMD CPD / jscpd. Calibrates itself on first run.**

</div>

---

## Why now

[GitClear's 2025 report](https://www.gitclear.com/ai_assistant_code_quality_2025_research) (211M lines
of code analyzed): **copy-pasted lines grew from 8.3% to 12.3%** of all changes 2021→2024, while
refactored lines **dropped from 25% to under 10%**. For the first time in measurable history,
copy/paste exceeded code reuse. AI assistants don't know your project's `_helper.py` — they emit the
copy.

`find-dup-defs` is the gate.

---

## What it does

```console
$ find-dup-defs ./src --calibrate
=== thickness calibration (ERROR): 76 clusters analyzed ===

distribution (each ▇ ≈ one ERROR cluster, scaled to fit):
  T [0.0, 0.1)
  T [0.1, 0.2)  ▇▇▇▇ 2
  T [0.2, 0.3)  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 25
  T [0.3, 0.4)  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 27
  T [0.4, 0.5)  ▇▇▇▇▇▇▇▇▇ 8
  T [0.5, 0.6)  ▇▇▇▇▇▇▇▇▇ 8
  T [0.6, 0.7)  ▇▇▇▇▇ 4
  T [0.7, 0.8)  ▇▇▇ 2

suggested thresholds (p50/p75/p90):

  permissive   --error-thickness 0.27  →  38 ERROR remain  (median dup: 8 loc, 2 args)
    e.g. ASTConverter.visit_Try/ASTConverter.visit_TryStar  [T=0.27, loc=16, args=2]
         mypy/fastparse.py:1367, :1384
    ┌──
    │ def visit_Try(self, n: ast3.Try) -> TryStmt:
    │     vs = [self.set_line(NameExpr(h.name), h) if h.name is not None else None
    │           for h in n.handlers]
    │     types = [self.visit(h.type) for h in n.handlers]
    │     handlers = [self.as_required_block(h.body) for h in n.handlers]
    │     node = TryStmt(...)
    │     return self.set_line(node, n)
    └──

=== inferred directives (auto-detected noise patterns) ===

  → -D 'de-escalate:*:*@*tests/*=test parametrize/fixture candidates'
    rationale: 21 clusters live entirely in test paths
    affects: 21 total (10 ERROR, 11 WARNING, 0 INFO)
```

One calibrate call gives you:
- Histogram of finding "thickness" (refactor urgency)
- Three suggested thresholds (`permissive` / `balanced` / `strict`) with **a code sample at each level**
- Auto-detected noise patterns as ready-to-paste `-D` directives

Then your CI:
```bash
find-dup-defs ./src \
  --error-thickness 0.5 --warning-thickness 0.4 --escalate-thickness 0.55 \
  -D 'de-escalate:*:*@*tests/*=test fixtures' \
  --errors-only --json
```

…exits non-zero only on actionable refactor candidates.

---

## Install

```bash
cargo install find-dup-defs
```

…or grab a prebuilt binary from the [Releases page](https://github.com/prostomarkeloff/find-dup-defs/releases/latest).

---

## The three detection passes

Every `.py` / `.ts` / `.tsx` / `.mts` / `.cts` / `.rs` file is parsed **once** (Ruff for Python — PEP
695 / 701 ready; oxc for TypeScript — TS 5.x, JSX/TSX, decorators; syn for Rust — full item grammar),
each callable yielded as top-level functions **and methods** (`Foo.bar` / `Type::method`):

1. **name-gated** — same-`(kind, name)` defs clustered by exact Ratcliff–Obershelp similarity on the `ast.dump`-shape canonical.
2. **cross-name** — renamed copy-paste: alpha-renamed canonical bucketed, ≥2 distinct names across ≥2 sites.
3. **Type-3** (ECScan) — IDF-weighted cosine over name-agnostic lines; catches edited renamed copies the exact pass misses.

### Smart filters (no false-positives from these patterns)

- `@overload` / `@abstractmethod` / Protocol stubs — bodies of `...` / `pass` / docstring filtered at extraction
- `raise NotImplementedError` ABC declarations
- `return False / None / 0 / "x"` dispatch overrides (huge cross-name FP source — gone)
- `@property` + `@x.setter` / `.deleter` — accessor role baked into the name
- `self` / `cls` receivers — stripped so methods can match equivalent free functions

---

## Severity model

Three tiers, controllable via three thresholds:

```
ERROR  ←→  WARNING  ←→  INFO
  ↑           ↑          ↑
gate       review     hidden by default (JSON-only, --show-info to display)
```

- `--error-thickness X` — ERROR demotes to WARNING if T < X
- `--warning-thickness X` — WARNING demotes to INFO if T < X
- `--escalate-thickness X` — any cluster with T ≥ X is forced to ERROR (catches fat multi-copy patterns the name-gated heuristics demote by sim alone)

### Thickness

A normalized [0, 1] "GET ME REFACTORED" score combining:
- **Dedup volume** = `(n_members - 1) × loc` — how many lines you'd actually delete (dominant signal)
- **Args** — wide signatures register as architecturally chunkier
- **Similarity** — higher confidence dups score higher

```
T = 0.7 · sat(volume, 30) + 0.1 · sat(args, 5) + 0.2 · sim
sat(x, k) = 1 − exp(−x/k)
```

Sort findings by T → biggest refactor wins first.

---

## Directives

User-authored overrides for repo-specific intentional duplication.

```
ACTION : [KIND:] NAME [@PATH] [=NOTE]
```

| Action | Effect | Severity |
|---|---|---|
| `suppress` | Drop entirely | gone |
| `de-escalate` | One step down | ERROR→WARNING→INFO |
| `escalate` | One step up | INFO→WARNING→ERROR |
| `note` | Annotate only | unchanged |

```bash
# Plugin no-op API: intentional, don't gate
-D 'de-escalate:METHOD:Plugin.get_*_hook=intentional plugin no-op API'

# Bootstrap copy that can't be deduplicated
-D 'suppress:FUNCTION:spawn@*mypyc/lib-rt/*=bootstrap copy: lib-rt cannot import from mypyc'

# Architectural blocker — escalate to ERROR even if thickness is mid-range
-D 'escalate:METHOD:Lock.*@*/storage/*=Lock/LockExtend must share impl before v1.0'

# Just leave a note
-D 'note:METHOD:For*.begin_body=v2 refactor target (see issue #42)'
```

Notes show up inline:
```
DUPLICATE METHOD [ERROR]: Lock.hold/LockExtend.hold  [normalized-exact, T=0.67, n=2, loc=28]
  # Lock/LockExtend must share impl before v1.0
```

---

## Auto-inferred directives

The `--calibrate` step pattern-matches across findings and surfaces ready-to-paste directives for repeating noise patterns. No manual config; just paste suggestions you agree with into CI.

| Pattern | Suggestion |
|---|---|
| ≥5 CONSTANT clusters where ≥80% members are in `*/locale*` | `suppress:CONSTANT:*@*locale*` |
| ≥3 clusters where all members live in test paths | `de-escalate:*:*@*/{test,tests,__tests__,test_cases,test-cases,__fixtures__,fixtures,integration,e2e}/*` |
| ≥3 clusters where all members are `.test.*` / `.spec.*` (jest/vitest/mocha) | `de-escalate:*:*@*.{test,spec}.*` |
| ≥5 clusters where all members are i18n / translation files | `suppress:*:*@*/{locale,locales,i18n,translations}/*` |
| ≥3 clusters touching `*_pb2*` / `*_grpc*` files | `suppress:*:*@*_pb2*` |
| ≥3 clusters all under `*/migrations/*` / `*/alembic/versions/*` | `suppress:*:*@*migrations/*` |
| ≥5 clusters all under `*/docs_src/*` / `*/examples/*` / `*/tutorial/*` | `de-escalate:*:*@*/{docs_src,examples,tutorial,samples}/*` |
| ≥5 clusters all in `.d.ts` declaration files | `suppress:*:*@*.d.ts` |
| ≥5 clusters all in `*.stories.*` Storybook files | `de-escalate:*:*@*.stories.*` |
| ≥30 clusters with same-name files across a vendored marker dir (`/util/vs/`, `/fixtures/`, `/vendor/`, `/third_party/`) and a parallel source root | `suppress:*:*@*<vendored-prefix>*` (per-snapshot, auto-derived) |

Directive globs support `{a,b,c}` brace alternation so one paste covers every convention of a
family (`*/{test,tests,__tests__}/*` is one directive, not three).

The vendored-pattern detector is **gated by a marker score** on the longer path — without a
recognized vendored marker (test/fixtures/vendor/util/vs), same-name files across different
dirs are treated as architectural duplication (real refactor candidates) rather than auto-
suppressed. Prevents over-suppression of legitimate cross-layer reuse patterns (e.g. FSD's
`pages/foo` ↔ `shared/foo`).

Verified on real benchmarks: **67% noise reduction** across 28 large Python repos and
**~94% average reduction** across 10 production TypeScript repos (≈6M SLOC total).

---

## Performance

`hyperfine --warmup 1 --runs 3` on macOS arm64 (M-series), against [`jscpd@4`](https://github.com/kucherenko/jscpd) and [PMD CPD 7.24](https://pmd.github.io/) — both Python-mode, same target tree:

| Repo (Python files)  | find-dup-defs | PMD CPD     | jscpd        |
|----------------------|---------------|-------------|--------------|
| `pip` (633)          | **0.18 s**    | 0.87 s (4.9×) | 3.21 s (18.2×) |
| `mypy/mypy` (155)    | **0.18 s**    | 0.81 s (4.6×) | 1.47 s (8.4×)  |
| `sympy` (1 589)      | **1.22 s**    | 4.29 s (3.5×) | 15.18 s (12.4×)|
| `django` (2 910)     | **1.01 s**    | 2.08 s (2.1×) | 9.67 s (9.6×)  |

PMD ran with `--minimum-tokens=100`; jscpd with defaults (min-lines=5, min-tokens=50). `find-dup-defs` does **more semantic work** (alpha-renamed canonical, IDF cosine, severity grading, calibration) and is still **3-12× faster** end-to-end — Rust + rayon-parallel extraction, single-parse Ruff frontend, no JVM/Node startup tax.

### Throughput

On `django` (426K SLOC, 2 910 files):
- `find-dup-defs`: **~422K SLOC/sec**
- PMD CPD: ~205K SLOC/sec
- jscpd: ~44K SLOC/sec

---

## Real benchmark — 10 production TypeScript repos

Across vscode, the TypeScript compiler itself, vue, angular, svelte, nest, astro, prisma, next.js,
excalidraw (≈6M SLOC total), `--calibrate` + auto-inferred directives + balanced thickness cut
raw ERROR count by **94% on average**:

| Repo               | LOC   | Raw ERROR | After auto-directives | %cut | Top remaining cluster |
|--------------------|------:|----------:|----------------------:|-----:|-----------------------|
| microsoft/vscode   | 3.1M  | 5428      | 174                   | **97%** | `registerCLIChatCommands` 771 LOC (copilotCLIChatSessions ↔ copilotCLIChatSessionsContribution) |
| microsoft/TypeScript | 265k| 1840      | 9                     | **100%** | `NavigationBarItem` interface (protocol.ts ↔ services/types.ts) |
| vercel/next.js     | 756k  | 489       | 26                    | **95%** | `defaultLoader` 115 LOC (legacy/image.tsx ↔ shared/lib/image-loader.ts) |
| nestjs/nest        | 112k  | 172       | 10                    | **94%** | `callOperator` × 5 lifecycle hooks |
| withastro/astro    | 223k  | 135       | 9                     | **93%** | `getFrontmatterLanguagePlugin` 50 LOC (language-server ↔ ts-plugin) |
| angular/angular    | 1.0M  | 627       | 54                    | **91%** | `templateBase/conditionalCreate/conditionalBranchCreate` cross-name |
| prisma/prisma      | 222k  | 322       | 68                    | **79%** | `fieldToColumnType` 95 LOC × 3 (adapter-neon / -pg / -ppg) |
| vuejs/core         | 151k  | 8         | 1                     | **88%** | (clean codebase) |
| excalidraw         | 170k  | 23        | 5                     | **78%** | `intersectDiamondWithLineSegment/intersectRectanguloidWithLineSegment` |
| sveltejs/svelte    | 17k   | 3         | 0                     | **100%** | (clean) |

Top remaining clusters on these corpora are textbook PR candidates — the tool is finding the
architectural duplication a human reviewer would also flag, with the noise (vendored snapshots,
test fixtures, declaration files, Storybook stories) automatically removed.

## Real benchmark — 28 large Python repos

Across 14 repos with ≥150K SLOC + 14 with 50K-150K each (≈8M SLOC total), `find-dup-defs --calibrate` auto-applied directives reduce raw ERROR count by **67% on average**:

| Repo               | Raw ERROR | After CI flags + auto-inferred | %cut | Top remaining cluster |
|--------------------|-----------|--------------------------------|------|-----------------------|
| django/django      | 559       | 71                             | 87%  | `TupleGreaterThan.get_fallback_sql` (n=4 SQL ops) |
| wagtail/wagtail    | 496       | 65                             | 86%  | `set_privacy` (n=2) |
| apache/airflow     | 2203      | 337                            | 84%  | `CloudComposerGetEnvironmentOperator` (n=18) |
| home-assistant/core| 4475      | 850                            | 81%  | `ConfigFlow.async_step_*` (n=178) |
| pandas-dev/pandas  | 406       | 78                             | 80%  | `read_csv/read_table` (n=2) |
| scipy/scipy        | 492       | 140                            | 71%  | `dct/dst/idct/idst` (n=4) |
| numpy/numpy        | 316       | 96                             | 69%  | `std/var` (n=2) |

Top findings on this corpus are textbook PR candidates:
- **`pip`** Version `__lt__/__le__/__eq__/__ge__/__gt__` × 6 — minus 130 lines via one `_compare` helper
- **`scipy`** `dct/dst/idct/idst` × 4 — minus ~330 lines via factory generator
- **`django`** `TupleGreaterThan/...` × 4 — minus ~75 lines via a `TupleLookupMixin` method
- **`scikit-learn`** `BaseSGDClassifier._fit / BaseSGDRegressor._fit` — classic dupe between sibling estimators

---

## AI-agent integration

One-shot workflow for autonomous refactor agents:

```bash
# 1. Calibrate (1-30s)
find-dup-defs ./repo --calibrate --json > calib.json

# 2. Agent reads calib.json:
#    - inferred_directives[]            → -D flags ready to paste
#    - error.suggestions[].error_thickness → starting threshold
#    - warning.suggestions[].error_thickness → WARNING threshold

# 3. Full scan with agent's chosen tuning
find-dup-defs ./repo \
  --error-thickness <calib> --warning-thickness <calib> \
  $(jq -r '.inferred_directives[].directive | "-D \"" + . + "\""' calib.json) \
  --errors-only --json > findings.json

# 4. For each finding, agent has everything to write a PR:
#    - groups[].snippet     — full source of one representative member
#    - groups[].members[]   — every duplicate location (file:line)
#    - groups[].thickness   — refactor priority
#    - groups[].notes[]     — directive-attached annotations
```

No file-system roundtrips needed — the snippet ships in JSON.

---

## What it finds — quick map

```console
$ find-dup-defs ./mypy
--- duplicate functions (cross-file, AST sim warn=0.5 error=0.85) ---
DUPLICATE FUNCTION [ERROR]: generate_hash_wrapper/generate_len_wrapper
  [normalized-exact, T=0.63, n=2, loc=24, args=3]
  mypyc/codegen/emitwrapper.py:546, :573

--- duplicate methods (cross-file, ...) ---
DUPLICATE METHOD [ERROR]: CallableType.formal_arguments/Parameters.formal_arguments
  [normalized-exact, T=0.56, n=2, loc=19, args=2]
  mypy/types.py:1992, :2341

--- duplicate methods (cross-name, exact AST-normalized) ---
DUPLICATE METHOD [ERROR]: For{Async,,Native}Iterable.begin_body
  [normalized-exact, T=0.53, n=3, loc=9, args=1]
  mypyc/irbuild/for_helpers.py:716, :778, :848

# summary: 14 ERROR, 47 WARNING groups
```

…or on TypeScript:

```console
$ find-dup-defs ./packages --only ts
--- duplicate functions (cross-file, AST sim warn=0.5 error=0.85) ---
DUPLICATE FUNCTION [ERROR]: formatDateTime
  [ast sim 1.00, T=0.85, n=5, loc=18, args=1]
  packages/adapter-mariadb/src/conversion.ts:180
  packages/adapter-mssql/src/conversion.ts:174
  packages/adapter-neon/src/conversion.ts:433
  packages/adapter-pg/src/conversion.ts:446
  packages/adapter-planetscale/src/conversion.ts:155

--- duplicate methods (cross-name, exact AST-normalized) ---
DUPLICATE FUNCTION [ERROR]: apply{ActivityBar,Explorer,Keybindings,Localization,Profiler,StatusBar}Tools
  [normalized-exact, T=0.93, n=6, loc=117, args=2]
  ...
```

…or on Rust — `find-dup-defs` dogfooding on its own crates (the three language frontends share structure):

```console
$ find-dup-defs ./crates --only rs
--- duplicate functions (cross-file, AST sim warn=0.5 error=0.85) ---
DUPLICATE FUNCTION [WARNING]: keyword_start  [ast sim 0.69, T=0.75, n=3, loc=25, args=3]
  crates/py-canon/src/defs.rs:74
  crates/rs-canon/src/defs.rs:55
  crates/ts-canon/src/defs.rs:79

--- duplicate methods (cross-name, exact AST-normalized) ---
DUPLICATE METHOD [ERROR]: Python::scan/Rust::scan/TypeScript::scan
  [normalized-exact, T=0.45, n=3, loc=6, args=1]
  crates/py-canon/src/frontend.rs:61
  crates/rs-canon/src/frontend.rs:50
  crates/ts-canon/src/frontend.rs:50
```

---

## CLI reference (essentials)

```
USAGE:
  find-dup-defs [OPTIONS] <PATHS>...

LANGUAGES:
  --only <CODES>             Restrict scan to specific frontends (comma-separated:
                             py,ts,rs). Default: every supported frontend found in PATHS.
                             Unknown codes exit non-zero.

THICKNESS LADDER:
  --error-thickness <F>      Demote ERROR → WARNING if T < this (default 0.0 = off)
  --warning-thickness <F>    Demote WARNING → INFO  if T < this (default 0.0 = off)
  --escalate-thickness <F>   Promote anything → ERROR if T ≥ this (default 0.0 = off)

SIMILARITY (name-gated):
  -t, --threshold <F>        Cluster floor (default 0.5)
  -e, --error-threshold <F>  ERROR floor   (default 0.85)
  --type3-theta <F>          Type-3 cosine threshold (default 0.7)

FILTERS:
  -D, --directive <S>        ACTION:[KIND:]NAME[@PATH][=NOTE], repeatable.
                             PATH glob supports `{a,b,c}` brace alternation.
  --kinds <K1,K2,...>        functions,methods,classes,interfaces,constants,type-aliases
  --min-size <N>             Only clusters with ≥ N members (default 2)
  --errors-only              Filter output to ERROR severity
  --show-info                Include INFO in human report (default hidden)

MODES:
  --calibrate                Print histogram + suggestions + inferred directives
  --json                     Machine-readable output
  --repo-root <PATH>         Path-prefix for short paths in report (default .)

OUTPUT:
  --no-cross-name            Skip pass 2
  --no-type3                 Skip pass 3
```

---

## Architecture

Five crates, each useful on its own:

| crate | role |
|---|---|
| [`find-dup-defs`](crates/find-dup-defs) | Engine + CLI — three passes, severity, directives, calibration, reports. Frontend-agnostic: clusters a `Vec<Def>` and never names a language crate. |
| [`dup-defs-core`](crates/dup-defs-core) | **Engine contract** — `Def` / `KindSpec` / `Analysis` / the `Frontend` trait (+ `LineMap`). Each frontend declares its own open kind vocabulary and lowers definitions to `Def` with canon precomputed. |
| [`py-canon`](crates/py-canon) | **Python frontend** — Ruff parse → `ast.dump`-shape canonical + def scan |
| [`ts-canon`](crates/ts-canon) | **TypeScript frontend** — oxc parse → s-expr canonical + def scan |
| [`rs-canon`](crates/rs-canon) | **Rust frontend** — syn parse → s-expr canonical + def scan |

Adding a new language is one more frontend crate that implements the `Frontend` trait against
`dup-defs-core` (`scan` → `Vec<Def>`, each definition's canonical strings precomputed in the
single parse) — no engine changes. The similarity engine is the exact Ratcliff–Obershelp port
[`difflib-fast`](https://github.com/prostomarkeloff/difflib-fast).

---

## Limitations

- **Python, TypeScript, Rust** today (frontend crate per language; PRs welcome — Go / Java / C# would each be a `<lang>-canon` sibling)
- **Type 4 (semantic equivalence, different syntax → same logic)** — not done; neural-network research territory
- **Token-level fine-grained duplication** (a 30-token sub-expression copy-pasted around) — out of scope; use jscpd / PMD CPD alongside if you need that
- **Calibration is heuristic** — formula constants (loc=20, args=5, weight 0.7/0.1/0.2) were tuned on the 28-Python-repo + 10-TS-repo benchmarks; your codebase may want different

---

<div align="center">

**Copy-paste has nowhere to hide.**

Made with ⚡ by [@prostomarkeloff](https://github.com/prostomarkeloff)

</div>
