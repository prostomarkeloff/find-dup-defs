# find-dup-defs

[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/find-dup-defs.svg)](https://crates.io/crates/find-dup-defs)
[![exact difflib](https://img.shields.io/badge/similarity-byte--for--byte%20difflib-blue.svg)](https://crates.io/crates/difflib-fast)

Your coding agent is stateless, and your codebase doesn't fit in its context window. So when it
writes a new function, it can't see that you already wrote that helper three modules over — it
writes the copy. Over a year of AI-assisted commits, duplication stops being an accident and
becomes the default.

find-dup-defs is the gate that catches it. It clusters duplicate and near-duplicate *definitions*
— functions, methods, classes, constants, `type` aliases, TS interfaces, Rust traits — across
Python, TypeScript and Rust; grades each cluster by how much a refactor would actually pay off;
and calibrates its own noise filters to your tree. One parse per file, three frontends (Ruff, oxc,
syn), and **2–12× faster than PMD CPD and jscpd** while doing more semantic work than either.

```bash
cargo install find-dup-defs
```

or grab a prebuilt binary from the Releases page.

## Why

[GitClear's 2025 report](https://www.gitclear.com/ai_assistant_code_quality_2025_research)
measured 211M changed lines: copy-pasted lines grew from 8.3% to 12.3% of all changes between 2021
and 2024, while refactored lines fell from 25% to under 10%. For the first time on record,
copy/paste exceeded reuse.

That isn't a coincidence, it's a mechanism. A human who half-remembers writing something greps for
it. An agent can't — it holds a few thousand lines of your repo at once, your `_helpers.py` isn't
among them, and emitting a fresh copy is locally the path of least resistance. Every copy is
individually reasonable; the aggregate is a codebase that says the same thing five ways. A linter
won't flag it, because each copy is valid code. You need something that looks *across* files at the
definitions themselves.

## How to?

Start with calibration. It never gates anything — it reads your tree and reports back:

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

Three things come out: a histogram of how refactor-worthy your duplication is; threshold
suggestions at the 50th/75th/90th percentile, each with a real code sample at the cut so you see
what you'd be gating on; and inferred directives — ready-to-paste `-D` strings for the noise it
found in *your* tree, each with its rationale and blast radius. Twenty-one clusters living entirely
under `tests/`? It hands you the de-escalation rule for exactly that.

Then commit the suggestions you agree with and gate CI on the rest:

```bash
find-dup-defs ./src --error-thickness 0.5 -D @find-dup-defs.directives --errors-only
```

And, opt-in, surface the duplication that should become a helper rather than just being deleted:

```bash
find-dup-defs ./src --patternology
```

Nothing is filtered until a directive says so. Calibration suggests; the committed file decides.

## What it finds, and why not just CPD

Three passes, all from the same single parse per file.

| Pass | Catches | How |
|---|---|---|
| **name-gated** | same-named copies | defs sharing a `(kind, name)` clustered by exact Ratcliff–Obershelp similarity on the alpha-renamed canonical (via [`difflib-fast`](https://crates.io/crates/difflib-fast)) |
| **cross-name** | renamed copy-paste | the alpha-renamed canonical bucketed; ≥2 distinct names across ≥2 files |
| **Type-3** (ECScan) | renamed *and* edited copies | IDF-weighted cosine over name-agnostic lines, as an exact all-pairs cosine join — catches what byte-identity misses |

The thing token-based clone detectors (jscpd, PMD CPD) structurally can't do is the middle two
rows. They match token streams; rename the variables or edit a line and the match is gone.
find-dup-defs clusters on an **alpha-renamed AST canonical** — every bound local rewritten to
`_v0, _v1, …`, the def's own name blanked to `_fn` — so a function and its renamed-and-edited twin
collapse to the same shape. The Type-3 pass goes further still: it builds IDF-weighted per-line
vectors and runs them through [`difflib-fast`'s](https://crates.io/crates/difflib-fast) `simjoin`,
an exact L2AP weighted-cosine join (every pair with `cos ≥ θ`, no LSH approximation, asserted
bit-identical to brute force), then single-linkages the survivors.

So the answer to "why not CPD" isn't one feature, it's the stack: we cluster by *meaning* not
tokens, we **calibrate the noise ourselves**, we **rank by refactor-payoff** instead of dumping a
flat list — and we do all of that **2–12× faster** than CPD while doing strictly more work per
finding. ([Performance](#performance) has the numbers.)

Method receivers (`self`, `cls`, `&self`) are stripped, so a method matches the equivalent free
function. And the shapes that *look* like duplication but aren't never form clusters in the first
place:

- **Python / TS** — `@overload` / `@abstractmethod` / Protocol stubs (`...` / `pass` / docstring
  bodies), `raise NotImplementedError`, dispatch overrides that just `return None / False / 0 /
  self`, and `@property` setter/deleter accessors (suffixed so a getter never matches its setter).
- **Rust** — one-line `write!` / `writeln!` `Display`/`Debug` impls, `matches!` predicates,
  `todo!` / `unimplemented!` / `panic!` / `unreachable!` stubs; and `#[cfg(...)]`-gated same-name
  siblings (`#[cfg(unix)] fn x` + `#[cfg(windows)] fn x`) collapse to one logical item.

Each surviving cluster lands in a tier: ERROR gates CI, WARNING is for review, INFO is hidden
unless you ask (`--show-info`, or `--json` where it's always present). `--only py,ts,rs` scopes a
run to specific frontends.

## Thickness

What moves a cluster between tiers is its thickness — a normalized [0, 1] estimate of how much
deleting the duplication would pay. It's the number you sort by, and it's exactly this:

```
T = 0.7 · sat(volume, 30) + 0.1 · sat(args, 5) + 0.2 · sim       sat(x, k) = 1 − exp(−x/k)
volume = (n_members − 1) · loc        # lines a refactor would actually delete
```

Volume dominates on purpose — a 60-line function copied four times outranks a 3-line one copied
six, whatever the similarity scores say. Wide signatures and higher similarity nudge it up. Three
flags move the cut: `--error-thickness` demotes thin ERRORs to WARNING, `--warning-thickness`
demotes thin WARNINGs to INFO, and `--escalate-thickness` forces anything thick enough up to ERROR
(applied last, so it overrides the demotions). Each defaults to `0.0` — off — until calibration
tells you a number. Sort by T and the biggest refactor is on top.

## Calibration & directives

The tool is meant to tune itself once, then be gated by an explicit, committed config — never by
hidden heuristics.

`--calibrate` prints the thickness histogram, three percentile-anchored threshold suggestions
(permissive / balanced / strict at p50 / p75 / p90, each with a concrete code sample at the cut),
and **inferred directives**: ready-to-paste `-D` strings for the noise patterns it found in your
tree. It only fires a suggestion when the evidence clears a floor:

| Detected pattern | Floor | Suggested directive |
|---|---|---|
| clusters entirely in test dirs | ≥3 | `de-escalate:*@*/{test,tests,__tests__,fixtures,integration,e2e}/*` |
| clusters in `.test.*` / `.spec.*` files | ≥3 | `de-escalate:*@*.{test,spec}.*` |
| generated code (`*_pb2*`, `*_grpc*`, `*.gen.*`) | ≥3 | `suppress:*@*_pb2*` |
| schema migrations | ≥3 | `suppress:*@*migrations/*` |
| `.d.ts` declaration files | ≥3 | `suppress:*@*.d.ts` |
| i18n / locale / translation dirs | ≥5 | `suppress:*@*/{locale,locales,i18n,translations}/*` |
| doc / tutorial / example snippets | ≥5 | `de-escalate:*@*/{examples,tutorial,samples}/*` |
| Storybook stories | ≥5 | `de-escalate:*@*.stories.*` |
| vendored / fork snapshot roots | ≥30 | `suppress:*@*<prefix>*` (auto-derived, marker-gated) |
| `(kind,name)` group > 256 members | — | `settings:max-name-group=256` |
| patternology candidates present | ≥8 | `settings:pattern-min-thickness=<p75>` |

The vendored detector is marker-gated: it only fires on directories carrying a real vendoring
signal (`/vendor/`, `/third_party/`, `/util/vs/`, `/fixtures/`, …). Same-name files across dirs
*without* a marker stay visible — that's genuine cross-layer duplication, not vendoring.

The rule language is [`directiva`](https://crates.io/crates/directiva), one rule per line:

```
ACTION : [<KIND>] NAME [@PATH] [=NOTE]
```

`suppress` drops a finding, `de-escalate` / `escalate` move it one tier (stepped and clamped),
`note` annotates without touching severity, and `set` carries pipeline config
(`set:max-name-group=256`, `set:gpu=on`, `set:pattern-min-thickness=0.5`). The note travels with
the rule, so the *why* is still there when someone reads the file a year later:

```bash
-D 'de-escalate:<methods>Plugin.get_*_hook=intentional plugin no-op API'
-D 'suppress:<functions>spawn@*lib-rt/*=bootstrap copy, cannot import'
-D 'escalate:<methods>Lock.*@*/storage/*=must share impl before v1.0'

# keep them in a committed file and point CI at it (one per line; # comments; @- reads stdin)
-D @find-dup-defs.directives
```

Globs support `{a,b,c}` alternation, so one paste covers a whole convention family.

## Patternology

The passes above answer "are these two definitions the same?". Patternology answers the next
question: "this shape that recurs across seven functions — should it be one helper?" It's the same
engine carried one step further — same alpha-renamed canonical forms, same `Finding` / severity /
directive pipeline — not a separate tool bolted on. It's opt-in (`--patternology`) and advisory:
WARNING for a tight family, INFO otherwise, **never an ERROR gate**. A refactor map, not a CI
failure.

```console
$ find-dup-defs ./crates --only rs --patternology     # the tool on its own code
--- helper candidates in functions (patternology — collapsible duplication) ---
DUPLICATE FUNCTION [WARNING]: analyze_impl_fn/analyze_item_fn  [ast sim 1.00, n=2, loc=3, args=1]
  # helper: fn _fn(_v0: &?) -> AnalyzedFn { analyze(&_v0.sig.ident.to_string(), &_v0.sig, &_v0.block) }
  #         (1 param); collapses 2 sites, ~3 loc saved
```

### The mechanism

A family of instances is folded by **Plotkin anti-unification** (least general generalization) into
a template with holes `?` at the points where the instances diverge. Folding aligns same-tagged
nodes by their common prefix and lists by longest-common-subsequence, so it's robust to arity
divergence — `[A, B, C]` against `[A, C]` generalizes to `[A, ?, C]`, not to a single hole. It's
also async-insensitive: the fold strips the `Async` tag, so an `async def` and its sync twin
anti-unify cleanly (the botocore ↔ aiobotocore mirror case).

Then the template has to *survive*, and most don't. A candidate is kept only if its holes are
**bindable expression parameters** — things you could actually pass to a function. The filters,
with their real defaults:

- **no statement-holes.** A divergence in statement position can't be passed as an argument — you
  can't hand a function a missing `if`. Rejected.
- **no selector-holes.** A varying *method or attribute or keyword name* — `obj.?()`, `?=val` —
  would need `getattr` / `**{name: v}` reflection to parameterize. A helper that needs reflection
  isn't a helper, so it's rejected rather than surfaced.
- **a shared-anchor floor** (≥2). The instances must share real identifiers or literals, not just
  tree shape. This kills pure-structure coincidences like `? = ?; ? = ?` — two assignments that
  have nothing to do with each other.
- **a substantial fixed skeleton** (≥6 shared nodes), **a manageable arity** (≤6 expression-holes
  → parameters), and **a skeleton that dominates the variation** (fixed / (fixed + holes) ≥ 0.5).

What's left is a motif that genuinely collapses into one clean, reflection-free helper. The
proposed body is rendered as readable pseudo-source (`def …:` for Python, `fn …` for Rust, the
matching shape for TS), and the finding carries its parameter count and an estimated LOC saved.

### Two granularities

- **whole-function** — families that share an entire shape, found by structural tf·idf cosine over
  node-type q-grams and a **greedy maximal-clique cover** (not connected components, which would
  single-linkage a whole dense neighborhood into one blob).
- **sub-block** — a recurring statement-window idiom *embedded* inside otherwise-different
  functions, mined by **support** — how many functions contain it — not pairwise similarity, which
  is the case whole-function cosine structurally cannot reach. A fetch-one idiom shared across
  seven unrelated repository methods comes out as
  `? = await _v0.execute(?); return ?.scalar_one_or_none()` (3 params).

### Codometry

Every candidate carries a **stable signature key**: the fixed skeleton with holes as `?` and atoms
verbatim, rendered deterministically. The same idiom in different files — or different *packages* —
produces the same key. So an external loop turns patternology into a measurement instrument:

```bash
for pkg in $(ls ~/.cargo/registry/src/*/); do
  find-dup-defs "$pkg" --patternology --json
done | jq -s 'map(.groups[] | select(.pattern)) | group_by(.pattern.signature)'
```

Group by signature across an ecosystem and you get **codometry** — which idioms recur where, at
what support, weighted by the LOC each collapse would save. Nobody else can produce that number,
because nobody else carries a cross-package-stable structural key on each finding.

The dialect seam is a `Dialect` trait — slot classification plus a pseudo-source renderer — with
`PyDialect`, `RustDialect` and `TsDialect` behind it. A run partitions defs by language and folds
each group with its own dialect; Python, TypeScript and Rust functions never anti-unify against
each other.

Knobs: `--pattern-theta` (whole-fn cosine floor, default 0.85), `--pattern-support` (sub-block
support floor, default 3), and `-D settings:pattern-min-thickness=<F>` to drop the thin two-site
tail (`--calibrate` suggests the value).

## Performance

This is the part the tool is fastest at being smug about. `hyperfine --warmup 1 --runs 3`, macOS
arm64, against [jscpd@4](https://github.com/kucherenko/jscpd) and [PMD CPD 7.24](https://pmd.github.io/),
both in Python mode on the same trees:

| repo (Python files) | find-dup-defs | PMD CPD | jscpd |
|---|---|---|---|
| `pip` (633) | 0.18 s | 0.87 s (4.9×) | 3.21 s (18.2×) |
| `mypy` (155) | 0.18 s | 0.81 s (4.6×) | 1.47 s (8.4×) |
| `sympy` (1 589) | 1.22 s | 4.29 s (3.5×) | 15.18 s (12.4×) |
| `django` (2 910) | 1.01 s | 2.08 s (2.1×) | 9.67 s (9.6×) |

It does more semantic work than either — alpha-renamed canonicals, an exact IDF cosine join,
severity grading, calibration — and is still 3–12× faster, because it's Rust + rayon over
single-parse frontends with no JVM or Node startup to amortize. Throughput on `django` (426K SLOC)
is ~422K SLOC/s, against PMD's ~205K and jscpd's ~44K.

<details>
<summary>GPU acceleration (optional, macOS / Metal) — and why it rarely matters</summary>

[`difflib-fast`](https://crates.io/crates/difflib-fast) can offload the name-gated Ratcliff–Obershelp
clustering to the Apple-Silicon GPU via its `Rationer` handle. It's off by default and gated twice:
build with `--features gpu`, enable with `-D 'settings:gpu=on'` (`on` / `gpu+cpu` / `gpu` / `off`).
Only large all-ASCII same-name groups (≥ ~300 members) route to Metal; everything else stays on
CPU, and the output is byte-for-byte identical in every mode.

In practice it rarely helps end-to-end. The GPU accelerates clustering of a *single* large group
(1.1–1.4× in `difflib-fast`'s own bench), but this tool's real workload is many mostly-small
groups. On `rustc/tests/ui` (20 425 files, with `fn main` × 12 678): `gpu=off` 33.97 s,
`gpu=on` 33.62 s. A tie. Keep CPU for everyday runs.
</details>

## On real repos

Ten production TypeScript repos (vscode, the TS compiler, vue, angular, svelte, nest, astro,
prisma, next.js, excalidraw; ≈6M SLOC), with `--calibrate`, the inferred directives, and the
balanced thickness cut — raw ERROR count drops 94% on average:

| repo | LOC | raw ERROR | after | %cut | top remaining cluster |
|---|---:|---:|---:|---:|---|
| microsoft/vscode | 3.1M | 5428 | 174 | 97% | `registerCLIChatCommands` 771 LOC |
| microsoft/TypeScript | 265k | 1840 | 9 | 100% | `NavigationBarItem` interface |
| vercel/next.js | 756k | 489 | 26 | 95% | `defaultLoader` 115 LOC |
| angular/angular | 1.0M | 627 | 54 | 91% | `conditionalCreate/conditionalBranchCreate` |
| prisma/prisma | 222k | 322 | 68 | 79% | `fieldToColumnType` 95 LOC × 3 adapters |

Twenty-eight large Python repos (≈8M SLOC), auto-applied directives, 67% average cut:

| repo | raw ERROR | after | %cut | top remaining cluster |
|---|---:|---:|---:|---|
| home-assistant/core | 4475 | 850 | 81% | `ConfigFlow.async_step_*` (n=178) |
| apache/airflow | 2203 | 337 | 84% | `CloudComposerGetEnvironmentOperator` (n=18) |
| django/django | 559 | 71 | 87% | `TupleGreaterThan.get_fallback_sql` (n=4) |
| scipy/scipy | 492 | 140 | 71% | `dct/dst/idct/idst` (n=4) |
| pandas-dev/pandas | 406 | 78 | 80% | `read_csv/read_table` (n=2) |

What's left at the top is the kind of thing a human reviewer would also flag. `pip`'s Version
`__lt__…__gt__` ×6 collapse into one `_compare` helper, −130 lines. `scipy`'s `dct/dst/idct/idst`
×4 want a factory, −330 lines. `scikit-learn`'s `BaseSGD{Classifier,Regressor}._fit` is a
sibling-estimator dupe waiting for a shared impl. The vendored snapshots, test fixtures, `.d.ts`
and Storybook noise is gone before you read a line.

## For agents

The JSON output is built so an agent never has to round-trip to the filesystem. Each finding ships
the full source of one member (`groups[].snippet`), every location (`members[]` as file:line), the
thickness for prioritization, the kind/severity/similarity, and any directive annotations
(`notes[]`). Pattern findings additionally carry a structured `pattern` object — `template`,
`signature`, `params`, `granularity`, `support`, `loc_saved` — so a consumer groups by signature
without parsing prose.

```bash
# calibrate → JSON, then scan with the chosen tuning + inferred directives
find-dup-defs ./repo --calibrate --json > calib.json
find-dup-defs ./repo \
  --error-thickness <calib> \
  $(jq -r '.inferred_directives[].directive | "-D \"" + . + "\""' calib.json) \
  --errors-only --json > findings.json
```

## Architecture

Six crates, layered so the engine never depends on a frontend and the contract crate stays pure:

```
              dup-defs-core            ← the contract: Def / KindSpec / Analysis / CanonDialect /
                  ▲                       the Frontend trait / LineMap.  No deps.
        ┌─────────┴─────────┐
   find-dup-defs-canon         find-dup-defs   ← find-dup-defs-canon: shared frontend helpers (alpha-rename, the
        ▲                  (engine+CLI)   KindSpec vocabulary, count_loc, AnalyzedFn).
   ┌────┼────┐               │           find-dup-defs: the 3 passes + patternology + severity +
 py-   rs-   ts-canon ───────┘           directives + calibration + reports.
 canon canon            (engine depends on the contract + each frontend, NOT on find-dup-defs-canon)
```

[`find-dup-defs`](crates/find-dup-defs) is the engine and CLI; it clusters a `Vec<Def>` and never
names a language. [`dup-defs-core`](crates/dup-defs-core) is the engine↔frontend contract — `Def`,
`KindSpec`, `Analysis`, the `Frontend` trait. [`find-dup-defs-canon`](crates/find-dup-defs-canon)
holds the helpers the frontends share (the alpha-rename, the kind vocabulary, `count_loc`).
[`py-canon`](crates/py-canon), [`ts-canon`](crates/ts-canon) and [`rs-canon`](crates/rs-canon) are
the frontends (Ruff, oxc, syn). Adding a language is one more `<lang>-canon` crate implementing
`Frontend` — plus a `Dialect` impl if it wants patternology — and no engine changes.

The similarity engine underneath is [`difflib-fast`](https://github.com/prostomarkeloff/difflib-fast),
an exact Ratcliff–Obershelp + L2AP cosine-join port. And the tool eats its own cooking: this
workspace gates to **0 ERROR** under `find-dup-defs crates -D @find-dup-defs.directives`. (The file
`crates/find-dup-defs/src/simgraph.rs` exists because an earlier run flagged the cosine/union-find
helpers that `type3` and `patternology` had each copied — so they were extracted into one module.)

## CLI reference

```
USAGE:  find-dup-defs [OPTIONS] <PATHS>...

LANGUAGES
  --only <CODES>            Restrict to frontends (py,ts,rs). Default: all found in PATHS.
  --kinds <K,…>             functions,methods,classes,interfaces,constants,type-aliases

SEVERITY (thickness ladder)
  --error-thickness <F>     Demote ERROR → WARNING if T < F   (default 0.0 = off)
  --warning-thickness <F>   Demote WARNING → INFO  if T < F   (default 0.0 = off)
  --escalate-thickness <F>  Promote anything → ERROR if T ≥ F (default 0.0 = off, applied last)

SIMILARITY
  -t, --threshold <F>       Name-gated cluster floor   (default 0.5)
  -e, --error-threshold <F> Name-gated ERROR floor     (default 0.85)
  --type3-theta <F>         Type-3 cosine floor        (default 0.7)
  --max-name-group <N>      Skip name-gated clustering for (kind,name) groups > N

PATTERNOLOGY (opt-in · advisory, never ERROR)
  --patternology            Surface collapsible-duplication helper candidates
  --pattern-theta <F>       Whole-fn structural cosine floor (default 0.85)
  --pattern-support <N>     Sub-block idiom support floor     (default 3)

FILTERS / MODES
  -D, --directive <S>       ACTION:[<KIND>]NAME[@PATH][=NOTE], repeatable. ACTION ∈
                            suppress / de-escalate / escalate / note / set:KEY=VALUE.
                            `@PATH` reads a directive file (# comments; @- = stdin).
  --min-size <N>            Only clusters with ≥ N members (default 2)
  --errors-only             Filter output to ERROR
  --show-info               Include INFO in the human report
  --calibrate               Histogram + threshold suggestions + inferred directives
  --json                    Machine-readable output
  --no-cross-name / --no-type3   Skip pass 2 / pass 3
```

## Limitations

The honest ledger:

- Python, TypeScript and Rust today; patternology covers all three. A new language is a
  `<lang>-canon` sibling crate.
- Rust patternology is the youngest of the three: `rs-canon` splices statement bodies as node
  children rather than lists, so long-body alignment is prefix-only, and macro internals are
  opaque.
- TypeScript patternology sees top-level `function` declarations and arrow / function-expression
  `const`s. Class methods don't participate — their slice doesn't re-parse as a standalone
  function, so they carry no patternology canonical. The duplicate passes still cover them.
- Type-4 clones (same logic, different syntax) are out of scope.
- Token-level sub-expression duplication is out of scope too; pair with jscpd or PMD CPD if you
  need it.
- The thickness constants were tuned on the benchmark corpora above. Your codebase may want
  different ones — that's what `--calibrate` is for.

---

<div align="center">

**Copy-paste has nowhere left to hide.**

Made with ⚡ by [@prostomarkeloff](https://github.com/prostomarkeloff)

</div>
