# find-dup-defs

[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/find-dup-defs.svg)](https://crates.io/crates/find-dup-defs)
[![PyPI](https://img.shields.io/pypi/v/find-dup-defs.svg)](https://pypi.org/project/find-dup-defs/)
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

Run it without installing anything:

```bash
uvx find-dup-defs ./src
```

Prebuilt wheels for Linux, macOS and Windows are on PyPI — no Rust toolchain needed:

```bash
pip install find-dup-defs        # or: uv tool install find-dup-defs
```

Or install from crates.io (builds from source):

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

Three passes, all from the same single parse per file. Three more are opt-in and advisory —
[patternology](#patternology), [lenses](#lenses) and [converge](#converge) — and none of them ever
raises an ERROR.

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

## Lenses

Every pass above canonicalizes one text — the definition's body — and varies only how much identity
it strips. That is one axis. A **lens** varies the text instead: it projects the same definition
onto a different question and throws the rest away.

The case that motivates it: two caches, one storing a JSONB blob and one storing typed columns,
sharing no identifier anywhere. Same architecture, written twice.

```console
$ find-dup-defs ./cache_a ./cache_b
No cross-file duplicates.
```

Nothing, and the reason is exact — the model's own name is a *free* name in the body canonical, so
`session.get(JsonCache, k)` and `db.get(ThumbEntry, i)` never meet. Erase what the module itself
introduced (imports, sibling definitions, class fields — renamed in attribute position too, which
the local set never reaches) and what survives is the grammar of talking to things the module did
not define:

```console
$ find-dup-defs ./cache_a ./cache_b --kinds lenses
DUPLICATE LENSES [WARNING]: cache_get/thumb_lookup    [normalized-exact, T=0.60, n=2, loc=7]
    votes[5]: control×3 effects×1 outgoing×1 scope×1 signature×1
DUPLICATE LENSES [WARNING]: cache_put/thumb_store     [normalized-exact, T=0.65, n=2, loc=8]
    votes[5]: control×2 effects×2 outgoing×2 scope×1 signature×1
DUPLICATE LENSES [WARNING]: cache_evict/thumb_purge   [normalized-exact, T=0.73, n=2, loc=2]
    votes[4]: effects×3 outgoing×3 scope×1 signature×1
```

Ten lenses, each answering one question:

| lens | question | keeps |
|---|---|---|
| `outgoing` | what does it depend on? | the *set* of callees the module did not introduce |
| `effects` | what protocol does it drive? | the same callees in call *order* |
| `control` | how does it branch? | the if/for/while/try/return/raise skeleton, with nesting |
| `failures` | how does it fail? | raised and caught exception types |
| `resources` | what does it hold open? | context expressions of `with` blocks |
| `signature` | what contract does it offer? | arity shape and annotation names — what it *has*, never what it lacks |
| `decorators` | what role does it play? | decorator names |
| `schema` | what shape does it declare? | column types and their options, as an unordered *set* |
| `scope` | what does its body do? | the body with every name its module introduced erased |
| `use` | how is it handled? | the statements elsewhere in the tree that mention it |

Two of them needed their own treatment. **`schema`** compares declarations, where order is not
meaning and the literals *are* identities: facts are sorted as a set, and `__tablename__`, index
names and foreign-key targets are dropped while the `ForeignKey` / `Index` call survives — that a
column references something is shape, which table it references is identity. Column types stay
verbatim despite being imported: they are the grammar a schema is written in. **`use`** cannot be
computed from a definition alone, so its facts are merged in after the tree is walked; assembly is
by name, with no import resolution and no call graph — the assumption the name-gated pass has
always made.

### Agreement is the signal

All ten stitch into **one** record, each fact tagged with the lens it came from
(`control:if`, `outgoing:.commit`, `schema:col Text nullable`). The Type-3 pass's IDF-weighted
cosine over those lines then *is* the vote — nothing new had to be built. Agreeing through several
lenses raises the score, agreeing through one barely moves it, and a fact the whole corpus shares
(`control:return`) is weighted to nothing without anyone declaring it noise. A cross-name exact
match means every lens agreed at once.

Each finding reports which lenses agreed and by how many facts, because similarity alone says how
close two definitions are and never *through what*. Measured on one production tree, mean thickness
climbs with the count — 0.71 at one vote, 0.79 at three, 0.89 at five, 0.92 at six — even though the
score is computed from corpus IDF and knows nothing about votes. The two are independent estimates
of the same thing, which is the best evidence the weighting works that could be had without tuning
it to fit.

A lens is only safe if its facts are either rare (informative) or universal (IDF ≈ 0). Many facts of
*middling* frequency are the failure mode: `signature` used to emit `posonly 0` / `kwonly 0` /
`async 0` for every ordinary function — seven facts about nothing — and dominated two thirds of all
findings on that tree, collapsing thousands of unrelated definitions into one cluster. It now reports
only what a signature *has*. Worth checking for any lens you add.

### What it finds that the body passes cannot

| | |
|---|---|
| a `Timeout` and a `Delay` enricher differing in one call, fifteen lines of identical plumbing | `sim 0.98` |
| `MediaConfigResource` twice — the legacy and the authenticated media endpoint, differing in one regex | `sim 1.00` |
| five `Delete*Command` classes on one template, one of them annotated `list[Dashboard]` by copy-paste | `sim 1.00` |
| the same OAuth setup step in two self-hosted integrations, 49 lines each | `sim 1.00` |
| six `TypeGuard` predicates across three files, docstrings included | `normalized-exact` |
| `AmplitudeClient` in six places, the shared-library copy *behind* the forks that grew a feature | `sim 0.55` |

A finding carried by a single lens is weak by construction — the vote count is there to be read.

Opt in with `--kinds lenses` — Python, Rust and TypeScript. The ten questions were never
Python-specific; what was, until the machinery moved into `dup-defs-core`, is that a frontend had to
carry the vocabulary, the stitching and the corpus scoring itself. Now it contributes an AST walk
and nothing else.

Three answers were worth thinking about rather than transliterating. Rust has no `with`, and its
answer to *what does it hold open* is a **guard** — a binding whose value is never read again and
whose only job is to live until the scope ends; that is structurally detectable, so the lens finds
it without a list of blessed names. TypeScript's is `using` / `await using`, **not** `try`/`finally`
— a `finally` is a cleanup path, projected as control flow, and reading every one as a held resource
would fill the lens with the language's commonest idiom. And Rust's failures keep the whole path:
`Err(MyError::Empty)`, because the enum is the failure family and the variant the specific failure.

The kind exists exactly when asked for, so the section list, the default report and the default JSON
are byte-identical without it. Directives address it like any other kind —
`-D 'suppress:<lenses>*@*/legacy/*=deliberate parallel port'`.

## Converge

Every pass above answers *are these two the same*. This one answers *these two are about the same
thing — **where do they stop agreeing***, and reports the step rather than the cluster.

```console
$ find-dup-defs ./src --converge
--- divergences in functions (converge — one thing done twice, and where they part) ---
DUPLICATE FUNCTION [INFO]: _apply_source_rate_limit / _enforce_submit_budget
  votes[4]: text×0 shape×62 subject×6 fork×12
  A only: _v2 = utcnow()
  B only: _v1 = now(UTC)
  ~ _ = _ - _(_=_)
    A: _v3 = _v2 - timedelta(hours=1)
    B: _v2 = _v1 - timedelta(seconds=HOUR_SECONDS)
  ~ if _ > _:
    A: if _v6 > _v1.source_limit_per_hour
    B: if _v4 > landing.lead_submit_limit_per_hour
  subject: domain.rate_limit.repo (reached by 11 definitions)
```

`text×0` — those two share **no line**. One rate limiter written four times across a codebase, and
the four disagree about where "now" comes from.

### Two anchors, because one is structurally blind

A pass keyed on a **shared statement** can only see divergence that grew out of textual agreement: a
copy that drifted, or two paths that converged. It cannot reach the opposite case — two places
written independently about one thing, with no line in common. Measured, that blind spot is real: a
pair of functions answering one question ("does this channel fit the plan") shared exactly one name
between them, and that name was `int`.

What such a pair does share is a **subject**: both reach the same module. Imports are the corpus's
own declaration of what a definition is about, so the frontend resolves the dotted path each name it
uses stands for, and the engine takes prefixes of it — which is how *imported the module* and
*imported a member of it* are made to meet, instead of failing to meet as strings.

| anchor | agreement | divergence | reads as |
|---|---|---|---|
| **statement** | same words | different names | one decision made in two ways |
| **subject** | same entity, same shape | different words | one procedure written twice |

### One currency

The seed decides only what the report points at, never how a pair is weighed:

```text
score = (E_text + E_shape + E_subject) · D · sharpness · novelty / members
```

**E** is how surprising the coincidence is, in nats, over the three ways two definitions can
evidently be one thing done twice — the run they share line for line, the shapes they share among
the lines they word differently, and the rarity of the deepest module both reach. **D** is the
rarity of the rarest name they *part* on: evidence says they are the same, this says how sharp the
difference is. **novelty** is `1` when the two found each other again after the gap and `1 − jaccard`
when they parted for good — for a drifted copy alikeness is the premise, for a permanent fork it
means a similarity pass already has the pair.

**members** divides by how many places share the run, and it is the one factor measured rather than
assumed: read against real code, a divergence between exactly two places was worth acting on 74% of
the time and one among three or more 16%. Two places is the primitive that does not exist, written
out twice; many places is the primitive that does exist, with users who legitimately go on to differ.

### Families

Many places around one subject, all one shape, said **once**:

```console
--- families of functions (converge — many places, one subject, one shape) ---
DUPLICATE FUNCTION [INFO]: chart_week_plusminus / chart_week_plusminus_wave / chart_daily_unsubscribers
  votes[2]: shape×73 members×3   # 3 definitions around domain.report._chart_helpers…
  src/domain/report/tg_charts.py:173
  src/domain/report/tg_charts.py:233
  src/domain/report/tg_charts.py:298
```

A group of N definitions produces N(N−1)/2 pairs, and reporting those says one fact once per pair
while burying it. In a read of fifty findings on a real tree, four slots went to pairs among one
family of six sibling functions and three more to a set of chart builders — the finding was never
"these two are alike", it was "this family exists".

Grouped by **greedy maximal clique**, not by connected component: under single-linkage `1-2-3-4` is
one group whose ends share no edge. The clique proposes and the evidence disposes — members are
dropped while dropping raises the score, so a clique that grew past the real family shrinks back to
it, with no threshold and no notion of which member is the odd one.

### Reading it

Opt in with `--converge`; findings are **INFO** and never gate. `--converge-top` keeps the strongest
50 of each kind by default, which is the one place this differs from every other pass: they report a
*set*, where cutting off would drop findings as true as the ones kept, and this reports a *ranking*
with no threshold in it, where the tail is what the ordering exists to push away. `--converge-top 0`
prints all of them (198k lines on a mid-sized tree — the reason for the default).

Works for Python, Rust and TypeScript, and names none of them: the inputs are the statement stream
and the reach set on `Facets`, so a language lights the pass up by filling them.

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
without parsing prose. Lens findings carry `facets` — `[[lens, shared facts], …]`, strongest first —
so a consumer can rank by *how many perspectives agreed* rather than by similarity alone, or filter
to the ones a single lens carried. The field is omitted when a run produces no tagged facts, so the
default document is unchanged.

```bash
# calibrate → JSON, then scan with the chosen tuning + inferred directives
find-dup-defs ./repo --calibrate --json > calib.json
find-dup-defs ./repo \
  --error-thickness <calib> \
  $(jq -r '.inferred_directives[].directive | "-D \"" + . + "\""' calib.json) \
  --errors-only --json > findings.json
```

## Architecture

Five crates, layered so the engine never depends on a language:

```
              dup-defs-core       ← the shared ground: the Def / KindSpec / Facets / Frontend
                  ▲                 contract, the kind vocabulary, the alpha-rename, the lens
        ┌────┬────┼────┬────┐       machinery and the dotted-path form.  No deps.
      py-   rs-   ts-  find-dup-defs
     canon canon canon  (engine + CLI: the 3 passes + patternology + converge +
        └────┴────┴───────▲          severity + directives + calibration + reports)
                          │
                    the engine depends on the contract and on each frontend,
                    and on no frontend's internals
```

[`find-dup-defs`](crates/find-dup-defs) is the engine and CLI; it clusters a `Vec<Def>` and never
names a language. [`dup-defs-core`](crates/dup-defs-core) is everything both sides share: the
contract (`Def`, `Facets`, `KindSpec`, `Analysis`, the `Frontend` trait), and the pieces the
frontends would otherwise each keep a copy of — the kind vocabulary, `alpha_rename`, `count_loc`,
the lens vocabulary and stitching, the dotted-path form and its prefix walk.
[`py-canon`](crates/py-canon), [`ts-canon`](crates/ts-canon) and [`rs-canon`](crates/rs-canon) are
the frontends (Ruff, oxc, syn). Adding a language is one more `<lang>-canon` crate implementing
`Frontend` — plus a `Dialect` impl if it wants patternology — and no engine changes.

The contract used to be its own crate with a `find-dup-defs-canon` between it and the frontends, so
the engine would not pull in frontend implementation. The perspective passes ended that: the engine
reads `reach::prefixes` to walk the module tree the frontends' `Facets::reaches` names, and the lens
machinery needs `Def` itself. A boundary with holes on both sides is a version to bump and a publish
order to get right for nothing, so the two were folded into one.

**What a frontend must answer.** Beyond the body canonical, `Facets` asks for two things, and both
are empty when a frontend does not report them — every pass reading them is self-gating, so a
language lights up the moment its frontend fills them in, with no engine edit and no list of
supported languages anywhere:

| facet | what | why it cannot be derived later |
|---|---|---|
| `statements` | every statement at every nesting level, header first at depth 0 | flattened, `for x in xs: / f() / g()` is indistinguishable from the three statements where `g()` runs *after* the loop — only the walk that produced them knows |
| `reaches` | the dotted path each imported name it uses stands for, separator normalized to `.` | the corpus's own declaration of what a definition is *about*; two functions written independently about one entity share no line and often not one name |

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
                            + `lenses` (opt-in; py, rs, ts) — see Lenses

SEVERITY (thickness ladder)
  --error-thickness <F>     Demote ERROR → WARNING if T < F   (default 0.0 = off)
  --warning-thickness <F>   Demote WARNING → INFO  if T < F   (default 0.0 = off)
  --escalate-thickness <F>  Promote anything → ERROR if T ≥ F (default 0.0 = off, applied last)

SIMILARITY
  -t, --threshold <F>       Name-gated cluster floor   (default 0.5)
  -e, --error-threshold <F> Name-gated ERROR floor     (default 0.85)
  --type3-theta <F>         Type-3 cosine floor        (default 0.7)
  --max-name-group <N>      Skip name-gated clustering for (kind,name) groups > N

LENSES (opt-in · py, rs, ts)
  --kinds lenses            Cluster by perspectives other than the body; each finding reports
                            which lenses agreed (`votes[n]: control×3 outgoing×2 …`)

PATTERNOLOGY (opt-in · advisory, never ERROR)
  --patternology            Surface collapsible-duplication helper candidates
  --pattern-theta <F>       Whole-fn structural cosine floor (default 0.85)
  --pattern-support <N>     Sub-block idiom support floor     (default 3)

CONVERGE (opt-in · advisory, never ERROR · py, rs, ts)
  --converge                Where two definitions about the same thing stop agreeing, and the
                            families of many that do the same thing around one subject
  --converge-top <N>        Keep the strongest N of each kind (default 50; 0 = every one)

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
- Lenses are Python-only. Nine of the ten read the definition's own tree and would port to another
  frontend as-is; `use` needs the tree-wide mention index that `py-canon` builds.
- The `use` lens assembles by name with no import resolution, which holds while top-level names are
  effectively unique (measured: 2435 distinct across 2444 classes in one production tree) and
  degrades on trees where one name covers hundreds of definitions — the case `--max-name-group`
  exists for.
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
