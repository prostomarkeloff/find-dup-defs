## `--converge` is 4.5x faster

It was the one pass still running single-threaded while the rest of the tool sat at 563% CPU. On a
25k-definition tree it took **11.0s of the 11.7s total**. Now it takes 1.2s.

```
converge, whole corpus     8.32s → 1.84s   4.5x
converge, one tree         1.82s → 0.44s   4.1x
converge uncapped          2.74s → 1.38s   2.0x
default report             0.49s → 0.48s   unchanged
```

Four changes, each measured:

- **FxHash instead of SipHash.** Converge is almost entirely hash lookups over interned strings, and
  those keys are not adversarial input. Weighing 6.6s → 4.1s on that alone.
- **Explain a finding only if it survives the cap.** The pass reports a *ranking* capped at a few
  hundred, but built all hundred and twenty thousand findings whole first — rendering the agreed
  run, the fork, cloning each definition's entire source — then dropped 99.7% of them.
- **Three parallel phases where there was one loop.** Growing a block and anti-unifying a fork are
  per-pair work with no shared state; the sieves ("this run was already seeded by another
  statement", "a pair keeps its **longest** run", "this family already appeared at a more specific
  subject") read and write one table and their order is load-bearing. So: compute in parallel,
  decide sequentially, over the same pre-sorted key list. Statement seeding 2.8s → 0.42s, subject
  seeding 1.1s → 0.12s, family cliques 0.73s → 0.22s.
- **The cluster key off the sequential path** — it builds a string per pair, and there are hundreds
  of thousands of pairs.

Determinism is enforced rather than hoped for: the weighing walks pairs **sorted by key**, not the
hash table, and `collect` on an indexed parallel iterator keeps positions. The result depends on
neither the core count nor which worker finished first.

The human report is **byte-identical** to 0.9.0 in six modes — converge, the default report, lenses,
patternology, `--calibrate`, and converge on Rust — verified against a binary built from the
previous tag.

## `--json` now says which pass found a thing

The human report separates passes by section heading; the machine one merged them into a list where
the advisory converge ranking carried the same `dup-function` as a gating pass. A consumer had to
guess from side effects — an absent `min_sim` together with a present `facets` — which is to say,
build its own model of what the tool declined to state about itself.

`Finding` already knew its `pass`. It is now serialized:

```json
{ "kind": "FUNCTION", "name": "a / b", "severity": "INFO", "pass": "converge", … }
```

Values are `name`, `cross-name`, `type-3`, `pattern`, `converge`, `converge-family`.

Purely additive: on the reference document, **145 lines added, none removed**, and every one of them
the new field. Nothing else about `--json` changed.

## Fixed

- `--kinds lenses` still advertised itself as **"(Python only)"** in `--help`, three languages after
  that stopped being true.

## Versions

| crate | | why |
|---|---|---|
| `find-dup-defs` | 0.9.0 → **0.10.0** | converge performance; `pass` in the JSON document |

Minor rather than patch because the JSON document gained a field: additive for a reader of known
keys, but no longer byte-identical, and 0.9.0 shipped promising it would be.

`dup-defs-core`, `py-canon`, `rs-canon` and `ts-canon` are **untouched** and not republished —
0.2.0 / 0.4.0 / 0.2.0 / 0.2.0 remain current.

## Install

```console
uv tool install find-dup-defs        # or: pipx install find-dup-defs
cargo install find-dup-defs
```

Prebuilt binaries for linux (gnu/musl, x86_64/aarch64), macOS (arm64/x86_64) and Windows are
attached below.
