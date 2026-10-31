## The same report, on an eighth of the CPU

**13.8× less CPU** (365 s → 26.5 s) and **4.6× wall** on a 371k-definition tree. Every report is
byte-identical to 0.11.0 — ten output modes were diffed against a 0.11.0 binary, including the
uncapped converge ranking, and nothing moved.

Nothing about what the tool finds has changed. This is entirely about how often it computes the
same answer.

## Where it was going

The profile pointed at exact Ratcliff–Obershelp — suffix-automaton construction 28% of CPU,
matching-stats 17%, longest-common-block 15% — which reads as the floor of an exact algorithm. It
isn't. A profile says where time goes; it does not say that **the inputs are overwhelmingly repeats
of each other**:

| | measured on a real tree |
|---|---|
| files | 51 550 → **7 543 distinct** (85% duplicated) |
| definitions in a name group | 368 962 → **47 733 distinct canonicals** (87%) |
| converge bodies | 217 980 → **29 788 distinct** (86%) |
| type-3 functions | 214 040 → **29 519 distinct** (86%) |

Every pass paid for that redundancy, and paid quadratically. Each now works over the distinct inputs
and expands the answer back — exactly, not approximately.

- **Scanning** parses one file per distinct content. A definition is a function of its source and
  where in it the definition sits, so a twin's definitions are a copy with the path swapped.
- **Name-gated clustering** clusters the distinct canonicals. Two identical strings have a ratio of
  exactly 1.0, so they always join and can never lower a cluster's minimum. A group spelling *one*
  canonical needs no clustering at all — and most groups are that.
- **Converge** tokenizes each body once instead of re-lexing it for every candidate pair, which was
  54% of the pass on its own, and memoizes the block walk and the weighing on the pair of bodies.
- **Type-3** runs its cosine join over distinct `(line sequence, name)` pairs, with the IDF still
  taken over every function — that one is a property of the corpus.
- **Discovery** walks and routes in parallel; **rendering** builds one buffer instead of a million
  strings.

| phase | 0.11.0 | 0.12.0 | |
|---|---:|---:|---:|
| name-gated clustering | 21 424 ms | 1 198 ms | **17.9×** |
| cross-name | 623 ms | 127 ms | 4.9× |
| converge | 27 061 ms | 6 770 ms | 4.0× |
| type-3 | 934 ms | 396 ms | 2.4× |
| discovery | 645 ms | 326 ms | 2.0× |
| scan | 2 615 ms | 1 348 ms | 1.9× |
| **whole run** | **50.9 s** | **11.0 s** | **4.6×** |
| **CPU** | **365 s** | **26.5 s** | **13.8×** |

Wall time improves less than CPU because the pipeline was already ~8× parallel: removing parallel
work leaves the serial remainder exposed. The CPU figure is the one that barely moves with machine
temperature.

## New: `settings:converge-cap=N`

Converge's seeding cap — how many places a shared statement may occur in before the pass calls it an
idiom — is now settable. The work it admits is quadratic in it.

```console
find-dup-defs src -D 'settings:converge-cap=20'
```

**The default is unchanged at 60, and lowering it is not a free speedup.** On a duplication-heavy
monorepo cap 20 costs nothing measurable and runs converge 3.9× faster; on ordinary code — a
standard library, a framework — it drops **70% of the pass's findings**, because there statements
genuinely do occur in twenty to sixty places. Both measurements are in the constant's doc comment. A
knob whose cost depends on the shape of the input cannot be priced on one shape.

## Versions

| crate | | why |
|---|---|---|
| `find-dup-defs` | 0.11.0 → **0.12.0** | `PipelineOpts` gained a field; breaking for a library consumer, invisible to a CLI user |

`dup-defs-core`, `py-canon`, `rs-canon` and `ts-canon` are **untouched** and not republished —
0.2.1 / 0.4.0 / 0.2.0 / 0.2.0 remain current.

## Install

```console
uv tool install find-dup-defs        # or: pipx install find-dup-defs
cargo install find-dup-defs
```

Prebuilt binaries for linux (gnu/musl, x86_64/aarch64), macOS (arm64/x86_64) and Windows are
attached below.
