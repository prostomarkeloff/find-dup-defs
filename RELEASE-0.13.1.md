## The name-gated pass 2.5× faster, every byte the same

This release changes one line: the clustering library moves to
[difflib-fast 0.4](https://github.com/prostomarkeloff/difflib-fast/releases/tag/v0.4.0). That
library was reworked around the shape this tool issues — thousands of `cluster_canonicals` calls
per repository, most of two or three long canonical bodies, a few of hundreds — and every byte of
this tool's output stayed put: ten output modes diffed against the previous build on a mixed
standard-library tree and on a 371k-definition monorepo.

| pass 1 (name-gated), large monorepo | 0.13.0 | 0.13.1 |
|---|---:|---:|
| `--kinds lenses` | 1.47 s | **0.55 s** |
| default | 1.16 s | **0.50 s** |

Whole runs: 6.8 s → 5.8 s and 4.6 s → 4.0 s on that tree (the other passes do not touch this
library); user CPU 28.9 s → 19.2 s and 17.3 s → 10.0 s.

## What difflib-fast 0.4 does differently

- Builds an automaton only for the strings it actually scans (a pair scans one side against the
  other's automaton — in a group of two that is one build, not two).
- Tests only the edges that connect the components: candidates go most-similar-first in batches
  over a union-find, and a pair already connected is not tested (edge tests 104k → 32k on the
  replayed run).
- Takes each cluster's minimum under the cluster's shared running minimum, deciding most pairs
  from a common prefix, suffix, or a single long block before the recursion starts.
- Sends narrow recursion windows to a direct comparison and takes the wide ones largest-first.

## Versions

| crate | | why |
|---|---|---|
| `find-dup-defs` | 0.13.0 → **0.13.1** | dependency: `difflib-fast = "0.4"` |

`dup-defs-core`, `py-canon`, `ts-canon`, `rs-canon` are unchanged and not republished.

## Install

```console
uv tool install find-dup-defs        # or: pipx install find-dup-defs
cargo install find-dup-defs
```

Prebuilt binaries for linux (gnu/musl, x86_64/aarch64), macOS (arm64/x86_64) and Windows are
attached below.
