## `--json` stops walking the same path a thousand times

The JSON report resolved every member's file with its own `canonicalize` — a kernel walk down
every component of the path — and a file named by a thousand findings was walked a thousand
times. The human report had already stopped doing that: it resolves each *distinct* file once,
with the real path of each *directory* resolved once per directory. This release points the JSON
renderer and the directive audit at that same resolver.

Nothing about what the tool reports changed. Ten output modes were diffed against 0.13.1 on a
mixed Python/TypeScript/Rust tree, plus the directive path in both report and JSON form, plus a
tree of symlinked files, symlinked directories, a symlinked repo root and links escaping it.
Byte-identical everywhere, same exit codes.

| render phase, mixed tree | 0.13.1 | 0.13.2 |
|---|---:|---:|
| `--json` | 138 ms | **7 ms** |
| `--converge --converge-top 0 --json` | 2550 ms | **155 ms** |
| default report (unchanged path) | 5.1 ms | 5.4 ms |

Whole runs on that tree: `--json` 1.27 s → **1.09 s**, `--kinds lenses --json` 1.12 s → **1.04 s**,
and the uncapped converge ranking as JSON 4.25 s → **1.75 s**. On CPython's `Lib` alone that last
one goes 3.99 s → **1.68 s**. The default text report is not on this path and does not move.

## Directives match in parallel

`-D` matching is now a parallel pass over the findings; the effects are still applied serially, in
directive order, so the contracts hold exactly as before — notes accumulate from every matching
directive, severity steps by the *summed* escalate/de-escalate and is clamped once at the end,
`suppress` drops the finding last, `set:` is pipeline config and never a finding filter, and the
audit still reports what a suppressed finding matched before it was dropped.

On the uncapped converge ranking, 95454 findings against 21 directives:

| directive pass | serial | parallel |
|---|---:|---:|
| | 99 ms | **13 ms** |

This is the smaller of the two levers by a wide margin. The headline above is the path resolution.

## Versions

| crate | | why |
|---|---|---|
| `find-dup-defs` | 0.13.1 → **0.13.2** | report paths resolved once per file |

`dup-defs-core`, `py-canon`, `ts-canon`, `rs-canon` are unchanged and not republished.

## Install

```console
uv tool install find-dup-defs        # or: pipx install find-dup-defs
cargo install find-dup-defs
```

Prebuilt binaries for linux (gnu/musl, x86_64/aarch64), macOS (arm64/x86_64) and Windows are
attached below.
