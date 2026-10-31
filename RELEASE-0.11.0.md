## `--json` says what each directive actually matched

The binary matches every `-D` directive against every finding and said nothing about it. So the
question a directive file eventually raises — *is this suppression still suppressing anything?* —
could only be answered from outside the tool, and both routes out were bad. Measured on a real
126-line directive file: re-running once per directive costs 2.6s × 126 ≈ **5.5 minutes**;
re-implementing the matcher outside **disagreed with this binary** on the `a/b` cross-name form and
read 16 live directives as dead. Matching is globs × kind gating × path scoping — a second
implementation is a second answer.

`--json` now carries `directives[]`:

```json
{
  "directive": "suppress:<constants>KINDS=per-frontend kind subset",
  "origin": ".ast-grep/lint.directives:57",
  "action": "suppress",
  "matched": 1,
  "findings": [{ "key": "dup-constant KINDS", "files": ["crates/py-canon/src/frontend.rs", "…"] }]
}
```

- **`matched: 0`** is a directive that no longer suppresses anything.
- **`origin`** is `<file>:<line>` (or `<inline>`) — a consumer auditing a directive *file* is told
  which line it is looking at, not only what the line says. Comments and blanks do not shift it.
- **`set:` reports `null`, not `0`** — it is pipeline config, never a finding filter, so "matched
  nothing" is not a fact about it, and `0` would read as dead.
- **The key omits line numbers deliberately.** Lines move whenever an unrelated line above a
  definition changes; a fingerprint including them would report novelty on every edit. Files are
  sorted and de-duplicated, so member order stays out of the identity too.

The field is **omitted entirely when no `-D` was given**, so the default document is byte-identical
to 0.10.0.

## Fixed

- **`--kinds all` scanned nothing, quietly.** An unrecognized kind matched none, so the run reported
  zero findings from an empty scan — a gate that goes green having looked at nothing. It now exits 2
  and lists the vocabulary, the way `--only` has always treated an unknown language code. The kind
  vocabulary is public as `dup_defs_core::KIND_IDS`, so the check reads one list rather than a copy.

## Versions

| crate | | why |
|---|---|---|
| `find-dup-defs` | 0.10.0 → **0.11.0** | the JSON document gained a field |
| `dup-defs-core` | 0.2.0 → **0.2.1** | additive: `KIND_IDS` is now public |

`py-canon`, `rs-canon` and `ts-canon` are **untouched** and not republished — 0.4.0 / 0.2.0 / 0.2.0
remain current.

## Install

```console
uv tool install find-dup-defs        # or: pipx install find-dup-defs
cargo install find-dup-defs
```

Prebuilt binaries for linux (gnu/musl, x86_64/aarch64), macOS (arm64/x86_64) and Windows are
attached below.
