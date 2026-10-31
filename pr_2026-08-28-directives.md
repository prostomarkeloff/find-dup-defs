# `--json` says what each directive actually matched

The binary matches every `-D` directive against every finding and said nothing about it. So the
question a directive file eventually raises — *is this suppression still suppressing anything?* —
could only be answered from outside the tool.

One commit: 7 files, +276 / −14. Default document byte-identical; the field appears only when `-D`
is passed.

---

## Why it has to come from here

Both outside routes were measured against a real 126-line directive file:

| route | cost | result |
|---|---|---|
| re-run the tool once per directive | 2.6s × 126 ≈ **5.5 min** | correct, unusable in a 25s lint step |
| re-implement the matcher | free | **disagreed with this binary** on the `a/b` cross-name form |

The second is the interesting failure. Matching is globs × kind gating × path scoping over a
mini-language; a second implementation is a second answer, and it read 16 live directives as dead.

## What lands

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
- **`origin`** carries `<file>:<line>` (or `<inline>`). A consumer auditing a directive *file* needs
  to be told which line it is looking at, not only what the line says. Comments and blanks do not
  shift it.
- **`set:` reports `null`, not `0`.** It is pipeline config, never a finding filter, so "matched
  nothing" is not a fact about it — and `0` would read as dead.
- **The key omits line numbers on purpose.** Lines move whenever an unrelated line above a
  definition changes, so a fingerprint that included them would report novelty on every edit. Files
  are sorted and de-duplicated, so member order does not enter the identity either.

Recorded inside the pass that already calls `matches()` for every pair — free, and the only point
where the answer exists at all: after the suppress `retain`, the dropped findings are gone.

## Also: `--kinds all` scanned nothing, quietly

An unrecognized `--kinds` value matched no kind, so the scan collected nothing and the run reported
zero findings — a gate that goes green having looked at nothing. `--only` has rejected unknown codes
since it was written, for precisely this reason; `--kinds` did not.

```console
$ find-dup-defs crates --kinds all
find-dup-defs: unknown kind "all" for --kinds (known: functions, methods, classes, interfaces,
constants, type-aliases, lenses)
$ echo $?
2
```

The vocabulary is now public (`dup_defs_core::KIND_IDS`) so the check reads one list rather than a
second copy of it, with a test that every id in it still resolves.

## Verified

- **`cargo test --release` — 146 tests**, including 7 new CLI tests: a directive that matched, a dead
  one, `set:` reporting null, the absent key when no `-D` was passed, file origins reporting the
  right line across comments and blanks, and both halves of the `--kinds` check.
- `json_default_matches_golden` passes untouched — the default document did not move.
- `cargo clippy --release --all-targets` clean; the self-gate reports **0 ERROR**.
- Dogfooded: this repo's own 19 directives are all live; the 126-line file it was built for has
  **25 dead (20%)**.

## Versions

`find-dup-defs` **0.10.0 → 0.11.0** — minor, because the JSON document gained a field.
`dup-defs-core` **0.2.0 → 0.2.1** — additive, `KIND_IDS` is now public.
`py-canon`, `rs-canon`, `ts-canon` untouched.
