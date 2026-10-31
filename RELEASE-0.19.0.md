## A Java frontend

`find-dup-defs` now reads Java. A new `java-canon` frontend over the tree-sitter Java grammar
lowers every method and constructor, every `class` / `record` / `enum`, every `interface` /
`@interface`, and every `static final` constant (a constant an interface declares included) to
the same `Def` record the other six languages produce — so every pass works over Java with no
engine changes: name-gated clustering, the cross-name pass, Type-3, converge, decisions, the
lenses, and patternology (the engine grew a `JavaDialect` with a `int helper(int x) { … }`
pseudo-source renderer).

What is Java-specific, read the Java way:

- **Types are kept as written** (import-resolved to their dotted path) in the exact and
  cross-name canonicals; only the Type-3 lines slot them — a Java type selects the overload and
  the dispatch, so it is not inert.
- **`this` is the receiver**: a method matches the equivalent static helper, and a read off
  `this` is a field of the value the method is read off. Without a type checker, a bare
  `helper()` call is a call of the name `helper` — never resolved to a method on `this` — and
  `Foo.m()` / `Foo.BAR` are static, by the language's own capitalization convention.
- **Normal form**: a Yoda `.equals` meets its mirror (`x.equals("lit")` and `"lit".equals(x)`
  are one form, `Objects.equals` sorts its arguments); `if (c) return true; return false;` is
  `return c;`; a `switch` whose every case ends in `break` / `return` / `throw` is the ladder it
  abbreviates; the guard holds the lighter branch, and at equal weight the one that fails
  (throws, or returns `null` / `false` / a code); `x += e` is `x = x + e`, `while (true)` is
  `for (;;)`.
- **Reaches** are a file's imports and the fully qualified names its body writes; `java.lang`
  falls out by itself, never imported. **Decisions** read `instanceof` as membership, `.equals`
  / `compareTo(…) == 0` as equality, `.contains` / `.startsWith` / `.indexOf(…) != -1` as
  membership, `Optional.isPresent()` as presence, and `Objects.requireNonNull` /
  `Preconditions.checkArgument` as the proposition they check. **Lenses** answer in Java's
  spellings: try-with-resources is the scoped resource, a `.close()` in `finally` the release;
  failures are the classes a body throws and catches; decorators are the definition's
  annotations (`@Override` stays out — every override carries it).
- **Noise is filtered** as in the other frontends: bodies that only `return` a name, a literal
  or `null`, a `throw new UnsupportedOperationException(…)` stub, the accessors a class is
  mostly made of, and files that declare themselves generated (`DO NOT EDIT`).

Validation: the workspace test suite (the crate's own 49 unit tests, the engine's, and the
byte-for-byte CLI goldens — including new Java converge and decisions fixtures) is green, clippy
is clean under the workspace's all+pedantic gate, and the self-dogfood reports 0 ERROR. A
smoke run over a 23k-line Bukkit plugin (38 files) scans in 0.44 s and surfaces the plugin's two
copied sub-worlds of code — e.g. `Game.copyWorldFolder` / `KingdomWarsGame.copyWorldFolder` at
ast sim 1.00 — plus the converge pass's split point between the two creature movers.

## Versions

| crate | | why |
|---|---|---|
| `java-canon` | **0.1.0** (new) | the Java frontend |
| `dup-defs-core` | 0.4.3 → **0.4.4** | `CanonDialect::Java` |
| `find-dup-defs` | 0.18.0 → **0.19.0** | Java frontend + patternology `JavaDialect` |

`py-canon`, `ts-canon`, `rs-canon`, `go-canon`, `c-canon` and `php-canon` are unchanged and not
republished.

## Install

```console
uv tool install find-dup-defs        # or: pipx install find-dup-defs
cargo install find-dup-defs
```

Prebuilt binaries for linux (gnu/musl, x86_64/aarch64), macOS (arm64/x86_64) and Windows are
attached below.
