# find-dup-defs

Find **duplicate / near-duplicate top-level definitions** across a Python codebase — copy-pasted
(and lightly edited, or renamed) functions, classes, constants, and `type` aliases.

The name is deliberately language-agnostic. Today the frontend is Python; other languages can be added
later as additional frontend crates feeding the same CLI.

This is a Cargo **workspace** of two crates:

- [`crates/py-canon`](crates/py-canon) — the **Python frontend**: source → `ast.dump`-shape
  canonicalization + top-level definition scan (Ruff parser; PEP 695 / 701). Reusable library.
- [`crates/find-dup-defs`](crates/find-dup-defs) — the CLI. **See its
  [README](crates/find-dup-defs/README.md) for usage, the three detection passes, and speed.**

The similarity engine is the exact Ratcliff–Obershelp library
[`difflib-fast`](https://crates.io/crates/difflib-fast) (a crates.io dependency).

```bash
# install (needs a Rust toolchain):
cargo install find-dup-defs

# or from source:
cargo build --release
cargo run --release -p find-dup-defs -- path/to/project --kinds functions,classes
```

## License

MIT.
