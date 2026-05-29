# py-canon

The **Python frontend** for [`find-dup-defs`](https://github.com/prostomarkeloff/find-dup-defs):
Python source → a CPython `ast.dump`-shape **canonical form** plus a **top-level definition scan**.

Parses with the [Ruff](https://github.com/astral-sh/ruff) Python parser (modern syntax — PEP 695 /
PEP 701). It implements `find-dup-defs`'s `Frontend` trait:

- **`Python::scan`** walks each file once and lowers every module-level definition (function, class,
  `UPPER_CASE` constant, `type` alias) and class method to a `Def`, with its canonical strings
  precomputed off the AST node.
- The **canonical** is a structural form matching CPython's `ast.dump` shape (docstrings stripped) —
  the input to byte-for-byte Ratcliff–Obershelp similarity. `ast_canonical` / `analyze_functions`
  expose it over a source string (used for tooling / golden checks).

The canonicalization is validated **byte-for-byte** against a golden corpus produced by CPython's own
`ast` module (`examples/verify_golden.rs`).

```rust
use std::sync::Arc;
use dup_defs_core::Frontend;
use py_canon::Python;

let files = [Arc::<str>::from("m.py")];
let defs = Python.scan(&files);               // reads the files, returns Defs with canon precomputed
for d in &defs {
    println!("{} {} @ {}:{}", d.kind.id, d.name, d.file, d.line);
    // d.cluster_canonical / d.analysis are ready for difflib-fast::cluster_canonicals
}
```

Reusable on its own; pairs with [`difflib-fast`](https://crates.io/crates/difflib-fast) for the
similarity/clustering step.

## License

MIT.
