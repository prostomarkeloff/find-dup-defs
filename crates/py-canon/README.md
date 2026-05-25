# py-canon

The **Python frontend** for [`find-dup-defs`](https://github.com/prostomarkeloff/find-dup-defs):
Python source → a CPython `ast.dump`-shape **canonical form** plus a **top-level definition scan**.

Parses with the [Ruff](https://github.com/astral-sh/ruff) Python parser (modern syntax — PEP 695 /
PEP 701). Two layers:

- **`find_module_defs`** scans files for each module-level definition (function, class, `UPPER_CASE`
  constant, `type` alias) → `ModuleDef { kind, name, file, line, col, text }`.
- **canonicalization** of a definition's source text: `ast_canonical` (a structural canonical matching
  CPython's `ast.dump` shape, docstrings stripped — the input to byte-for-byte Ratcliff–Obershelp
  similarity), plus `normalize_functions` / `analyze_functions` for the alpha-renamed and
  name-agnostic forms used to detect *renamed* copy-paste.

The canonicalization is validated **byte-for-byte** against a golden corpus produced by CPython's own
`ast` module (`examples/verify_golden.rs`).

```rust
use py_canon::{find_module_defs, ast_canonical};

let defs = find_module_defs(&["m.py".to_string()]);   // reads the files, returns top-level defs
for d in &defs {
    println!("{} {} @ {}:{}", d.kind, d.name, d.file, d.line);
    let canonical = ast_canonical(&d.text);           // ast.dump-shape structural canonical
    // → feed canonicals to difflib-fast::cluster_canonicals to find near-duplicates
}
```

Reusable on its own; pairs with [`difflib-fast`](https://crates.io/crates/difflib-fast) for the
similarity/clustering step.

## License

MIT.
