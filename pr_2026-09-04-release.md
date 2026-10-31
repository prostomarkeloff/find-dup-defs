## TL;DR

Релиз 0.13.1: единственное изменение — версия крейта. Зависимость `difflib-fast = "0.4"` уже в `main` (#18); этот релиз доносит её до crates.io / PyPI / бинарников. Фаза `pass1-name` на большом монорепо 1 472 → 546 ms (`--kinds lenses`) и 1 158 → 504 ms (default), вывод во всех десяти режимах побайтово тот же.

## Что изменилось

- `crates/find-dup-defs/Cargo.toml`: `version = "0.13.1"`; `Cargo.lock` следом.

## Проверка

| что | результат |
|---|---|
| `cargo test --release`, `cargo clippy --release --all-targets` | ok |
| parity 10 режимов на `mixed` с crates.io `difflib-fast 0.4.0` (#18) | побайтово |
| parity 10 режимов на `mixed` + `mono` с тем же кодом difflib-fast до его релиза | побайтово |
