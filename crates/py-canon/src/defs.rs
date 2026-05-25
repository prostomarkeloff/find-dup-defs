//! Module-level definition scan — the "find-*" step of dup-defs, ported off ast-grep.
//!
//! Replaces four `ast-grep scan --rule find-module-{functions,classes,constants,type-aliases}`
//! subprocess calls with one native, parallel pass. Matches the same semantics: **top-level
//! only** (not nested in any function/class), `UPPER_CASE` constants, decorators excluded from
//! a def's text (the range starts at the `def`/`async`/`class` keyword, like tree-sitter's
//! `function_definition`). Emits each def's kind / name / location / source text — the shape
//! the cross-file grouping step consumes.
//!
//! Parses with **ruff** (same parser the canonicalization uses), so modern syntax — PEP 695
//! `type` aliases / generics, PEP 701 f-strings — is handled; rustpython silently dropped any
//! file containing them, hiding every def it held from the dup-defs passes.

use std::fs;

use rayon::prelude::*;
use ruff_python_ast::{Expr, Stmt};
use ruff_python_parser::parse_module;

use crate::loc::LineMap;

/// One module-level definition found by the scan (kind = `functions` / `classes` /
/// `constants` / `type-aliases`; `line`/`col` 0-indexed to match the prior ast-grep range).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ModuleDef {
    pub kind: String,
    pub name: String,
    pub file: String,
    pub line: usize,
    pub col: usize,
    pub text: String,
}

fn is_upper(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// The name of a module-level `UPPER_CASE` constant assignment (`NAME = …` / `NAME: T = …`).
fn const_name(stmt: &Stmt) -> Option<String> {
    match stmt {
        Stmt::Assign(node) => match node.targets.as_slice() {
            [Expr::Name(name)] if is_upper(name.id.as_str()) => Some(name.id.as_str().to_owned()),
            _ => None,
        },
        Stmt::AnnAssign(node) => match node.target.as_ref() {
            Expr::Name(name) if is_upper(name.id.as_str()) => Some(name.id.as_str().to_owned()),
            _ => None,
        },
        _ => None,
    }
}

/// The byte offset of the `def`/`async`/`class` keyword — i.e. the def text *excluding* decorators.
/// With no decorators the statement range already starts at the keyword; with decorators we skip
/// past the last decorator and any intervening whitespace / comment lines to the keyword token.
fn keyword_start(source: &str, range_start: usize, last_decorator_end: Option<usize>) -> usize {
    let Some(mut i) = last_decorator_end else { return range_start };
    let bytes = source.as_bytes();
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        break;
    }
    i
}

/// Classify one top-level statement → `(kind, name, text_start, text_end)` if it is a tracked
/// definition. `text_start` is decorator-excluded (the keyword offset) for functions / classes.
pub(crate) fn classify(source: &str, stmt: &Stmt) -> Option<(&'static str, String, usize, usize)> {
    match stmt {
        Stmt::FunctionDef(node) => {
            let deco_end = node.decorator_list.last().map(|d| usize::from(d.range.end()));
            let start = keyword_start(source, usize::from(node.range.start()), deco_end);
            Some(("functions", node.name.id.as_str().to_owned(), start, usize::from(node.range.end())))
        }
        Stmt::ClassDef(node) => {
            let deco_end = node.decorator_list.last().map(|d| usize::from(d.range.end()));
            let start = keyword_start(source, usize::from(node.range.start()), deco_end);
            Some(("classes", node.name.id.as_str().to_owned(), start, usize::from(node.range.end())))
        }
        Stmt::TypeAlias(node) => match node.name.as_ref() {
            Expr::Name(name) => {
                Some(("type-aliases", name.id.as_str().to_owned(), usize::from(node.range.start()), usize::from(node.range.end())))
            }
            _ => None,
        },
        Stmt::Assign(node) => {
            const_name(stmt).map(|name| ("constants", name, usize::from(node.range.start()), usize::from(node.range.end())))
        }
        Stmt::AnnAssign(node) => {
            const_name(stmt).map(|name| ("constants", name, usize::from(node.range.start()), usize::from(node.range.end())))
        }
        _ => None,
    }
}

fn module_defs_from(source: &str, file: &str) -> Vec<ModuleDef> {
    let Ok(parsed) = parse_module(source) else { return Vec::new() };
    let module = parsed.into_syntax();
    let lines = LineMap::new(source);
    let mut defs: Vec<ModuleDef> = Vec::new();
    for stmt in &module.body {
        let Some((kind, name, start, end)) = classify(source, stmt) else { continue };
        let (line, col) = lines.loc0(start);
        defs.push(ModuleDef {
            kind: kind.to_owned(),
            name,
            file: file.to_owned(),
            line,
            col,
            text: source[start..end].to_owned(),
        });
    }
    defs
}

fn module_defs_in(file: &str) -> Vec<ModuleDef> {
    match fs::read_to_string(file) {
        Ok(source) => module_defs_from(&source, file),
        Err(_) => Vec::new(),
    }
}

/// `find_module_defs(files) -> Vec<ModuleDef>`: the dup-defs find step, native + parallel.
#[must_use]
pub fn find_module_defs(files: &[String]) -> Vec<ModuleDef> {
    let per_file: Vec<Vec<ModuleDef>> = files.par_iter().map(|f| module_defs_in(f)).collect();
    per_file.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::module_defs_from;

    fn triples(src: &str) -> Vec<(String, String, String)> {
        module_defs_from(src, "<test>").into_iter().map(|d| (d.kind, d.name, d.text)).collect()
    }

    #[test]
    fn finds_top_level_kinds_only() {
        let src = "MAX = 5\nlower = 1\n\ntype Ids = list[int]\n\n\ndef top():\n    def nested():\n        pass\n    return 1\n\n\nclass C:\n    def method(self):\n        pass\n";
        let got = triples(src);
        let kinds: Vec<&str> = got.iter().map(|(k, _, _)| k.as_str()).collect();
        let names: Vec<&str> = got.iter().map(|(_, n, _)| n.as_str()).collect();
        // MAX (UPPER const), Ids (type-alias), top (fn), C (class). Excluded: lower (not
        // UPPER), nested (not top-level), method (inside class).
        assert!(names.contains(&"MAX") && names.contains(&"Ids"));
        assert!(names.contains(&"top") && names.contains(&"C"));
        assert!(!names.contains(&"lower") && !names.contains(&"nested") && !names.contains(&"method"));
        assert_eq!(kinds.iter().filter(|k| **k == "functions").count(), 1);
        assert_eq!(kinds.iter().filter(|k| **k == "classes").count(), 1);
    }

    #[test]
    fn function_text_excludes_decorators() {
        let got = triples("import functools\n\n\n@functools.cache\ndef memo():\n    return 1\n");
        let func = got.iter().find(|(k, _, _)| k == "functions").expect("a function");
        assert!(func.2.starts_with("def "), "text should start at def, got: {:?}", func.2);
    }

    #[test]
    fn pep695_and_modern_syntax_file_is_scanned() {
        // A file mixing PEP 695 `type` aliases, PEP 695 generics, and PEP 701 f-strings — the
        // exact shapes rustpython choked on, silently dropping every def in the file.
        let src = "type Alias = list[int]\n\n\ndef worker[T](x: T) -> T:\n    msg = f\"got {x['k']}\"\n    return x\n\n\nclass Repo[T]:\n    pass\n";
        let names: Vec<String> = module_defs_from(src, "<test>").into_iter().map(|d| d.name).collect();
        assert!(names.contains(&"Alias".to_owned()), "type alias missing: {names:?}");
        assert!(names.contains(&"worker".to_owned()), "generic fn missing: {names:?}");
        assert!(names.contains(&"Repo".to_owned()), "generic class missing: {names:?}");
    }
}
