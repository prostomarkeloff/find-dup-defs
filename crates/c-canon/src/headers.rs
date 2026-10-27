//! What a C definition **reaches**: the headers that declare the names it uses.
//!
//! The other languages name a module where they use it — `pkg.F`, `a::b::c`, `from a import b` — and
//! their frontends read the path off the use. C names nothing: `av_malloc(n)` says no more than that
//! `av_malloc` is declared somewhere the file `#include`d. Which header that is, only the headers say,
//! so they are read: every header of the run once, for the names it declares, and every file's
//! `#include`s followed through them. A definition reaches the header that declares a name it uses —
//! the nearest one, in the order the includes were followed.
//!
//! An include is resolved the ways a build would find it, without the build's `-I` flags: beside the
//! including file, then in each directory above it, then as the one header of the run whose path ends
//! the way the include is written. A system header (`<stdio.h>`) is in none of those, so the C library
//! is never what a definition is about — the role the Go frontend gives its stdlib vocabulary list,
//! here falling out of where the headers are.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use tree_sitter::Node;

use crate::prep::{prepare, Macros};
use crate::tree::{field, fields, kids, parse, text};

/// How many headers one file's includes are followed through. A translation unit of a large project
/// pulls in a few hundred of its own; past that the nearest declaration was found long ago.
const MAX_CLOSURE: usize = 512;

/// One header, read.
struct Header {
    /// The engine's dotted path for it: its path under the run's root, extension dropped.
    reach: Arc<str>,
    /// The headers it includes, resolved, as indices into [`Headers::all`].
    includes: Vec<usize>,
}

/// Every header of a run, read once.
#[derive(Default)]
pub(crate) struct Headers {
    root: PathBuf,
    all: Vec<Header>,
    /// Each header's path, by index.
    paths: Vec<PathBuf>,
    /// Each header's index by its path.
    index: HashMap<PathBuf, usize>,
    /// Each header by its file name — the last resort an include is resolved by.
    by_name: HashMap<String, Vec<usize>>,
    /// Which headers declare each name — usually one.
    declared_in: HashMap<String, Vec<usize>>,
}

/// A path with `.` and `..` resolved lexically — what two spellings of one header's path compare as.
fn normalized(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The dotted path of a file under `root`: `root/libavutil/mem.h` → `libavutil.mem`.
pub(crate) fn dotted_under(root: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(root).unwrap_or(file).with_extension("");
    let segments: Vec<String> = rel
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str().map(str::to_owned),
            _ => None,
        })
        .collect();
    dup_defs_core::reach::reach_path(&segments).to_string()
}

/// The paths a file's `#include` directives name, as written, wherever in the file they sit.
fn include_paths(root: Node<'_>, src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        for k in kids(n) {
            match k.kind() {
                "preproc_include" => {
                    if let Some(path) = field(k, "path") {
                        let spelled = text(path, src).trim();
                        let inner = spelled.trim_start_matches(['"', '<']).trim_end_matches(['"', '>']);
                        if !inner.is_empty() && matches!(path.kind(), "string_literal" | "system_lib_string") {
                            out.push(inner.to_owned());
                        }
                    }
                }
                kind if kind.starts_with("preproc_") || kind == "linkage_specification" || kind == "declaration_list" => {
                    stack.push(k);
                }
                _ => {}
            }
        }
    }
    out.reverse();
    out
}

/// The names a header declares: its functions and objects, its macros, its type names, and the tags
/// and enumerators of the types it writes out. A tag is keyed with its keyword (`struct foo`), the way
/// a body spells it.
fn declared_names(root: Node<'_>, src: &str) -> HashSet<String> {
    fn spec(t: Node<'_>, src: &str, out: &mut HashSet<String>) {
        let keyword = match t.kind() {
            "struct_specifier" => "struct",
            "union_specifier" => "union",
            "enum_specifier" => "enum",
            _ => return,
        };
        if let Some(name) = field(t, "name") {
            out.insert(format!("{keyword} {}", text(name, src)));
        }
        if keyword == "enum" {
            for e in field(t, "body").map(kids).unwrap_or_default() {
                if let Some(name) = field(e, "name") {
                    out.insert(text(name, src).to_owned());
                }
            }
        }
    }
    fn declarator_name(d: Node<'_>, src: &str) -> Option<String> {
        match d.kind() {
            "identifier" | "type_identifier" | "field_identifier" => Some(text(d, src).to_owned()),
            _ => field(d, "declarator")
                .or_else(|| kids(d).into_iter().find(|k| k.kind().ends_with("declarator") || k.kind().ends_with("identifier")))
                .and_then(|inner| declarator_name(inner, src)),
        }
    }
    let mut out = HashSet::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        for item in kids(n) {
            match item.kind() {
                "declaration" | "type_definition" | "function_definition" => {
                    if let Some(t) = field(item, "type") {
                        spec(t, src, &mut out);
                    }
                    for d in fields(item, "declarator") {
                        if let Some(name) = declarator_name(d, src) {
                            out.insert(name);
                        }
                    }
                }
                "preproc_def" | "preproc_function_def" => {
                    if let Some(name) = field(item, "name") {
                        out.insert(text(name, src).to_owned());
                    }
                }
                "struct_specifier" | "union_specifier" | "enum_specifier" => spec(item, src, &mut out),
                kind if kind.starts_with("preproc_") || kind == "linkage_specification" || kind == "declaration_list" || kind == "ERROR" => {
                    stack.push(item);
                }
                _ => {}
            }
        }
    }
    out
}

impl Headers {
    /// Read every header among `files` — what it declares and what it includes. `root` is the
    /// directory the run's paths are read under.
    pub(crate) fn read(root: &Path, files: &[Arc<str>], macros: &Macros) -> Self {
        let paths: Vec<PathBuf> =
            files.iter().filter(|f| f.ends_with(".h")).map(|f| normalized(Path::new(&**f))).collect();
        let parsed: Vec<(HashSet<String>, Vec<String>)> = paths
            .par_iter()
            .map(|p| {
                let Ok(src) = std::fs::read_to_string(p) else { return (HashSet::new(), Vec::new()) };
                let prepared = prepare(&src, macros);
                let Some(tree) = parse(&prepared) else { return (HashSet::new(), Vec::new()) };
                (declared_names(tree.root_node(), &prepared), include_paths(tree.root_node(), &prepared))
            })
            .collect();
        let mut headers = Self { root: root.to_path_buf(), paths: paths.clone(), ..Self::default() };
        for (i, p) in paths.iter().enumerate() {
            headers.index.insert(p.clone(), i);
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                headers.by_name.entry(name.to_owned()).or_default().push(i);
            }
            headers.all.push(Header { reach: Arc::from(dotted_under(root, p)), includes: Vec::new() });
        }
        for (i, (declares, _)) in parsed.iter().enumerate() {
            for name in declares {
                headers.declared_in.entry(name.clone()).or_default().push(i);
            }
        }
        let resolved: Vec<Vec<usize>> =
            paths.iter().zip(&parsed).map(|(p, (_, includes))| headers.resolve_all(p, includes)).collect();
        for (header, includes) in headers.all.iter_mut().zip(resolved) {
            header.includes = includes;
        }
        headers
    }

    /// The headers a file at `from` names in its includes, resolved — each once, in order.
    fn resolve_all(&self, from: &Path, includes: &[String]) -> Vec<usize> {
        let mut seen = HashSet::new();
        includes.iter().filter_map(|inc| self.resolve(from, inc)).filter(|i| seen.insert(*i)).collect()
    }

    /// The header of the run an include written in the file at `from` names: beside `from`, then in
    /// each directory above it, then the one header of the run whose path ends as the include is
    /// written — the nearest to `from` when several do.
    fn resolve(&self, from: &Path, include: &str) -> Option<usize> {
        if self.all.is_empty() {
            return None;
        }
        let mut dir = from.parent();
        while let Some(d) = dir {
            if let Some(&i) = self.index.get(&normalized(&d.join(include))) {
                return Some(i);
            }
            if d.as_os_str().is_empty() || !d.starts_with(&self.root) && d != self.root {
                break;
            }
            dir = d.parent();
        }
        let name = Path::new(include).file_name()?.to_str()?;
        let wanted = normalized(Path::new(include));
        let candidates: Vec<usize> = self
            .by_name
            .get(name)?
            .iter()
            .copied()
            .filter(|&i| self.path_of(i).is_some_and(|p| p.ends_with(&wanted)))
            .collect();
        candidates.into_iter().max_by_key(|&i| {
            let shared = self.path_of(i).map_or(0, |p| p.components().zip(from.components()).take_while(|(a, b)| a == b).count());
            (shared, std::cmp::Reverse(i))
        })
    }

    fn path_of(&self, i: usize) -> Option<&PathBuf> {
        self.paths.get(i)
    }

    /// Every header a file's includes reach, nearest first: its own, then theirs, breadth first.
    pub(crate) fn closure(&self, file: &Path, includes: &[String]) -> Vec<usize> {
        let from = normalized(file);
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        let mut queue: VecDeque<usize> = self.resolve_all(&from, includes).into();
        while let Some(i) = queue.pop_front() {
            if order.len() >= MAX_CLOSURE {
                break;
            }
            if !seen.insert(i) {
                continue;
            }
            order.push(i);
            queue.extend(self.all[i].includes.iter().copied().filter(|j| !seen.contains(j)));
        }
        order
    }

    /// The headers a definition reaches: for every name it uses, the nearest header of `closure` that
    /// declares it. Sorted and deduplicated.
    pub(crate) fn reaches<'n>(&self, closure: &[usize], names: impl Iterator<Item = &'n String>) -> Vec<Arc<str>> {
        if closure.is_empty() {
            return Vec::new();
        }
        let rank: HashMap<usize, usize> = closure.iter().enumerate().map(|(r, &h)| (h, r)).collect();
        let mut out: Vec<Arc<str>> = names
            .filter_map(|name| {
                let declared = self.declared_in.get(name)?;
                let nearest = declared.iter().filter_map(|h| rank.get(h).map(|r| (*r, *h))).min()?;
                Some(Arc::clone(&self.all[nearest.1].reach))
            })
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// The include paths a source names — exposed for the per-file closure.
    pub(crate) fn includes_of(root: Node<'_>, src: &str) -> Vec<String> {
        include_paths(root, src)
    }
}

#[cfg(test)]
mod tests {
    use super::Headers;
    use crate::prep::Macros;
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn a_name_reaches_the_header_that_declares_it_through_the_includes() {
        let dir = std::env::temp_dir().join(format!("c-canon-headers-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("util")).expect("dir");
        std::fs::create_dir_all(dir.join("codec")).expect("dir");
        std::fs::write(dir.join("util/mem.h"), "void *av_malloc(unsigned long n);\n#define AV_MAX 4\n").expect("mem.h");
        std::fs::write(dir.join("util/frame.h"), "#include \"mem.h\"\ntypedef struct AVFrame { int w; } AVFrame;\n").expect("frame.h");
        std::fs::write(dir.join("codec/codec.h"), "#include \"util/frame.h\"\n#include <stdio.h>\nint decode(AVFrame *f);\n").expect("codec.h");
        let files: Vec<Arc<str>> = ["util/mem.h", "util/frame.h", "codec/codec.h"]
            .iter()
            .map(|f| Arc::from(dir.join(f).to_string_lossy().as_ref()))
            .collect();
        let headers = Headers::read(&dir, &files, &Macros::default());
        let closure = headers.closure(&dir.join("codec/dec.c"), &["codec.h".to_owned()]);
        let names = ["av_malloc", "AVFrame", "printf", "AV_MAX"].map(str::to_owned);
        let reached: Vec<String> = headers.reaches(&closure, names.iter()).iter().map(ToString::to_string).collect();
        // `printf` is the C library's: no header of the run declares it.
        assert_eq!(reached, vec!["util.frame", "util.mem"]);
        assert!(Path::new(&dir).exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
