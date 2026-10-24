//! Build constraints: which definitions are one definition written for two configurations.
//!
//! A Go package may declare a name once per build, and a file can opt out of builds — by a
//! `//go:build` line before its `package` clause, or by a `_GOOS` / `_GOARCH` suffix on its name.
//! So two files of one package that both declare `Default` are the same definition written for two
//! configurations (`binding.go` under `!nomsgpack`, `binding_nomsgpack.go` under `nomsgpack`), and
//! only one of them is ever compiled — the Go reading of Rust's `#[cfg(unix)] fn x` beside
//! `#[cfg(windows)] fn x`, which the Rust frontend reports once. Reported twice, they would be the
//! commonest duplicate a Go tree has, and not one a refactor removes.
//!
//! A file under `//go:build ignore` is no build at all — conventionally a standalone program run by
//! `go run`, several of which share a directory — so it stays out of this.

use std::collections::HashSet;
use std::path::Path;

use dup_defs_core::Def;

/// Every `GOOS` a file name may end in (`go tool dist list`, plus the names the toolchain still
/// recognizes in file suffixes).
const GOOS: &[&str] = &[
    "aix", "android", "darwin", "dragonfly", "freebsd", "hurd", "illumos", "ios", "js", "linux", "nacl", "netbsd",
    "openbsd", "plan9", "solaris", "wasip1", "windows", "zos",
];

/// Every `GOARCH` a file name may end in.
const GOARCH: &[&str] = &[
    "386", "amd64", "amd64p32", "arm", "armbe", "arm64", "arm64be", "loong64", "mips", "mipsle", "mips64",
    "mips64le", "mips64p32", "mips64p32le", "ppc", "ppc64", "ppc64le", "riscv", "riscv64", "s390", "s390x",
    "sparc", "sparc64", "wasm",
];

/// What a file's build constraint makes of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Build {
    /// Built in every configuration.
    Always,
    /// Built in some configurations only.
    Constrained,
    /// Built in none: a standalone program (`//go:build ignore`).
    Ignored,
}

/// The constraint a file's header and name put on it.
fn build_of(src: &str, file: &str) -> Build {
    for line in src.lines() {
        let line = line.trim();
        if line.starts_with("package ") {
            break;
        }
        let expr = line.strip_prefix("//go:build ").or_else(|| line.strip_prefix("// +build "));
        if let Some(expr) = expr {
            let tags = expr.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '!' || c == '.'));
            return if tags.into_iter().any(|t| t == "ignore") { Build::Ignored } else { Build::Constrained };
        }
    }
    let stem = Path::new(file).file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let stem = stem.strip_suffix("_test").unwrap_or(stem);
    let mut parts = stem.rsplit('_');
    let last = parts.next().unwrap_or("");
    let before = parts.next();
    // `name_GOOS`, `name_GOARCH`, `name_GOOS_GOARCH` — a bare `linux.go` is just a name.
    let suffixed = before.is_some() && (GOOS.contains(&last) || GOARCH.contains(&last));
    if suffixed { Build::Constrained } else { Build::Always }
}

/// The package a constrained file belongs to — its directory and its `package` clause — or `None`
/// for a file every build compiles, or none does.
pub(crate) fn alternative_unit(src: &str, file: &str) -> Option<(String, String)> {
    if build_of(src, file) != Build::Constrained {
        return None;
    }
    let package = src.lines().find_map(|l| l.trim().strip_prefix("package "))?.split_whitespace().next()?.to_owned();
    let dir = Path::new(file).parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
    Some((dir, package))
}

/// One definition per name per package among the files built only in some configurations: the first
/// as the files were handed, the rest dropped. `unit_of` names a file's package when it is such a
/// file. Every `init` stays — a package may hold as many as it likes, and none is another's alternative.
pub(crate) fn collapse_alternatives(defs: Vec<Def>, unit_of: impl Fn(&str) -> Option<(String, String)>) -> Vec<Def> {
    let mut seen = HashSet::new();
    defs.into_iter()
        .filter(|d| match unit_of(&d.file) {
            Some(unit) if d.name != "init" => seen.insert((unit, d.kind.id, d.name.clone())),
            _ => true,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{alternative_unit, build_of, Build};

    #[test]
    fn a_constraint_is_read_off_the_header_or_the_name() {
        assert_eq!(build_of("//go:build !nomsgpack\n\npackage binding\n", "binding/binding.go"), Build::Constrained);
        assert_eq!(build_of("// +build linux\n\npackage p\n", "p/x.go"), Build::Constrained);
        assert_eq!(build_of("package p\n", "p/poll_windows.go"), Build::Constrained);
        assert_eq!(build_of("package p\n", "p/cpu_linux_arm64_test.go"), Build::Constrained);
        assert_eq!(build_of("package p\n", "p/linux.go"), Build::Always, "a file named for a platform is not built for it");
        assert_eq!(build_of("package p\n//go:build linux\n", "p/x.go"), Build::Always, "only before the package clause");
        assert_eq!(build_of("//go:build ignore\n\npackage main\n", "gen/gen.go"), Build::Ignored);
    }

    #[test]
    fn a_unit_is_the_directory_and_the_package_clause() {
        let unit = alternative_unit("//go:build windows\n\npackage poll_test\n", "a/b/x_windows_test.go");
        assert_eq!(unit, Some(("a/b".to_owned(), "poll_test".to_owned())));
        assert_eq!(alternative_unit("package p\n", "a/b/x.go"), None);
    }
}
