//! Build configurations: which definitions are one definition written for two configurations.
//!
//! C chooses between implementations two ways, and neither is visible to a reader of one file as a
//! choice:
//!
//! * **Within a file**, by the preprocessor: `#ifdef _WIN32 int f(…) {…} #else int f(…) {…} #endif`.
//!   A translation unit cannot define one name twice, so a name a file defines more than once is
//!   defined once per configuration — the file's first definition of it is reported
//!   ([`crate::defs`] keeps it).
//! * **Across files**, by the build: `os/unix/process.c` and `os/win32/process.c`,
//!   `x86/dsp_init.c` and `arm/dsp_init.c`, `ae_epoll.c` and `ae_kqueue.c` — one is compiled, the
//!   others are not, and which is the build system's to say. A file is read as one such alternative
//!   when a directory on its path or the last word of its name is a platform, an architecture or an
//!   event backend; a name two alternatives of one unit both define is reported once, as the Go
//!   frontend reports a name two `_GOOS` files of one package declare. Reported twice, they would be
//!   the commonest duplicate a portable C tree has, and not one a refactor removes.

use std::collections::HashSet;
use std::path::Path;

use dup_defs_core::Def;

/// Directory names and file-name suffixes that select a platform, an architecture, or an I/O backend.
const ALTERNATIVES: &[&str] = &[
    // operating systems
    "win32", "win64", "windows", "win", "mingw", "cygwin", "msvc", "unix", "posix", "linux", "darwin", "macos", "osx",
    "mac", "ios", "android", "freebsd", "openbsd", "netbsd", "dragonfly", "bsd", "solaris", "sunos", "illumos", "aix",
    "hpux", "haiku", "qnx", "vms", "os2", "beos", "emscripten", "wasm", "wasi", "msdos", "dos", "amiga", "nt",
    "zos", "fuchsia",
    // architectures and their vector units
    "x86", "x86_64", "x64", "i386", "i686", "amd64", "ia32", "arm", "arm64", "aarch64", "armv7", "neon", "mips",
    "mips64", "ppc", "ppc64", "powerpc", "riscv", "riscv64", "rvv", "s390", "s390x", "sparc", "loongarch",
    "loongarch64", "alpha", "ia64", "sh4", "m68k", "avr32", "bfin", "sse", "sse2", "sse4", "avx", "avx2",
    "avx512", "msa", "mmi", "lsx", "lasx", "vsx", "altivec", "sve",
    // event backends
    "epoll", "kqueue", "devpoll", "evport", "iocp", "eventport",
];

fn is_alternative(word: &str) -> bool {
    ALTERNATIVES.contains(&word.to_ascii_lowercase().as_str())
}

/// The unit a file is an alternative within, or `None` for a file every build compiles: the path up
/// to its platform directory (`src/os` for `src/os/unix/process.c`), or its own directory when the
/// platform is the last word of its name (`src` for `src/ae_epoll.c`).
pub(crate) fn alternative_unit(file: &str) -> Option<String> {
    let path = Path::new(file);
    let dir = path.parent()?;
    // The outermost platform directory on the path: `src/os/unix/x.c` is an alternative within `src/os`.
    let platform = dir.ancestors().filter(|a| a.file_name().and_then(|n| n.to_str()).is_some_and(is_alternative)).last();
    if let Some(platform) = platform {
        return Some(spelled(platform.parent().unwrap_or(Path::new(""))));
    }
    let stem = path.file_stem()?.to_str()?;
    let last = stem.rsplit(['_', '-']).next()?;
    (last.len() < stem.len() && is_alternative(last)).then(|| spelled(dir))
}

/// A directory as the units are compared: its path, `/`-separated.
fn spelled(dir: &Path) -> String {
    dir.to_string_lossy().replace('\\', "/")
}

/// One definition per name per unit among the files built only in some configurations: the first as
/// the files were handed, the rest dropped. `unit_of` names a file's unit when it is such a file.
///
/// A unit's own directory may hold the portable definition the alternatives override
/// (`libavutil/bswap.h` beside `libavutil/arm/bswap.h`, guarded by `#ifndef av_bswap32`): then the
/// portable one is the definition, and every alternative of it is another configuration's.
pub(crate) fn collapse_alternatives(defs: Vec<Def>, unit_of: impl Fn(&str) -> Option<String>) -> Vec<Def> {
    let dir_of = |file: &str| Path::new(file).parent().map(spelled).unwrap_or_default();
    let portable: HashSet<(String, &str, &str)> = defs
        .iter()
        .filter(|d| unit_of(&d.file).is_none())
        .map(|d| (dir_of(&d.file), d.kind.id, d.name.as_str()))
        .collect();
    let mut seen = HashSet::new();
    let keep: Vec<bool> = defs
        .iter()
        .map(|d| match unit_of(&d.file) {
            Some(unit) => {
                !portable.contains(&(unit.clone(), d.kind.id, d.name.as_str())) && seen.insert((unit, d.kind.id, d.name.clone()))
            }
            None => true,
        })
        .collect();
    defs.into_iter().zip(keep).filter_map(|(d, k)| k.then_some(d)).collect()
}


#[cfg(test)]
mod tests {
    use super::alternative_unit;

    #[test]
    fn a_platform_is_read_off_a_directory_or_the_last_word_of_a_name() {
        assert_eq!(alternative_unit("src/os/unix/ngx_process.c"), Some("src/os".to_owned()));
        assert_eq!(alternative_unit("src/os/win32/ngx_process.c"), Some("src/os".to_owned()));
        assert_eq!(alternative_unit("libavcodec/x86/h264dsp_init.c"), Some("libavcodec".to_owned()));
        assert_eq!(alternative_unit("src/ae_epoll.c"), Some("src".to_owned()));
        assert_eq!(alternative_unit("src/ae_kqueue.c"), Some("src".to_owned()));
        assert_eq!(alternative_unit("src/ae.c"), None);
        assert_eq!(alternative_unit("src/linux.c"), None, "a file named for a platform is not built for it");
        assert_eq!(alternative_unit("lib/armor.c"), None);
        assert_eq!(alternative_unit("/abs/src/os/win32/x.c"), Some("/abs/src/os".to_owned()), "an absolute path keeps its root");
    }

    #[test]
    fn a_portable_definition_beside_its_overrides_is_the_definition() {
        use dup_defs_core::{kinds::FUNCTIONS, Def, Facets};
        use std::sync::Arc;
        let def = |file: &str| Def {
            lang: "c",
            kind: &FUNCTIONS,
            name: "av_bswap32".to_owned(),
            file: Arc::from(file),
            line: 0,
            col: 0,
            loc: 3,
            args: 1,
            text_orig: String::new(),
            cluster_canonical: None,
            analysis: None,
            thickness: None,
            facets: Facets::default(),
        };
        let defs = vec![def("lib/arm/bswap.h"), def("lib/x86/bswap.h"), def("lib/bswap.h"), def("other/bswap.h")];
        let kept: Vec<String> = super::collapse_alternatives(defs, alternative_unit).into_iter().map(|d| d.file.to_string()).collect();
        assert_eq!(kept, vec!["lib/bswap.h".to_owned(), "other/bswap.h".to_owned()]);
    }
}
