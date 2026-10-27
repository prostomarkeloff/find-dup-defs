//! What the preprocessor would have done before the parser looks — the part of it the parser cannot
//! do without, and nothing more.
//!
//! tree-sitter reads C *before* preprocessing, which is the only way it can be read at all without a
//! build configuration, and mostly it reads it well. Two habits of real C defeat it:
//!
//! * **Attribute macros.** `static av_cold int init(…)`, `int f(int x UNUSED)`, `G_BEGIN_DECLS`: a
//!   project spells its compiler attributes through macros that expand to `__attribute__((…))` or to
//!   nothing, and an unknown word where a type or a declarator belongs is a syntax error. Which words
//!   those are, the corpus says itself — every `#define` of them is an attribute or empty — so they
//!   are learned from it ([`Macros::learn`]), not listed.
//! * **`extern "C"` braces under `#ifdef __cplusplus`.** The opening brace sits in one conditional
//!   block and the closing one in another, so read without the preprocessor the brace never closes and
//!   the rest of the header is swallowed. A C compiler never sees either block.
//!
//! Both are removed by overwriting them with spaces — byte for byte, newlines kept — so every offset
//! the parser reports is still an offset into the file as written, and a definition's text and
//! position are read from the original.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

/// The words C reserves. A project may `#define` one away (`#define inline`, `#define const` for an
/// old compiler), but erasing a keyword everywhere would change what the parser sees far more than
/// the one configuration it was written for.
const KEYWORDS: &[&str] = &[
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern",
    "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "_Bool",
    "_Complex", "_Imaginary", "_Alignas", "_Alignof", "_Atomic", "_Generic", "_Noreturn", "_Static_assert",
    "_Thread_local", "bool", "true", "false", "NULL",
];

/// Calling conventions and qualifiers compilers provide without a header defining them, which the
/// grammar does not know in every position they are written in.
const BUILTIN: &[&str] = &["__cdecl", "__stdcall", "__fastcall", "__thiscall", "__vectorcall", "__w64", "__ptr32", "__ptr64"];

/// Tokens an attribute macro's expansion may consist of: attributes, the function specifiers, the
/// storage classes, and the linkage block an `extern "C"` wrapper opens or closes.
const ATTRIBUTE_WORDS: &[&str] = &[
    "inline", "__inline", "__inline__", "__forceinline", "static", "extern", "__extension__", "__cdecl",
    "__stdcall", "__fastcall", "__thread", "_Noreturn", "__restrict", "__restrict__",
];

/// The macros whose every definition is an attribute or nothing — the words erased before parsing.
#[derive(Debug, Default)]
pub(crate) struct Macros {
    /// Object-like: `av_cold`, `UNUSED`, `G_BEGIN_DECLS`.
    object: HashSet<String>,
    /// Function-like, erased with their arguments: `av_printf_format(1, 2)`.
    function: HashSet<String>,
}

/// One `#define`'s expansion, as far as attributes go.
enum Expansion {
    /// Nothing at all.
    Empty,
    /// Attributes, specifiers and linkage only.
    Attribute,
    /// Attributes and these other macros — an attribute macro if each of them is one.
    Through(Vec<String>),
    /// Anything else.
    Other,
}

impl Macros {
    /// Only the words no project defines for itself.
    pub(crate) fn builtin() -> Self {
        Self { object: BUILTIN.iter().map(|s| (*s).to_owned()).collect(), function: HashSet::new() }
    }

    /// The attribute macros a corpus defines: every name each of whose `#define`s expands to
    /// attributes, specifiers, a linkage brace or nothing — through other such macros, to a fixed
    /// point. A function-like macro counts only when some definition actually expands to an
    /// attribute: an empty `#define DBG(x)` is a statement that compiles to nothing, which the parser
    /// reads fine as the call it looks like.
    pub(crate) fn learn<'a>(sources: impl Iterator<Item = &'a str>) -> Self {
        let mut defs: HashMap<(String, bool), Vec<Expansion>> = HashMap::new();
        for src in sources {
            for (name, function_like, body) in defines(src) {
                defs.entry((name, function_like)).or_default().push(expansion(&body));
            }
        }
        defs.retain(|(name, _), _| !KEYWORDS.contains(&name.as_str()));
        let mut known: HashSet<String> = BUILTIN.iter().map(|s| (*s).to_owned()).collect();
        let mut out = Self::builtin();
        loop {
            let mut grew = false;
            for ((name, function_like), expansions) in &defs {
                let set = if *function_like { &out.function } else { &out.object };
                if set.contains(name) {
                    continue;
                }
                let all = expansions.iter().all(|e| match e {
                    Expansion::Empty | Expansion::Attribute => true,
                    Expansion::Through(refs) => refs.iter().all(|r| known.contains(r)),
                    Expansion::Other => false,
                });
                let substantive = !*function_like || expansions.iter().any(|e| !matches!(e, Expansion::Empty));
                if all && substantive {
                    known.insert(name.clone());
                    if *function_like {
                        out.function.insert(name.clone());
                    } else {
                        out.object.insert(name.clone());
                    }
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        out
    }

    /// Whether an object-like macro of this name is erased before parsing — an attribute, not a value.
    pub(crate) fn erases(&self, name: &str) -> bool {
        self.object.contains(name)
    }

    fn is_empty(&self) -> bool {
        self.object.is_empty() && self.function.is_empty()
    }
}

/// Every `#define` in a source: name, whether it takes arguments, and its expansion with comments
/// and line continuations removed.
fn defines(src: &str) -> Vec<(String, bool, String)> {
    let mut out = Vec::new();
    let mut lines = src.lines();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix('#') else { continue };
        let Some(rest) = rest.trim_start().strip_prefix("define") else { continue };
        if !rest.starts_with([' ', '\t']) {
            continue;
        }
        let mut logical = rest.to_owned();
        while logical.ends_with('\\') {
            logical.pop();
            match lines.next() {
                Some(next) => {
                    logical.push(' ');
                    logical.push_str(next);
                }
                None => break,
            }
        }
        let logical = strip_comments(&logical);
        let logical = logical.trim_start();
        let name_end = logical.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(logical.len());
        let name = &logical[..name_end];
        if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        let after = &logical[name_end..];
        // `#define F(x)` — the parenthesis touching the name is what makes a macro function-like.
        let (function_like, body) = match after.strip_prefix('(') {
            Some(params) => (true, params.split_once(')').map_or("", |(_, b)| b)),
            None => (false, after),
        };
        out.push((name.to_owned(), function_like, body.trim().to_owned()));
    }
    out
}

/// A line with its `/* … */` and `// …` comments removed (strings are not looked inside: an
/// attribute expansion holds none that matter).
fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        let block = rest.find("/*");
        let line = rest.find("//");
        match (block, line) {
            (Some(b), l) if l.is_none_or(|l| b < l) => {
                out.push_str(&rest[..b]);
                out.push(' ');
                rest = rest[b + 2..].split_once("*/").map_or("", |(_, after)| after);
            }
            (_, Some(l)) => {
                out.push_str(&rest[..l]);
                return out;
            }
            _ => {
                out.push_str(rest);
                return out;
            }
        }
    }
}

/// What a macro's expansion is, as far as attributes go.
fn expansion(body: &str) -> Expansion {
    let mut refs = Vec::new();
    let mut tokens = tokens(body).into_iter().peekable();
    if tokens.peek().is_none() {
        return Expansion::Empty;
    }
    while let Some(tok) = tokens.next() {
        match tok.as_str() {
            "__attribute__" | "__attribute" | "__declspec" | "__pragma" | "_Pragma" | "alignas" | "_Alignas" => {
                // The argument list is the attribute's; it says nothing about whether this is one.
                if tokens.peek().map(String::as_str) != Some("(") {
                    return Expansion::Other;
                }
                let mut depth = 0usize;
                for t in tokens.by_ref() {
                    match t.as_str() {
                        "(" => depth += 1,
                        ")" => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
            // `extern "C" {` and the brace that closes it.
            "\"C\"" | "{" | "}" => {}
            word if ATTRIBUTE_WORDS.contains(&word) => {}
            word if word.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') && !KEYWORDS.contains(&word) => {
                refs.push(word.to_owned());
            }
            _ => return Expansion::Other,
        }
    }
    if refs.is_empty() {
        Expansion::Attribute
    } else {
        Expansion::Through(refs)
    }
}

/// A crude tokenizer for macro expansions: identifiers, string literals, and single punctuation.
fn tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c.is_whitespace() {
            continue;
        }
        if c.is_ascii_alphanumeric() || c == '_' {
            let mut end = i + c.len_utf8();
            while let Some(&(j, d)) = chars.peek() {
                if d.is_ascii_alphanumeric() || d == '_' {
                    end = j + d.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            out.push(s[i..end].to_owned());
        } else if c == '"' {
            let mut end = s.len();
            let mut escaped = false;
            for (j, d) in chars.by_ref() {
                if escaped {
                    escaped = false;
                } else if d == '\\' {
                    escaped = true;
                } else if d == '"' {
                    end = j + 1;
                    break;
                }
            }
            out.push(s[i..end].to_owned());
        } else {
            out.push(c.to_string());
        }
    }
    out
}

/// The source as the parser should read it: attribute macros and `extern "C"` wrappers overwritten
/// with spaces, every other byte where it was. Borrowed when there is nothing to erase.
pub(crate) fn prepare<'s>(src: &'s str, macros: &Macros) -> Cow<'s, str> {
    let mut spans: Vec<(usize, usize)> = cplusplus_blocks(src);
    if !macros.is_empty() {
        spans.extend(macro_uses(src, macros));
    }
    if spans.is_empty() {
        return Cow::Borrowed(src);
    }
    let mut bytes = src.as_bytes().to_vec();
    for (start, end) in spans {
        for b in &mut bytes[start..end] {
            // Whole characters only ever fall inside a span, so writing every byte of one keeps the
            // text valid UTF-8; a newline stays, so line numbers do too.
            if *b != b'\n' {
                *b = b' ';
            }
        }
    }
    Cow::Owned(String::from_utf8(bytes).unwrap_or_else(|_| src.to_owned()))
}

/// The `#ifdef __cplusplus` blocks that hold nothing but an `extern "C" {` or its closing `}`, each
/// with its directives, as byte spans.
fn cplusplus_blocks(src: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if !src.contains("__cplusplus") {
        return out;
    }
    let mut starts = Vec::new();
    let mut at = 0usize;
    for line in src.split_inclusive('\n') {
        starts.push((at, line));
        at += line.len();
    }
    let directive = |line: &str| -> Option<String> {
        let t = line.trim_start().strip_prefix('#')?;
        Some(t.split_whitespace().collect::<Vec<_>>().join(" "))
    };
    let mut i = 0;
    while i < starts.len() {
        let opens = directive(starts[i].1).is_some_and(|d| {
            d == "ifdef __cplusplus" || d == "if defined(__cplusplus)" || d == "if defined __cplusplus"
        });
        if !opens {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        let mut only_linkage = true;
        while j < starts.len() {
            let line = starts[j].1;
            if let Some(d) = directive(line) {
                if d.starts_with("endif") {
                    break;
                }
                only_linkage = false;
                break;
            }
            let code = strip_comments(line);
            let code: String = code.split_whitespace().collect();
            if !(code.is_empty() || code == "extern\"C\"{" || code == "}") {
                only_linkage = false;
            }
            j += 1;
        }
        if only_linkage && j < starts.len() {
            let end = starts[j].0 + starts[j].1.len();
            out.push((starts[i].0, end));
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Every use of an attribute macro in code — not in a directive, a comment, a string or a character
/// literal — with a function-like macro's argument list.
fn macro_uses(src: &str, macros: &Macros) -> Vec<(usize, usize)> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut line_start = true;
    while i < b.len() {
        let c = b[i];
        match c {
            b'\n' => {
                line_start = true;
                i += 1;
            }
            b' ' | b'\t' | b'\r' | b'\x0c' => i += 1,
            b'#' if line_start => {
                // A directive runs to the end of its line, continuations included.
                while i < b.len() && b[i] != b'\n' {
                    if b[i] == b'\\' && b.get(i + 1) == Some(&b'\n') {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i += 2;
            }
            b'"' | b'\'' => {
                line_start = false;
                i = skip_quoted(b, i);
            }
            // A number's suffix and hex digits are not words.
            c if c.is_ascii_digit() => {
                line_start = false;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.') {
                    i += 1;
                }
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                line_start = false;
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                let word = &src[start..i];
                if macros.object.contains(word) {
                    out.push((start, i));
                } else if macros.function.contains(word) {
                    let mut j = i;
                    while j < b.len() && matches!(b[j], b' ' | b'\t') {
                        j += 1;
                    }
                    if b.get(j) == Some(&b'(') {
                        let end = skip_parens(b, j);
                        out.push((start, end));
                        i = end;
                    }
                }
            }
            _ => {
                line_start = false;
                i += 1;
            }
        }
    }
    out
}

/// Past a string or character literal starting at `i`.
fn skip_quoted(b: &[u8], i: usize) -> usize {
    let quote = b[i];
    let mut j = i + 1;
    while j < b.len() && b[j] != quote && b[j] != b'\n' {
        if b[j] == b'\\' {
            j += 1;
        }
        j += 1;
    }
    (j + 1).min(b.len())
}

/// Past a balanced parenthesized list starting at `i`, literals inside skipped.
fn skip_parens(b: &[u8], i: usize) -> usize {
    let mut depth = 0usize;
    let mut j = i;
    while j < b.len() {
        match b[j] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return j + 1;
                }
            }
            b'"' | b'\'' => {
                j = skip_quoted(b, j);
                continue;
            }
            _ => {}
        }
        j += 1;
    }
    b.len()
}

#[cfg(test)]
mod tests {
    use super::{prepare, Macros};

    #[test]
    fn attribute_macros_are_learned_from_their_definitions() {
        let header = "#ifdef __GNUC__\n#define av_cold __attribute__((cold))\n#define av_always_inline __attribute__((always_inline)) inline\n#define av_printf_format(a, b) __attribute__((__format__(__printf__, a, b)))\n#else\n#define av_cold\n#define av_always_inline inline\n#define av_printf_format(a, b)\n#endif\n#define UNUSED(x) (void)(x)\n#define MAX_LEN 4096\n#define G_BEGIN_DECLS extern \"C\" {\n#define LOCAL_COLD av_cold static\n#define DBG(x)\n";
        let m = Macros::learn(std::iter::once(header));
        for name in ["av_cold", "av_always_inline", "G_BEGIN_DECLS", "LOCAL_COLD"] {
            assert!(m.object.contains(name), "{name}: {m:?}");
        }
        assert!(m.function.contains("av_printf_format"), "{m:?}");
        // A statement macro, a value, and a no-op call are not attributes.
        assert!(!m.function.contains("UNUSED") && !m.object.contains("MAX_LEN") && !m.function.contains("DBG"), "{m:?}");
    }

    #[test]
    fn erasing_keeps_every_offset() {
        let m = Macros::learn(std::iter::once("#define av_cold __attribute__((cold))\n#define fmt(a) __attribute__((format(a)))\n"));
        let src = "static av_cold int f(int x) fmt(1);\n/* av_cold */ const char *s = \"av_cold\";\n#if av_cold\n";
        let out = prepare(src, &m);
        assert_eq!(out.len(), src.len());
        assert_eq!(out.lines().next(), Some("static         int f(int x)       ;"));
        assert!(out.contains("/* av_cold */") && out.contains("\"av_cold\"") && out.contains("#if av_cold"), "{out}");
    }

    #[test]
    fn an_extern_c_wrapper_is_erased_with_its_directives() {
        let src = "#ifdef __cplusplus\nextern \"C\" {\n#endif\nint f(void);\n#ifdef __cplusplus\n}\n#endif\n";
        let out = prepare(src, &Macros::default());
        assert_eq!(out.trim(), "int f(void);");
        // A block that does more than open the linkage is the program's.
        let real = "#ifdef __cplusplus\n#define EXT extern \"C\"\n#endif\n";
        assert_eq!(prepare(real, &Macros::default()), real);
    }
}
