//! Byte-offset → line/column mapping. Used by every frontend (ruff's parser ships no bundled
//! `SourceCode`, oxc exposes byte spans the same way), so it lives here in the shared core.
//!
//! Columns are counted in **characters** (Unicode scalar values), 1-indexed in `loc1` — the
//! convention rustpython's `SourceLocation` used, preserved for backward compatibility.

/// Precomputed line-start offsets for one source string (`starts[i]` = byte offset of line `i`).
pub struct LineMap<'a> {
    src: &'a str,
    starts: Vec<usize>,
}

impl<'a> LineMap<'a> {
    #[must_use]
    pub fn new(src: &'a str) -> Self {
        let mut starts = vec![0usize];
        for (i, byte) in src.bytes().enumerate() {
            if byte == b'\n' {
                starts.push(i + 1);
            }
        }
        Self { src, starts }
    }

    /// 0-indexed line containing `offset`.
    fn line_index(&self, offset: usize) -> usize {
        match self.starts.binary_search(&offset) {
            Ok(line) => line,
            Err(next) => next - 1, // `next` is the first start > offset; the line is the one before
        }
    }

    /// 1-indexed `(line, column)` with column counted in characters — rustpython parity.
    #[must_use]
    pub fn loc1(&self, offset: usize) -> (usize, usize) {
        let line = self.line_index(offset);
        let col = self.src.get(self.starts[line]..offset).map_or(0, |s| s.chars().count());
        (line + 1, col + 1)
    }

    /// 0-indexed `(line, column)` — the convention [`crate::Def`] reports.
    #[must_use]
    pub fn loc0(&self, offset: usize) -> (usize, usize) {
        let line = self.line_index(offset);
        let col = self.src.get(self.starts[line]..offset).map_or(0, |s| s.chars().count());
        (line, col)
    }
}
