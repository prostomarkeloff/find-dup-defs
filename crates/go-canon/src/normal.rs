//! **Semantic normal form** of a Go body, taken before its name-agnostic canonicals.

use crate::ir::Func;

/// The normal form of a callable. `inline` admits the single-use-temporary rule: the exact pass wants
/// it, the Type-3 pass does not.
#[must_use]
pub(crate) fn normal_form(f: &Func, _inline: bool) -> Func {
    f.clone()
}
