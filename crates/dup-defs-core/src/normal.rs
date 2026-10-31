//! What the normal-form rewrites of every frontend share: how long they run, and the name facts they
//! read a body by. The rules themselves walk each language's own syntax tree and live in its frontend
//! (`py-canon`, `rs-canon`, `ts-canon`, each a `normal` module).

use std::collections::{HashMap, HashSet};

/// Rounds of a normal-form rewrite before it is cut off. Every rule either shrinks the tree or fixes
/// an orientation no rule undoes, so the fixed point comes in a handful of rounds; the bound only
/// guards against a rule pair that would disagree.
pub const MAX_ROUNDS: usize = 12;

/// The ordering a comparison's negation is: `!(a < b)` is `a >= b`. `None` for any other operator.
///
/// The form is compared, never run, so a NaN's opinion of `!(a < b)` does not enter into it.
#[must_use]
pub fn complement(op: &str) -> Option<&'static str> {
    Some(match op {
        "<" => ">=",
        ">=" => "<",
        ">" => "<=",
        "<=" => ">",
        _ => return None,
    })
}

/// The operator a comparison becomes with its operands swapped: `a < b` is `b > a`; `==` and `!=` are
/// their own mirrors. `None` for an operator that has none.
#[must_use]
pub fn mirror(op: &str) -> Option<&'static str> {
    Some(match op {
        "==" => "==",
        "!=" => "!=",
        "<" => ">",
        ">" => "<",
        "<=" => ">=",
        ">=" => "<=",
        _ => return None,
    })
}

/// What a definition binds and reads, counted once per round of the rewrite.
#[derive(Default)]
pub struct Scope {
    /// Names the definition binds anywhere (parameters included).
    pub locals: HashSet<String>,
    pub reads: HashMap<String, usize>,
    pub stores: HashMap<String, usize>,
    /// Reads from inside a nested function — run later, it may see a later binding, so a temporary
    /// read there is not inlined.
    pub nested_reads: HashSet<String>,
    /// Names nothing may be assumed about: what Python's `global`, `nonlocal` and `del` name, what a
    /// Rust macro's tokens mention.
    pub pinned: HashSet<String>,
}

impl Scope {
    /// A binding of `name`: a local, stored once more.
    pub fn bind(&mut self, name: &str) {
        if !self.locals.contains(name) {
            self.locals.insert(name.to_owned());
        }
        self.store(name);
    }

    /// A store to `name` that binds nothing new — an assignment.
    pub fn store(&mut self, name: &str) {
        if let Some(n) = self.stores.get_mut(name) {
            *n += 1;
        } else {
            self.stores.insert(name.to_owned(), 1);
        }
    }

    /// A read of `name`, from inside a nested function when `nested`.
    pub fn read(&mut self, name: &str, nested: bool) {
        if let Some(n) = self.reads.get_mut(name) {
            *n += 1;
        } else {
            self.reads.insert(name.to_owned(), 1);
        }
        if nested && !self.nested_reads.contains(name) {
            self.nested_reads.insert(name.to_owned());
        }
    }

    #[must_use]
    pub fn reads(&self, name: &str) -> usize {
        self.reads.get(name).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn stores(&self, name: &str) -> usize {
        self.stores.get(name).copied().unwrap_or(0)
    }
}
