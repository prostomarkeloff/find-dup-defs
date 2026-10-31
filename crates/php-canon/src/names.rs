//! What a name in a PHP file stands for: its namespace, and the names its `use` statements import.
//!
//! PHP resolves a name at compile time, against the file alone:
//!
//! * a **fully qualified** name (`\App\Models\Order`) is itself;
//! * a **qualified** one (`Models\Order`) is read through an import of its first segment, or under the
//!   current namespace;
//! * an **unqualified class** name is an import's alias, or a class of the current namespace;
//! * an **unqualified function or constant** is an import's alias (`use function`, `use const`), or —
//!   the interpreter tries the namespace first and falls back to the global one at run time — a global
//!   one: nearly every such call is to the language's own library (`strlen`, `PHP_EOL`), which a
//!   namespace does not redefine.
//!
//! A name is written here as PHP prints it, without the leading `\`; [`dotted`] gives the engine's form.
//! What a definition **reaches** is every name it uses that the file *imported* — the file's own
//! declaration of what it is about — and every one it wrote fully qualified, which says the same thing
//! inline.

use std::collections::HashMap;

use tree_sitter::Node;

use crate::canon::NameKind;
use crate::tree::{field, kids, name_of, text};

/// One namespace block of a file: its name and its imports, by kind, keyed by lowercased alias (PHP's
/// names are case-insensitive).
#[derive(Clone, Debug, Default)]
pub(crate) struct Names {
    pub namespace: String,
    classes: HashMap<String, String>,
    functions: HashMap<String, String>,
    constants: HashMap<String, String>,
}

impl Names {
    pub(crate) fn new(namespace: String) -> Self {
        Self { namespace, ..Self::default() }
    }

    /// Add the imports one `use` declaration makes.
    pub(crate) fn add_use(&mut self, decl: Node<'_>, src: &str) {
        let kind_of = |n: Node<'_>| field(n, "type").map(|t| text(t, src).to_ascii_lowercase());
        let decl_kind = kind_of(decl);
        // `use A\{B, C as D};` — a group shares its prefix.
        let (prefix, clauses) = match field(decl, "body") {
            Some(group) => {
                let prefix = kids(decl).into_iter().find(|k| k.kind() == "namespace_name").map(|p| name_of(p, src)).unwrap_or_default();
                (prefix, kids(group))
            }
            None => (String::new(), kids(decl)),
        };
        for clause in clauses.into_iter().filter(|c| c.kind() == "namespace_use_clause") {
            let alias = field(clause, "alias").map(|a| text(a, src).to_owned());
            let Some(target) = kids(clause).into_iter().find(|k| matches!(k.kind(), "name" | "qualified_name") && Some(k.id()) != field(clause, "alias").map(|a| a.id())) else {
                continue;
            };
            let spelled = name_of(target, src);
            let full = if prefix.is_empty() { spelled.trim_start_matches('\\').to_owned() } else { format!("{prefix}\\{spelled}") };
            let alias = alias.unwrap_or_else(|| full.rsplit('\\').next().unwrap_or(&full).to_owned()).to_ascii_lowercase();
            let table = match kind_of(clause).or_else(|| decl_kind.clone()).as_deref() {
                Some("function") => &mut self.functions,
                Some("const") => &mut self.constants,
                _ => &mut self.classes,
            };
            table.insert(alias, full);
        }
    }

    /// What a name of `kind` stands for, fully qualified, and whether the file **imported** it or wrote
    /// it fully qualified — the names a definition reaches.
    pub(crate) fn resolve(&self, kind: NameKind, spelled: &str) -> (String, bool) {
        if let Some(full) = spelled.strip_prefix('\\') {
            return (full.to_owned(), full.contains('\\'));
        }
        if let Some(rest) = spelled.strip_prefix("namespace\\") {
            return (self.under_namespace(rest), false);
        }
        // `Models\Order` — its first segment may be an imported namespace.
        if let Some((head, rest)) = spelled.split_once('\\') {
            return match self.classes.get(&head.to_ascii_lowercase()) {
                Some(full) => (format!("{full}\\{rest}"), true),
                None => (self.under_namespace(spelled), false),
            };
        }
        let table = match kind {
            NameKind::Class => &self.classes,
            NameKind::Function => &self.functions,
            NameKind::Constant => &self.constants,
        };
        match table.get(&spelled.to_ascii_lowercase()) {
            Some(full) => (full.clone(), true),
            None if kind == NameKind::Class => (self.under_namespace(spelled), false),
            None => (spelled.to_owned(), false),
        }
    }

    fn under_namespace(&self, name: &str) -> String {
        if self.namespace.is_empty() {
            name.to_owned()
        } else {
            format!("{}\\{name}", self.namespace)
        }
    }

    /// A name this namespace declares, fully qualified.
    pub(crate) fn declared(&self, name: &str) -> String {
        self.under_namespace(name)
    }
}

/// A fully qualified name in the engine's dotted form: `App\Models\Order` → `App.Models.Order`.
pub(crate) fn dotted(full: &str) -> String {
    full.trim_start_matches('\\').replace('\\', ".")
}

/// The namespace blocks of a file in order, each with the top-level statements it holds: a braced
/// `namespace X { … }` holds its body, a `namespace X;` everything up to the next one. Every block's
/// imports are read before its definitions, so a definition sees them all. `facets` false skips the
/// import-table build: the `Names` tables are what the perspective passes read, so a run that never
/// reads facets need not resolve a single `use`.
pub(crate) fn blocks<'t>(root: Node<'t>, src: &str, facets: bool) -> Vec<(Names, Vec<Node<'t>>)> {
    let mut out: Vec<(Names, Vec<Node<'t>>)> = vec![(Names::default(), Vec::new())];
    for item in kids(root) {
        if matches!(item.kind(), "php_tag" | "php_end_tag" | "text" | "empty_statement") {
            continue;
        }
        if item.kind() == "namespace_definition" {
            let name = field(item, "name").map(|n| name_of(n, src)).unwrap_or_default();
            match field(item, "body") {
                Some(body) => {
                    out.push((Names::new(name), kids(body)));
                    // Code after a braced block is in the global namespace again.
                    out.push((Names::default(), Vec::new()));
                }
                None => out.push((Names::new(name), Vec::new())),
            }
        } else if let Some(last) = out.last_mut() {
            last.1.push(item);
        }
    }
    out.retain(|(_, items)| !items.is_empty());
    if facets {
        for (names, items) in &mut out {
            for item in items.iter().filter(|i| i.kind() == "namespace_use_declaration") {
                names.add_use(*item, src);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{blocks, dotted};
    use crate::canon::NameKind;
    use crate::tree::parse;

    #[test]
    fn a_name_is_read_through_the_file_s_imports_and_namespace() {
        let src = "<?php\nnamespace App\\Http;\nuse App\\Models\\{Order, User as Member};\nuse Psr\\Log;\nuse function App\\Support\\fmt;\nfunction f() {}\n";
        let tree = parse(src).expect("parse");
        let found = blocks(tree.root_node(), src, true);
        let (names, _) = &found[0];
        assert_eq!(names.namespace, "App\\Http");
        assert_eq!(names.resolve(NameKind::Class, "Order"), ("App\\Models\\Order".to_owned(), true));
        assert_eq!(names.resolve(NameKind::Class, "Member"), ("App\\Models\\User".to_owned(), true));
        assert_eq!(names.resolve(NameKind::Class, "Log\\LoggerInterface"), ("Psr\\Log\\LoggerInterface".to_owned(), true));
        assert_eq!(names.resolve(NameKind::Class, "Request"), ("App\\Http\\Request".to_owned(), false));
        assert_eq!(names.resolve(NameKind::Function, "fmt"), ("App\\Support\\fmt".to_owned(), true));
        assert_eq!(names.resolve(NameKind::Function, "strlen"), ("strlen".to_owned(), false), "the global library");
        assert_eq!(names.resolve(NameKind::Class, "\\DateTime"), ("DateTime".to_owned(), false), "a global class reaches nothing");
        assert_eq!(dotted("App\\Models\\Order"), "App.Models.Order");
    }
}
