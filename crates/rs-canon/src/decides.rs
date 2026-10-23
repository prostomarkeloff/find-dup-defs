//! What a Rust definition **decides on** — the frontend half of [`Facets::decides`].
//!
//! Rust says which expressions are propositions, and this module only reads them off — by the same
//! rules the Python and TypeScript frontends read their own, written in the atom grammar all three
//! share ([`dup_defs_core::decide`]):
//!
//! - every **comparison**, wherever it stands — `==` and `!=` as one relation, the four orders as
//!   another;
//! - every expression in a position the language **tests for truth** — the condition of an `if` or a
//!   `while`, the operand of `!`, the left of `&&` and `||` (and the right when the whole is tested),
//!   a `match` guard, an `assert!`, the predicate handed to `filter` / `find` / `any` / `all` and kin;
//! - every **pattern** a value is tested against — a `match` arm, `if let`, `while let`, `let … else`,
//!   `matches!` — as the comparison it abbreviates: a unit variant or a constant is equality, a variant
//!   with a payload or a struct pattern is membership (Python's `case Point(x=0)`), a range is the two
//!   orders that bound it;
//! - `assert_eq!` / `assert_ne!` as the equality they assert, `.contains` / `.contains_key` as
//!   membership (`[a, b].contains(&x)` as the equalities it abbreviates).
//!
//! **Absence and failure are one test each, however spelled.** `if let Some(x) = o`, `o.is_some()`,
//! `o.is_none()`, `matches!(o, None)` and `o == None` all ask whether `o` is absent — `None = o`, the
//! atom Python's `o is None` and TypeScript's `o === undefined` write. An `Ok(_)` or `Err(_)` pattern,
//! `.is_ok()` and `.is_err()` all ask whether a result failed: `r in @Err`. A `while let Some(x) = …`
//! is iteration — Python's `for x in it` — not a test of absence, and `?` hands an error back rather
//! than deciding on it.
//!
//! A term's identity follows the binding the language gives the name it starts from: a name the
//! definition binds is a hole; a `use`d name is the path it was imported from; a name the module
//! defines is the module's path and that name; `crate::`, `self::`, `super::` and `Self` are resolved
//! where they stand. Paths are crate-qualified. A file under a package's `src/` is the module its
//! place there names, under the package directory's name read as a crate name (`actix-web` is
//! `actix_web`, the name `cargo new` gives it) — so `use actix_web::…` in one crate and the definition
//! in the other meet as one path. Unlike Python's, a Rust path needs no source root learned from the
//! corpus: `crate::` already says where the root is.
//!
//! [`Facets::decides`]: dup_defs_core::Facets::decides

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::decide::{self, DecisionFacts};
use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{BinOp, Expr, Fields, FnArg, Item, Lit, Member, Pat, Stmt, Token, Type, UnOp, UseTree};

/// What the names of one file stand for, read once per file and shared by its definitions.
pub(crate) struct FileScope {
    /// Each `use`d name, as the dotted path it stands for.
    imports: HashMap<String, String>,
    /// The file's own module, crate-qualified and dotted.
    module: String,
    /// The module `crate::` names.
    root: String,
    /// Names each module of the file binds other than by `use`: the file's own, and each inline
    /// module's under its dotted path.
    defined: HashMap<String, HashSet<String>>,
    /// The structs the module defines, each field with the type it is declared as where that names
    /// one — what a chain of typed fields hops through.
    records: HashMap<String, Vec<(String, Option<String>)>>,
}

fn dotted(head: &str, rest: &[&str]) -> String {
    let mut out = head.to_owned();
    for attr in rest {
        if !out.is_empty() {
            out.push('.');
        }
        out.push_str(attr);
    }
    out
}

fn parent(module: &str) -> String {
    module.rsplit_once('.').map_or_else(|| module.to_owned(), |(up, _)| up.to_owned())
}

/// The crate root and the module a file is, both dotted. Under a package's `src/` a file is the module
/// its place there names — `lib.rs` and `main.rs` the root, a `mod.rs` its directory — under the
/// package directory's name with `-` read as `_`. A test, example or bench file is a crate of its own.
fn crate_paths(file: &str) -> (String, String) {
    let mut segments: Vec<String> =
        file.split(['/', '\\']).filter(|s| !s.is_empty() && *s != ".").map(str::to_owned).collect();
    if let Some(last) = segments.last_mut() {
        if let Some(stem) = last.strip_suffix(".rs") {
            *last = stem.to_owned();
        }
    }
    let at = segments
        .iter()
        .rposition(|s| s == "src")
        .or_else(|| segments.iter().rposition(|s| matches!(s.as_str(), "tests" | "examples" | "benches")));
    match at {
        Some(at) if at > 0 => {
            let package = segments[at - 1].replace('-', "_");
            let mut mods = segments[at + 1..].to_vec();
            if mods.last().is_some_and(|s| s == "mod") || (mods.len() == 1 && matches!(mods[0].as_str(), "lib" | "main")) {
                mods.pop();
            }
            if segments[at] == "src" {
                let module = dotted(&package, &mods.iter().map(String::as_str).collect::<Vec<_>>());
                (package, module)
            } else {
                let base = format!("{package}.{}", segments[at]);
                let root = dotted(&base, &mods.first().map(String::as_str).into_iter().collect::<Vec<_>>());
                let module = dotted(&base, &mods.iter().map(String::as_str).collect::<Vec<_>>());
                (root, module)
            }
        }
        _ => {
            let module = segments.iter().map(|s| s.replace('-', "_")).collect::<Vec<_>>().join(".");
            (module.clone(), module)
        }
    }
}

fn item_name(item: &Item) -> Option<String> {
    Some(match item {
        Item::Fn(f) => f.sig.ident.to_string(),
        Item::Struct(s) => s.ident.to_string(),
        Item::Enum(e) => e.ident.to_string(),
        Item::Union(u) => u.ident.to_string(),
        Item::Trait(t) => t.ident.to_string(),
        Item::Type(t) => t.ident.to_string(),
        Item::Const(c) => c.ident.to_string(),
        Item::Static(s) => s.ident.to_string(),
        Item::Mod(m) => m.ident.to_string(),
        _ => return None,
    })
}

impl FileScope {
    pub(crate) fn new(file: &str, ast: &syn::File) -> Self {
        let (root, module) = crate_paths(file);
        let mut scope =
            Self { imports: HashMap::new(), module: module.clone(), root, defined: HashMap::new(), records: HashMap::new() };
        scope.read_items(&ast.items, &module);
        let records = ast
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Struct(s) => Some((s.ident.to_string(), scope.record_fields(&s.fields))),
                _ => None,
            })
            .collect();
        scope.records = records;
        scope
    }

    /// The module a definition stands in: the file's, or an inline module's inside it (`prefix`, as
    /// definitions are qualified).
    pub(crate) fn here(&self, prefix: &str) -> String {
        if prefix.is_empty() {
            self.module.clone()
        } else {
            format!("{}.{}", self.module, prefix.replace("::", "."))
        }
    }

    /// Every module of the file — what it binds, then what it `use`s — the file first and its inline
    /// modules after: a `mod tests` imports what its definitions name, and a module's own import of a
    /// name is not overridden by a nested one's.
    fn read_items(&mut self, items: &[Item], here: &str) {
        self.defined.insert(here.to_owned(), items.iter().filter_map(item_name).collect());
        let mut found = HashMap::new();
        for item in items {
            if let Item::Use(u) = item {
                self.walk_use(&u.tree, &mut Vec::new(), here, &mut found);
            }
        }
        for (name, path) in found {
            self.imports.entry(name).or_insert(path);
        }
        for item in items {
            if let Item::Mod(m) = item {
                if let Some((_, inner)) = &m.content {
                    self.read_items(inner, &format!("{here}.{}", m.ident));
                }
            }
        }
    }

    fn walk_use(&self, tree: &UseTree, prefix: &mut Vec<String>, here: &str, out: &mut HashMap<String, String>) {
        match tree {
            UseTree::Path(p) => {
                prefix.push(p.ident.to_string());
                self.walk_use(&p.tree, prefix, here, out);
                prefix.pop();
            }
            // `use a::b::{self}` binds `b`.
            UseTree::Name(n) if n.ident == "self" => {
                if let Some(last) = prefix.last().cloned() {
                    out.insert(last, self.absolute(prefix, here));
                }
            }
            UseTree::Name(n) => {
                prefix.push(n.ident.to_string());
                out.insert(n.ident.to_string(), self.absolute(prefix, here));
                prefix.pop();
            }
            UseTree::Rename(r) => {
                prefix.push(r.ident.to_string());
                out.insert(r.rename.to_string(), self.absolute(prefix, here));
                prefix.pop();
            }
            UseTree::Group(g) => {
                for tree in &g.items {
                    self.walk_use(tree, prefix, here, out);
                }
            }
            // A glob binds names this cannot enumerate.
            UseTree::Glob(_) => {}
        }
    }

    /// A path as written in module `here`, dotted from where it really starts: `crate`, `self` and
    /// `super` resolved, a name the file imported or defined read as what it stands for.
    fn absolute(&self, segments: &[String], here: &str) -> String {
        let Some((head, rest)) = segments.split_first() else { return String::new() };
        let mut rest: Vec<&str> = rest.iter().map(String::as_str).collect();
        let base = match head.as_str() {
            "crate" => self.root.clone(),
            "self" => here.to_owned(),
            "super" => {
                let mut module = parent(here);
                while rest.first() == Some(&"super") {
                    module = parent(&module);
                    rest.remove(0);
                }
                module
            }
            name => self.named_in(name, here),
        };
        dotted(&base, &rest)
    }

    /// What a name stands for in module `here`: what the file imported it as; what `here` or — through
    /// the `use super::*` a nested module of a file writes — a module of the file enclosing it defines;
    /// or itself, a name from outside the file.
    fn named_in(&self, name: &str, here: &str) -> String {
        if let Some(path) = self.imports.get(name) {
            return path.clone();
        }
        let mut module = here.to_owned();
        loop {
            if self.defined.get(&module).is_some_and(|names| names.contains(name)) {
                return format!("{module}.{name}");
            }
            if module.len() <= self.module.len() {
                return name.to_owned();
            }
            module = parent(&module);
        }
    }

    /// The term a path written in module `here` stands for. `Self` is the type the definition is
    /// implemented on, and with none known it is some value's type, read as a hole.
    fn resolve(&self, segments: &[String], here: &str, self_ty: Option<&str>) -> String {
        match segments.split_first() {
            Some((head, rest)) if head == "Self" => {
                let rest: Vec<&str> = rest.iter().map(String::as_str).collect();
                match self_ty {
                    Some(ty) => decide::path(&dotted(&self.absolute(&[ty.to_owned()], here), &rest)),
                    None => decide::field(&rest),
                }
            }
            Some(_) => decide::path(&self.absolute(segments, here)),
            None => decide::HOLE.to_owned(),
        }
    }

    /// The term a name the definition did not bind stands for, with `rest` read off it.
    fn free(&self, name: &str, rest: &[&str], here: &str) -> String {
        decide::path(&dotted(&self.named_in(name, here), rest))
    }

    /// A definition's identity: what the corpus's calls and reads name it by.
    pub(crate) fn identity(&self, qualified: &str) -> Arc<str> {
        Arc::from(decide::path(&format!("{}.{}", self.module, qualified.replace("::", "."))))
    }

    /// The fields of a struct this module defines, by name — a record's constructor takes them.
    pub(crate) fn record(&self, name: &str) -> Option<&Vec<(String, Option<String>)>> {
        self.records.get(name)
    }

    fn record_fields(&self, fields: &Fields) -> Vec<(String, Option<String>)> {
        fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let name = f.ident.as_ref().map_or_else(|| i.to_string(), ToString::to_string);
                (name, self.type_class(&f.ty, &self.module, None))
            })
            .collect()
    }

    /// The type a declaration in module `here` names, as a path — through a reference, and through
    /// the wrappers that hold one value of it (`Option`, `Box`, `Arc`, `Rc`), the way Python reads
    /// `Optional[C]` as `C`.
    fn type_class(&self, ty: &Type, here: &str, self_ty: Option<&str>) -> Option<String> {
        match ty {
            Type::Reference(r) => self.type_class(&r.elem, here, self_ty),
            Type::Paren(p) => self.type_class(&p.elem, here, self_ty),
            Type::Group(g) => self.type_class(&g.elem, here, self_ty),
            Type::Path(p) if p.qself.is_none() => {
                let last = p.path.segments.last()?;
                if matches!(last.ident.to_string().as_str(), "Option" | "Box" | "Arc" | "Rc") {
                    let syn::PathArguments::AngleBracketed(args) = &last.arguments else { return None };
                    let mut types = args.args.iter().filter_map(|a| match a {
                        syn::GenericArgument::Type(t) => Some(t),
                        _ => None,
                    });
                    return match (types.next(), types.next()) {
                        (Some(inner), None) => self.type_class(inner, here, self_ty),
                        _ => None,
                    };
                }
                let segments: Vec<String> = p.path.segments.iter().map(|s| s.ident.to_string()).collect();
                let class = self.resolve(&segments, here, self_ty);
                class.starts_with('@').then_some(class)
            }
            _ => None,
        }
    }

    /// The class a field of a struct this module defines is declared as.
    fn field_class(&self, class: &str, field: &str) -> Option<String> {
        let (_, fields) =
            self.records.iter().find(|(name, _)| decide::path(&format!("{}.{name}", self.module)) == class)?;
        fields.iter().find(|(f, _)| f == field).and_then(|(_, ty)| ty.clone())
    }
}

fn member_name(member: &Member) -> String {
    match member {
        Member::Named(id) => id.to_string(),
        Member::Unnamed(i) => i.index.to_string(),
    }
}

/// A member chain from its root outwards: the path it starts from, or `None` when it starts at a
/// computed value, and the fields read off it. References, dereferences and `?` are seen through —
/// they change how a value is held, not which value it is.
fn chain(mut expr: &Expr) -> (Option<&syn::Path>, Vec<String>) {
    let mut attrs = Vec::new();
    loop {
        match expr {
            Expr::Field(f) => {
                attrs.push(member_name(&f.member));
                expr = &f.base;
            }
            Expr::Paren(p) => expr = &p.expr,
            Expr::Group(g) => expr = &g.expr,
            Expr::Reference(r) => expr = &r.expr,
            Expr::Try(t) => expr = &t.expr,
            Expr::Unary(u) if matches!(u.op, UnOp::Deref(_)) => expr = &u.expr,
            _ => break,
        }
    }
    attrs.reverse();
    match expr {
        Expr::Path(p) if p.qself.is_none() => (Some(&p.path), attrs),
        _ => (None, attrs),
    }
}

fn lit_text(lit: &Lit) -> Option<String> {
    Some(match lit {
        Lit::Str(s) => decide::literal(&s.value()),
        Lit::ByteStr(b) => format!("b{}", decide::literal(&String::from_utf8_lossy(&b.value()))),
        Lit::CStr(c) => decide::literal(&c.value().to_string_lossy()),
        Lit::Byte(b) => b.value().to_string(),
        Lit::Char(c) => decide::literal(&c.value().to_string()),
        Lit::Int(i) => i.base10_digits().to_owned(),
        Lit::Float(f) => decide::number(f.base10_parse::<f64>().ok()?),
        Lit::Bool(b) => if b.value { "True" } else { "False" }.to_owned(),
        _ => return None,
    })
}

/// The text of a literal, `None` the absent value, or `None` for anything that is neither.
fn literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Lit(l) => lit_text(&l.lit),
        Expr::Path(p) if p.qself.is_none() && p.path.is_ident("None") => Some("None".to_owned()),
        Expr::Paren(p) => literal(&p.expr),
        Expr::Group(g) => literal(&g.expr),
        _ => None,
    }
}

fn strip(mut expr: &Expr) -> &Expr {
    loop {
        match expr {
            Expr::Paren(p) => expr = &p.expr,
            Expr::Group(g) => expr = &g.expr,
            Expr::Reference(r) => expr = &r.expr,
            _ => return expr,
        }
    }
}

fn is_comparison(op: BinOp) -> bool {
    matches!(op, BinOp::Eq(_) | BinOp::Ne(_) | BinOp::Lt(_) | BinOp::Le(_) | BinOp::Gt(_) | BinOp::Ge(_))
}

fn is_compound_assign(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_)
    )
}

fn operator(op: BinOp) -> Option<&'static str> {
    Some(match op {
        BinOp::Add(_) => "+",
        BinOp::Sub(_) => "-",
        BinOp::Mul(_) => "*",
        BinOp::Div(_) => "/",
        BinOp::Rem(_) => "%",
        BinOp::BitXor(_) => "^",
        BinOp::BitAnd(_) => "&",
        BinOp::BitOr(_) => "|",
        BinOp::Shl(_) => "<<",
        BinOp::Shr(_) => ">>",
        _ => return None,
    })
}

fn macro_name(mac: &syn::Macro) -> String {
    mac.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default()
}

/// A macro's arguments as expressions, when it takes a comma-separated list of them.
fn macro_args(mac: &syn::Macro) -> Vec<Expr> {
    mac.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated)
        .map(|args| args.into_iter().collect())
        .unwrap_or_default()
}

/// `matches!(subject, pattern if guard)`.
fn parse_matches(input: syn::parse::ParseStream<'_>) -> syn::Result<(Expr, Pat, Option<Expr>)> {
    let subject: Expr = input.parse()?;
    input.parse::<Token![,]>()?;
    let pat = Pat::parse_multi_with_leading_vert(input)?;
    let guard = if input.peek(Token![if]) {
        input.parse::<Token![if]>()?;
        Some(input.parse::<Expr>()?)
    } else {
        None
    };
    let _ = input.parse::<Option<Token![,]>>()?;
    Ok((subject, pat, guard))
}

/// The methods a predicate is handed to: what the closure returns is tested for truth, as a
/// comprehension's `if` is.
const PREDICATE_METHODS: &[&str] = &[
    "filter", "find", "any", "all", "position", "rposition", "take_while", "skip_while", "retain",
    "retain_mut", "partition", "is_some_and", "is_none_or", "is_ok_and", "is_err_and",
];

/// What a method call tests wherever it stands, when it is one of the spellings of membership,
/// absence or failure.
enum Proposition {
    Contains,
    Absent,
    Failed,
}

fn proposition(m: &syn::ExprMethodCall) -> Option<Proposition> {
    match (m.method.to_string().as_str(), m.args.len()) {
        ("contains" | "contains_key", 1) => Some(Proposition::Contains),
        ("is_some" | "is_none", 0) | ("is_some_and" | "is_none_or", 1) => Some(Proposition::Absent),
        ("is_ok" | "is_err", 0) | ("is_ok_and" | "is_err_and", 1) => Some(Proposition::Failed),
        _ => None,
    }
}

/// The values a closure body hands back: its tail, and what it `return`s at the top level.
fn returned(body: &Expr) -> Vec<&Expr> {
    let Expr::Block(b) = body else { return vec![body] };
    let mut out = Vec::new();
    for stmt in &b.block.stmts {
        match stmt {
            Stmt::Expr(Expr::Return(r), _) => out.extend(r.expr.as_deref()),
            Stmt::Expr(e, None) => out.push(e),
            _ => {}
        }
    }
    out
}

/// Every name a definition binds, at any depth but inside nested items, how many times each is bound
/// or assigned, and the plain `let`s — what a name bound once may be read as.
#[derive(Default)]
struct Locals<'ast> {
    bound: HashSet<String>,
    sites: HashMap<String, usize>,
    lets: Vec<(String, &'ast Expr)>,
}

impl Locals<'_> {
    fn mark(&mut self, name: String) {
        *self.sites.entry(name).or_insert(0) += 1;
    }

    fn assigned(&mut self, target: &Expr) {
        if let Expr::Path(p) = target {
            if let Some(id) = p.path.get_ident() {
                self.mark(id.to_string());
            }
        }
    }
}

impl<'ast> Visit<'ast> for Locals<'ast> {
    fn visit_pat_ident(&mut self, it: &'ast syn::PatIdent) {
        // A capitalized name in a pattern is a variant or a constant, by the convention the compiler
        // warns about breaking — a test, not a binding.
        if !it.ident.to_string().starts_with(char::is_uppercase) {
            self.bound.insert(it.ident.to_string());
            self.mark(it.ident.to_string());
        }
        visit::visit_pat_ident(self, it);
    }

    fn visit_local(&mut self, it: &'ast syn::Local) {
        let pat = match &it.pat {
            Pat::Type(t) => &*t.pat,
            other => other,
        };
        if let (Pat::Ident(id), Some(init)) = (pat, &it.init) {
            if id.mutability.is_none() && id.by_ref.is_none() && id.subpat.is_none() && init.diverge.is_none() {
                self.lets.push((id.ident.to_string(), &init.expr));
            }
        }
        visit::visit_local(self, it);
    }

    fn visit_expr_assign(&mut self, it: &'ast syn::ExprAssign) {
        self.assigned(&it.left);
        visit::visit_expr_assign(self, it);
    }

    fn visit_expr_binary(&mut self, it: &'ast syn::ExprBinary) {
        if is_compound_assign(it.op) {
            self.assigned(&it.left);
        }
        visit::visit_expr_binary(self, it);
    }

    fn visit_item(&mut self, _: &'ast Item) {}
}

/// The walk: one definition's propositions, as atoms.
struct Atoms<'s> {
    scope: &'s FileScope,
    /// The module the definition stands in.
    here: &'s str,
    /// The type the definition is implemented on, what `Self` names.
    self_ty: Option<&'s str>,
    locals: HashSet<String>,
    /// Locals bound once to a value fixed by the text: read wherever they appear as that value.
    constants: HashMap<String, String>,
    /// Locals bound once to a call's result: read bare, they are that computation.
    computed: HashMap<String, String>,
    /// Parameters bound nowhere but by the call, read bare as themselves (see [`decide::param`]).
    params: HashSet<String>,
    /// Locals whose type the text names — an annotated parameter, a struct built in place. A field
    /// read off one is a typed field ([`decide::typed_field`]).
    types: HashMap<String, String>,
    /// Set while the condition of a `while` is read: its `let Some(..)` is iteration.
    iterating: bool,
    out: Vec<Arc<str>>,
}

impl Atoms<'_> {
    fn push(&mut self, atom: Option<Arc<str>>) {
        if let Some(atom) = atom {
            self.out.push(atom);
        }
    }

    fn named(&self, name: &str, rest: &[&str]) -> String {
        if name == "self" {
            return decide::field(rest);
        }
        if let Some(value) = self.constants.get(name) {
            return Self::extended(value, rest);
        }
        if rest.is_empty() {
            if let Some(computation) = self.computed.get(name) {
                return computation.clone();
            }
            if self.params.contains(name) {
                return decide::param(name);
            }
        } else if let Some(class) = self.types.get(name) {
            return self.typed(class, rest);
        }
        if self.locals.contains(name) {
            decide::field(rest)
        } else {
            self.scope.free(name, rest, self.here)
        }
    }

    /// A chain of fields read off a value of `class`, hopping through the types the module knows.
    fn typed(&self, class: &str, rest: &[&str]) -> String {
        decide::typed_chain(class, rest, |current, attr| self.scope.field_class(current, attr))
    }

    /// A constant's members: a path grows by them; any other constant has none a text could read,
    /// so what was read off it is a field of a value.
    fn extended(value: &str, rest: &[&str]) -> String {
        if rest.is_empty() {
            value.to_owned()
        } else if value.starts_with('@') || value.starts_with(decide::HOLE) {
            dotted(value, rest)
        } else {
            decide::field(rest)
        }
    }

    fn path_term(&self, path: &syn::Path, attrs: &[String]) -> String {
        let attrs: Vec<&str> = attrs.iter().map(String::as_str).collect();
        let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        if path.leading_colon.is_some() {
            return decide::path(&dotted(&segments.join("."), &attrs));
        }
        match segments.as_slice() {
            [only] if only == "None" && attrs.is_empty() => "None".to_owned(),
            [only] if only != "Self" => self.named(only, &attrs),
            _ => Self::extended(&self.scope.resolve(&segments, self.here, self.self_ty), &attrs),
        }
    }

    fn call_term(&self, callee: &str, arguments: &Punctuated<Expr, Token![,]>) -> String {
        decide::call(callee, arguments.iter().filter_map(|a| self.constant(a)).collect())
    }

    /// What a method is reached as: one more member of the receiver's chain, read the way a field of it
    /// is — so a local holding a computation is a value the method is read off, not that computation.
    fn method(&self, receiver: &Expr, method: &syn::Ident) -> String {
        let (root, mut attrs) = chain(receiver);
        attrs.push(method.to_string());
        match root {
            Some(path) => self.path_term(path, &attrs),
            None => decide::field(&attrs.iter().map(String::as_str).collect::<Vec<_>>()),
        }
    }

    /// The identity of an operand.
    fn term(&self, expr: &Expr) -> String {
        if let Some(lit) = literal(expr) {
            return lit;
        }
        match expr {
            Expr::Paren(p) => self.term(&p.expr),
            Expr::Group(g) => self.term(&g.expr),
            Expr::Reference(r) => self.term(&r.expr),
            Expr::Try(t) => self.term(&t.expr),
            Expr::Await(a) => self.term(&a.base),
            Expr::Cast(c) => self.term(&c.expr),
            Expr::Unary(u) if matches!(u.op, UnOp::Deref(_)) => self.term(&u.expr),
            Expr::Path(_) | Expr::Field(_) => match chain(expr) {
                (Some(path), attrs) => self.path_term(path, &attrs),
                (None, attrs) => decide::field(&attrs.iter().map(String::as_str).collect::<Vec<_>>()),
            },
            Expr::MethodCall(m) => self.call_term(&self.method(&m.receiver, &m.method), &m.args),
            Expr::Call(c) => self.call_term(&self.term(&c.func), &c.args),
            // `vec![a, b]` is the list it builds, as Python's `[a, b]` is; any other macro's expansion is
            // computed text, no identity of its own.
            Expr::Macro(m) if macro_name(&m.mac) == "vec" => {
                decide::container(macro_args(&m.mac).iter().map(|e| self.term(e)).collect())
            }
            Expr::Struct(s) => decide::call(
                &self.path_term(&s.path, &[]),
                s.fields.iter().filter_map(|f| self.constant(&f.expr)).collect(),
            ),
            Expr::Index(i) => {
                let key = self.constant(&i.index).unwrap_or_else(|| decide::HOLE.to_owned());
                format!("{}[{key}]", self.term(&i.expr))
            }
            Expr::Array(a) => decide::container(a.elems.iter().map(|e| self.term(e)).collect()),
            Expr::Binary(b) => match operator(b.op) {
                Some(op) => {
                    let rendered = format!("({} {op} {})", self.term(&b.left), self.term(&b.right));
                    if decide::has_identity(&rendered) {
                        rendered
                    } else {
                        decide::HOLE.to_owned()
                    }
                }
                None => decide::HOLE.to_owned(),
            },
            Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) => {
                let inner = self.term(&u.expr);
                if decide::has_identity(&inner) {
                    format!("-{inner}")
                } else {
                    inner
                }
            }
            _ => decide::HOLE.to_owned(),
        }
    }

    /// A value handed to a call that is fixed by the text rather than computed: a literal, a path the
    /// definition did not bind, or an array of those.
    fn constant(&self, expr: &Expr) -> Option<String> {
        if let Some(lit) = literal(expr) {
            return Some(lit);
        }
        match expr {
            Expr::Path(_) | Expr::Field(_) => match chain(expr) {
                (Some(path), attrs) => {
                    if let (None, Some(root)) = (path.leading_colon, path.get_ident()) {
                        let root = root.to_string();
                        if let Some(value) = self.constants.get(&root) {
                            return Some(Self::extended(value, &attrs.iter().map(String::as_str).collect::<Vec<_>>()));
                        }
                        if root == "self" || self.locals.contains(&root) {
                            return None;
                        }
                    }
                    Some(self.path_term(path, &attrs))
                }
                (None, _) => None,
            },
            Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) => self.constant(&u.expr).map(|inner| format!("-{inner}")),
            Expr::Array(a) => {
                a.elems.iter().map(|e| self.constant(e)).collect::<Option<Vec<_>>>().map(decide::container)
            }
            Expr::Reference(r) => self.constant(&r.expr),
            Expr::Paren(p) => self.constant(&p.expr),
            Expr::Group(g) => self.constant(&g.expr),
            Expr::Cast(c) => self.constant(&c.expr),
            _ => None,
        }
    }

    /// An expression in a position the language tests for truth.
    fn truth(&mut self, expr: &Expr) {
        match expr {
            // Structure, not a proposition: its parts are read where the walk reaches them.
            Expr::Binary(b) if is_comparison(b.op) => {}
            Expr::Unary(u) if matches!(u.op, UnOp::Not(_)) => {}
            Expr::Let(_) => {}
            Expr::MethodCall(m) if proposition(m).is_some() => {}
            Expr::Macro(m) if macro_name(&m.mac) == "matches" => {}
            // `&&` and `||` test their left operand and hand the right one back as their value — so the
            // right one is tested exactly when the whole expression is, which is now.
            Expr::Binary(b) if matches!(b.op, BinOp::And(_) | BinOp::Or(_)) => self.truth(&b.right),
            Expr::Paren(p) => self.truth(&p.expr),
            Expr::Group(g) => self.truth(&g.expr),
            _ => {
                let atom = decide::truth(&self.term(expr));
                self.push(atom);
            }
        }
    }

    /// `subject` tested against `pat`: a tuple against a tuple position by position, an alternative
    /// each on its own, anything else by the term the subject is.
    fn test(&mut self, subject: &Expr, pat: &Pat, iterating: bool) {
        match (strip(subject), pat) {
            (Expr::Tuple(t), Pat::Tuple(p)) if t.elems.len() == p.elems.len() => {
                for (e, q) in t.elems.iter().zip(&p.elems) {
                    self.test(e, q, false);
                }
            }
            (_, Pat::Or(o)) => {
                for case in &o.cases {
                    self.test(subject, case, iterating);
                }
            }
            _ => {
                let term = self.term(subject);
                self.pattern(&term, pat, iterating);
            }
        }
    }

    fn pattern(&mut self, subject: &str, pat: &Pat, iterating: bool) {
        let field = |attr: &str| {
            if subject.starts_with(decide::HOLE) {
                format!("{subject}.{attr}")
            } else {
                decide::field(&[attr])
            }
        };
        match pat {
            Pat::Lit(l) => {
                if let Some(lit) = lit_text(&l.lit) {
                    let atom = decide::equal(subject, &lit);
                    self.push(atom);
                }
            }
            Pat::Ident(i) => {
                if let Some((_, sub)) = &i.subpat {
                    self.pattern(subject, sub, iterating);
                } else if i.ident == "None" {
                    if !iterating {
                        let atom = decide::equal(subject, "None");
                        self.push(atom);
                    }
                } else if i.ident.to_string().starts_with(char::is_uppercase) {
                    let atom = decide::equal(subject, &self.scope.free(&i.ident.to_string(), &[], self.here));
                    self.push(atom);
                }
            }
            Pat::Path(p) => {
                let value = self.path_term(&p.path, &[]);
                let value = if p.path.segments.last().is_some_and(|s| s.ident == "None") { "None".to_owned() } else { value };
                let atom = decide::equal(subject, &value);
                self.push(atom);
            }
            Pat::TupleStruct(t) => {
                let atom = match t.path.segments.last().map(|s| s.ident.to_string()).as_deref() {
                    Some("Some") if iterating => None,
                    Some("Some") => decide::equal(subject, "None"),
                    Some("Ok" | "Err") => decide::member(subject, &decide::path("Err")),
                    _ => decide::member(subject, &self.path_term(&t.path, &[])),
                };
                self.push(atom);
                for inner in &t.elems {
                    self.pattern(decide::HOLE, inner, false);
                }
            }
            Pat::Struct(s) => {
                let atom = decide::member(subject, &self.path_term(&s.path, &[]));
                self.push(atom);
                for f in &s.fields {
                    self.pattern(&field(&member_name(&f.member)), &f.pat, false);
                }
            }
            Pat::Tuple(t) => {
                for inner in &t.elems {
                    self.pattern(decide::HOLE, inner, false);
                }
            }
            Pat::Slice(s) => {
                for inner in &s.elems {
                    self.pattern(decide::HOLE, inner, false);
                }
            }
            Pat::Or(o) => {
                for case in &o.cases {
                    self.pattern(subject, case, iterating);
                }
            }
            Pat::Reference(r) => self.pattern(subject, &r.pat, iterating),
            Pat::Paren(p) => self.pattern(subject, &p.pat, iterating),
            Pat::Type(t) => self.pattern(subject, &t.pat, iterating),
            Pat::Range(r) => {
                for bound in [&r.start, &r.end].into_iter().flatten() {
                    let atom = decide::order(subject, &self.term(bound));
                    self.push(atom);
                }
            }
            _ => {}
        }
    }

    /// The method calls that are propositions wherever they stand, read as the test they spell.
    fn proposition_call(&mut self, m: &syn::ExprMethodCall) {
        let Some(kind) = proposition(m) else { return };
        let receiver = strip(&m.receiver);
        let atoms = match kind {
            Proposition::Contains => {
                let element = self.term(&m.args[0]);
                match receiver {
                    // `[a, b].contains(&x)` is `x == a || x == b`.
                    Expr::Array(a) => a.elems.iter().map(|e| decide::equal(&element, &self.term(e))).collect(),
                    // `(lo..hi).contains(&x)` is the two orders that bound `x`.
                    Expr::Range(r) => {
                        [&r.start, &r.end].into_iter().flatten().map(|b| decide::order(&element, &self.term(b))).collect()
                    }
                    _ => vec![decide::member(&element, &self.term(receiver))],
                }
            }
            Proposition::Absent => vec![decide::equal(&self.term(receiver), "None")],
            Proposition::Failed => vec![decide::member(&self.term(receiver), &decide::path("Err"))],
        };
        for atom in atoms {
            self.push(atom);
        }
    }

    /// What a predicate closure returns, each tested for truth.
    fn predicate(&mut self, callback: &Expr) {
        if let Expr::Closure(c) = strip(callback) {
            for value in returned(&c.body) {
                self.truth(value);
            }
        }
    }

    fn read_macro(&mut self, mac: &syn::Macro) {
        let name = macro_name(mac);
        if name == "matches" {
            if let Ok((subject, pat, guard)) = mac.parse_body_with(parse_matches) {
                self.test(&subject, &pat, false);
                if let Some(guard) = &guard {
                    self.truth(guard);
                    Visit::visit_expr(self, guard);
                }
                Visit::visit_expr(self, &subject);
            }
            return;
        }
        let args = macro_args(mac);
        match (name.as_str(), args.as_slice()) {
            ("assert" | "debug_assert", [condition, ..]) => self.truth(condition),
            ("assert_eq" | "assert_ne" | "debug_assert_eq" | "debug_assert_ne", [a, b, ..]) => {
                let atom = decide::equal(&self.term(a), &self.term(b));
                self.push(atom);
            }
            _ => {}
        }
        for arg in &args {
            Visit::visit_expr(self, arg);
        }
    }
}

impl<'ast> Visit<'ast> for Atoms<'_> {
    fn visit_expr_binary(&mut self, it: &'ast syn::ExprBinary) {
        let atom = match it.op {
            BinOp::Eq(_) | BinOp::Ne(_) => decide::equal(&self.term(&it.left), &self.term(&it.right)),
            BinOp::Lt(_) | BinOp::Le(_) | BinOp::Gt(_) | BinOp::Ge(_) => {
                decide::order(&self.term(&it.left), &self.term(&it.right))
            }
            BinOp::And(_) | BinOp::Or(_) => {
                self.truth(&it.left);
                None
            }
            _ => None,
        };
        self.push(atom);
        visit::visit_expr_binary(self, it);
    }

    fn visit_expr_unary(&mut self, it: &'ast syn::ExprUnary) {
        if matches!(it.op, UnOp::Not(_)) {
            self.truth(&it.expr);
        }
        visit::visit_expr_unary(self, it);
    }

    fn visit_expr_if(&mut self, it: &'ast syn::ExprIf) {
        self.truth(&it.cond);
        visit::visit_expr_if(self, it);
    }

    fn visit_expr_while(&mut self, it: &'ast syn::ExprWhile) {
        self.truth(&it.cond);
        self.iterating = true;
        self.visit_expr(&it.cond);
        self.iterating = false;
        self.visit_block(&it.body);
    }

    fn visit_expr_let(&mut self, it: &'ast syn::ExprLet) {
        let iterating = std::mem::take(&mut self.iterating);
        self.test(&it.expr, &it.pat, iterating);
        visit::visit_expr_let(self, it);
    }

    fn visit_expr_match(&mut self, it: &'ast syn::ExprMatch) {
        for arm in &it.arms {
            self.test(&it.expr, &arm.pat, false);
            if let Some((_, guard)) = &arm.guard {
                self.truth(guard);
            }
        }
        visit::visit_expr_match(self, it);
    }

    fn visit_local(&mut self, it: &'ast syn::Local) {
        // `let P = e else { … }` refutes `e` against `P`: a test, as `if let` is.
        if let Some(init) = &it.init {
            if init.diverge.is_some() {
                self.test(&init.expr, &it.pat, false);
            }
        }
        visit::visit_local(self, it);
    }

    fn visit_expr_method_call(&mut self, it: &'ast syn::ExprMethodCall) {
        self.proposition_call(it);
        if PREDICATE_METHODS.contains(&it.method.to_string().as_str()) {
            if let Some(callback) = it.args.first() {
                self.predicate(callback);
            }
        }
        visit::visit_expr_method_call(self, it);
    }

    fn visit_macro(&mut self, it: &'ast syn::Macro) {
        self.read_macro(it);
    }

    fn visit_item(&mut self, _: &'ast Item) {}
}

/// Calls and value reads, by the same identities the atoms are written in.
struct Refs<'w, 's> {
    walk: &'w Atoms<'s>,
    calls: Vec<decide::CallSite>,
    mentions: Vec<Arc<str>>,
    /// Fields written other than through a constructor the corpus can name: assigned, or set by a
    /// struct literal of a type that is no path.
    loose: Vec<Arc<str>>,
}

impl<'ast> Visit<'ast> for Refs<'_, '_> {
    fn visit_expr(&mut self, it: &'ast Expr) {
        match it {
            Expr::Call(c) => {
                let callee = self.walk.term(&c.func);
                if matches!(strip(&c.func), Expr::Path(_)) && callee.starts_with('@') {
                    let args = c.args.iter().map(|a| (None, Arc::from(self.walk.term(a)))).collect();
                    self.calls.push(decide::CallSite { callee: Arc::from(callee), args, open: false });
                } else {
                    self.visit_expr(&c.func);
                }
                for arg in &c.args {
                    self.visit_expr(arg);
                }
            }
            // A struct literal is its type's constructor, called with every field by name.
            Expr::Struct(s) => {
                let callee = self.walk.path_term(&s.path, &[]);
                if callee.starts_with('@') {
                    let args = s
                        .fields
                        .iter()
                        .map(|f| (Some(Arc::from(member_name(&f.member))), Arc::from(self.walk.term(&f.expr))))
                        .collect();
                    self.calls.push(decide::CallSite { callee: Arc::from(callee), args, open: s.rest.is_some() });
                } else {
                    self.loose.extend(s.fields.iter().map(|f| Arc::from(member_name(&f.member))));
                }
                for f in &s.fields {
                    self.visit_expr(&f.expr);
                }
                if let Some(rest) = &s.rest {
                    self.visit_expr(rest);
                }
            }
            Expr::Path(_) | Expr::Field(_) if chain(it).0.is_some() => {
                let read = self.walk.term(it);
                if read.starts_with('@') {
                    self.mentions.push(Arc::from(read));
                }
            }
            _ => visit::visit_expr(self, it),
        }
    }

    fn visit_expr_assign(&mut self, it: &'ast syn::ExprAssign) {
        if let Expr::Field(f) = strip(&it.left) {
            self.loose.push(Arc::from(member_name(&f.member)));
        }
        visit::visit_expr_assign(self, it);
    }

    fn visit_expr_binary(&mut self, it: &'ast syn::ExprBinary) {
        if is_compound_assign(it.op) {
            if let Expr::Field(f) = strip(&it.left) {
                self.loose.push(Arc::from(member_name(&f.member)));
            }
        }
        visit::visit_expr_binary(self, it);
    }

    fn visit_macro(&mut self, it: &'ast syn::Macro) {
        let args = if macro_name(it) == "matches" {
            it.parse_body_with(parse_matches).map(|(subject, _, guard)| std::iter::once(subject).chain(guard).collect()).unwrap_or_default()
        } else {
            macro_args(it)
        };
        for arg in &args {
            Visit::visit_expr(self, arg);
        }
    }

    /// A type names a type; it does not hand a value anywhere.
    fn visit_type(&mut self, _: &'ast Type) {}

    fn visit_item(&mut self, _: &'ast Item) {}
}

/// What a `let` binds once its `?` and `.await` have done their work: the computation itself.
fn settled(mut value: &Expr) -> &Expr {
    loop {
        match value {
            Expr::Try(t) => value = &t.expr,
            Expr::Await(a) => value = &a.base,
            _ => return value,
        }
    }
}

/// Everything the decisions pass reads off one callable standing in module `here`. `self_ty` is what
/// `Self` names; `open` — the
/// callable is reached by calls the corpus cannot show (a trait method, dispatched through the trait;
/// a function an attribute registers), so what its parameters hold is not the corpus's to say.
pub(crate) fn decides(
    sig: &syn::Signature,
    block: &syn::Block,
    scope: &FileScope,
    here: &str,
    self_ty: Option<&str>,
    open: bool,
) -> DecisionFacts {
    let mut locals = Locals::default();
    for input in &sig.inputs {
        if let FnArg::Typed(t) = input {
            locals.visit_pat(&t.pat);
        }
    }
    locals.visit_block(block);
    let mut bound = locals.bound;
    // A type parameter is a type the caller picks, no more a fact than a value the caller passes.
    bound.extend(sig.generics.type_params().map(|p| p.ident.to_string()));
    let annotated: Vec<(String, &Type)> = sig
        .inputs
        .iter()
        .filter_map(|input| match input {
            FnArg::Typed(t) => match &*t.pat {
                Pat::Ident(id) => Some((id.ident.to_string(), &*t.ty)),
                _ => None,
            },
            FnArg::Receiver(_) => None,
        })
        .collect();
    let order: Vec<String> = annotated.iter().map(|(name, _)| name.clone()).collect();
    let untouched: HashSet<String> = order.iter().filter(|name| locals.sites.get(*name) == Some(&1)).cloned().collect();
    let types: HashMap<String, String> =
        annotated.iter().filter_map(|(name, ty)| Some((name.clone(), scope.type_class(ty, here, self_ty)?))).collect();
    let mut single = locals.lets;
    single.retain(|(name, _)| locals.sites.get(name) == Some(&1));
    let mut walk = Atoms {
        scope,
        here,
        self_ty,
        locals: bound,
        constants: HashMap::new(),
        computed: HashMap::new(),
        params: untouched,
        types,
        iterating: false,
        out: Vec::new(),
    };
    // A `let` whose value is itself a constant — possibly through another such `let` — is that
    // constant. Resolved to a fixed point: each round can only add, and there are finitely many names.
    loop {
        let found: Vec<(String, String)> = single
            .iter()
            .filter(|(name, _)| !walk.constants.contains_key(name))
            .filter_map(|(name, value)| walk.constant(value).map(|c| (name.clone(), c)))
            .collect();
        if found.is_empty() {
            break;
        }
        walk.constants.extend(found);
    }
    for (name, value) in &single {
        if walk.constants.contains_key(name) {
            continue;
        }
        match settled(value) {
            value @ (Expr::Call(_) | Expr::MethodCall(_)) => {
                let computation = walk.term(value);
                walk.computed.insert(name.clone(), computation);
            }
            // A value built from a struct literal is of that struct: its fields are typed.
            Expr::Struct(s) => {
                let class = walk.path_term(&s.path, &[]);
                if class.starts_with('@') && !walk.types.contains_key(name) {
                    walk.types.insert(name.clone(), class);
                }
            }
            _ => {}
        }
    }
    walk.visit_block(block);
    let mut refs = Refs { walk: &walk, calls: Vec::new(), mentions: Vec::new(), loose: Vec::new() };
    refs.visit_block(block);
    let (calls, mut mentions, mut loose) = (refs.calls, refs.mentions, refs.loose);
    loose.sort_unstable();
    loose.dedup();
    mentions.sort_unstable();
    mentions.dedup();
    let mut atoms = walk.out;
    atoms.sort_unstable();
    atoms.dedup();
    let params = if open { Vec::new() } else { order.iter().map(|p| Arc::from(p.as_str())).collect() };
    DecisionFacts { atoms, params, calls, mentions, loose }
}

#[cfg(test)]
mod tests {
    use super::{crate_paths, decides, FileScope};
    use dup_defs_core::decide::{canonical, render, resolve_typed_fields};
    use syn::{ImplItem, Item};

    /// The atoms of the first function or method in `src` as the engine reads them with nothing
    /// inferred (a typed field is the field of some value), rendered, as if the file lived at `file`.
    fn atoms_at(file: &str, src: &str) -> Vec<String> {
        let ast = syn::parse_file(src).expect("fixture parses");
        let scope = FileScope::new(file, &ast);
        let facts = ast
            .items
            .iter()
            .find_map(|item| match item {
                Item::Fn(f) => Some(decides(&f.sig, &f.block, &scope, &scope.here(""), None, false)),
                Item::Impl(im) => im.items.iter().find_map(|ii| match ii {
                    ImplItem::Fn(f) => Some(decides(&f.sig, &f.block, &scope, &scope.here(""), Some("Order"), false)),
                    _ => None,
                }),
                _ => None,
            })
            .expect("fixture has a function");
        let mut out: Vec<String> =
            facts.atoms.iter().map(|a| render(&canonical(&resolve_typed_fields(a, |_, _| None)))).collect();
        out.sort();
        out.dedup();
        out
    }

    fn atoms(src: &str) -> Vec<String> {
        atoms_at("shop/src/orders.rs", src)
    }

    #[test]
    fn a_file_is_the_module_its_place_under_src_names() {
        assert_eq!(crate_paths("/w/shop-core/src/lib.rs"), ("shop_core".to_owned(), "shop_core".to_owned()));
        assert_eq!(crate_paths("/w/shop-core/src/orders/mod.rs"), ("shop_core".to_owned(), "shop_core.orders".to_owned()));
        assert_eq!(crate_paths("/w/shop-core/src/orders/line.rs"), ("shop_core".to_owned(), "shop_core.orders.line".to_owned()));
        assert_eq!(crate_paths("/w/shop-core/tests/flow.rs"), ("shop_core.tests.flow".to_owned(), "shop_core.tests.flow".to_owned()));
    }

    #[test]
    fn negation_spelling_and_the_path_a_variant_is_reached_by_do_not_change_the_decision() {
        let refuse = atoms("use crate::model::Status;\nfn refuse(o: &Order) -> R { if o.status != Status::Open { return Err(E); } Ok(()) }\n");
        let accept = atoms("fn accept(o: &Order) -> bool { o.status == crate::model::Status::Open }\n");
        let matched = atoms("use crate::model::Status;\nfn m(o: &Order) -> bool { matches!(o.status, Status::Open) }\n");
        assert_eq!(refuse, vec!["@shop.model.Status.Open = _.status"]);
        assert_eq!(accept, refuse);
        assert_eq!(matched, refuse);
    }

    #[test]
    fn absence_and_failure_are_one_test_each_however_spelled() {
        let absent = vec!["None = _.paid_at"];
        assert_eq!(atoms("fn f(o: &Order) -> u8 { if let Some(t) = o.paid_at { t } else { 0 } }\n"), absent);
        assert_eq!(atoms("fn f(o: &Order) -> bool { o.paid_at.is_none() }\n"), absent);
        assert_eq!(atoms("fn f(o: &Order) -> u8 { match o.paid_at { Some(t) => t, None => 0 } }\n"), absent);
        assert_eq!(atoms("fn f(o: &Order) -> bool { o.paid_at == None }\n"), absent);
        let failed = vec!["_.charge() in @Err"];
        assert_eq!(atoms("fn f(o: &Order) -> bool { o.charge().is_err() }\n"), failed);
        assert_eq!(atoms("fn f(o: &Order) -> u8 { match o.charge() { Ok(_) => 1, Err(_) => 0 } }\n"), failed);
    }

    #[test]
    fn iteration_is_not_a_test_of_absence() {
        assert_eq!(atoms("fn f(xs: &mut Vec<u8>) { while let Some(x) = xs.pop() { if x > 3 { g(); } } }\n"), vec!["3 < _"]);
    }

    #[test]
    fn a_struct_pattern_tests_its_fields_and_a_range_its_bounds() {
        let found = atoms("use crate::model::{Event, Key};\nfn f(e: &Event) -> bool { match e { Event::Press { key: Key::Enter, .. } => true, _ => false } }\n");
        assert_eq!(found, vec!["$e in @shop.model.Event.Press", "@shop.model.Key.Enter = _.key"]);
        let digit = atoms("fn f(c: char) -> bool { matches!(c, '0'..='9') }\n");
        let spelled = atoms("fn f(c: char) -> bool { c >= '0' && c <= '9' }\n");
        assert_eq!(digit, spelled);
    }

    #[test]
    fn membership_spellings_meet() {
        let listed = atoms("use crate::model::Kind;\nfn f(o: &Order) -> bool { [Kind::A, Kind::B].contains(&o.kind) }\n");
        let matched = atoms("use crate::model::Kind;\nfn f(o: &Order) -> bool { matches!(o.kind, Kind::A | Kind::B) }\n");
        assert_eq!(listed, matched);
        assert_eq!(atoms("fn f(o: &Order, seen: &Set) -> bool { seen.contains(&o.id) }\n"), vec!["_.id in $seen"]);
    }

    #[test]
    fn self_is_the_type_the_method_is_implemented_on() {
        let found = atoms("struct Order;\nimpl Order { fn open(&self) -> bool { self.state == Self::OPEN } }\n");
        assert_eq!(found, vec!["@shop.orders.Order.OPEN = _.state"]);
    }

    #[test]
    fn a_predicate_and_an_assertion_are_truth_positions() {
        let filtered = atoms("fn f(xs: &[Order]) -> usize { xs.iter().filter(|o| o.paid).count() }\n");
        let asserted = atoms("fn f(o: &Order) { assert!(o.paid, \"unpaid\"); }\n");
        assert_eq!(filtered, vec!["? _.paid"]);
        assert_eq!(asserted, filtered);
        assert_eq!(atoms("fn f(o: &Order) { assert_eq!(o.total, 0); }\n"), vec!["0 = _.total"]);
    }

    #[test]
    fn a_nested_module_resolves_from_where_it_stands() {
        let src = "pub enum Kind { A }\nmod tests {\n    use super::Kind;\n    use crate::model::Status;\n    fn t(o: &O) -> bool { o.kind == Kind::A && o.status == Status::Open && o.x == super::LIMIT }\n}\n";
        let ast = syn::parse_file(src).expect("fixture parses");
        let scope = FileScope::new("shop/src/orders.rs", &ast);
        let Item::Mod(m) = &ast.items[1] else { panic!("a module") };
        let Some((_, items)) = &m.content else { panic!("inline") };
        let Item::Fn(f) = &items[2] else { panic!("a function") };
        let facts = decides(&f.sig, &f.block, &scope, &scope.here("tests"), None, false);
        let mut found: Vec<String> = facts.atoms.iter().map(|a| render(&canonical(&resolve_typed_fields(a, |_, _| None)))).collect();
        found.sort();
        assert_eq!(found, vec!["@shop.model.Status.Open = _.status", "@shop.orders.Kind.A = _.kind", "@shop.orders.LIMIT = _.x"]);
    }

    #[test]
    fn a_let_bound_to_a_constant_is_that_constant_and_a_parameter_is_marked() {
        let direct = atoms("use crate::cfg::LIMIT;\nfn f(o: &Order) -> bool { o.total < LIMIT }\n");
        let through = atoms("use crate::cfg::LIMIT;\nfn f(o: &Order) -> bool { let cap = LIMIT; o.total < cap }\n");
        assert_eq!(direct, through);
        assert_eq!(atoms("fn f(flag: bool, o: &Order) -> u8 { if !flag { o.x } else { 0 } }\n"), vec!["? $flag"]);
    }
}
