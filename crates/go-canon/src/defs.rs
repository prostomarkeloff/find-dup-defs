//! Package-level definition scan for Go — the "find-*" step of `find-dup-defs`.
//!
//! Parses each file **once** with tree-sitter and lowers every tracked declaration straight to the
//! engine's [`Def`], computing its canonical strings off the syntax tree (no re-parse). Surfaces:
//!
//! * **functions** — `func f(…)` with a body.
//! * **methods** — `func (r *T) m(…)`, named `T.m` (pointer and type arguments stripped from the
//!   receiver type). The receiver is not a parameter, and the body's reads of it are written `self`.
//! * **classes** — `type T struct { … }`.
//! * **interfaces** — `type T interface { … }`.
//! * **constants** — package-level `const` names with a value of their own. Go marks a constant by
//!   declaring it one, so there is no naming convention to test, unlike Python's `UPPER_SNAKE`; an
//!   `iota` continuation (a bare name under a typed first line) repeats the line above it and is left
//!   out.
//! * **type-aliases** — every other `type` declaration: `type X = Y` and defined types such as
//!   `type Duration int64` or `type HandlerFunc func(…)`.
//!
//! Only package-level declarations: a type or function literal inside a body is part of that body,
//! the "top-level only" rule every frontend follows.
//!
//! A file that declares itself generated (`// Code generated … DO NOT EDIT.` before its `package`
//! clause — the Go convention `go generate` tools write and linters honour) yields nothing: its
//! duplication is the generator's, and editing it is not how it would be fixed.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dup_defs_core::lens::{enabled_lenses, lens_def, DefSite, Lens, LensFacts};
use dup_defs_core::{count_loc, Analysis, AnalyzedFn, CanonDialect, Def, Facets, KindSpec, LineMap, ScanOpts};
use tree_sitter::Node;

use crate::canon::{analyze, file_imports, interface_canon, locals_of, reached_paths, scope_canonical, struct_canon, Callable};
use crate::frontend::{CLASSES, CONSTANTS, FUNCTIONS, INTERFACES, METHODS, TYPE_ALIASES};
use crate::ir::lower_func;
use crate::tree::{field, fields, kids, parse, statements, text};

/// Whether a file declares itself generated: a `// Code generated … DO NOT EDIT.` line before the
/// `package` clause.
pub(crate) fn is_generated(src: &str) -> bool {
    for line in src.lines() {
        let line = line.trim();
        if line.starts_with("package ") {
            return false;
        }
        if line.starts_with("// Code generated") && line.ends_with("DO NOT EDIT.") {
            return true;
        }
    }
    false
}

/// Number of value parameters — `a, b int` is two, an unnamed `int` one; the receiver is not one.
fn count_args(params: Option<Node<'_>>) -> usize {
    params.map(kids).unwrap_or_default().iter().map(|d| fields(*d, "name").len().max(1)).sum()
}

/// True when a body is a no-op or a stub: empty, a single `return` of names and literals only
/// (`return nil`, `return 0, nil`, `return x`), or a single `panic(…)` — the shapes that are not
/// refactor clusters, the Go reading of the other frontends' trivial-body filters. A field read
/// (`return s.name`), a call, and anything structural fall through and are compared.
fn is_trivial(body: Node<'_>, src: &str) -> bool {
    match statements(body).as_slice() {
        [] => true,
        [only] => match only.kind() {
            "return_statement" => kids(*only).first().is_none_or(|values| kids(*values).iter().all(|v| is_atom(*v))),
            "expression_statement" => kids(*only).first().is_some_and(|e| {
                e.kind() == "call_expression"
                    && field(*e, "function").is_some_and(|f| f.kind() == "identifier" && text(f, src) == "panic")
            }),
            _ => false,
        },
        _ => false,
    }
}

fn is_atom(n: Node<'_>) -> bool {
    matches!(
        n.kind(),
        "identifier"
            | "nil"
            | "true"
            | "false"
            | "iota"
            | "int_literal"
            | "float_literal"
            | "imaginary_literal"
            | "rune_literal"
            | "interpreted_string_literal"
            | "raw_string_literal"
    )
}

/// A function or method declaration's parts; `None` for one without a body (an assembly-backed
/// declaration).
pub(crate) fn callable_of<'t>(decl: Node<'t>, src: &'t str) -> Option<Callable<'t>> {
    let body = field(decl, "body")?;
    let method = decl.kind() == "method_declaration";
    let receiver = field(decl, "receiver").and_then(|list| kids(list).into_iter().next());
    let receiver = receiver
        .and_then(|r| fields(r, "name").into_iter().next())
        .map(|n| text(n, src))
        .filter(|n| *n != "_");
    Some(Callable {
        method,
        receiver,
        type_params: field(decl, "type_parameters"),
        params: field(decl, "parameters"),
        results: field(decl, "result"),
        body,
    })
}

/// The named type a receiver is declared on: `*Server[K]` → `Server`.
fn receiver_type(ty: Node<'_>, src: &str) -> String {
    match ty.kind() {
        "pointer_type" | "parenthesized_type" => {
            kids(ty).first().map_or_else(|| "<ty>".to_owned(), |t| receiver_type(*t, src))
        }
        "generic_type" => field(ty, "type").map_or_else(|| "<ty>".to_owned(), |t| text(t, src).to_owned()),
        _ => text(ty, src).to_owned(),
    }
}

/// What a definition contributes beyond its location: its canonical forms and, for a callable, the
/// facets the perspective passes read.
struct DefCanon {
    cluster: Option<String>,
    analysis: Option<Analysis>,
    facets: Facets,
}

impl DefCanon {
    /// A body-bearing kind clustered by its canonical text alone (struct, interface), or a raw-text
    /// kind (`None`).
    fn body(cluster: Option<String>) -> Self {
        Self { cluster, analysis: None, facets: Facets::default() }
    }
}

/// One builder so every push site stays uniform.
struct Builder<'a> {
    source: &'a str,
    lines: &'a LineMap<'a>,
    file: &'a Arc<str>,
    /// What each package name this file imports stands for — resolved once, read by every callable.
    imports: HashMap<String, Arc<str>>,
    /// Which lenses this run asked for; empty when it did not ask.
    lenses: Vec<Lens>,
    /// Names the file itself introduced — what every lens erases before it looks.
    file_names: HashSet<String>,
    out: Vec<Def>,
}

impl Builder<'_> {
    /// One lens [`Def`] for this definition, when the run asked for lenses and the projection has
    /// enough to say. The record, the thinness floor and the scoring are the shared module's.
    fn push_lens(&mut self, name: &str, span: Node<'_>, facts: &LensFacts) {
        if self.lenses.is_empty() {
            return;
        }
        let start = span.start_byte();
        let (line, col) = self.lines.loc0(start);
        let loc = count_loc(self.source.get(start..span.end_byte()).unwrap_or(""));
        let site = DefSite { name, file: self.file, line, col, loc };
        if let Some(def) = lens_def("go", CanonDialect::Go, &site, &self.lenses, facts) {
            self.out.push(def);
        }
    }

    fn push(&mut self, kind: &'static KindSpec, name: String, span: Node<'_>, args: usize, canon: DefCanon) {
        let DefCanon { cluster: cluster_canonical, analysis, facets } = canon;
        let (start, end) = (span.start_byte(), span.end_byte());
        let (line, col) = self.lines.loc0(start);
        let text_orig = self.source.get(start..end).unwrap_or("").to_owned();
        let loc = count_loc(&text_orig);
        self.out.push(Def {
            lang: "go",
            kind,
            name,
            file: Arc::clone(self.file),
            line,
            col,
            loc,
            args,
            text_orig,
            cluster_canonical,
            analysis,
            thickness: None,
            facets,
        });
    }

    /// A function or method: canonical forms, statement stream, and the packages it reaches.
    fn callable(&mut self, kind: &'static KindSpec, name: String, own_name: &str, decl: Node<'_>, c: &Callable<'_>) {
        if is_trivial(c.body, self.source) {
            return;
        }
        let func = lower_func(c, self.source);
        if !self.lenses.is_empty() {
            let scope = scope_canonical(&self.imports, &func, &self.file_names);
            let locals = locals_of(&func);
            let facts = crate::lenses::callable_facts(c, self.source, &self.imports, &locals, &self.file_names, scope);
            self.push_lens(&name, decl, &facts);
        }
        let AnalyzedFn { cluster_canonical, xname_canonical, type3_lines, type_refs, type3_units, statements, size } =
            analyze(&self.imports, own_name, &func);
        let canon = DefCanon {
            cluster: Some(cluster_canonical),
            analysis: Some(Analysis {
                xname_canonical,
                type3_lines,
                type_refs,
                type3_units,
                size,
                canon_dialect: CanonDialect::Go,
            }),
            facets: Facets {
                statements,
                reaches: reached_paths(c.body, self.source, &self.imports),
                ..Facets::default()
            },
        };
        self.push(kind, name, decl, count_args(c.params), canon);
    }

    fn function(&mut self, decl: Node<'_>) {
        let Some(c) = callable_of(decl, self.source) else { return };
        let Some(name) = field(decl, "name").map(|n| text(n, self.source)) else { return };
        self.callable(&FUNCTIONS, name.to_owned(), name, decl, &c);
    }

    fn method(&mut self, decl: Node<'_>) {
        let Some(c) = callable_of(decl, self.source) else { return };
        let Some(name) = field(decl, "name").map(|n| text(n, self.source)) else { return };
        let owner = field(decl, "receiver")
            .and_then(|list| kids(list).into_iter().next())
            .and_then(|r| field(r, "type"))
            .map_or_else(|| "<ty>".to_owned(), |t| receiver_type(t, self.source));
        self.callable(&METHODS, format!("{owner}.{name}"), name, decl, &c);
    }

    /// `type` declarations. A declaration of one spec reports its whole text, `type` keyword
    /// included; a grouped `type ( … )` reports each spec on its own.
    fn types(&mut self, decl: Node<'_>) {
        let specs: Vec<Node<'_>> =
            kids(decl).into_iter().filter(|k| matches!(k.kind(), "type_spec" | "type_alias")).collect();
        let single = specs.len() == 1;
        for spec in specs {
            let Some(name) = field(spec, "name").map(|n| text(n, self.source)) else { continue };
            let Some(ty) = field(spec, "type") else { continue };
            let span = if single { decl } else { spec };
            let type_params = field(spec, "type_parameters");
            if spec.kind() == "type_spec" && ty.kind() == "struct_type" && !self.lenses.is_empty() {
                let facts = crate::lenses::struct_facts(ty, self.source, &self.file_names);
                self.push_lens(name, span, &facts);
            }
            let (kind, canon) = match (spec.kind(), ty.kind()) {
                ("type_spec", "struct_type") => (&CLASSES, Some(struct_canon(self.source, name, type_params, ty))),
                ("type_spec", "interface_type") => {
                    (&INTERFACES, Some(interface_canon(self.source, name, type_params, ty)))
                }
                _ => (&TYPE_ALIASES, None),
            };
            self.push(kind, name.to_owned(), span, 0, DefCanon::body(canon));
        }
    }

    fn consts(&mut self, decl: Node<'_>) {
        let specs: Vec<Node<'_>> = kids(decl).into_iter().filter(|k| k.kind() == "const_spec").collect();
        let single = specs.len() == 1;
        for spec in specs {
            if field(spec, "value").is_none() {
                continue; // an `iota` continuation repeats the line above
            }
            let span = if single { decl } else { spec };
            for name in fields(spec, "name") {
                let name = text(name, self.source);
                if name != "_" {
                    self.push(&CONSTANTS, name.to_owned(), span, 0, DefCanon::body(None));
                }
            }
        }
    }
}

/// Scan one Go source string → its definitions as [`Def`]s with canon precomputed. A declaration
/// holding a syntax error is skipped; the rest of the file is still read.
#[must_use]
pub fn scan_source(source: &str, file: &Arc<str>, opts: &ScanOpts) -> Vec<Def> {
    if is_generated(source) {
        return Vec::new();
    }
    let Some(tree) = parse(source) else { return Vec::new() };
    let root = tree.root_node();
    let lines = LineMap::new(source);
    let lenses = enabled_lenses(opts);
    // Every lens reads the file's own names, so they are collected once per file — and only when asked.
    let file_names =
        if lenses.is_empty() { HashSet::new() } else { crate::lenses::file_bound_names(root, source) };
    let mut b = Builder {
        source,
        lines: &lines,
        file,
        imports: file_imports(root, source),
        lenses,
        file_names,
        out: Vec::new(),
    };
    for item in kids(root) {
        if item.has_error() {
            continue;
        }
        match item.kind() {
            "function_declaration" => b.function(item),
            "method_declaration" => b.method(item),
            "type_declaration" => b.types(item),
            "const_declaration" => b.consts(item),
            _ => {}
        }
    }
    b.out
}

#[cfg(test)]
mod tests {
    use super::scan_source;
    use dup_defs_core::{Analysis, Def, ScanOpts};
    use std::sync::Arc;

    fn scan(src: &str) -> Vec<Def> {
        let f: Arc<str> = Arc::from("t.go");
        scan_source(src, &f, &ScanOpts::default())
    }

    fn defs(src: &str) -> Vec<(String, String)> {
        scan(src).into_iter().map(|d| (d.kind.id.to_owned(), d.name)).collect()
    }

    fn names_of_kind(src: &str, kind: &str) -> Vec<String> {
        defs(src).into_iter().filter(|(k, _)| k == kind).map(|(_, n)| n).collect()
    }

    fn analysis(src: &str, name: &str) -> Analysis {
        scan(src).into_iter().find(|d| d.name == name).and_then(|d| d.analysis).expect(name)
    }

    #[test]
    fn surfaces_each_kind() {
        let src = r#"package p

import "fmt"

const MaxRetries = 5

const (
	modeA Mode = iota
	modeB
	greeting = "hi"
)

type Ids []uint64
type Alias = Ids

func Compute(values []int, weight int) int {
	total := 0
	for _, v := range values {
		total += v * weight
	}
	return total
}

type Repo struct {
	store []byte
	fmt.Stringer
}

type Fetcher interface {
	Get(id uint64) (uint64, error)
}

func (r *Repo) FetchItem(id int) byte {
	rec := r.store[id]
	return rec + 1
}
"#;
        let d = defs(src);
        let has = |k: &str, n: &str| d.iter().any(|(kk, nn)| kk == k && nn == n);
        assert!(has("constants", "MaxRetries"), "{d:?}");
        assert!(has("constants", "modeA"), "{d:?}");
        assert!(has("constants", "greeting"), "{d:?}");
        assert!(!has("constants", "modeB"), "an iota continuation repeats the line above: {d:?}");
        assert!(has("type-aliases", "Ids"), "{d:?}");
        assert!(has("type-aliases", "Alias"), "{d:?}");
        assert!(has("functions", "Compute"), "{d:?}");
        assert!(has("classes", "Repo"), "{d:?}");
        assert!(has("interfaces", "Fetcher"), "{d:?}");
        assert!(has("methods", "Repo.FetchItem"), "{d:?}");
    }

    #[test]
    fn a_receiver_is_named_by_its_base_type() {
        let src = "package p\nfunc (s *Server[K]) Run(n int) int {\n\tx := n + 1\n\treturn x * 2\n}\nfunc (Plain) Go(n int) int {\n\tx := n + 2\n\treturn x\n}\n";
        let methods = names_of_kind(src, "methods");
        assert_eq!(methods, vec!["Server.Run".to_owned(), "Plain.Go".to_owned()]);
    }

    #[test]
    fn trivial_bodies_skipped() {
        let src = r#"package p
func empty() {}
func lit() bool { return true }
func ident(x int) int { return x }
func pair() (int, error) { return 0, nil }
func stub() int { panic("todo") }
func bare() { return }
func getter(s *S) string { return s.name }
func real(x int) int { y := x + 1; return y * 2 }
"#;
        let fns = names_of_kind(src, "functions");
        assert_eq!(fns, vec!["getter".to_owned(), "real".to_owned()], "a field read and a real body survive: {fns:?}");
    }

    #[test]
    fn a_generated_file_yields_nothing() {
        let src = "// Code generated by protoc-gen-go. DO NOT EDIT.\n\npackage p\n\nfunc Real(x int) int {\n\ty := x + 1\n\treturn y\n}\n";
        assert!(scan(src).is_empty());
        // The marker counts only before the package clause.
        let late = "package p\n\n// Code generated by hand. DO NOT EDIT.\nfunc Real(x int) int {\n\ty := x + 1\n\treturn y\n}\n";
        assert_eq!(scan(late).len(), 1);
    }

    #[test]
    fn a_method_aligns_with_a_free_function_whatever_its_receiver_is_called() {
        let src = "package p\nfunc (s *S) Add(a, b int) int {\n\tt := a + b\n\treturn t\n}\nfunc AddFree(x, y int) int {\n\tu := x + y\n\treturn u\n}\nfunc (c *C) Size() int {\n\tn := len(c.items)\n\treturn n + 1\n}\nfunc (other *D) Size() int {\n\tn := len(other.items)\n\treturn n + 1\n}\n";
        assert_eq!(analysis(src, "S.Add").xname_canonical, analysis(src, "AddFree").xname_canonical);
        // The receiver is `self` in every mode, so two types' same-shaped methods agree by name too.
        let all = scan(src);
        let cluster = |n: &str| all.iter().find(|d| d.name == n).and_then(|d| d.cluster_canonical.clone());
        assert_eq!(cluster("C.Size"), cluster("D.Size"));
        assert!(cluster("C.Size").is_some_and(|c| c.contains("Path('self')")), "{:?}", cluster("C.Size"));
    }

    #[test]
    fn locals_rename_and_free_names_stay() {
        let a = analysis("package p\nfunc F(items []Item) int {\n\tcount := 0\n\tfor _, it := range items {\n\t\tif it.Ok {\n\t\t\tcount++\n\t\t}\n\t}\n\treturn count\n}\n", "F");
        let b = analysis("package p\nfunc G(rows []Item) int {\n\tn := 0\n\tfor _, r := range rows {\n\t\tif r.Ok {\n\t\t\tn++\n\t\t}\n\t}\n\treturn n\n}\n", "G");
        assert_eq!(a.xname_canonical, b.xname_canonical);
        assert!(a.xname_canonical.contains("Field(Path('_v2'), 'Ok')"), "{}", a.xname_canonical);
    }

    #[test]
    fn a_package_member_is_a_path_and_a_value_member_a_field() {
        let src = "package p\nimport (\n\t\"strings\"\n\tlg \"github.com/acme/logx\"\n)\nfunc F(s string, b *Buf) int {\n\tlg.Info(strings.ToUpper(s))\n\tb.Write(s)\n\treturn b.n\n}\n";
        let x = analysis(src, "F").xname_canonical;
        assert!(x.contains("Call(Path('lg.Info'), Call(Path('strings.ToUpper'), Path('_v0')))"), "{x}");
        assert!(x.contains("Method(Path('_v1'), 'Write', Path('_v0'))"), "{x}");
        assert!(x.contains("Field(Path('_v1'), 'n')"), "{x}");
    }

    #[test]
    fn a_callable_reports_the_packages_it_reaches() {
        let src = "package p\nimport (\n\t\"fmt\"\n\tlg \"github.com/acme/logx\"\n\t\"gopkg.in/yaml.v3\"\n\t\"github.com/redis/go-redis/v9\"\n\t_ \"embed\"\n)\nfunc F(c *redis.Client) error {\n\tlg.Info(\"x\")\n\tvar v yaml.Node\n\t_ = v\n\treturn nil\n}\n";
        let d = scan(src).into_iter().find(|d| d.name == "F").expect("F");
        let reaches: Vec<&str> = d.facets.reaches.iter().map(|r| &**r).collect();
        // `fmt` is imported and never touched; the parameter's type is not in the body.
        assert_eq!(reaches, vec!["github.com.acme.logx", "gopkg.in.yaml.v3"]);
    }

    #[test]
    fn a_callable_reports_its_statement_stream_with_nesting() {
        let src = "package p\nfunc F(xs []int) int {\n\tfor _, x := range xs {\n\t\tg(x)\n\t}\n\treturn 0\n}\n";
        let d = scan(src).into_iter().find(|d| d.name == "F").expect("F");
        let depths: Vec<u16> = d.facets.statements.iter().map(|s| s.depth).collect();
        assert_eq!(depths, vec![0, 1, 2, 1], "statements: {:?}", d.facets.statements);
        assert!(d.facets.statements[0].line.starts_with("Func("), "the head is the definition itself");
        assert!(d.facets.statements[1].line.starts_with("For("), "the loop contributes a header");
    }

    #[test]
    fn an_if_initializer_is_a_step_of_its_own() {
        // `if v, err := f(); err != nil` and the same code split over two lines read alike.
        let joined = "package p\nfunc F() error {\n\tif v, err := load(); err != nil {\n\t\treturn wrap(err, v)\n\t}\n\treturn nil\n}\n";
        let split = "package p\nfunc G() error {\n\tv, err := load()\n\tif err != nil {\n\t\treturn wrap(err, v)\n\t}\n\treturn nil\n}\n";
        let stream = |src: &str, name: &str| -> Vec<(u16, String)> {
            scan(src)
                .into_iter()
                .find(|d| d.name == name)
                .expect(name)
                .facets
                .statements
                .into_iter()
                .skip(1)
                .map(|s| (s.depth, s.line))
                .collect()
        };
        assert_eq!(stream(joined, "F"), stream(split, "G"));
        assert_eq!(analysis(joined, "F").type3_lines[1..], analysis(split, "G").type3_lines[1..]);
    }

    #[test]
    fn an_else_if_chain_keeps_its_branches_as_siblings() {
        let src = "package p\nfunc F(n int) int {\n\tif n > 1 {\n\t\tg()\n\t} else if n > 0 {\n\t\th()\n\t} else {\n\t\tk()\n\t}\n\treturn n\n}\n";
        let d = scan(src).into_iter().find(|d| d.name == "F").expect("F");
        let shape: Vec<(u16, String)> =
            d.facets.statements.iter().map(|s| (s.depth, s.line.split('(').next().unwrap_or("").to_owned())).collect();
        assert_eq!(
            shape,
            vec![
                (0, "Func".into()),
                (1, "If".into()),
                (2, "ExprStmt".into()),
                (1, "Else".into()),
                (1, "If".into()),
                (2, "ExprStmt".into()),
                (1, "Else".into()),
                (2, "ExprStmt".into()),
                (1, "Return".into()),
            ]
        );
    }

    #[test]
    fn switch_cases_sit_under_their_switch() {
        let src = "package p\nfunc F(v any) int {\n\tswitch t := v.(type) {\n\tcase int, int64:\n\t\treturn use(t)\n\tdefault:\n\t\treturn 0\n\t}\n}\n";
        let d = scan(src).into_iter().find(|d| d.name == "F").expect("F");
        let shape: Vec<(u16, String)> =
            d.facets.statements.iter().map(|s| (s.depth, s.line.split('(').next().unwrap_or("").to_owned())).collect();
        assert_eq!(
            shape,
            vec![(0, "Func".into()), (1, "TypeSwitch".into()), (2, "TypeCase".into()), (3, "Return".into()), (2, "Default".into()), (3, "Return".into())]
        );
    }

    #[test]
    fn the_exact_canonical_keeps_types_the_near_copy_lines_slot_them() {
        let a = analysis("package p\nfunc F(p Token) (int, error) {\n\tv := p.Get(\"uid\")\n\treturn v, nil\n}\n", "F");
        let b = analysis("package p\nfunc G(q Claims) (int, error) {\n\tv := q.Get(\"uid\")\n\treturn v, nil\n}\n", "G");
        assert_ne!(a.xname_canonical, b.xname_canonical);
        assert_eq!(a.type3_lines, b.type3_lines);
        assert_eq!(a.type3_lines[0], "func _fn(_v0 Ty(_T0, )) (Ty(_T1, ), Ty(_T2, ))");
        assert_eq!(a.type_refs, ["Token", "int", "error"]);
    }

    #[test]
    fn a_declared_type_parameter_is_bound_not_vocabulary() {
        let a = analysis("package p\nfunc First[T any](items []T) T {\n\tv := items[0]\n\treturn v\n}\n", "First");
        assert_eq!(a.type_refs, Vec::<String>::new());
        assert!(a.type3_lines[0].starts_with("func _fn[_T0]("), "{:?}", a.type3_lines[0]);
    }

    #[test]
    fn the_parts_of_a_line_are_its_units() {
        let a = analysis("package p\nfunc F(rows []Row) int {\n\tfor _, r := range rows {\n\t\tif r.Ok && 0 < r.V {\n\t\t\tout = append(out, r.Render(true))\n\t\t}\n\t}\n\treturn 0\n}\n", "F");
        let units_of = |line: u32| -> Vec<&str> {
            a.type3_units.iter().filter(|(l, _)| *l == line).map(|(_, u)| u.as_str()).collect()
        };
        // The `if` header: the comparison and the field reads are parts; the whole test is the line.
        let test = units_of(2);
        assert!(test.iter().any(|u| u.starts_with("Bin(Int('0'), '<', Field(")), "{test:?}");
        assert!(!test.iter().any(|u| u.contains("'&&'")), "the test itself is the line: {test:?}");
        // The append: the method call inside is a part of its line.
        assert!(units_of(3).iter().any(|u| u.starts_with("Method(Path('_v1'), 'Render'")), "{:?}", units_of(3));
    }

    #[test]
    fn a_declaration_with_a_syntax_error_is_skipped_and_the_rest_read() {
        let src = "package p\nfunc Broken(x int) int {\n\ty := x +\n}\nfunc Fine(x int) int {\n\ty := x + 1\n\treturn y\n}\n";
        let fns = names_of_kind(src, "functions");
        assert_eq!(fns, vec!["Fine".to_owned()], "{fns:?}");
    }

    fn lens_record(src: &str, name: &str) -> Vec<String> {
        let f: Arc<str> = Arc::from("t.go");
        let kinds = vec!["lenses".to_owned()];
        let opts = ScanOpts { kinds: Some(&kinds) };
        scan_source(src, &f, &opts)
            .into_iter()
            .find(|d| d.kind.id == "lenses" && d.name == name)
            .map(|d| d.analysis.expect("lens analysis").type3_lines)
            .unwrap_or_default()
    }

    #[test]
    fn lenses_are_absent_unless_the_run_asks_for_them() {
        let src = "package p\nfunc F(x int) int {\n\tif x > 0 {\n\t\tg(x)\n\t}\n\treturn x\n}\n";
        assert!(scan(src).iter().all(|d| d.kind.id != "lenses"), "the default run stays byte-identical");
        assert!(!lens_record(src, "F").is_empty());
    }

    #[test]
    fn a_go_callable_projects_through_every_lens_it_can_answer() {
        let src = r#"package p

import (
	"errors"
	"os"
)

var ErrEmpty = errors.New("empty")

func Load(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf, err := readAll(f)
	if len(buf) == 0 {
		return nil, ErrEmpty
	}
	for _, b := range buf {
		emit(b)
	}
	return buf, err
}
"#;
        let facts = lens_record(src, "Load");
        let has = |f: &str| facts.iter().any(|x| x == f);
        assert!(has("outgoing:os.Open"), "{facts:?}");
        assert!(has("outgoing:emit"), "{facts:?}");
        assert!(has("control:if"), "{facts:?}");
        assert!(has("control:for"), "{facts:?}");
        assert!(has("control:+return"), "nesting rides on the tag: {facts:?}");
        // Go's failures: the error handed back, and the sentinel returned.
        assert!(has("failures:?"), "{facts:?}");
        assert!(has("failures:ErrEmpty"), "{facts:?}");
        // What it holds open is what the deferred release acquired.
        assert!(has("resources:os.Open"), "{facts:?}");
        assert!(has("signature:arity=1"), "{facts:?}");
        assert!(has("signature:ret:[]"), "{facts:?}");
        assert!(facts.iter().any(|f| f.starts_with("scope:")), "the widest rung rides as one fact: {facts:?}");
        // The file's own name is erased from the lenses: `ErrEmpty` is a failure, not a callee.
        assert!(!has("outgoing:ErrEmpty"), "{facts:?}");
    }

    #[test]
    fn a_release_of_something_not_bound_here_is_the_release_itself() {
        let src = "package p\nfunc (s *S) Bump(n int) int {\n\ts.mu.Lock()\n\tdefer s.mu.Unlock()\n\ts.n += n\n\treturn s.n\n}\n";
        let facts = lens_record(src, "S.Bump");
        assert!(facts.contains(&"resources:.Unlock".to_owned()), "{facts:?}");
        assert!(facts.contains(&"signature:method".to_owned()), "{facts:?}");
    }

    #[test]
    fn a_struct_declares_a_shape_as_a_set_with_its_tag_keys_as_options() {
        // Three fields: a record of fewer facts carries no shape (`MIN_FACTS`), and Go has no
        // attribute to add a fourth.
        let a = "package p\ntype S struct {\n\tA int `json:\"a\" db:\"a\"`\n\tB []string\n\tC *time.Time\n}\n";
        let b = "package p\ntype S struct {\n\tC *time.Time\n\tB []string\n\tA int `json:\"x\" db:\"x\"`\n}\n";
        // A declaration is a set; a tag's keys are the field's options, its values are not compared.
        assert_eq!(lens_record(a, "S"), lens_record(b, "S"));
        assert!(lens_record(a, "S").contains(&"schema:int json db".to_owned()), "{:?}", lens_record(a, "S"));
    }

    #[test]
    fn divergent_bodies_with_identical_handling_share_their_use_facts() {
        use dup_defs_core::uses::use_facts;
        // Two caches whose bodies share nothing, written by key, read by key and swept by age the
        // same way — one used from inside its package, the other through its package name.
        let a = concat!(
            "package a\n",
            "type ImageCache struct { key string; blob []byte; deadAt int64 }\n",
            "func PutImage(s *Session, k string, v []byte) { s.Add(&ImageCache{key: k, blob: v, deadAt: soon()}) }\n",
            "func GetImage(s *Session, k string) *Row { return s.Find(ByKey(ImageCache{}, k)) }\n",
            "func ReapImages(s *Session) { s.Delete(OlderThan(ImageCache{}, now())) }\n",
        );
        let b = concat!(
            "package b\n",
            "import \"x/store\"\n",
            "func PutPrompt(s *Session, k string, v Json) { s.Add(&store.PromptStore{key: k, blob: v, deadAt: soon()}) }\n",
            "func GetPrompt(s *Session, k string) *Row { return s.Find(ByKey(store.PromptStore{}, k)) }\n",
            "func ReapPrompts(s *Session) { s.Delete(OlderThan(store.PromptStore{}, now())) }\n",
        );
        let c = "package store\ntype PromptStore struct { payload Json }\n";
        let (fa, fb, fc): (Arc<str>, Arc<str>, Arc<str>) = (Arc::from("a.go"), Arc::from("b.go"), Arc::from("c.go"));
        let opts = ScanOpts::default();
        let mut defs = scan_source(a, &fa, &opts);
        defs.extend(scan_source(b, &fb, &opts));
        defs.extend(scan_source(c, &fc, &opts));
        let facts = use_facts(&[(a, "a.go", 1), (b, "b.go", 1), (c, "c.go", 1)], &defs, crate::canon::collect_sites);
        let get = |n: &str| facts.get(n).unwrap_or_else(|| panic!("no use facts for {n}: {facts:?}")).clone();
        assert_eq!(get("ImageCache"), get("PromptStore"), "identical handling ⇒ identical use facts");
    }

    #[test]
    fn a_struct_canonical_reads_fields_not_tags() {
        let src = "package p\ntype A struct {\n\tID int `json:\"id\"`\n\tName, Email string\n\tBase\n}\ntype B struct {\n\tID int `db:\"id\"`\n\tName, Email string\n\tBase\n}\n";
        let all = scan(src);
        let canon = |n: &str| all.iter().find(|d| d.name == n).and_then(|d| d.cluster_canonical.clone()).expect(n);
        assert_eq!(canon("A").replace("'A'", "'X'"), canon("B").replace("'B'", "'X'"));
        assert!(canon("A").contains("Field('Email', Ty('string', ))"), "{}", canon("A"));
        assert!(canon("A").contains("Embed(Ty('Base', ))"), "{}", canon("A"));
    }
}
