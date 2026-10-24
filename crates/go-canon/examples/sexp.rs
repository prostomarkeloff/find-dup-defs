//! Dev aid: print the tree-sitter S-expression of a Go file (`cargo run -p go-canon --example sexp -- f.go`).
fn main() {
    let path = std::env::args().nth(1).expect("usage: sexp <file.go>");
    let src = std::fs::read_to_string(&path).expect("read");
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_go::LANGUAGE.into()).expect("grammar");
    let tree = p.parse(&src, None).expect("parse");
    println!("{}", tree.root_node().to_sexp());
}
