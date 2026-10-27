//! Dev aid: print the tree-sitter S-expression of a C file (`cargo run -p c-canon --example c_sexp -- f.c`).
fn main() {
    let path = std::env::args().nth(1).expect("usage: c_sexp <file.c>");
    let src = std::fs::read_to_string(&path).expect("read");
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_c::LANGUAGE.into()).expect("grammar");
    let tree = p.parse(&src, None).expect("parse");
    println!("{}", tree.root_node().to_sexp());
}
