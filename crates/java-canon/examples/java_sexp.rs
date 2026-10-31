//! Dev aid: print the tree-sitter S-expression of a Java file (`cargo run -p java-canon --example java_sexp -- f.java`).
fn main() {
    let path = std::env::args().nth(1).expect("usage: java_sexp <file.java>");
    let src = std::fs::read_to_string(&path).expect("read");
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_java::LANGUAGE.into()).expect("java grammar");
    let tree = parser.parse(&src, None).expect("parse");
    println!("{}", tree.root_node().to_sexp());
}
