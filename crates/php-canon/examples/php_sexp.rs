//! Dev aid: print the tree-sitter S-expression of a PHP file (`cargo run -p php-canon --example php_sexp -- f.php`).
fn main() {
    let path = std::env::args().nth(1).expect("usage: php_sexp <file.php>");
    let src = std::fs::read_to_string(&path).expect("read");
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_php::LANGUAGE_PHP.into()).expect("grammar");
    let tree = p.parse(&src, None).expect("parse");
    println!("{}", tree.root_node().to_sexp());
}
