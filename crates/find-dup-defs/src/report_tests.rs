use std::path::Path;

use super::{short_path, short_paths};

#[test]
fn directives_preserve_note_order_summed_severity_and_suppressed_hits() {
    use super::{apply_directives, Finding, LintAction, Severity};
    use dup_defs_core::kinds::FUNCTIONS;

    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mixed"));
    let finding = |name: &str, severity| Finding {
        pass: "cross-name",
        kind: &FUNCTIONS,
        name: name.to_owned(),
        severity,
        min_sim: Some(1.0),
        loc: 3,
        args: 1,
        thickness: 0.5,
        snippet: String::new(),
        notes: vec!["original".to_owned()],
        pattern: None,
        facets: Vec::new(),
        members: vec![
            (root.join("py/core_b.py").display().to_string(), 2, 0),
            (root.join("py/core_a.py").display().to_string(), 1, 0),
            (root.join("py/core_a.py").display().to_string(), 3, 0),
        ],
    };
    let mut findings = vec![
        finding("first/alias", Severity::Error),
        finding("hidden", Severity::Warning),
        finding("second", Severity::Info),
    ];
    let directives: Vec<_> = [
        "de-escalate:<functions>*@*core_a.py=down",
        "escalate:alias=up1",
        "escalate:alias=up2",
        "de-escalate:alias=down2",
        "note:alias=last",
        "suppress:hidden=removed",
        "note:hidden=still audited",
        "note:<methods>*=wrong kind",
        "note:*@*absent.py=wrong path",
        "set:max-name-group=256",
    ]
    .iter()
    .map(|s| directiva::parse_as::<LintAction>(s).unwrap())
    .collect();

    let hits = apply_directives(&mut findings, &directives, root);
    assert_eq!(
        findings.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
        ["first/alias", "second"]
    );
    assert_eq!(findings[0].severity, Severity::Error);
    assert_eq!(findings[1].severity, Severity::Info);
    assert_eq!(findings[0].notes, ["original", "down", "up1", "up2", "down2", "last"]);
    assert_eq!(findings[1].notes, ["original", "down"]);
    assert_eq!(
        hits[0].iter().map(|h| h.key.as_str()).collect::<Vec<_>>(),
        ["dup-xname first/alias", "dup-xname hidden", "dup-xname second"]
    );
    assert_eq!(
        hits[0][0].files,
        ["core_a.py", "core_b.py"].map(|name| Path::new("py").join(name).display().to_string())
    );
    assert_eq!(hits[5][0].key, "dup-xname hidden");
    assert_eq!(hits[6][0].key, "dup-xname hidden");
    assert!(hits[7..].iter().all(Vec::is_empty));
}

#[test]
fn batched_paths_match_scalar_resolution() {
    let root = std::env::temp_dir().join(format!("fdd-report-paths-{}", std::process::id()));
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/real.py"), "VALUE = 1\n").unwrap();
    std::fs::write(root.join("outside.py"), "VALUE = 1\n").unwrap();
    let paths = vec![
        repo.join("src/real.py"),
        repo.join("src/../src/real.py"),
        repo.join("src/missing.py"),
        root.join("outside.py"),
    ];
    #[cfg(unix)]
    let paths = {
        use std::os::unix::fs::symlink;
        let mut paths = paths;
        symlink("src/real.py", repo.join("alias.py")).unwrap();
        symlink("../outside.py", repo.join("external.py")).unwrap();
        symlink("missing.py", repo.join("dangling.py")).unwrap();
        symlink("src", repo.join("linked")).unwrap();
        paths.extend([
            repo.join("alias.py"),
            repo.join("external.py"),
            repo.join("dangling.py"),
            repo.join("linked/real.py"),
        ]);
        paths
    };
    let files: Vec<&str> = paths.iter().map(|p| p.to_str().unwrap()).collect();
    for base in [&repo, &root.join("missing-root")] {
        let canonical = std::fs::canonicalize(base).unwrap_or_else(|_| base.clone());
        let shown = short_paths(&files, &canonical);
        for file in &files {
            assert_eq!(shown[file], short_path(file, base), "{file} under {}", base.display());
        }
    }
    assert_eq!(short_path(files[0], &repo), Path::new("src").join("real.py").to_string_lossy());
    std::fs::remove_dir_all(root).unwrap();
}
