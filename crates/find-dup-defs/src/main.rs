//! `find-dup-defs` CLI — thin shell over the library pipeline.
//!
//! Pipeline + types + 3 passes live in `lib.rs`; this binary parses CLI flags,
//! sorts findings, applies user directives + calibration, and renders the
//! human or JSON report.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use clap::Parser;
use dup_defs_core::{Frontend, KindSpec};
use find_dup_defs::{cluster, collect_defs, large_name_groups, section_index, Finding, GpuMode, PipelineOpts, Severity};
use serde::Serialize;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(about, version)]
#[allow(clippy::struct_excessive_bools)] // CLI flags, not a state machine
struct Cli {
    /// Files or directories to scan (directories are walked for `*.py`).
    #[arg(required = true)]
    paths: Vec<PathBuf>,
    /// Name-gated clustering floor: same-named defs cluster if their exact RO ratio is ≥ this.
    #[arg(short, long, default_value_t = 0.5)]
    threshold: f64,
    /// Name-gated ERROR floor: a cluster's min pairwise ratio ≥ this gates as ERROR (else WARNING).
    #[arg(short, long, default_value_t = 0.85)]
    error_threshold: f64,
    /// De-escalate any cluster whose `thickness` < this from ERROR to WARNING — the calibration
    /// knob for "what counts as a real refactor candidate" per codebase. WARNINGs stay WARNINGs
    /// at this stage (`--warning-thickness` handles their tier); findings are NEVER dropped.
    /// `0.0` (default) leaves severities untouched. Try `0.3` to mute 2-3 line copy-pastes,
    /// `0.5` for genuinely fat candidates only.
    #[arg(long, default_value_t = 0.0)]
    error_thickness: f64,
    /// De-escalate any WARNING below this thickness to INFO — symmetric to `--error-thickness`,
    /// keeps WARNING meaningful by routing low-confidence stuff to the INFO tier instead of
    /// letting it pile up. Default `0.0` leaves WARNINGs untouched.
    #[arg(long, default_value_t = 0.0)]
    warning_thickness: f64,
    /// Escalate any non-ERROR cluster whose `thickness` ≥ this to ERROR — symmetric inverse of
    /// `--error-thickness`. Catches the "fat cluster that landed in WARNING because of mid-sim
    /// or small-canonical heuristics, but is actually a big-mass refactor target." Default
    /// `0.0` disabled. Applied LAST so it overrides the de-escalation knobs above.
    #[arg(long, default_value_t = 0.0)]
    escalate_thickness: f64,
    /// Include INFO-severity findings in the human-readable report. JSON output always
    /// contains them. Default hides INFO so the normal report stays focused on the actionable
    /// ERROR/WARNING list.
    #[arg(long)]
    show_info: bool,
    /// Repo root for relative paths in the report (paths under it are shown repo-relative).
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    /// Type-3 cosine detection floor (candidate edge when cosine > this).
    #[arg(long, default_value_t = 0.7)]
    type3_theta: f64,
    /// Only report clusters with at least this many definitions.
    #[arg(long, default_value_t = 2)]
    min_size: usize,
    /// Skip name-gated clustering for any (kind, name) group with more than N members. A name
    /// shared by hundreds of definitions — `fn main` across thousands of test fixtures,
    /// `new` / `default` — is a convention or entry point, not a refactor cluster, and the
    /// within-group O(n²) Ratcliff–Obershelp comparison can dominate runtime on huge monorepos
    /// (it was ~90% of CPU on the full rust compiler tree). Off by default (no cap); renamed-
    /// identical copies among the members still surface via the cross-name pass.
    #[arg(long)]
    max_name_group: Option<usize>,
    /// Restrict to these kinds (comma-separated:
    /// functions,methods,classes,interfaces,constants,type-aliases).
    #[arg(long, value_delimiter = ',')]
    kinds: Option<Vec<String>>,
    /// Restrict the scan to specific language frontends (comma-separated). Default: every
    /// supported language found in the target paths. Known codes:
    ///   `py` — Python via `py-canon` (Ruff parser)
    ///   `ts` — TypeScript via `ts-canon` (oxc parser)
    ///   `rs` — Rust via `rs-canon` (syn parser)
    /// Unknown codes exit non-zero. Useful in mixed-language repos where you only care about
    /// one frontend per CI job (`--only py`, `--only ts`), or when a frontend's parser is
    /// crashing on bad input and you need to scan around it.
    #[arg(long, value_delimiter = ',')]
    only: Option<Vec<String>>,
    /// Skip the cross-name (renamed-identical) pass.
    #[arg(long)]
    no_cross_name: bool,
    /// Skip the Type-3 (renamed near-copy) pass.
    #[arg(long)]
    no_type3: bool,
    /// Only report ERROR-severity clusters.
    #[arg(long)]
    errors_only: bool,
    /// Emit JSON instead of the human-readable report.
    #[arg(long)]
    json: bool,
    /// Print a thickness-calibration report instead of the normal duplicate list — distribution
    /// of current ERROR thicknesses + three percentile-anchored `--error-thickness` candidates
    /// (`permissive`/`balanced`/`strict` at p50/p75/p90). Pairs with `--json` for machine
    /// output. Respects `--kinds` / `--min-size` so you can calibrate against a focused subset.
    #[arg(long)]
    calibrate: bool,
    /// Filter findings by a compact glob-rule. Repeatable. Format:
    ///   `ACTION:[KIND:]NAME[@PATH][=NOTE]`
    ///
    /// `ACTION` ∈ `suppress` (drop entirely) | `de-escalate` (ERROR → WARNING) | `escalate`
    ///            (WARNING → ERROR) | `note` (attach text, no severity change). Same vocabulary
    ///            as iilint's `[tool.iilint].directives`.
    /// `KIND`   ∈ `METHOD` | `FUNCTION` | `CLASS` | `CONSTANT` | `TYPE_ALIAS` (optional).
    /// `NAME`   glob on the cluster's dup name (`Class.method` or `a/b/c` for cross-name).
    ///          `*` matches any chars, `?` matches one. Tested against each `/`-separated alias.
    /// `PATH`   glob on member file paths (any member match wins).
    /// `NOTE`   free-form annotation surfaced next to the finding (required for `note`,
    ///          optional self-documentation for the other three).
    ///
    /// Examples:
    ///   `-D de-escalate:Plugin.get_*_hook=intentional plugin no-op API`
    ///   `-D suppress:FUNCTION:spawn@*mypyc/lib-rt/*=bootstrap copy`
    ///   `-D de-escalate:METHOD:*.test_*@*/test/*=parametrize candidate`
    ///   `-D escalate:METHOD:Lock.*@*/storage/*=Lock dups block this release`
    ///   `-D note:METHOD:For*.begin_body=v2 refactor target`
    #[arg(long = "directive", short = 'D', value_name = "DIRECTIVE")]
    directives: Vec<String>,
}

// ───────────────────────────── directives (glob filters) ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DirectiveAction {
    /// Drop the finding from the report entirely.
    Suppress,
    /// ERROR → WARNING. WARNING stays WARNING.
    Deescalate,
    /// WARNING → ERROR. ERROR stays ERROR.
    Escalate,
    /// Pure annotation — attach a note to the finding, no severity change.
    Note,
    /// Pipeline configuration, not a finding filter: `settings:KEY=VALUE` (e.g.
    /// `settings:max-name-group=256`). Applied to [`PipelineOpts`] before the scan; arbitrary
    /// keys, so new knobs can be configured through the same `-D` channel CI already passes.
    Settings,
}

/// One compact directive: `ACTION:[KIND:]NAME[@PATH]`. See the `--directive` CLI doc for the
/// grammar and examples. Parsed once into globs + a kind filter; matched against each Finding.
#[derive(Debug)]
struct Directive {
    action: DirectiveAction,
    /// Internal kind tag (`functions` / `methods` / `classes` / `constants` / `type-aliases`).
    /// `None` matches any kind.
    kind: Option<&'static str>,
    name_pat: String,
    path_pat: Option<String>,
    /// Self-documentation attached after `=` in the spec — surfaced alongside the finding so
    /// the next reviewer sees WHY this directive was added without grepping the directive file.
    /// Required for `note`; optional but encouraged for `suppress`/`de-escalate`/`escalate`.
    note: Option<String>,
}

impl Directive {
    fn matches(&self, f: &Finding) -> bool {
        // `settings:` is pipeline config, applied before the scan — it never filters findings.
        if self.action == DirectiveAction::Settings {
            return false;
        }
        if let Some(k) = self.kind {
            if f.kind.id != k {
                return false;
            }
        }
        // Cross-name passes join aliases with '/'. Match any individual alias OR the joined
        // form, so a pattern like `Foo.bar` lands on `Foo.bar/Baz.bar` without needing wildcards.
        let any_alias = f.name.split('/').any(|alias| glob_match(&self.name_pat, alias));
        if !(any_alias || glob_match(&self.name_pat, &f.name)) {
            return false;
        }
        if let Some(pp) = &self.path_pat {
            if !f.members.iter().any(|(file, _, _)| glob_match(pp, file)) {
                return false;
            }
        }
        true
    }
}

/// Parse `ACTION:[KIND:]NAME[@PATH]` into a [`Directive`]. The KIND segment is optional — if the
/// first `:`-delimited chunk after the action matches a known kind label (`METHOD`/`FUNCTION`/
/// `CLASS`/`CONSTANT`/`TYPE_ALIAS`, case-insensitive, `_` or `-` for the alias), it's consumed;
/// otherwise the whole remainder is treated as the NAME glob.
fn parse_directive(spec: &str) -> Result<Directive, String> {
    // Strip the optional `=NOTE` tail first — note text is free-form (may contain `:` or `@`).
    let (head, note) = match spec.split_once('=') {
        Some((h, n)) => (h, Some(n.trim().to_owned())),
        None => (spec, None),
    };
    let (action_str, rest) = head
        .split_once(':')
        .ok_or_else(|| format!("expected `ACTION:…` in directive {spec:?}"))?;
    let action = match action_str.trim().to_ascii_lowercase().replace('-', "").as_str() {
        "suppress" => DirectiveAction::Suppress,
        "deescalate" => DirectiveAction::Deescalate,
        "escalate" => DirectiveAction::Escalate,
        "note" => DirectiveAction::Note,
        "settings" => DirectiveAction::Settings,
        other => {
            return Err(format!(
                "unknown action {other:?} in directive {spec:?} \
                 (expected `suppress` / `de-escalate` / `escalate` / `note` / `settings`)"
            ));
        }
    };
    if matches!(action, DirectiveAction::Note | DirectiveAction::Settings) && note.is_none() {
        return Err(format!(
            "`{action_str}` directive requires `=…` ({}) (directive: {spec:?})",
            if action == DirectiveAction::Settings { "the setting value, e.g. settings:max-name-group=256" } else { "note text" }
        ));
    }
    let (kind, after_kind) = match rest.split_once(':') {
        Some((maybe_kind, after)) => {
            let token = maybe_kind.trim().to_ascii_uppercase().replace('-', "_");
            match token.as_str() {
                // `*` is an explicit "any kind" — consume the segment, no kind filter.
                "*" => (None, after),
                "METHOD" | "METHODS" => (Some("methods"), after),
                "FUNCTION" | "FUNCTIONS" => (Some("functions"), after),
                "CLASS" | "CLASSES" => (Some("classes"), after),
                "INTERFACE" | "INTERFACES" => (Some("interfaces"), after),
                "CONSTANT" | "CONSTANTS" => (Some("constants"), after),
                "TYPE_ALIAS" | "TYPE_ALIASES" => (Some("type-aliases"), after),
                _ => (None, rest), // not a kind — the whole `rest` is NAME[@PATH]
            }
        }
        None => (None, rest),
    };
    let (name, path) = match after_kind.split_once('@') {
        Some((n, p)) => (n.trim(), Some(p.trim())),
        None => (after_kind.trim(), None),
    };
    if name.is_empty() {
        return Err(format!("empty name pattern in directive {spec:?}"));
    }
    Ok(Directive {
        action,
        kind,
        name_pat: name.to_owned(),
        path_pat: path.map(str::to_owned),
        note,
    })
}

/// Name-gated groups larger than this are reported by the directive-inferrer as a suggested
/// `settings:max-name-group` cap (a name shared by this many definitions is a convention or
/// entry point), and it's the cap value the suggestion proposes.
const SUGGEST_CAP: usize = 256;

/// Apply one `settings:KEY=VALUE` directive to the pipeline options. Known keys are validated
/// (a bad value for a known key is a hard error); unknown keys are warned-and-ignored so a
/// directive file written for a newer build degrades gracefully on an older one.
fn apply_setting(opts: &mut PipelineOpts, key: &str, value: &str) {
    match key {
        "max-name-group" => {
            let Ok(n) = value.parse::<usize>() else {
                eprintln!("find-dup-defs: settings:max-name-group expects an integer, got {value:?}");
                std::process::exit(2);
            };
            opts.max_name_group = Some(n);
        }
        // Backend for the name-gated clustering. Accepts boolean-ish words and the three
        // `difflib-fast` concurrency names; `on` maps to the recommended GPU+CPU mode. Only takes
        // effect on a `--features gpu` macOS build (else the Rationer runs CPU with identical
        // output), but the directive parses and validates everywhere so a shared config is portable.
        "gpu" => {
            opts.gpu = match value.trim().to_ascii_lowercase().as_str() {
                "off" | "cpu" | "false" | "no" | "0" => GpuMode::Cpu,
                "on" | "true" | "yes" | "1" | "gpu+cpu" | "gpucpu" | "gpu_cpu" => GpuMode::GpuPlusCpu,
                "gpu" => GpuMode::Gpu,
                other => {
                    eprintln!(
                        "find-dup-defs: settings:gpu expects on/off (or cpu/gpu/gpu+cpu), got {other:?}"
                    );
                    std::process::exit(2);
                }
            };
        }
        other => eprintln!("find-dup-defs: ignoring unknown settings key {other:?}"),
    }
}

/// Minimal glob matcher — `*` (any run), `?` (single char), `{a,b,…}` (alternation, single
/// level). Recursive-backtracking; cheap for the short patterns directives carry. Alternation
/// lets one directive cover multiple test-tree conventions (`*/{test,tests,__tests__}/*`) in
/// one paste, instead of asking the user to chain three `-D` flags.
fn glob_match(pat: &str, s: &str) -> bool {
    for expanded in expand_braces(pat) {
        if glob_match_simple(&expanded, s) {
            return true;
        }
    }
    false
}

fn glob_match_simple(pat: &str, s: &str) -> bool {
    fn go(p: &[u8], pi: usize, t: &[u8], ti: usize) -> bool {
        if pi == p.len() {
            return ti == t.len();
        }
        match p[pi] {
            b'*' => {
                // Collapse runs of `*` so `**name` doesn't blow up branching.
                let mut j = pi + 1;
                while j < p.len() && p[j] == b'*' {
                    j += 1;
                }
                for k in ti..=t.len() {
                    if go(p, j, t, k) {
                        return true;
                    }
                }
                false
            }
            b'?' => ti < t.len() && go(p, pi + 1, t, ti + 1),
            c => ti < t.len() && t[ti] == c && go(p, pi + 1, t, ti + 1),
        }
    }
    go(pat.as_bytes(), 0, s.as_bytes(), 0)
}

/// Expand one level of `{a,b,c}` brace alternation, recursively. `{test,tests}` ⇒ two
/// patterns. Empty alternatives are legal (`{,s}` ⇒ both empty and `s`). Nesting beyond one
/// `{...}` is supported via recursion — leftmost group is expanded first, the rest of the
/// pattern is re-expanded on the result. Unmatched / empty braces fall through as literals so a
/// pattern with no alternation is one-call cheap.
#[cfg(test)]
mod glob_tests {
    use super::{expand_braces, glob_match, parse_directive, vendored_score, DirectiveAction, GpuMode, PipelineOpts};

    #[test]
    fn settings_directive_parses_key_and_value() {
        let d = parse_directive("settings:max-name-group=256").expect("parses");
        assert_eq!(d.action, DirectiveAction::Settings);
        assert_eq!(d.name_pat, "max-name-group");
        assert_eq!(d.note.as_deref(), Some("256"));
        assert!(d.kind.is_none() && d.path_pat.is_none());
        // a settings directive without `=VALUE` is rejected.
        assert!(parse_directive("settings:max-name-group").unwrap_err().contains("requires"));
    }

    #[test]
    fn apply_setting_sets_max_name_group() {
        let mut opts = PipelineOpts::with_paths(vec![]);
        assert_eq!(opts.max_name_group, None);
        super::apply_setting(&mut opts, "max-name-group", "256");
        assert_eq!(opts.max_name_group, Some(256));
        // unknown key is ignored (forward-compatible), not fatal.
        super::apply_setting(&mut opts, "future-knob", "x");
        assert_eq!(opts.max_name_group, Some(256));
    }

    #[test]
    fn apply_setting_sets_gpu_mode() {
        let mut opts = PipelineOpts::with_paths(vec![]);
        assert_eq!(opts.gpu, GpuMode::Cpu); // default
        super::apply_setting(&mut opts, "gpu", "on");
        assert_eq!(opts.gpu, GpuMode::GpuPlusCpu);
        super::apply_setting(&mut opts, "gpu", "gpu");
        assert_eq!(opts.gpu, GpuMode::Gpu);
        super::apply_setting(&mut opts, "gpu", "GPU+CPU"); // case-insensitive
        assert_eq!(opts.gpu, GpuMode::GpuPlusCpu);
        super::apply_setting(&mut opts, "gpu", "off");
        assert_eq!(opts.gpu, GpuMode::Cpu);
    }

    #[test]
    fn vendored_score_marks_known_snapshot_paths() {
        // Strong markers — pair should produce a vendored directive.
        assert!(vendored_score("extensions/copilot/src/util/vs/base/common") > 0);
        assert!(vendored_score("extensions/copilot/test/simulation/fixtures/codeMapper") > 0);
        assert!(vendored_score("packages/third_party/upstream-snapshot") > 0);
        // No markers — generic project paths, architectural-drift candidates, not vendored.
        assert_eq!(vendored_score("src/pages/dashboard/analytics/audience/components/tags-table"), 0);
        assert_eq!(vendored_score("src/shared/ui/lists/tags-table"), 0);
        assert_eq!(vendored_score("src/features/admin-channel-add/model"), 0);
        assert_eq!(vendored_score("src/components/Button"), 0);
    }


    #[test]
    fn brace_expansion_covers_all_test_dir_conventions() {
        let pat = "*/{test,tests,__tests__}/*";
        let expanded = expand_braces(pat);
        assert_eq!(expanded.len(), 3);
        assert!(expanded.contains(&"*/test/*".to_owned()));
        assert!(expanded.contains(&"*/tests/*".to_owned()));
        assert!(expanded.contains(&"*/__tests__/*".to_owned()));
    }

    #[test]
    fn glob_match_handles_alternation_in_path() {
        let pat = "*/{test,tests,__tests__}/*";
        // `/test/` (singular) — angular convention
        assert!(glob_match(pat, "packages/compiler-cli/test/compliance/foo.ts"));
        // `/tests/` (plural) — svelte / standard pytest
        assert!(glob_match(pat, "packages/svelte/tests/css/test.ts"));
        // `/__tests__/` — jest / RTL
        assert!(glob_match(pat, "packages/next/src/__tests__/foo.test.ts"));
        // Non-test path should NOT match
        assert!(!glob_match(pat, "src/components/Button.ts"));
    }

    #[test]
    fn glob_match_file_extension_alternation() {
        let pat = "*.{test,spec}.*";
        assert!(glob_match(pat, "src/foo.test.ts"));
        assert!(glob_match(pat, "src/foo.spec.ts"));
        assert!(glob_match(pat, "packages/next/src/server/foo.external.test.ts"));
        assert!(!glob_match(pat, "src/foo.ts"));
    }

    #[test]
    fn empty_alternation_branch_is_legal() {
        let pat = "*/{,s}post*";
        let expanded = expand_braces(pat);
        assert!(expanded.contains(&"*/post*".to_owned()));
        assert!(expanded.contains(&"*/spost*".to_owned()));
    }

    #[test]
    fn no_braces_is_one_pattern() {
        assert_eq!(expand_braces("*tests/*"), vec!["*tests/*".to_owned()]);
    }
}

fn expand_braces(pat: &str) -> Vec<String> {
    let bytes = pat.as_bytes();
    let Some(open) = bytes.iter().position(|&b| b == b'{') else {
        return vec![pat.to_owned()];
    };
    let Some(close_rel) = bytes[open + 1..].iter().position(|&b| b == b'}') else {
        return vec![pat.to_owned()];
    };
    let close = open + 1 + close_rel;
    let prefix = &pat[..open];
    let alts = &pat[open + 1..close];
    let suffix = &pat[close + 1..];
    let mut out = Vec::new();
    for alt in alts.split(',') {
        let combined = format!("{prefix}{alt}{suffix}");
        out.extend(expand_braces(&combined));
    }
    out
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn main() {
    let cli = Cli::parse();
    // Frontend registry — the binary is the composition root that knows every language; the
    // engine itself is frontend-agnostic. Bound to locals so the `&dyn Frontend` refs outlive
    // the whole run.
    let (py, ts, rs) = (py_canon::Python, ts_canon::TypeScript, rs_canon::Rust);
    let registry: Vec<&dyn Frontend> = vec![&py, &ts, &rs];
    // `--only` selects a subset by language code. Unknown codes are an error (exit 2) rather
    // than a silent no-op — a typo'd `--only py,rs` should fail loud so a CI job that meant to
    // include Rust doesn't ship an empty report.
    let selected: Vec<&dyn Frontend> = match &cli.only {
        None => registry.clone(),
        Some(codes) => {
            for code in codes {
                if !registry.iter().any(|f| f.lang() == code) {
                    let known: Vec<&str> = registry.iter().map(|f| f.lang()).collect();
                    eprintln!(
                        "find-dup-defs: unknown language code {code:?} for --only (known: {})",
                        known.join(", ")
                    );
                    std::process::exit(2);
                }
            }
            registry.iter().copied().filter(|f| codes.iter().any(|c| c == f.lang())).collect()
        }
    };
    // User-authored directives, parsed once (exit-2 on a typo so CI fails loud). `settings:`
    // entries configure the pipeline before the scan; the rest filter findings afterwards.
    let directives: Vec<Directive> = cli
        .directives
        .iter()
        .map(|s| {
            parse_directive(s).unwrap_or_else(|e| {
                eprintln!("find-dup-defs: invalid --directive: {e}");
                std::process::exit(2);
            })
        })
        .collect();

    // `--max-name-group` is the base; `settings:max-name-group=…` directives override it (the
    // portable, inferrer-suggested way to configure the same knob).
    let mut opts = PipelineOpts {
        paths: cli.paths.clone(),
        threshold: cli.threshold,
        error_threshold: cli.error_threshold,
        type3_theta: cli.type3_theta,
        min_size: cli.min_size,
        error_thickness: cli.error_thickness,
        warning_thickness: cli.warning_thickness,
        escalate_thickness: cli.escalate_thickness,
        kinds: cli.kinds.clone(),
        no_cross_name: cli.no_cross_name,
        no_type3: cli.no_type3,
        max_name_group: cli.max_name_group,
        gpu: GpuMode::Cpu,
    };
    for d in directives.iter().filter(|d| d.action == DirectiveAction::Settings) {
        apply_setting(&mut opts, &d.name_pat, d.note.as_deref().unwrap_or_default());
    }

    // One scan → defs. The cheap `(kind, name)` group sizes feed the directive-inferrer's
    // `settings:max-name-group` suggestion (independent of clustering and the cap); then cluster.
    let defs = collect_defs(&selected, &opts.paths);
    let large_groups = large_name_groups(&defs, SUGGEST_CAP);
    let mut findings = cluster(defs, &opts);
    if !directives.is_empty() {
        // Attach notes from EVERY matching directive first, even ones whose action will drop
        // the finding — but we run suppress last, so a `suppress` with a `=note` still has its
        // note visible if some non-suppressing directive also matches. Order of effects:
        //   1. Notes accumulate from every match.
        //   2. Severity adjusts: escalate before de-escalate (so a conflicting pair lands at
        //      ERROR, the louder of the two — matches "the user explicitly asked to raise it").
        //   3. Suppress drops findings entirely.
        for f in &mut findings {
            for d in &directives {
                if d.matches(f) {
                    if let Some(n) = &d.note {
                        f.notes.push(n.clone());
                    }
                }
            }
            // Each matching directive contributes one step on the severity ladder — same
            // semantic as iilint's `severity_steps`. Multiple `de-escalate`s chain (ERROR →
            // WARNING → INFO); `escalate` cancels out (1 escalate + 1 de-escalate = no-op).
            let step = directives
                .iter()
                .filter(|d| d.matches(f))
                .map(|d| match d.action {
                    DirectiveAction::Deescalate => 1,
                    DirectiveAction::Escalate => -1,
                    _ => 0,
                })
                .sum::<i32>();
            if step != 0 {
                f.severity = Severity::from_index(f.severity.to_index() + step);
            }
        }
        findings.retain(|f| {
            !directives
                .iter()
                .any(|d| d.action == DirectiveAction::Suppress && d.matches(f))
        });
    }
    if cli.errors_only {
        findings.retain(|f| f.severity == Severity::Error);
    }
    // Detection/section order (constants, fn-name-gated, fn-cross-name, fn-Type-3, classes,
    // type-aliases), then within a section by name and first member — deterministic + reproducible.
    findings.sort_by(|a, b| {
        section_index(a)
            .cmp(&section_index(b))
            .then(a.name.cmp(&b.name))
            .then(a.members[0].cmp(&b.members[0]))
    });

    if cli.calibrate {
        // Calibration is informational — never exits non-zero, never prints the dup list. Runs
        // AFTER demotion/escalation knobs so each invocation answers "given my current
        // configuration, what's the next nudge worth?". Both ladders surface: ERROR drives
        // `--error-thickness` suggestions, WARNING drives `--warning-thickness`.
        let errs: Vec<&Finding> = findings.iter().filter(|f| f.severity == Severity::Error).collect();
        let warns: Vec<&Finding> = findings.iter().filter(|f| f.severity == Severity::Warning).collect();
        let report = if cli.json {
            render_calibration_json(&errs, &warns, &findings, &large_groups, &cli.repo_root)
        } else {
            format_calibration(&errs, &warns, &findings, &large_groups, &cli.repo_root)
        };
        print!("{report}");
        return;
    }

    let report = if cli.json {
        // JSON consumers get every severity unconditionally — it's their job to filter.
        render_json(&findings, &cli.repo_root)
    } else {
        // Human report hides INFO by default — that's the whole point of the tier. JSON path
        // unchanged so downstream tooling never loses data.
        let visible: Vec<Finding> = if cli.show_info {
            findings.clone()
        } else {
            findings.iter().filter(|f| f.severity != Severity::Info).cloned().collect()
        };
        format_report(&visible, &selected, cli.threshold, cli.error_threshold, &cli.repo_root)
    };
    print!("{report}");

    if findings.iter().any(|f| f.severity == Severity::Error) {
        std::process::exit(1);
    }
}

// ───────────────────────────── thickness calibration ─────────────────────────────

/// Linear-interpolated percentile on a value list, *sorted ascending*. `p` in `[0, 1]`. Returns
/// `0.0` on empty input — the format/JSON paths handle the "no errors to calibrate" case by
/// reading the list length, so the value here just needs to be a stable sentinel.
#[allow(
    clippy::cast_precision_loss, // cluster counts fit in f64 mantissa
    clippy::cast_possible_truncation, // floor/ceil already discrete and bounded by sorted.len()
    clippy::cast_sign_loss, // p ∈ [0, 1] and len ≥ 0 → rank is non-negative
)]
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn median_usize_sorted(sorted: &[usize]) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    sorted[sorted.len() / 2]
}

#[derive(Serialize)]
struct CalibSuggestion {
    label: &'static str,
    percentile: u8,
    error_thickness: f64,
    errors_kept: usize,
    median_loc: usize,
    median_args: usize,
    /// "What does this threshold actually catch?" — smallest cluster whose `thickness ≥
    /// error_thickness`, i.e. the threshold's lower-edge exemplar. Pairs with `example_snippet`.
    example_name: String,
    example_thickness: f64,
    example_loc: usize,
    example_args: usize,
    /// `<short-path>:<line>` for every member, so the user can jump to any copy. Full member
    /// list (not truncated) — JSON consumer can render however they want; the human formatter
    /// trims to a few entries inline.
    example_members: Vec<String>,
    /// Full source text of one representative member (pre-strip for methods). JSON keeps it
    /// verbatim; the human formatter dedents Python class-method indentation and box-frames it.
    example_snippet: String,
}

#[derive(Serialize)]
struct CalibHistBin {
    thickness_lo: f64,
    thickness_hi: f64,
    count: usize,
}

/// Per-severity calibration block — emitted twice in the JSON report, once for ERROR
/// (driving `--error-thickness`) and once for WARNING (driving `--warning-thickness`).
#[derive(Serialize)]
struct CalibTier {
    total: usize,
    /// CLI flag the suggestions would set — `error-thickness` for the ERROR block,
    /// `warning-thickness` for the WARNING block.
    target_flag: &'static str,
    histogram: Vec<CalibHistBin>,
    suggestions: Vec<CalibSuggestion>,
}

#[derive(Serialize)]
struct CalibReport {
    error: CalibTier,
    warning: CalibTier,
    /// Auto-detected noise patterns + ready-to-paste directives. Each entry quotes the exact
    /// `-D` string the user can copy. See [`infer_directives`] for what's checked.
    inferred_directives: Vec<InferredDirective>,
}

/// One auto-discovered noise pattern: a CLI-ready directive string + rationale + effect size,
/// so the calibrate user can decide whether to paste it into their CI invocation.
#[derive(Serialize, Clone)]
struct InferredDirective {
    /// Exact `-D <…>` string the user can paste verbatim.
    directive: String,
    /// One-sentence "why this matches your codebase" explanation, grounded in counts.
    rationale: String,
    /// How many existing findings the directive would touch (ERROR + WARNING + INFO combined).
    affects_total: usize,
    affects_error: usize,
    affects_warning: usize,
    affects_info: usize,
}

/// 10 bins of width 0.1 over `[0.0, 1.0]`; the top bin captures anything ≥ 1.0 (saturated
/// thickness never *exceeds* 1, but we collapse the boundary to make the upper-edge bucket
/// non-empty for huge defs).
fn thickness_histogram(values: &[f64]) -> Vec<CalibHistBin> {
    let bins: i32 = 10;
    let step = 1.0 / f64::from(bins);
    (0..bins)
        .map(|i| {
            let lo = f64::from(i) * step;
            let hi = f64::from(i + 1) * step;
            let count = values.iter().filter(|&&v| v >= lo && (v < hi || (i == bins - 1 && v >= hi))).count();
            CalibHistBin { thickness_lo: lo, thickness_hi: hi, count }
        })
        .collect()
}

/// Build the three percentile-anchored suggestions. For each anchor we report what the user
/// would actually get: how many ERROR clusters survive `T >= anchor`, the median size of those
/// survivors, AND the cluster sitting at the threshold's lower edge — its name + a snippet of
/// its source — so the user can SEE what kind of dup would still be ERROR after dialing this
/// knob. That's the difference between picking a number blindly and picking one against a
/// concrete example.
fn calibration_suggestions(errs: &[&Finding], repo_root: &Path) -> Vec<CalibSuggestion> {
    if errs.is_empty() {
        return Vec::new();
    }
    let mut ts: Vec<f64> = errs.iter().map(|f| f.thickness).collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    [("permissive", 50u8), ("balanced", 75), ("strict", 90)]
        .into_iter()
        .map(|(label, p)| {
            let t = percentile_sorted(&ts, f64::from(p) / 100.0);
            let kept: Vec<&&Finding> = errs.iter().filter(|f| f.thickness >= t).collect();
            let mut locs: Vec<usize> = kept.iter().map(|f| f.loc).collect();
            let mut args: Vec<usize> = kept.iter().map(|f| f.args).collect();
            locs.sort_unstable();
            args.sort_unstable();
            // The "smallest survivor" — the cluster closest to (but not below) the threshold.
            // Tiebreak by thickness ascending, then by name for determinism.
            let example = kept
                .iter()
                .min_by(|a, b| {
                    a.thickness
                        .partial_cmp(&b.thickness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.name.cmp(&b.name))
                })
                .copied();
            let (ex_name, ex_t, ex_loc, ex_args, ex_members, ex_snippet) = match example {
                Some(f) => {
                    let members: Vec<String> = f
                        .members
                        .iter()
                        .map(|(file, line, _)| format!("{}:{}", short_path(file, repo_root), line))
                        .collect();
                    (f.name.clone(), f.thickness, f.loc, f.args, members, f.snippet.clone())
                }
                None => (String::new(), 0.0, 0, 0, Vec::new(), String::new()),
            };
            CalibSuggestion {
                label,
                percentile: p,
                error_thickness: t,
                errors_kept: kept.len(),
                median_loc: median_usize_sorted(&locs),
                median_args: median_usize_sorted(&args),
                example_name: ex_name,
                example_thickness: ex_t,
                example_loc: ex_loc,
                example_args: ex_args,
                example_members: ex_members,
                example_snippet: ex_snippet,
            }
        })
        .collect()
}

/// Normalize Python class-method indentation for standalone display: the `def` line lives at
/// column 0 (we strip leading whitespace at extraction time), but the body still carries the
/// class-level extra 4-space step. We dedent the body to 4 spaces so a method snippet reads as
/// a regular function would in isolation. Top-level functions and bodies already at ≤4-space
/// indent are left untouched.
fn dedent_python_def_body(s: &str) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() < 2 {
        return s.to_owned();
    }
    let body_min = lines[1..]
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.chars().take_while(|c| *c == ' ').count())
        .min()
        .unwrap_or(0);
    if body_min <= 4 {
        return s.to_owned();
    }
    let strip = body_min - 4;
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    out.push(lines[0].to_owned());
    for l in &lines[1..] {
        if l.len() >= strip && l.chars().take(strip).all(|c| c == ' ') {
            out.push(l[strip..].to_owned());
        } else {
            out.push((*l).to_owned());
        }
    }
    out.join("\n")
}

/// Render a snippet inside a left-bar box (`│ …`) under the suggestion bullet — the visual
/// frame separates code from prose without compounding indentation. Long bodies collapse to
/// `max_lines` rows + an elision marker so the calibration view stays one screen.
fn snippet_box(snippet: &str, max_lines: usize) -> String {
    let cleaned = dedent_python_def_body(snippet);
    let lines: Vec<&str> = cleaned.lines().collect();
    let shown_count = lines.len().min(max_lines);
    let extra = lines.len().saturating_sub(shown_count);
    let mut out = String::new();
    out.push_str("    ┌──\n");
    for l in &lines[..shown_count] {
        out.push_str("    │ ");
        out.push_str(l);
        out.push('\n');
    }
    if extra > 0 {
        let _ = writeln!(out, "    │ … (+{extra} more lines)");
    }
    out.push_str("    └──");
    out
}

// ───────────────────────────── directive inference ─────────────────────────────

fn path_is_i18n(p: &str) -> bool {
    let lo = p.to_ascii_lowercase();
    lo.contains("/locale/") || lo.contains("/locales/") || lo.contains("/i18n/") || lo.contains("/translations/")
}
fn path_is_test(p: &str) -> bool {
    let lo = p.to_ascii_lowercase();
    // Directory markers seen in the wild across languages: pytest's `tests/`, Go-style
    // `test/`, jest/RTL's `__tests__/`, angular compliance harness `test_cases/`, framework
    // doc fixtures `__fixtures__/` / `fixtures/`. Any of these in the path makes the file a
    // test-tree resident.
    if lo.contains("/tests/")
        || lo.contains("/test/")
        || lo.contains("/__tests__/")
        || lo.contains("/test_cases/")
        || lo.contains("/test-cases/")
        || lo.contains("/__fixtures__/")
        || lo.contains("/fixtures/")
        || lo.contains("/integration/")
        || lo.contains("/e2e/")
    {
        return true;
    }
    let Some(fname) = p.rsplit('/').next() else { return false };
    // Python: `test_*.py` / `*_test.py`. TS/JS: `*.test.*` / `*.spec.*`.
    fname.starts_with("test_")
        || fname.ends_with("_test.py")
        || fname.contains(".test.")
        || fname.contains(".spec.")
}
fn path_is_generated(p: &str) -> bool {
    // Python protobuf / gRPC / hand-rolled `.gen.py`, plus JS/TS conventions for codegen output
    // (`*.generated.{ts,tsx,…}`, `__generated__/`).
    p.ends_with("_pb2.py")
        || p.ends_with("_pb2_grpc.py")
        || p.ends_with("_grpc.py")
        || p.ends_with(".gen.py")
        || p.ends_with(".generated.ts")
        || p.ends_with(".generated.tsx")
        || p.ends_with(".generated.js")
        || p.ends_with(".generated.jsx")
        || p.contains("/_generated/")
        || p.contains("/generated/")
        || p.contains("/__generated__/")
}
fn path_is_migration(p: &str) -> bool {
    p.contains("/migrations/") || p.contains("/alembic/versions/")
}
/// `.d.ts` declaration files — they parse cleanly as TypeScript and would otherwise inflate
/// "duplicate interface" / "duplicate type alias" findings with what's by design a type-only
/// distribution mechanism (each consumer re-declares its slice of an external API).
fn path_is_dts(p: &str) -> bool {
    p.ends_with(".d.ts") || p.ends_with(".d.tsx") || p.ends_with(".d.mts") || p.ends_with(".d.cts")
}
/// Storybook stories — each `.stories.tsx` typically pastes the same component-wrapping
/// boilerplate; refactoring it out defeats Storybook's "stories are self-contained examples"
/// model.
fn path_is_story(p: &str) -> bool {
    let fname = p.rsplit('/').next().unwrap_or(p);
    fname.contains(".stories.")
}
/// Tutorial / docs / example code — each snippet is a self-contained illustration that
/// reasonably shares boilerplate with siblings (same `app = FastAPI()` scaffold, same login
/// example, …). Refactoring them into shared helpers would defeat the "this snippet is
/// runnable on its own" point of doc examples.
fn path_is_doc_example(p: &str) -> bool {
    p.contains("/docs_src/")
        || p.contains("/docs/src/")
        || p.contains("/examples/")
        || p.contains("/example/")
        || p.contains("/tutorial/")
        || p.contains("/tutorials/")
        || p.contains("/samples/")
}

/// `(filename, dir)` of a member's path. Empty `dir` if the path is bare.
fn split_path(file: &str) -> (&str, &str) {
    match file.rfind('/') {
        Some(i) => (&file[i + 1..], &file[..i]),
        None => (file, ""),
    }
}

/// True iff every member of the cluster has the same filename, but they live in at least two
/// distinct directories. This is the **vendored-copy / fork signature**: the same physical file
/// has been checked into multiple roots (e.g. `src/vs/base/common/async.ts` plus
/// `extensions/copilot/src/util/vs/base/common/async.ts`), so every definition inside it
/// produces a "duplicate" cluster that is duplication by design. Real refactor candidates
/// almost always live in files with *different* names.
fn cluster_is_same_filename_across_dirs(f: &Finding) -> bool {
    let mut fnames: BTreeSet<&str> = BTreeSet::new();
    let mut dirs: BTreeSet<&str> = BTreeSet::new();
    for (file, _, _) in &f.members {
        let (fname, dir) = split_path(file);
        fnames.insert(fname);
        dirs.insert(dir);
    }
    fnames.len() == 1 && dirs.len() >= 2
}

/// Longest common SUFFIX-paths of `a` and `b` (path components, not raw chars). E.g.
/// `extensions/copilot/src/util/vs/base/common` vs `src/vs/base/common` ⇒
/// `("extensions/copilot/src/util", "")` since the trailing `vs/base/common` is shared. The
/// returned tuple is `(a_unique_prefix, b_unique_prefix)` — whichever is non-empty marks the
/// "extra" side, which for vendored snapshots is the suppressible vendored root.
fn split_at_shared_suffix<'a>(a: &'a str, b: &'a str) -> (String, String) {
    let ap: Vec<&str> = a.split('/').collect();
    let bp: Vec<&str> = b.split('/').collect();
    let mut shared = 0;
    while shared < ap.len() && shared < bp.len()
        && ap[ap.len() - 1 - shared] == bp[bp.len() - 1 - shared]
    {
        shared += 1;
    }
    let a_prefix = ap[..ap.len() - shared].join("/");
    let b_prefix = bp[..bp.len() - shared].join("/");
    (a_prefix, b_prefix)
}

/// Heuristic "this path looks like a vendored / fixture / fork copy" score. Higher = more
/// likely vendored. Used to pick the **source** directory from a same-filename cluster (the
/// lowest-scoring member) so the vendored-side glob lands on the right dirs.
///
/// Pure-shortest-path heuristic doesn't work in practice: e.g. vscode's real
/// `src/vs/editor/browser/widget/codeEditor` (5 components) is *shorter* than some vendored
/// counterparts but *longer* than `extensions/copilot/test/simulation/fixtures/vscode`
/// (5 components, alphabetically smaller), which would mis-identify the source.
fn vendored_score(dir: &str) -> i32 {
    let mut score = 0i32;
    if dir.contains("/fixtures/") || dir.contains("/__fixtures__/") {
        score += 10;
    }
    if dir.contains("/test/") || dir.contains("/tests/") || dir.contains("/__tests__/") {
        score += 5;
    }
    if dir.contains("/vendor/") || dir.contains("/vendored/") || dir.contains("/third_party/") {
        score += 8;
    }
    if dir.contains("/util/vs/") || dir.contains("/util/") {
        // `util/vs/` is the specific copilot-vendoring shape; `/util/` is a weaker generic
        // signal that some "utility" snapshot might live here.
        score += if dir.contains("/util/vs/") { 5 } else { 1 };
    }
    if dir.contains("/extensions/") || dir.contains("/extension/") {
        score += 1; // weak — extension dirs commonly mirror core source
    }
    score
}

/// Auto-detect "vendored / fork" patterns: clusters whose members all share a filename but
/// live in different directories. For each cluster the **lowest-scoring member directory** is
/// treated as the source (vendored-marker score from [`vendored_score`]); every other
/// directory contributes a `(vendored_dir, source_dir)` pair. The
/// longer path's unique prefix (divergent portion above any shared trailing components) is the
/// vendored root, and findings are aggregated under that root + rolled up to the deepest
/// ancestor whose total finding count clears the `MIN_CLUSTERS` floor — so we emit one
/// directive per *snapshot* root, not one per leaf-dir.
///
/// Handles both 2-dir clusters (one source, one vendored copy) and N-way snapshots (one
/// source + several vendored copies in different roots — common for test fixtures that mirror
/// the real source in multiple locations).
///
/// Findings clusters short-circuit to `>= MIN_CLUSTERS` after rollup — emit one directive per
/// distinct vendored *root*, not one per leaf-dir.
const VENDORED_MIN_CLUSTERS: usize = 30;

#[allow(clippy::too_many_lines)] // multi-stage pipeline (per-pair grouping, anchor rollup, parent-drop, directive emission) reads more clearly straight-through than broken across helpers
fn infer_vendored_directives(findings: &[Finding], repo_root: &Path) -> Vec<InferredDirective> {
    use std::collections::HashMap;
    let mut by_prefix: HashMap<String, (Vec<&Finding>, BTreeSet<String>)> = HashMap::new();
    for f in findings {
        if !cluster_is_same_filename_across_dirs(f) {
            continue;
        }
        let mut dirs: BTreeSet<&str> = BTreeSet::new();
        for (file, _, _) in &f.members {
            dirs.insert(split_path(file).1);
        }
        // Pick source by lowest vendored-marker score (ties → shortest path → lex order). This
        // beats pure shortest-path: a `src/vs/editor/...` source is correctly preferred over a
        // similarly-deep `extensions/.../fixtures/vscode` vendored copy.
        let source_dir: String = dirs
            .iter()
            .min_by(|a, b| {
                vendored_score(a)
                    .cmp(&vendored_score(b))
                    .then_with(|| a.split('/').count().cmp(&b.split('/').count()))
                    .then_with(|| a.cmp(b))
            })
            .map(|s| (*s).to_owned())
            .unwrap_or_default();
        for &dir in &dirs {
            if dir == source_dir {
                continue;
            }
            // Gating signal: only treat a pair as "vendored" when the longer-pathed side
            // actually carries a vendored-marker (`/fixtures/`, `/vendor/`, `/util/vs/`,
            // `/test/`, etc. — see [`vendored_score`]). Without a marker, two same-named files
            // across different dirs more often signal architectural duplication (a generic
            // utility re-implemented for a specific call site) than a fork snapshot — emitting
            // a broad `suppress` glob over those paths silently hides real refactor candidates
            // and is over-broad enough to catch unrelated clusters that happen to have one
            // member in the same subtree. Architectural-drift findings still surface as
            // regular ERRORs; we just don't auto-suppress them.
            if vendored_score(dir) == 0 {
                continue;
            }
            let (vendored_prefix, _) = split_at_shared_suffix(dir, &source_dir);
            let key = if vendored_prefix.is_empty() { dir.to_owned() } else { vendored_prefix };
            let entry = by_prefix.entry(key).or_default();
            entry.0.push(f);
            entry.1.insert(source_dir.clone());
        }
    }

    // Build "anchor → all findings under it" by rolling each leaf-prefix up through every
    // ancestor path. Then for each leaf-prefix pick its DEEPEST ancestor whose total finding
    // count clears `VENDORED_MIN_CLUSTERS`. That gives one directive per distinct vendored
    // *root* (the copilot `src/util/vs` snapshot, the copilot `test/simulation/fixtures`
    // snapshot, …) without rolling up so far that the glob becomes "anything in the repo".
    let mut by_anchor: HashMap<String, Vec<&Finding>> = HashMap::new();
    for (prefix, (fs, _)) in &by_prefix {
        if prefix.is_empty() {
            continue;
        }
        let parts: Vec<&str> = prefix.split('/').collect();
        for end in 1..=parts.len() {
            let ancestor = parts[..end].join("/");
            // Dedup findings within an ancestor's list: a single finding might roll into
            // multiple ancestors (always its own chain), but per-ancestor we want a set.
            let entry = by_anchor.entry(ancestor).or_default();
            for f in fs {
                entry.push(*f);
            }
        }
    }
    // For each leaf-prefix pick the deepest ancestor whose finding count clears the floor.
    // Collect the chosen anchors into a set so two leaves under the same anchor only produce
    // one directive (their findings are already aggregated into `by_anchor`).
    let mut chosen: BTreeMap<String, Vec<&Finding>> = BTreeMap::new();
    for prefix in by_prefix.keys() {
        let parts: Vec<&str> = prefix.split('/').collect();
        for end in (1..=parts.len()).rev() {
            let ancestor = parts[..end].join("/");
            if let Some(fs) = by_anchor.get(&ancestor) {
                if fs.len() >= VENDORED_MIN_CLUSTERS {
                    chosen.insert(ancestor, fs.clone());
                    break;
                }
            }
        }
    }
    // Drop ancestors that are *parents* of another chosen ancestor — the deeper one is more
    // specific and already covers the same findings; keeping both would emit duplicate
    // directives at different granularities.
    let chosen_keys: Vec<String> = chosen.keys().cloned().collect();
    chosen.retain(|k, _| {
        !chosen_keys.iter().any(|other| other != k && other.starts_with(&format!("{k}/")))
    });
    let mut directives: Vec<(String, Vec<&Finding>, BTreeSet<String>)> = chosen
        .into_iter()
        .map(|(anchor, fs)| {
            let srcs: BTreeSet<String> = by_prefix
                .iter()
                .filter(|(k, _)| k.starts_with(&anchor))
                .flat_map(|(_, (_, s))| s.iter().cloned())
                .collect();
            (anchor, fs, srcs)
        })
        .collect();
    directives.sort_by_key(|(_, fs, _)| std::cmp::Reverse(fs.len()));

    directives.into_iter()
        .take(5)
        .map(|(prefix, fs, sources)| {
            let n = fs.len();
            let by_sev = |s: Severity| fs.iter().filter(|f| f.severity == s).count();
            // Strip the repo root so the emitted glob is portable across machines — same
            // policy the rest of the report uses via `short_path`.
            let short_prefix = short_path(&prefix, repo_root);
            let glob = format!("*{short_prefix}*");
            let one_source: String = sources
                .into_iter()
                .next()
                .map(|s| short_path(&s, repo_root))
                .unwrap_or_default();
            InferredDirective {
                directive: format!(
                    "suppress:*:*@{glob}=likely vendored / fork snapshot mirroring {one_source}"
                ),
                rationale: format!(
                    "{n} clusters have all members in same-named files across `{short_prefix}` and a parallel source root"
                ),
                affects_total: n,
                affects_error: by_sev(Severity::Error),
                affects_warning: by_sev(Severity::Warning),
                affects_info: by_sev(Severity::Info),
            }
        })
        .collect()
}

/// Build one suggestion from findings matching `predicate`; returns `None` below `min_clusters`
/// so the inferred-directives section only surfaces patterns with real evidence.
fn build_suggestion(
    findings: &[Finding],
    predicate: impl Fn(&Finding) -> bool,
    min_clusters: usize,
    directive: &str,
    rationale_template: impl Fn(usize) -> String,
) -> Option<InferredDirective> {
    let matched: Vec<&Finding> = findings.iter().filter(|f| predicate(f)).collect();
    if matched.len() < min_clusters {
        return None;
    }
    let by_sev = |s: Severity| matched.iter().filter(|f| f.severity == s).count();
    Some(InferredDirective {
        directive: directive.to_owned(),
        rationale: rationale_template(matched.len()),
        affects_total: matched.len(),
        affects_error: by_sev(Severity::Error),
        affects_warning: by_sev(Severity::Warning),
        affects_info: by_sev(Severity::Info),
    })
}

/// Pattern-match across findings to emit ready-to-paste `-D` directives for the recurring noise
/// shapes (i18n locales, all-test clusters, generated `_pb2`/`_grpc`, schema migrations). The
/// user still owns the final list; we shortcut the first 80% of project-specific tuning.
fn infer_directives(
    findings: &[Finding],
    large_groups: &[(&'static KindSpec, String, usize)],
    repo_root: &Path,
) -> Vec<InferredDirective> {
    let mut out: Vec<InferredDirective> = Vec::new();
    // Huge same-name groups (`fn main` across test fixtures, `async_setup_entry` across HA
    // integrations) are conventions / entry points, not refactor clusters — and clustering them
    // is the O(n²) cost that dominates big monorepos. Suggest a `settings:max-name-group` cap.
    if !large_groups.is_empty() {
        let total: usize = large_groups.iter().map(|(_, _, n)| n).sum();
        let sample: Vec<String> =
            large_groups.iter().take(4).map(|(k, n, c)| format!("{}:{n} ×{c}", k.id)).collect();
        let more = large_groups.len().saturating_sub(4);
        let tail = if more > 0 { format!(", +{more} more") } else { String::new() };
        out.push(InferredDirective {
            directive: format!("settings:max-name-group={SUGGEST_CAP}"),
            rationale: format!(
                "{} name-group(s) exceed {SUGGEST_CAP} members ({}{tail}) — shared / entry-point names, not duplication; their O(n²) name-gated clustering dominates runtime, so the cap skips them",
                large_groups.len(),
                sample.join(", ")
            ),
            affects_total: total,
            affects_error: 0,
            affects_warning: 0,
            affects_info: 0,
        });
    }
    if let Some(s) = build_suggestion(
        findings,
        |f| {
            f.kind.id == "constants" && {
                // Majority-in-locale heuristic — Django-style codebases have one canonical
                // declaration in `global_settings.py` plus the per-locale overrides; requiring
                // ALL members in locale paths would miss exactly that legit case.
                #[allow(clippy::cast_precision_loss)] // member counts always small
                {
                    let n = f.members.len() as f64;
                    let in_locale = f.members.iter().filter(|(p, _, _)| path_is_i18n(p)).count() as f64;
                    in_locale / n >= 0.8
                }
            }
        },
        5,
        "suppress:CONSTANT:*@*locale*=i18n locale tables, duplication is by design",
        |n| format!("{n} CONSTANT clusters are ≥80% inside locale/i18n paths"),
    ) {
        out.push(s);
    }
    // i18n locale / translation files for ANY kind — per-locale function families (Angular's
    // `plural_locale_*` rules, Django-style translation modules) are duplication-by-design;
    // each locale has its own near-identical implementation.
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().all(|(p, _, _)| path_is_i18n(p)),
        5,
        "suppress:*:*@*/{locale,locales,i18n,translations}/*=i18n / translation tables — per-locale near-duplicates are by design",
        |n| format!("{n} clusters live entirely in locale / i18n / translation paths"),
    ) {
        out.push(s);
    }
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().all(|(p, _, _)| path_is_test(p)),
        3,
        // Brace alternation expands to every test-tree convention `path_is_test` recognizes,
        // so one paste covers `/test/`, `/tests/`, `/__tests__/`, `/test_cases/`, `/integration/`,
        // `/e2e/`, etc., without catching unrelated words like `/testimony/`.
        "de-escalate:*:*@*/{test,tests,__tests__,test_cases,test-cases,__fixtures__,fixtures,integration,e2e}/*=test parametrize/fixture candidates — review for conftest",
        |n| format!("{n} clusters live entirely in test paths — parametrize/conftest candidates"),
    ) {
        out.push(s);
    }
    // TS/JS files using the `.test.*` / `.spec.*` naming convention (jest, vitest, mocha) that
    // live OUTSIDE a `/test/` directory — the file-extension test marker is independent of the
    // directory marker, so we emit it as a parallel directive.
    if let Some(s) = build_suggestion(
        findings,
        |f| {
            f.members.iter().all(|(p, _, _)| {
                let fname = p.rsplit('/').next().unwrap_or(p);
                fname.contains(".test.") || fname.contains(".spec.")
            })
        },
        3,
        "de-escalate:*:*@*.{test,spec}.*=test files by .test.* / .spec.* naming",
        |n| format!("{n} clusters live entirely in `*.test.*` / `*.spec.*` files (jest/vitest/mocha)"),
    ) {
        out.push(s);
    }
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().any(|(p, _, _)| path_is_generated(p)),
        3,
        "suppress:*:*@*_pb2*=generated protobuf/gRPC code, suppress wholesale",
        |n| format!("{n} clusters touch `*_pb2*`/`*_grpc*` files — generated code"),
    ) {
        out.push(s);
    }
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().all(|(p, _, _)| path_is_migration(p)),
        3,
        "suppress:*:*@*migrations/*=schema migrations are snapshots, not refactor targets",
        |n| format!("{n} clusters live entirely under migrations/ — schema-history files"),
    ) {
        out.push(s);
    }
    // Tutorial / docs / example code — fastapi/docs_src, sklearn/examples, angular/adev
    // examples, etc. Brace alternation covers every convention `path_is_doc_example` recognizes
    // so users don't have to chain multiple `-D` flags for the same pattern family.
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().all(|(p, _, _)| path_is_doc_example(p)),
        5,
        "de-escalate:*:*@*/{docs_src,docs/src,examples,example,tutorial,tutorials,samples}/*=tutorial/doc-example code — snippet duplication is expected",
        |n| format!("{n} clusters live entirely under docs_src/ / examples/ / tutorial/ paths"),
    ) {
        out.push(s);
    }
    // TS-specific: `.d.ts` type declarations — distribution mechanism for type-only API
    // surface; cross-package duplication is intentional. Suppress wholesale unless overridden.
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().all(|(p, _, _)| path_is_dts(p)),
        3,
        "suppress:*:*@*.d.ts=TypeScript declaration files, type-only duplication is by design",
        |n| format!("{n} clusters live entirely in `.d.ts` declaration files"),
    ) {
        out.push(s);
    }
    // TS-specific: Storybook stories. Each `.stories.tsx` deliberately repeats render
    // boilerplate; the per-story scaffolding is the API, not a refactor target.
    if let Some(s) = build_suggestion(
        findings,
        |f| f.members.iter().all(|(p, _, _)| path_is_story(p)),
        5,
        "de-escalate:*:*@*.stories.*=Storybook stories — boilerplate duplication is the docstring",
        |n| format!("{n} clusters live entirely in `*.stories.*` Storybook files"),
    ) {
        out.push(s);
    }
    // Vendored / fork pattern — same filename across multiple dir roots. Dominates large
    // codebases that include forked snapshots of upstream sources (e.g. vscode's
    // `extensions/copilot/src/util/vs/*` mirroring `src/vs/*`). Path-glob suggestions land here
    // because the heuristic is non-obvious from a single cluster — only the aggregate reveals
    // the snapshot root.
    out.extend(infer_vendored_directives(findings, repo_root));
    out
}

/// Compact "where are the duplicates?" line. Consecutive members sharing the same file collapse
/// to `path:lineA, :lineB, :lineC` so multi-copy intra-file clusters don't smear the same path
/// 3×. Truncates the tail with `(+N more)` after `max` entries.
fn fmt_member_locations(members: &[String], max: usize) -> String {
    if members.is_empty() {
        return String::new();
    }
    let take = members.len().min(max);
    let mut parts: Vec<String> = Vec::with_capacity(take);
    let mut last_file: Option<&str> = None;
    for entry in &members[..take] {
        let (file, line) = match entry.rsplit_once(':') {
            Some((f, l)) => (f, l),
            None => (entry.as_str(), ""),
        };
        if last_file == Some(file) {
            parts.push(format!(":{line}"));
        } else {
            parts.push(entry.clone());
            last_file = Some(file);
        }
    }
    let mut s = parts.join(", ");
    let extra = members.len() - take;
    if extra > 0 {
        let _ = write!(s, " (+{extra} more)");
    }
    s
}

fn render_calibration_json(errs: &[&Finding], warns: &[&Finding], all: &[Finding], large_groups: &[(&'static KindSpec, String, usize)], repo_root: &Path) -> String {
    let report = CalibReport {
        error: CalibTier {
            total: errs.len(),
            target_flag: "error-thickness",
            histogram: thickness_histogram(&errs.iter().map(|f| f.thickness).collect::<Vec<_>>()),
            suggestions: calibration_suggestions(errs, repo_root),
        },
        warning: CalibTier {
            total: warns.len(),
            target_flag: "warning-thickness",
            histogram: thickness_histogram(&warns.iter().map(|f| f.thickness).collect::<Vec<_>>()),
            suggestions: calibration_suggestions(warns, repo_root),
        },
        inferred_directives: infer_directives(all, large_groups, repo_root),
    };
    serde_json::to_string_pretty(&report).unwrap_or_default() + "\n"
}

/// Render one severity tier — histogram + percentile-anchored suggestions + a sample snippet
/// per suggestion. Both `--error-thickness` and `--warning-thickness` are calibrated this way;
/// `tier_label` / `target_flag` switch the prose, the rest is identical so the two blocks read
/// as a parallel pair.
fn format_calibration_tier(
    out: &mut String,
    tier_label: &str,
    target_flag: &str,
    findings: &[&Finding],
    repo_root: &Path,
) {
    if findings.is_empty() {
        let _ = writeln!(out, "=== thickness calibration ({tier_label}): 0 clusters — skip.\n");
        return;
    }
    let _ = writeln!(
        out,
        "=== thickness calibration ({tier_label}): {} clusters analyzed ===\n",
        findings.len()
    );
    let hist = thickness_histogram(&findings.iter().map(|f| f.thickness).collect::<Vec<_>>());
    let max_count = hist.iter().map(|b| b.count).max().unwrap_or(1).max(1);
    let bar_max = 30usize;
    let _ = writeln!(out, "distribution (each ▇ ≈ one {tier_label} cluster, scaled to fit):");
    for b in &hist {
        let bar_len = (b.count * bar_max).div_ceil(max_count);
        let bar = "▇".repeat(bar_len);
        let _ = writeln!(
            out,
            "  T [{:.1}, {:.1})  {bar} {}",
            b.thickness_lo, b.thickness_hi, b.count
        );
    }
    out.push('\n');
    let _ = writeln!(
        out,
        "suggested thresholds (p50/p75/p90 of current {tier_label} thickness distribution):"
    );
    for s in calibration_suggestions(findings, repo_root) {
        out.push('\n');
        let _ = writeln!(
            out,
            "  {:<11}  --{target_flag} {:.2}  → {} {tier_label} remain  (median dup: {} loc, {} args)",
            s.label, s.error_thickness, s.errors_kept, s.median_loc, s.median_args
        );
        let _ = writeln!(
            out,
            "    e.g. {}  [T={:.2}, loc={}, args={}]",
            s.example_name, s.example_thickness, s.example_loc, s.example_args
        );
        let locs = fmt_member_locations(&s.example_members, 3);
        if !locs.is_empty() {
            let _ = writeln!(out, "         {locs}");
        }
        out.push_str(&snippet_box(&s.example_snippet, 15));
        out.push('\n');
    }
    out.push('\n');
}

fn format_calibration(errs: &[&Finding], warns: &[&Finding], all: &[Finding], large_groups: &[(&'static KindSpec, String, usize)], repo_root: &Path) -> String {
    if errs.is_empty() && warns.is_empty() {
        return "=== thickness calibration: 0 findings — nothing to calibrate against. ===\n".to_owned();
    }
    let mut out = String::new();
    format_calibration_tier(&mut out, "ERROR", "error-thickness", errs, repo_root);
    format_calibration_tier(&mut out, "WARNING", "warning-thickness", warns, repo_root);

    let inferred = infer_directives(all, large_groups, repo_root);
    if !inferred.is_empty() {
        out.push_str("=== inferred directives (auto-detected noise patterns) ===\n\n");
        for d in &inferred {
            let _ = writeln!(out, "  → -D '{}'", d.directive);
            let _ = writeln!(out, "    rationale: {}", d.rationale);
            let _ = writeln!(
                out,
                "    affects: {} total ({} ERROR, {} WARNING, {} INFO)",
                d.affects_total, d.affects_error, d.affects_warning, d.affects_info
            );
            out.push('\n');
        }
        out.push_str("(Paste any of these into your CI invocation — patterns matched repeatably,\n");
        out.push_str("not heuristics on individual clusters. Review the rationale before applying.)\n\n");
    }
    out.push_str("workflow: dial `--error-thickness` to focus the gate; dial `--warning-thickness`\n");
    out.push_str("to control how much low-confidence noise stays in WARNING vs falls to INFO.\n");
    out
}

// ───────────────────────── report (identical to the Python reference) ─────────────────────────

/// A group is "cross-name" when found by a name-agnostic pass (renamed copy-paste).
fn is_cross_name(pass: &str) -> bool {
    pass == "cross-name" || pass == "type-3"
}

// `section_index` is provided by the library — see `find_dup_defs::section_index`.

/// Best-effort repo-relative path; raw string when not under `repo_root` (mirrors `short_path`).
fn short_path(file: &str, repo_root: &Path) -> String {
    let p = std::fs::canonicalize(file).unwrap_or_else(|_| PathBuf::from(file));
    let root = std::fs::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf());
    p.strip_prefix(&root).map_or_else(|_| file.to_owned(), |rel| rel.to_string_lossy().into_owned())
}

/// The trailing marker on a `DUPLICATE` line: similarity / pass tag + thickness triage signals
/// (`T=…`, `loc=…`, `args=…`). `args` is dropped when 0 (constants, type-aliases, classes) to
/// keep the line scannable. Constants / type-aliases with no similarity tag still get a `loc`
/// hint so the user can tell a multi-line constant from a one-line alias at a glance.
fn group_suffix(f: &Finding) -> String {
    let tag: Option<String> = if is_cross_name(f.pass) && f.min_sim.is_none() {
        Some("normalized-exact".to_owned())
    } else {
        f.min_sim.map(|s| format!("ast sim {s:.2}"))
    };
    let n = f.members.len();
    let metrics = if f.args > 0 {
        format!("T={:.2}, n={}, loc={}, args={}", f.thickness, n, f.loc, f.args)
    } else {
        format!("T={:.2}, n={}, loc={}", f.thickness, n, f.loc)
    };
    let base = match tag {
        Some(t) => format!("  [{t}, {metrics}]"),
        None => format!("  [{metrics}]"),
    };
    // Notes from matching directives — visible right next to the finding so the next reviewer
    // sees the override reason without grepping. Multiple notes join with `; ` to stay on one
    // line; the JSON form keeps them as a structured array.
    if f.notes.is_empty() {
        base
    } else {
        format!("{base}  # {}", f.notes.join("; "))
    }
}

/// Build the ordered report sections `(section_index, header)` for the active frontends. Each
/// kind contributes its name-pass section (`cross-file, {sim}` for body kinds, `cross-file, by
/// name` for raw-text kinds) plus, for callables, its cross-name and Type-3 sections. The list
/// is the union of the selected frontends' kinds, deduped by `id` and sorted by section index —
/// so the default run reproduces the historical 10-section layout, while `--only py` prints only
/// Python's sections (no empty `interfaces`).
fn report_sections(frontends: &[&dyn Frontend], warn: f64, error: f64) -> Vec<(usize, String)> {
    let sim = format!("AST sim warn={warn} error={error}");
    let mut kinds: Vec<&'static KindSpec> = Vec::new();
    for f in frontends {
        for &k in f.kinds() {
            if !kinds.iter().any(|e| e.id == k.id) {
                kinds.push(k);
            }
        }
    }
    let mut rows: Vec<(usize, String)> = Vec::new();
    for k in kinds {
        let base = k.section as usize;
        let name_desc = if k.body { format!("cross-file, {sim}") } else { "cross-file, by name".to_owned() };
        rows.push((base, format!("duplicate {} ({name_desc})", k.noun_plural)));
        if k.fn_like {
            rows.push((base + 1, format!("duplicate {} (cross-name, exact AST-normalized)", k.noun_plural)));
            rows.push((base + 2, format!("duplicate {} (cross-name Type-3, IDF-weighted cosine)", k.noun_plural)));
        }
    }
    rows.sort_by_key(|(idx, _)| *idx);
    rows
}

/// Human-readable per-section report — byte-for-byte the Python `format_report`.
fn format_report(findings: &[Finding], frontends: &[&dyn Frontend], warn: f64, error: f64, repo_root: &Path) -> String {
    let sections = report_sections(frontends, warn, error);

    let mut lines: Vec<String> = Vec::new();
    for (index, (sect, header)) in sections.iter().enumerate() {
        if index > 0 {
            lines.push(String::new());
        }
        lines.push(format!("--- {header} ---"));
        for f in findings.iter().filter(|&f| section_index(f) == *sect) {
            lines.push(format!("DUPLICATE {} [{}]: {}{}", f.kind.label, f.severity.label(), f.name, group_suffix(f)));
            for (file, line, _col) in &f.members {
                lines.push(format!("  {}:{}", short_path(file, repo_root), line));
            }
            lines.push(String::new());
        }
    }

    if findings.is_empty() {
        lines.push("No cross-file duplicates.".to_owned());
        return lines.join("\n") + "\n";
    }
    let errs = findings.iter().filter(|f| f.severity == Severity::Error).count();
    let warns = findings.len() - errs;
    lines.push(format!("# summary: {errs} ERROR, {warns} WARNING groups"));
    lines.join("\n") + "\n"
}

#[derive(Serialize)]
struct JsonMember {
    file: String,
    line: usize,
}

#[derive(Serialize)]
struct JsonGroup {
    kind: String,
    name: String,
    severity: String,
    min_sim: Option<f64>,
    cross_name: bool,
    /// Composite [0,1] "fat function" score — see [`thickness`].
    thickness: f64,
    /// Max non-blank-line count across cluster members.
    loc: usize,
    /// Max parameter count across cluster members (0 for non-callable kinds).
    args: usize,
    members: Vec<JsonMember>,
    allowlist_key: String,
    notes: Vec<String>,
}

#[derive(Serialize)]
struct JsonReport {
    groups: Vec<JsonGroup>,
    summary: serde_json::Map<String, serde_json::Value>,
}

/// Machine-readable groups + summary — byte-for-byte the Python `render_json` (indent=2).
fn render_json(findings: &[Finding], repo_root: &Path) -> String {
    let groups: Vec<JsonGroup> = findings
        .iter()
        .map(|f| {
            let cross = is_cross_name(f.pass);
            let rule = if cross { "dup-xname".to_owned() } else { format!("dup-{}", f.kind.label.to_ascii_lowercase()) };
            JsonGroup {
                kind: f.kind.label.to_owned(),
                name: f.name.clone(),
                severity: f.severity.label().to_owned(),
                min_sim: f.min_sim,
                cross_name: cross,
                thickness: f.thickness,
                loc: f.loc,
                args: f.args,
                members: f.members.iter().map(|(file, line, _)| JsonMember { file: short_path(file, repo_root), line: *line }).collect(),
                allowlist_key: format!("{rule} {}", f.name),
                notes: f.notes.clone(),
            }
        })
        .collect();

    // summary: counts in first-seen severity order (matches the reference dict), then total.
    let mut summary = serde_json::Map::new();
    for f in findings {
        let key = f.severity.label();
        let n = summary.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0) + 1;
        summary.insert(key.to_owned(), serde_json::Value::from(n));
    }
    summary.insert("total".to_owned(), serde_json::Value::from(findings.len()));

    let report = JsonReport { groups, summary };
    serde_json::to_string_pretty(&report).unwrap_or_default() + "\n"
}
