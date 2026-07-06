# Spaceborne Mega-Review — Orchestration Brief


## Scope
- Full spaceborne source tree. Exclude generated files, build artifacts,
  `.egg-info`, and notebooks unless they contain logic other code depends on.
- If a subsystem is large (roughly >1500 lines across its files), split it
  further rather than widening one subagent's file list. Recall drops fast
  once a subagent's attention has to spread across too many files — that's
  the main way a broader net actually costs you something. Depth over
  breadth for any single subagent.

## Known context (background only — do not let this narrow the search)
These are the issues I'm already aware of and actively debugging. Mention
them if a subagent happens to touch this code, but the entire point of this
review is to surface what I'm *not* already looking for — no subagent
should treat this list as its checklist:
- Analytic vs. ensemble covariance discrepancy, currently attributed to
  mask-induced noise leakage rather than signal leakage
- The correct noise spectrum to pass to `nmt.gaussian_covariance` under
  inhomogeneous noise is still unresolved
- Past packaging friction: `cloelib` / `pyproject.toml` cross-platform
  (Linux/macOS) compiler toolchain mismatches

## What to look for
Not a checklist to sort files into — a set of lenses to apply to
everything. Don't assign one lens per file based on what it's named or
where it lives; apply all of them to each file you read.

- **Correctness** — logic errors, off-by-one, sign errors, wrong array
  axis/broadcasting, indexing mistakes, silently-wrong results that would
  run without crashing or raising
- **Numerical fragility** — ill-conditioned operations, unstable
  integration/interpolation schemes, resolution mismatches (grid /
  ell-binning / NSIDE), places where a NaN or Inf could propagate silently
  instead of raising
- **Physics/formalism vs. implementation** — anywhere the code's actual
  math doesn't match what a docstring, comment, or cited reference claims
  it does
- **Convention/unit consistency** — radians vs. degrees, normalization
  factors (e.g. ell(ell+1)/2π), fsky application, sign/ordering conventions
  applied inconsistently across functions
- **Error handling** — exceptions swallowed, wrong exception types, missing
  validation on inputs that would produce garbage instead of failing loudly
- **State and mutation** — unexpected in-place mutation of shared
  arrays/objects, global or module-level state that breaks under repeated
  calls or parallel execution
- **API/interface drift** — functions whose actual signature or behavior
  has drifted from how they're called elsewhere, stale defaults, arguments
  accepted but silently ignored downstream
- **Resource/performance footguns** — accidental O(n²) where O(n) is
  available, redundant recomputation inside loops, unnecessary full-array
  copies. Flag, don't rabbit-hole on micro-optimization.
- **Test coverage** — code paths (especially edge cases: empty input,
  single-element arrays, extreme parameter values) with no test exercising
  them
- **Packaging/environment** — anything that would break on a clean install
  or a different platform

If something looks wrong but doesn't fit any of these, flag it anyway —
the list exists to prompt thinking, not to gate what gets reported.

## Output format per finding
`[severity: critical/warning/suggestion] path/to/file.py:line — one-line description — why it matters — suggested direction (no fix implementation needed)`

Group findings by subsystem in the final report. End with a short top-of-report
summary of the 5-10 highest-priority items across the whole codebase.

