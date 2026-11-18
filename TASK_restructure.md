# TASK: restructure for a robust test suite

Goal: make the full pipeline (not just the library modules) testable in-process, so that
config-branching / wiring bugs are caught by pytest + CI rather than by end-to-end runs.

## Starting point (2026-11-17)

- Unit tests: 653 pass in ~14 s (`python -m pytest tests/ -q`). Library modules are
  reasonably covered; `cov_harmonic_space.py` and `plot_lib.py` have no test file.
- `main.py` (2210 lines) is a top-level script: 3 functions, ~100 top-level `if/elif`
  branches. Importing it parses `sys.argv` (`load_config`), sets JAX/OMP env vars and runs
  the pipeline → untestable except via subprocess.
- Wiring bugs that unit tests can't catch, all historically in main.py: psky-closure
  `noise_model` discarded `.get()` (fixed 2f2255eb), `separate_universe` SSC broken,
  `iNKA_eb_leakage` KeyError on old cfgs.
- Effective config is created *inside* main.py: `setdefault`s (L302-303), hardcoded block
  L279-313 (`n_probes=2`, `use_h_units`, `shift_nz_interpolation_kind`, OC settings),
  `pvt_cfg` built at L651 and mutated at L710-712.
- Classes take whole `cfg`/`pvt_cfg` and read nested keys in `__init__` (CovHarmonicSpace,
  NmtCov, CovSSC, Responses, OneCovarianceInterface, IOHandler, ...).
- `tests/bench_set_check.py`: not pytest-collected, hardcoded `DATA_ROOT`/`CODE_ROOT`,
  external `.npz` benches, `CFG_RENAMES` patch table, known-mismatch list judged by eye.
- Uncommitted work in tree at time of writing (ell_utils, sb_lib, cov_harmonic_space +
  tests) — commit/land it before starting step 1 to avoid a messy diff.

## Plan (in order; each step = its own commit, suite green after each)

### 1. Turn main.py into functions (blocks everything else)
Split along the existing `# ! ===` section markers into `spaceborne/pipeline.py`:

| function                         | main.py section(s)                                   |
|----------------------------------|------------------------------------------------------|
| `setup_runtime(cfg)`             | JAX/threads env, branch/conda/update checks (L55-190) |
| `build_effective_cfg(cfg)`       | hardcoded opts, setdefaults, legacy migration (L187, L279-313) |
| `resolve_probes_and_terms(cfg)`  | probe selection, req terms, codes, HS-from-RS (L317-485) |
| `build_grids(cfg, ccl_obj)`      | k/z grids, k_max checks, CCL grids (L525-645)        |
| `build_pvt_cfg(...)`             | L651-712                                             |
| `build_ell_binning`, `load_mask`, `load_nz`, `build_ia/galbias/magbias`, `build_kernels`, `compute_cls`, `compute_nls` | L695-1007 |
| `compute_covs(...)`              | NaMaster/RS/COSEBIs init, Gaussian, OC, PyCCL, SB SSC/cNG, combine (L1008-1674) |
| `save_outputs(...)`              | select cov_dict, OC overwrite/compare, npz/fits, benchmark (L1679-1812, L1967-) |
| `run_diagnostics(...)`, `make_plots(...)` | L1813-2190                                  |
| `run(cfg) -> results`            | orchestrates the above, returns a results object/dict |

- `main.py` shrinks to: argparse → `load yaml` → `run(cfg)`. No side effects on import.
- `setup_runtime` must run before JAX import → keep env-var setting in `main.py`/first call;
  check where JAX is first imported.
- Pure move first (no behaviour change); verify with a bench config before/after
  (bitwise-identical npz).

### 2. Single source of truth for the effective config
- Move all defaults/hardcodes/legacy migrations into `config_checker` (or a `defaults.py`)
  → `normalize_cfg(raw) -> cfg`, then `check(cfg)`.
- Tests: every key read anywhere in the code exists after `normalize_cfg(minimal_cfg)`;
  old-format configs (bench yamls) normalize without KeyError (would have caught
  `iNKA_eb_leakage`). `CFG_RENAMES` from bench_set_check moves here.

### 3. In-process end-to-end smoke tests (`tests/test_pipeline.py`, `@pytest.mark.slow`)
- Tiny configs: zbins=2, few ℓ bins, coarse k/z grids, low nside for NaMaster.
- Parametrize over space {harmonic, real, cosebis} × terms {G, G+SSC, G+cNG} × code
  {Spaceborne, PyCCL} × partial-sky {none, NaMaster}; OC cases skipped unless OC available.
- Invariants: shapes, finite, symmetric, PD (or documented non-PD cases), tot = G+SSC+cNG,
  probe blocks consistent between 2D/4D/6D, G-only fsky scaling.
- Keep default `pytest` fast: register `slow` marker, CI runs it in a separate job.

### 4. Benchmark regression as pytest
- `tests/test_benchmarks.py`, data root from env var (`SB_BENCH_ROOT`), skip if unset.
- Expected mismatches → explicit per-(config, term) tolerances in a versioned file, with a
  reason string; no by-eye checks.
- Regenerate a fresh bench set after step 1 (see memory: pending bench TODO).

### 5. Narrow class interfaces (incremental, opportunistic)
- Replace `cfg`/`pvt_cfg` constructor args with explicit args or small dataclasses when a
  class is touched. Start with `CovHarmonicSpace` (and add its missing test file).
- Minor: replace global `np.random.randn` in `cov_partial_sky.py:438` with a passed `rng`;
  consider splitting `sb_lib.py` (75 functions) by topic.

## Status / log
- [ ] step 1
- [ ] step 2
- [ ] step 3
- [ ] step 4
- [ ] step 5
