# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Spaceborne computes the **covariance matrix** of the 3×2pt photometric probes — weak lensing (`LL`),
photometric galaxy clustering (`GG`), and galaxy-galaxy lensing (`GL`) — in **harmonic and real space**, meaning it can work with three statistics: Cls, 2PCF, and COSEBIs. It computes the covariance
for Gaussian (G), super-sample (SSC) and connected non-Gaussian (cNG) terms. Pure Python, JAX-accelerated,
interfaced with CCL (cosmology), NaMaster (partial-sky harmonic cov) and OneCovariance (cross-checks).

## Commands

Environment (conda, Python 3.13):
```bash
conda env create -f environment.yaml   # or: mamba env create -f environment.yaml
conda activate spaceborne
pip install .
```

Run (the whole pipeline is driven by a single YAML config):
```bash
python main.py                                   # uses ./config.yaml
python main.py --config=/path/to/config.yaml     # custom config
python main.py --config=... --show-plots         # also display figures
```

Tests (CI runs these on pushes/PRs to `develop` and `main`):
```bash
python -m pytest tests/ -v                        # full suite
python -m pytest tests/test_mask_utils.py -v      # one file
python -m pytest tests/test_ell_utils.py::test_name -v   # one test
```

Lint/format (ruff, enforced via `.pre-commit-config.yaml`):
```bash
ruff check . && ruff format .
```
Style is non-default: line-length 88, **single quotes**, `skip-magic-trailing-comma = true`
(so do not add trailing commas that force multi-line collapses). Many naming rules (N802/N803/N806…)
are intentionally ignored because the code uses physics notation (`C_ell`, `cov_4D_to_6D`, etc.) — match
the surrounding style, don't "fix" casing.

## Architecture

`main.py` is **not** a `main()` function — it's a long top-to-bottom script that executes the whole
pipeline, with `# ! ===` section banners. It is the orchestrator; the `spaceborne/` package holds the
implementation. Reading order to understand a run:

1. **Config** (`config.yaml`) is the single source of truth. Every option is documented inline with its
   type. There is no hidden default file — the YAML *is* the API. `config_checker.py` validates it.
2. **Cosmology & power spectra**: `ccl_interface.py` (CCL cosmology, `p_of_k`, tracers) →
   `wf_cl_lib.py`/`cl_utils.py` (radial kernels, `C(ell)`), `responses.py` (P(k) responses for SSC).
3. **Geometry**: `mask_utils.py` — `Mask` objects per probe. Distinguishes the **binary footprint**
   (used for `fsky` scalars) from fractional **weight maps** (per-bin, per-probe; used only by the
   NaMaster partial-sky cov). `footprint_fsky_ab` builds the probe-pair effective fskys.
4. **Covariance terms** (each builds a nested `cov_dict`):
   - **Gaussian**: `cov_harmonic_space.py::CovHarmonicSpace.set_gauss_cov` dispatches on
     `covariance.partial_sky_method`:
     - `Knox` — analytic 1/fsky rescaling (fast).
     - `NaMaster` — `cov_partial_sky.py::CovNaMaster.build_psky_cov`, using NaMaster mode-coupling +
       `gaussian_covariance` under the NKA / **iNKA** approximation (`use_iNKA`, `spin0`,
       `cov_type: coupled|decoupled`).
     - `ensemble` — Monte-Carlo cov from masked healpy Gaussian realizations (`synalm`→mask→pseudo-Cl),
       parallelized with joblib in `cov_partial_sky.py` (`compute_ensemble_covariance_parallel`,
       `_compute_one_realization`, `sim_cls_to_ensemble_cov`).
   - **SSC / cNG**: `cov_ssc.py` + CCL trispectrum (PyCCL section of `main.py`).
   - **Real space / COSEBIs**: `cov_real_space.py`, `cov_cosebis.py`, projected from harmonic space via
     `cov_projector.py` / `cov_transform.py` (uses `twobessel_fang.py`, optional `pylevin`).
   - **External cross-check**: `oc_interface.py` shells out to OneCovariance.
5. **Assemble & save**: terms combined, then reshaped and written. `io_handler.py` + `sb_lib.py` hold the
   shape machinery and I/O.

### Covariance data model (important, easy to get wrong)
- `cov_dict[term][probe_ab, probe_cd]` where `term ∈ {'g','ssc','cng'}` and probe blocks are 2-tuples of
  `'LL'|'GL'|'GG'`. Each block is stored as `'4d'` `(nbl, nbl, zpair_ab, zpair_cd)` and/or `'6d'`
  `(nbl, nbl, zbins, zbins, zbins, zbins)`, then compressed to a single 2D matrix for output.
- Blocks are only allocated for **requested** probes (`probe_selection`). Code that fills blocks must
  iterate over the keys that exist, not hardcode all 9 — hardcoding crashes single-probe runs.
- 5D Cl/noise arrays use probe indices `[0,0]=LL`, `[1,1]=GG`, `[1,0]=GL` (shear=0/spin-2,
  clustering=1/spin-0).
- 2D ordering is set by `covariance.covariance_ordering_2D` (e.g. `probe_zpair_scale`); auto-pairs use the
  `triu`/`tril` + `row-major`/`column-major` convention.

### NaMaster specifics (`cov_partial_sky.py`)
- Input to `nmt.gaussian_covariance` is the **full-sky** spectra. The iNKA feeds
  `couple_cell(Cl)/mean(w²)` for the signal (≈ full-sky Cl but with realistic coupling); shape **noise is
  added flat** (`+ N_ℓ`) and its mask coupling is handled internally by the covariance workspace `cw` — do
  not divide the noise by fsky.
- `spin0=True` treats all fields as spin-0 (much faster, less accurate). The accurate path is spin-2 +
  iNKA for shear.
- Workspaces (`w00/w02/w22`) and covariance workspaces (`cw`) are cached under
  `<output>/cache/nmt/`; `load_cached_nmt_workspaces` / `save_mcms` control reuse.

## Typical workflow

Single runs are launched from external config files (not the repo's `config.yaml`). Batch parameter
sweeps use `launch_jobs.py` + `batch_run_utils.py` to clone the base config, override keys
(`partial_sky_method`, `spin0`, `use_iNKA`, output path, …) and run them in sequence with
`continue_on_error`. Outputs (per variant): `covmats_2D.npz` (`['Gauss']`), `covmats_6D.npz`,
`ell_values.txt`, `cl_*.txt`, `run_config.yaml`, `figs/`.

## Releases

Only tagged releases are considered stable; `main` may be mid-development. Version in `pyproject.toml`
(date-based, e.g. `2026.05.0`).
