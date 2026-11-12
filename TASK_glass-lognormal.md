# Lognormal (GLASS) ensemble covariance — status

Branch `glass-lognormal` (from `nowcast-v02` @ a692b565), not merged. 2026-11-12.

## What was added

- `spaceborne/lognormal_sims.py`: matter shells sampled with GLASS (lognormal or
  Gaussian), projected onto the tomographic fields with the radial kernels of the
  C_ells (`wf_lensing_arr`, `wf_galaxy_w_gal_bias_arr`: galaxy bias, IA, magnification
  included). Shell spectra: non-Limber (linear theory × non-linear boost) with
  correlations between `ncorr` neighbours for ell <= min(`ell_max_nonlimber`,
  0.1 Mpc⁻¹ · χ); Limber auto-spectra above. Pairs straddling the transition get half
  the cross-spectrum, which keeps the projected spectra continuous.
- `ensemble_covariance`: `field_type: gaussian | lognormal | gaussian_shells`,
  `lognormal_shell_thickness`, `lognormal_ncorr`, `lognormal_shift`,
  `lognormal_ell_max_nonlimber` (defaults set in `main.py`, documented in
  `config.yaml`). GLASS is imported lazily; nothing changes for `field_type: gaussian`.
- `covariance: which_pk_responses: lognormal`: dlnP/dδ_b = 2 / shift, galaxies
  linearly biased w.r.t. the global mean (no counterterms).
- Output: `ensemble_cov_sim_theory_cls.npz` (spectra the sims realise, shell edges and
  kernels). Caches of the shell spectra in `<output>/cache/`.
- `tests/test_lognormal_sims.py` (9 tests; generator tests skipped without GLASS).

## Validation

`Spaceborne_jobs_src/validate_lognormal/README.md` (branch
`worktree-validate-lognormal`): the SSC of the sims agrees with the Spaceborne SSC with
the lognormal response in all six probe blocks (amplitudes 0.86–1.14, errors 0.12–0.33),
and Spaceborne agrees with an independent shell-based calculation to 1–4 % (lensing)
and ~3 % (clustering, thin shells).

## Open

- GLASS is not in `environment.yaml` / the `spaceborne` env (runs used a venv on top
  of it). Decide whether to add it as an optional dependency.
- The 15 failures of `tests/test_config_checker.py` are the same on `nowcast-v02`
  (repo `config.yaml` lacks `has_mag_bias` etc.), unrelated to this branch.
- Setup cost: ~4 min for the non-Limber spectra of 74 shells (single thread, cached).
- RSD are not simulated; input C_ells are not used by the lognormal sims (warnings).
- The in-survey trispectrum of the lognormal sims has no analytic counterpart here.
- Top band of the coupled non-Gaussian covariance: sims above analytic for GL/GG
  (no buffer above ell_max in `_couple_cov_ng`).
- Delete this file once the branch is merged or dropped.
