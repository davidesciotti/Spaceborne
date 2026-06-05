"""Shared pytest configuration for the Spaceborne test suite."""

import matplotlib

# Use a non-interactive backend so tests that exercise plotting code paths
# (e.g. bnt.compute_bnt_matrix with plot_nz=True) never try to open a window.
# This is required for the tests to run headless in CI.
matplotlib.use('Agg')
