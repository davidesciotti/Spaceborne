"""Unit tests for spaceborne.cov_partial_sky module."""

import multiprocessing

import healpy as hp
import numpy as np
import pytest

from spaceborne import cov_dict as cd
from spaceborne import cov_partial_sky as cps
from spaceborne import sb_lib as sl

nmt = pytest.importorskip('pymaster')

PROBE_PAIRS = ['LL', 'GL', 'GG']
UNIQUE_PROBE_COMBS = ['LLLL', 'LLGL', 'LLGG', 'GLGL', 'GLGG', 'GGGG']


class _SameForAllBins(dict):
    """Footprint-only mode: the same NaMaster object for every z-bin index."""

    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    def __missing__(self, key):
        return self.obj


@pytest.fixture(scope='module')
def nmt_inputs():
    nside, zbins, nbl = 16, 2, 4
    lmax = 3 * nside - 1
    ells = np.arange(lmax + 1)

    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    mask = nmt.mask_apodization((theta < 1.0).astype(float), 10.0, apotype='C1')
    fields = {
        'G': nmt.NmtField(mask, None, spin=0, lmax=lmax),
        'L': nmt.NmtField(mask, None, spin=2, lmax=lmax),
    }
    ell_edges = np.linspace(2, lmax + 1, nbl + 1).astype(int)
    nmt_bin = nmt.NmtBin.from_edges(ell_edges[:-1], ell_edges[1:])

    wsp = {
        probe: _SameForAllBins(
            nmt.NmtWorkspace.from_fields(fields[probe[0]], fields[probe[1]], nmt_bin)
        )
        for probe in PROBE_PAIRS
    }
    cw_dict = {}
    for probe_abcd in UNIQUE_PROBE_COMBS:
        probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
        cw = nmt.NmtCovarianceWorkspace.from_fields(
            *(fields[probe] for probe in probe_abcd)
        )
        cw_dict[probe_ab, probe_cd] = _SameForAllBins(cw)

    # spectra that differ between (zi, zj) and (zj, zi) for GL, to catch index swaps
    rng = np.random.default_rng(1)
    amp = rng.uniform(0.5, 1.5, size=(6, zbins, zbins))
    amp[[0, 2, 5]] = (amp[[0, 2, 5]] + amp[[0, 2, 5]].transpose(0, 2, 1)) / 2
    shape = 1 / (ells + 10.0)
    cl_tt, cl_te, cl_ee, cl_tb, cl_eb, cl_bb = (
        shape[:, None, None] * amp[i][None] for i in range(6)
    )

    ind = sl.build_full_ind('triu', 'row-major', zbins)
    zpairs_auto, zpairs_cross, _ = sl.get_zpairs(zbins)
    ind_auto = ind[:zpairs_auto]
    ind_cross = ind[zpairs_auto : zpairs_auto + zpairs_cross]

    return {
        'spin0': False,
        'cl_tt': cl_tt,
        'cl_te': cl_te,
        'cl_ee': cl_ee,
        'cl_tb': cl_tb,
        'cl_eb': cl_eb,
        'cl_bb': cl_bb,
        'nbl': nbl,
        'zbins': zbins,
        'ind_dict': {'LL': ind_auto, 'GL': ind_cross, 'GG': ind_auto},
        'cw_dict': cw_dict,
        'wgg_dict': wsp['GG'],
        'wgl_dict': wsp['GL'],
        'wll_dict': wsp['LL'],
        'unique_probe_combs': UNIQUE_PROBE_COMBS,
        'nonreq_probe_combs': [],
        'ells_in': ells,
        'ells_out': (ell_edges[:-1] + ell_edges[1:]) / 2,
        'ells_out_edges': ell_edges,
        'which_binning': 'sum',
        'weights': None,
    }


def _run_nmt_gaussian_cov(nmt_inputs, *, coupled, n_jobs):
    probe_pairs = [(a, b) for a in PROBE_PAIRS for b in PROBE_PAIRS]
    cov_dict = cd.create_cov_dict(
        required_terms=['g'], probe_pairs=probe_pairs, dims=['4d']
    )
    cps.nmt_gaussian_cov(
        cov_dict=cov_dict, coupled=coupled, n_jobs=n_jobs, **nmt_inputs
    )
    return {pair: cov_dict['g'][pair]['4d'] for pair in probe_pairs}


@pytest.mark.skipif(
    multiprocessing.get_start_method() != 'fork', reason='parallel path needs fork'
)
@pytest.mark.parametrize('coupled', [True, False])
def test_nmt_gaussian_cov_parallel_matches_serial(nmt_inputs, coupled):
    cov_serial = _run_nmt_gaussian_cov(nmt_inputs, coupled=coupled, n_jobs=1)
    cov_parallel = _run_nmt_gaussian_cov(nmt_inputs, coupled=coupled, n_jobs=3)

    for pair, cov in cov_serial.items():
        assert np.any(cov), f'{pair} block is empty'
        np.testing.assert_array_equal(cov_parallel[pair], cov, err_msg=str(pair))
