import warnings

import healpy as hp
import numpy as np

from spaceborne import constants, cosmo_lib, io_handler
from spaceborne import constants as const
from spaceborne import sb_lib as sl


def get_footprint_cl_abcd_dicts(
    footp_ab_dict: dict, fsky_ab_dict: dict, unique_probe_combs_hs: list
) -> tuple:
    """Compute the Cls and the normalised Cls for all footprints in
    unique_probe_combs_hs"""

    footp_cl_abcd_dict = {}
    footp_cl_norm_abcd_dict = {}
    for probe_abcd in unique_probe_combs_hs:
        probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')

        # compute and store the footprint cross-spectrum...
        _ells, _cls = get_maps_cl(footp_ab_dict[probe_ab], footp_ab_dict[probe_cd])

        # ...and its normalised version
        footp_cl_abcd_dict[probe_ab, probe_cd] = (_ells, _cls)
        denominator = (4 * np.pi) ** 2 * fsky_ab_dict[probe_ab] * fsky_ab_dict[probe_cd]
        _cls_norm = _cls * (2 * _ells + 1) / denominator
        footp_cl_norm_abcd_dict[probe_ab, probe_cd] = (_ells, _cls_norm)
    return footp_cl_abcd_dict, footp_cl_norm_abcd_dict


def get_fsky_abcd_dict(fsky_ab_dict: dict, req_probe_combs_hs_2d: list) -> tuple:
    """Given the fsky for each probe pair AB, compute the max(fsky_ab, fsky_cd)
    and convert it into steradians"""
    fsky_max_abcd_dict = {}
    for probe_abcd in req_probe_combs_hs_2d:
        probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
        # compute and store max(fsky_ab, fsky_cd)
        fsky_max_abcd_dict[probe_ab, probe_cd] = max(
            fsky_ab_dict[probe_ab], fsky_ab_dict[probe_cd]
        )

    # turn into A_max (in sr)
    amax_abcd_dict = {
        k: v * const.DEG2_IN_SPHERE * const.DEG2_TO_SR
        for k, v in fsky_max_abcd_dict.items()
    }

    return fsky_max_abcd_dict, amax_abcd_dict


def plot_footprint(footprint: np.ndarray, probe: str):
    hp.mollview(
        footprint, cmap='inferno_r', title=f'Footprint {probe} - Mollweide view'
    )
    hp.graticule()


def footprint_fsky_ab(mask_obj_ll, mask_obj_gg):
    """
    Given the LL and GG mask objects, computes the AB masks and their fsky.
    """
    m_ll = mask_obj_ll.footprint
    m_gg = mask_obj_gg.footprint

    # The SSC window of a probe pair AB is the *product* of the two fields'
    # masks, W_A * W_B (TJPCov convention, see covariance_fourier_ssc.py). For a
    # non-binary mask (e.g. a fractional weight map) this differs from the single mask, so the
    # auto-pairs must be squared too (W_A^2) to stay consistent with the
    # mean(W_A * W_B) effective fsky used in the normalisation downstream.
    footp_ab_dict = {'LL': m_ll * m_ll, 'GL': m_ll * m_gg, 'GG': m_gg * m_gg}
    # NB: fsky is the mean of the product of the *single* masks, mean(W_A * W_B),
    # NOT the mean of the squared window above (which would be mean(W_A^2 W_B^2)).
    fsky_ab_dict = {
        'LL': combined_fsky(m_ll, m_ll),
        'GL': combined_fsky(m_ll, m_gg),
        'GG': combined_fsky(m_gg, m_gg),
    }

    if not np.isclose(fsky_ab_dict['LL'], fsky_ab_dict['GG'], atol=0, rtol=1e-5):
        warnings.warn(
            'LL and GG footprints have different fsky. Using probe-dependent sky '
            'fractions:\n'
            f'LL = {fsky_ab_dict["LL"]:.4f}, '
            f'GG = {fsky_ab_dict["GG"]:.4f}.',
            stacklevel=2,
        )

    if (
        np.isclose(fsky_ab_dict['GL'], 0, atol=1e-12, rtol=0)
        or np.isnan(fsky_ab_dict['GL'])
        or np.isinf(fsky_ab_dict['GL'])
    ):
        warnings.warn(
            "The footprints seem to have zero overlap; fsky_ab_dict['GL'] = "
            f'{fsky_ab_dict["GL"]:.4f}',
            stacklevel=2,
        )

    return footp_ab_dict, fsky_ab_dict


def combined_fsky(map1: np.ndarray, map2: np.ndarray) -> float:
    """Combine two masks (e.g. footprint and weight map) by multiplying them
    and compute the resulting fsky."""
    fsky_combined = np.mean(map1 * map2)
    return float(fsky_combined)


def get_maps_cl(map1: np.ndarray, map2: np.ndarray) -> tuple:
    cl = hp.anafast(map1, map2)
    ells = np.arange(len(cl))
    return ells, cl


def generate_polar_cap_func(area_deg2, nside):
    _fsky_expected = cosmo_lib.deg2_to_fsky(area_deg2)

    # Convert the area to radians squared for the angular radius calculation
    area_rad2 = area_deg2 * (np.pi / 180) ** 2

    # The area of a cap is given by A = 2*pi*(1 - cos(theta)),
    # so solving for theta gives the angular radius of the cap
    theta_cap_rad = np.arccos(1 - area_rad2 / (2 * np.pi))

    # Convert the angular radius to degrees for visualization
    theta_cap_deg = np.degrees(theta_cap_rad)
    print(f'Angle subtended by the cap: {theta_cap_deg:.4f} deg')

    # Vector pointing to the North Pole (θ=0); φ can take any value
    vec = hp.ang2vec(0, 0)
    pixels_in_cap = hp.query_disc(nside, vec, theta_cap_rad)

    # Set the pixels within the cap to 1
    mask = np.zeros(hp.nside2npix(nside))
    mask[pixels_in_cap] = 1

    # Calculate the actual sky fraction of the generated mask
    # fsky_actual = np.sum(mask) / len(mask)
    # print(f'Measured fsky from the mask: {fsky_actual:.4f}')

    return mask


def up_downgrade_map(map_in, nside_out):
    """Very simple wrapper, basically gets nside_in and prints a message
    if up/downgrading is needed"""

    nside_in = hp.get_nside(map_in)
    if nside_out is not None and nside_out != nside_in:
        print(f'Changing map resolution from nside = {nside_in} to nside = {nside_out}')
        map_out = hp.ud_grade(map_in=map_in, nside_out=nside_out)
        return map_out
    else:
        return map_in


class Mask:
    def __init__(self, mask_cfg, probe):
        self.probe = probe
        self.geometry = mask_cfg[probe]['geometry']

        self.use_weight_maps = mask_cfg[probe]['weight_maps_filename'] is not None

        self.footprint_filename = mask_cfg[probe]['footprint_filename']
        self.weight_maps_filename = mask_cfg[probe]['weight_maps_filename']

        self.footprint = None
        self.weight_maps = None

        self.nside_cfg = mask_cfg['nside']
        self.desired_survey_area_deg2 = mask_cfg['survey_area_deg2']

    def load(self):
        # ! 1. load footprint/weight maps or generate polar cap
        if self.geometry == 'footprint_file':
            # load
            print(
                f'\nLoading footprint file for {self.probe} '
                f'from {self.footprint_filename}\n'
            )

            self.footprint = io_handler.load_footprint(
                path=self.footprint_filename, nside=self.nside_cfg
            )
            # get nside and up/downgrade if needed
            self.footprint = up_downgrade_map(self.footprint, self.nside_cfg)

        elif self.geometry == 'polar_cap':
            print(
                f'\nGenerating a polar cap mask for {self.probe} with area '
                f'{self.desired_survey_area_deg2} deg^2 and nside {self.nside_cfg}'
            )
            self.footprint = generate_polar_cap_func(
                self.desired_survey_area_deg2, self.nside_cfg
            )
        else:
            raise ValueError(
                f'Unsupported geometry type: {self.geometry} for probe {self.probe}. '
                'Supported types are: footprint_file and polar_cap'
            )

        if self.use_weight_maps:
            # load
            print(
                f'\nLoading weight map file for {self.probe} '
                f'from {self.weight_maps_filename}\n'
            )
            self.weight_maps = io_handler.load_weight_map_fits(
                self.weight_maps_filename
            )
            # get nside and up/downgrade if needed. Rebuild the array rather than
            # assigning into rows: up/downgrading changes the pixel count, so the
            # regraded maps don't fit back into the original fixed-width 2D array.
            self.weight_maps = np.array(
                [up_downgrade_map(wmap, self.nside_cfg) for wmap in self.weight_maps]
            )

    def get_cls_fsky(self):
        """get footprint angular power spectrum and effective fsky"""
        self.ells_footprint, self.cl_footprint = get_maps_cl(
            self.footprint, self.footprint
        )
        self.fsky_footprint = combined_fsky(self.footprint, self.footprint)

        self.cl_footprint_norm = (
            self.cl_footprint
            * (2 * self.ells_footprint + 1)
            / (4 * np.pi * self.fsky_footprint) ** 2
        )

        # 4. finally, set survey area in steradians and other useful quantities
        self.survey_area_deg2 = self.fsky_footprint * constants.DEG2_IN_SPHERE
        self.survey_area_sr = self.survey_area_deg2 * constants.DEG2_TO_SR

        # else:
        #     print(
        #         'No mask provided or requested. The covariance terms will be '
        #         'rescaled by 1/fsky'
        #     )
        #     self.ell_mask = None
        #     self.cl_mask = None
        #     self.fsky = self.survey_area_deg2 / constants.DEG2_IN_SPHERE

        print(f'fsky = {self.fsky_footprint:.4f}')
        print(f'survey_area_sr = {self.survey_area_sr:.4f}')
        print(f'survey_area_deg2 = {self.survey_area_deg2:.4f}\n')

    def process(self):
        self.load()
        self.get_cls_fsky()

    def plot_footprint(self):
        plot_footprint(footprint=self.footprint, probe=self.probe)

    def plot_weight_maps(self):
        if not self.use_weight_maps:
            return
        for zi in range(self.weight_maps.shape[0]):
            hp.mollview(
                self.weight_maps[zi],
                cmap='inferno_r',
                title=f'Weight map {self.probe}, zi={zi} - Mollweide view',
            )
        hp.graticule()
