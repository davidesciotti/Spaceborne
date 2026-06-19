import healpy as hp
import numpy as np

from spaceborne import constants, cosmo_lib, io_handler


def estimate_ell_cutoff(ells, cl: np.ndarray, threshold: float = 1e-7):
    """Given an input power spectrum, estimates the ell at which the spectrum has
    decayed to a fraction threshold of its peak.
    Uses the maxima to avoid issues with oscillations close to 0"""
    maxima = np.maximum.accumulate(cl[::-1])[::-1]
    peak = maxima[0]
    cross = np.where(maxima < threshold * peak)[0]
    bandwidth = ells[cross[0]] if cross.size else ells[-1]
    return bandwidth


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
    print(
        f'\nGenerating a polar cap mask with area {area_deg2} deg^2 and nside {nside}'
    )

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
            self.footprint = io_handler.load_footprint(
                path=self.footprint_filename, nside=self.nside_cfg
            )
            # get nside and up/downgrade if needed
            self.footprint = up_downgrade_map(self.footprint, self.nside_cfg)

        elif self.geometry == 'polar_cap':
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

    def plot_maps(self):
        hp.mollview(
            self.footprint,
            cmap='inferno_r',
            title=f'Footprint {self.probe} - Mollweide view',
        )
        if self.use_weight_maps:
            for zi in range(self.weight_maps.shape[0]):
                hp.mollview(
                    self.weight_maps[zi],
                    cmap='inferno_r',
                    title=f'Weight map {self.probe}, zi={zi} - Mollweide view',
                )
        hp.graticule()
