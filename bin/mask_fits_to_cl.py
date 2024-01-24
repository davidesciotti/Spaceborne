
import copy
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
import healpy as hp
import sys

ROOT = '/home/davide/Documenti/Lavoro/Programmi'
sys.path.append(f'{ROOT}/Spaceborne')
import bin.cosmo_lib as cosmo_lib


def get_mask_quantities(clmask=None, mask=None, mask2=None, verbose=True):
    """Auxiliary routine to compute different mask quantities (ell,Cl,fsky) for partial sky Sij routines.

    Parameters
    ----------
    clmask : str or numpy.ndarray, default None
        Array or path to fits file containing the angular power spectrum of the mask.
        To be used when the observable(s) have a single mask.

    mask : str or numpy.ndarray, default None
        Array or path to fits file containing the mask in healpix form.
        PySSC will use hp to compute the mask power spectrum.
        It is faster to directly give clmask if you have it (particularly when calling PySSC several times).
        To be used when the observable(s) have a single mask.

    mask2 : str or numpy.ndarray, default None
        Array or path to fits file containing a potential second mask in healpix form.
        In the case where you want the covariance between observables measured on different areas of the sky.
        PySSC will use hp to compute the mask cross-spectrum.
        Again, it is faster to directly give clmask if you have it.
        If mask is set and mask2 is None, PySSC assumes that all observables share the same mask.

    verbose : bool, default False
        Verbosity of the routine.

    Returns
    -------
    tuple
        ell, cl_mask, fsky
    """
    if mask is None:  # User gives Cl(mask)
        if verbose:
            print('Using given Cls')
        if isinstance(clmask, str):
            cl_mask = hp.fitsfunc.read_cl(str(clmask))
        elif isinstance(clmask, np.ndarray):
            cl_mask = clmask
        ell = np.arange(len(cl_mask))
        lmaxofcl = ell.max()
    else:  # User gives mask as a map
        if verbose:
            print('Using given mask map')
        if isinstance(mask, str):
            map_mask = hp.read_map(mask, dtype=np.float64)
        elif isinstance(mask, np.ndarray):
            map_mask = mask
        nside = hp.pixelfunc.get_nside(map_mask)
        lmaxofcl = 2 * nside
        if mask2 is None:
            map_mask2 = copy.copy(map_mask)
        else:
            if isinstance(mask2, str):
                map_mask2 = hp.read_map(mask2, dtype=np.float64)
            elif isinstance(mask2, np.ndarray):
                map_mask2 = mask2
        cl_mask = hp.anafast(map_mask, map2=map_mask2, lmax=lmaxofcl)
        ell = np.arange(lmaxofcl + 1)

    # Compute fsky from the mask
    fsky = np.sqrt(cl_mask[0] / (4 * np.pi))
    if verbose:
        print('f_sky = %.4f' % (fsky))

    return ell, cl_mask, fsky


def find_lmax(ell, cl_mask, var_tol=0.05, debug=False):
    """Auxiliary routine to search the best lmax for all later sums on multipoles.

    Computes the smallest lmax so that we reach convergence of the variance
    ..math ::
        var = \sum_\ell  \\frac{(2\ell + 1)}{4\pi} C_\ell^{mask}

    Parameters
    ----------
    ell : array_like
        Full vector of multipoles. As large as possible of shape (nell,)
    cl_mask : array_like
        power spectrum of the mask at the supplied multipoles of shape (nell,).
    var_tol : float, default 0.05
         Float that drives the target precision for the sum over angular multipoles.
         Default is 5%. Lowering it means increasing the number of multipoles thus increasing computational time.

    Returns
    -------
    float
        lmax
    """

    assert ell.ndim == 1, 'ell must be a 1-dimensional array'
    assert cl_mask.ndim == 1, 'cl_mask must be a 1-dimensional array'
    assert len(ell) == len(cl_mask), 'ell and cl_mask must have the same size'
    lmaxofcl = ell.max()
    summand = (2 * ell + 1) / (4 * np.pi) * cl_mask
    var_target = np.sum(summand)
    # Initialisation
    lmax = 0
    var_est = np.sum(summand[:(lmax + 1)])
    while (abs(var_est - var_target) / var_target > var_tol and lmax < lmaxofcl):
        lmax = lmax + 1
        var_est = np.sum(summand[:(lmax + 1)])
        if debug:
            print('In lmax search', lmax, abs(var_est - var_target) / var_target, var_target, var_est)
    lmax = min(lmax, lmaxofcl)  # make sure we didnt overshoot at the last iteration
    return lmax


# ! generate my own polar cap
def generate_polar_cap(area_deg2, nside=2048):

    print('Generating a polar cap mask with area %.2f deg2 and resolution nside {nside}' % area_deg2)

    # Expected sky fraction
    fsky_expected = cosmo_lib.deg2_to_fsky(area_deg2)
    print(f"Expected f_sky: {fsky_expected}")

    # Convert the area to radians squared for the angular radius calculation
    area_rad2 = area_deg2 * (np.pi / 180)**2

    # The area of a cap is given by A = 2*pi*(1 - cos(theta)),
    # so solving for theta gives us the angular radius of the cap
    theta_cap_rad = np.arccos(1 - area_rad2 / (2 * np.pi))

    # Convert the angular radius to degrees for visualization
    theta_cap_deg = np.degrees(theta_cap_rad)
    print(f"Angular radius of the cap in degrees: {theta_cap_deg}")

    # Calculate the corresponding nside for the HEALPix map
    # The resolution parameter nside should be chosen so lmax ~ 3*nside

    # Create an empty mask with the appropriate number of pixels for our nside
    mask = np.zeros(hp.nside2npix(nside))

    # Find the pixels within our cap
    # Vector pointing to the North Pole (θ=0, φ can be anything since θ=0 defines the pole)
    vec = hp.ang2vec(0, 0)
    pixels_in_cap = hp.query_disc(nside, vec, theta_cap_rad)

    # Set the pixels within the cap to 1
    mask[pixels_in_cap] = 1

    # Calculate the actual sky fraction of the generated mask
    fsky_actual = np.sum(mask) / len(mask)
    print(f"Actual f_sky from the mask: {fsky_actual}")

    return mask


def mask_path_to_cl(mask_path, plot_title, coord=['C', 'E']):

    mask = hp.read_map(mask_path)
    hp.mollview(mask, coord=coord, title=plot_title, cmap='inferno_r')
    ell_mask, cl_mask, fsky = get_mask_quantities(clmask=None, mask=mask, mask2=None, verbose=True)
    nside = hp.get_nside(mask)
    print(f'nside = {nside}')
    return ell_mask, cl_mask, fsky, nside


# ! settings
area_deg2 = 14700
# area_deg2 = 15000
# nside = 4096
nside = 2048
coord = ['C', 'E']
# ! end settings


# Path to the FITS files
mask_path = '/home/davide/Documenti/Lavoro/Programmi/common_data/mask'
mask_lowres_path = f'{mask_path}/mask_circular_1pole_15000deg2.fits'
mask_circular_path = f'{mask_path}/mask_circular_1pole_{area_deg2:d}deg2_nside{nside}_davide.fits'
mask_dr1_path = f'{mask_path}/euclid_dr1_mask.fits'
mask_csst_npz = np.load('/home/davide/Documenti/Lavoro/Programmi/CSSTforecasts/mask_nside4096.npz')


# TODO understand why the plot is different, it's probably vec = hp.ang2vec(0, 0)

# compute Cl(mask) and fsky computed from user input (mask(s) or clmask)
ell_mask_circular, cl_mask_circular, fsky_circular, nside_circular = mask_path_to_cl(
    mask_circular_path, 'pole highres', coord=coord)
ell_mask_dr1, cl_mask_dr1, fsky_dr1, nside_dr1 = mask_path_to_cl(mask_dr1_path, 'dr1', coord=coord)

area_deg2_circular = int(cosmo_lib.fsky_to_deg2(fsky_circular))
area_deg2_dr1 = int(cosmo_lib.fsky_to_deg2(fsky_dr1))

plt.figure()
plt.loglog(ell_mask_circular, cl_mask_circular, ls='--', label=f'high res, fsky = {fsky_circular}', alpha=0.5)
plt.loglog(ell_mask_dr1, cl_mask_dr1, ls='--', label=f'dr1, fsky = {fsky_dr1}', alpha=0.5)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{mask}$')
plt.legend()

np.save(f'{mask_path}/ell_circular_1pole_{area_deg2_circular:d}deg2_nside{nside_circular}_davide.npy', ell_mask_circular)
np.save(f'{mask_path}/Cell_circular_1pole_{area_deg2_circular:d}deg2_nside{nside_circular}_davide.npy', cl_mask_circular)

np.save(f'{mask_path}/ell_DR1_{area_deg2_dr1:d}deg2_nside{nside_dr1}.npy', ell_mask_dr1)
np.save(f'{mask_path}/Cell_DR1_{area_deg2_dr1:d}deg2_nside{nside_dr1}.npy', cl_mask_dr1)


# mask_circular = generate_polar_cap(area_deg2=area_deg2, nside=nside)
# hp.write_map(mask_circular_path, mask_circular, overwrite=True)


# ! csst mask, very slow to load (more than 3 GB)
# mask_csst_full = mask_csst_npz['map_area_only']
# map_csst_nobright = mask_csst_npz['map_remove_bright']
# ell_csst = mask_csst_npz['l']
# cl_mask_csst = mask_csst_npz['Cmask']
# cl_mask_csst = hp.anafast(mask_csst_full)  # should give the same as above?
# hp.mollview(mask_csst_full, coord=coord, cmap='inferno_r')
