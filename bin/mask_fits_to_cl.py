
import copy
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
import healpy as hp


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

    print('generating a polar cap mask with area %.2f deg2 and resolution nside {nside}' % area_deg2)

    # Total area of the sphere in square degrees
    total_area_deg2 = 41253

    # Expected sky fraction
    fsky_expected = area_deg2 / total_area_deg2
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


area_deg2 = 14700
# nside = 4096
nside = 2048

# change to 15000 to match the Euclid mask, for a check
mask_euclid_highres = generate_polar_cap(area_deg2=area_deg2, nside=nside)


# Path to the FITS files
mask_euclid_lowres_path = '/home/davide/Documenti/Lavoro/Programmi/common_data/sylvain/mask/mask_circular_1pole_15000deg2.fits'
mask_euclid_highres_path = f'/home/davide/Documenti/Lavoro/Programmi/common_data/sylvain/mask/mask_circular_1pole_{area_deg2:d}deg2_nside{nside}_davide.fits'
mask_csst_npz = np.load('/home/davide/Documenti/Lavoro/Programmi/CSSTforecasts/mask_nside4096.npz')

# load sylvain's mask (which is low-resolution)
mask_euclid_lowres = hp.read_map(mask_euclid_lowres_path, verbose=True)

# TODO understand why the plot is different, it's probably vec = hp.ang2vec(0, 0)
# Plot the masks using mollview
coord = ['C', 'E']
hp.mollview(mask_euclid_lowres, coord=coord, title='Polar Cap Mask low res', cmap='inferno_r')
hp.mollview(mask_euclid_highres, coord=coord, title='Polar Cap Mask high res', cmap='inferno_r')

# compute Cl(mask) and fsky computed from user input (mask(s) or clmask)
ell_euclid_lowres, cl_mask_euclid_lowres, fsky_euclid_lowres = get_mask_quantities(
    clmask=None, mask=mask_euclid_lowres, mask2=None, verbose=True)
ell_euclid_highres, cl_mask_euclid_highres, fsky_euclid__highres = get_mask_quantities(
    clmask=None, mask=mask_euclid_highres, mask2=None, verbose=True)


plt.figure()
plt.loglog(ell_euclid_lowres, cl_mask_euclid_lowres, label='low res')
plt.loglog(ell_euclid_highres, cl_mask_euclid_highres, ls='--', label='high res, area = %i deg2' % area_deg2)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{mask}$')

np.save('/home/davide/Documenti/Lavoro/Programmi/common_data/sylvain/mask/ell_circular_1pole_{area_deg2:d}deg2_nside{nside}_davide.npy', ell_euclid_highres)
np.save('/home/davide/Documenti/Lavoro/Programmi/common_data/sylvain/mask/Cell_circular_1pole_{area_deg2:d}deg2_nside{nside}_davide.npy', cl_mask_euclid_highres)
hp.write_map(mask_euclid_highres_path, mask_euclid_highres, overwrite=True)

# (re-) get nside, just to check
nside_euclid_lowres = hp.get_nside(mask_euclid_lowres)
nside_euclid_highres = hp.get_nside(mask_euclid_highres)

print(f'nside_euclid_lowres, {nside_euclid_lowres}')
print(f'nside_euclid_highres, {nside_euclid_highres}')



# ! csst mask, very slow to load (more than 3 GB)
# mask_csst_full = mask_csst_npz['map_area_only']
# map_csst_nobright = mask_csst_npz['map_remove_bright']
# ell_csst = mask_csst_npz['l']
# cl_mask_csst = mask_csst_npz['Cmask']
# cl_mask_csst = hp.anafast(mask_csst_full)  # should give the same as above?
# hp.mollview(mask_csst_full, coord=coord, cmap='inferno_r')





