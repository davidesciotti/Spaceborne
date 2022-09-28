import camb
import numpy as np
from camb import model, initialpower
from matplotlib import pyplot as plt

# taken from https://camb.readthedocs.io/en/latest/CAMBdemo.html

pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)

# Now get matter power spectra and sigma8 at redshift 0 and 0.8
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
# Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[0., 0.8], kmax=2.0)

# Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)
s8 = np.array(results.get_sigma8())

# Non-Linear spectra (Halofit)
pars.NonLinear = model.NonLinear_both
results.calc_power_spectra(pars)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)

for i, (redshift, line) in enumerate(zip(z, ['-', '--'])):
    plt.loglog(kh, pk[i, :], color='k', ls=line)
    plt.loglog(kh_nonlin, pk_nonlin[i, :], color='r', ls=line)
plt.xlabel('k/h Mpc')
plt.legend(['linear', 'non-linear'], loc='lower left')
plt.title('Matter power at z=%s and z= %s' % tuple(z))

nz = 100  # number of steps to use for the radial/redshift integration
kmax = 10  # kmax to use
# First set up parameters as usual
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)

# For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
# so get background results to find chistar, set up a range in chi, and calculate corresponding redshifts

# comoving radial distance (in Mpc), scalar or arr
results = camb.get_background(pars)
chistar = results.conformal_time(0) - results.tau_maxvis
chis = np.linspace(0, chistar, nz)


# ! this is to show that, if the output (z, in this case) is adimensional, the plot only shifts on the x axis
# ! (the domain is shifted rigidly)
# z of chi (in Mph)
plt.figure()
zs = results.redshift_at_comoving_radial_distance(chis)
plt.plot(chis, zs, '.', label = 'chi in Mpc')

# z of chi (in Mph/h)
chis /= 0.67
zs = results.redshift_at_comoving_radial_distance(chis)
plt.plot(chis, zs, '.', label = 'chi in Mpc/h')

plt.grid()
plt.legend()

# the results don't change, z is dimensionless