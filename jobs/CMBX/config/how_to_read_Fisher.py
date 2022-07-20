import numpy as np

filename = "/path/to/fisher/file.npz"
fish = np.load(filename)

# The resulting "fish" is a dictionary containing:
# - fish['all_params'] == parameter names
# - fish['conds']      == covariance matrix condition number for each ell
# - fish['cov']        == covariance matrix (inverse Fisher)
# - fish['fish']       == Fisher matrix
# - fish['me']         == fiducial values of parameters
# - fish['prec']       == precision of each parameters, i.e. "wid / me"
# - fish['wid']        == marginalised 1 sigma constraints

# Convention for filenames:
#    "fish" 
#  + "_Euclid-opti" or "_Euclid-pess" or nothing == self-explanatory (NB: nothing means it's CMB only)
#  + "_with-GCS" or nothing                      == with or without GC spectro
#  + "_CMB" or "_CMBphionly"                     == all CMB probes (T, E, phi) or just phi
#  + "-planck" or "-SO" or "-S4"                 == self-explanatory
#  + "_mode-a" or "mode-c" or "mode-p"           == * p ("plus")  == CMB + Euclid as independent probes
#                                                   * a ("add")   == proper CMBxEuclid covariance but no cross-probes in data
#                                                   * c ("cross") == full CMBxEucld covariance and data
#  + "_flat" or "_nonflat"                       == self-explanatory
#  + "_LCDM" or "_w0waCDM"                       == self-explanatory
#  + "_gamma" or nothing                         == self-explanatory
#  + "_max-bins_super-prec_21point.npz"          == don't pay attention to that :-)
