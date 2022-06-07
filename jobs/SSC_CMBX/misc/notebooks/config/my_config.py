ell_min_GC = 10
ell_max_GC = 3000
ell_max_XC = ell_max_GC
ell_min_WL = 10
ell_max_WL = 5000

ell_min    = 10
zbins      = 10
nbl        = 30
nProbes = 2

Rl = 4
fsky_sylvain = 0.375 # used by Sylvain in SSC comparison
Cij_folder = 'Cij_14may'

survey_area = 15000 # deg^2
deg2_in_sphere = 41252.96 # deg^2 in a spere
fsky_IST = survey_area/deg2_in_sphere


# which_forecast = "IST"
which_forecast = "sylvain"

if   which_forecast == "IST":     fsky = fsky_IST
elif which_forecast == "sylvain": fsky = fsky_sylvain


ind_ordering = 'vincenzo'
# ind_ordering = 'CLOE'
# ind_ordering = 'SEYFERT'


# this is better:
GL_or_LG = "GL"
# GL_or_LG = "LG"

# use_LG = True
# use_LG = False

transpose_derivatives = False # check bene, cancella...


# reminder: sylvain's settings is:
# which_forecast = "sylvain"
# ind_ordering = 'vincenzo'
# GL_or_LG = "LG"


# reminder: CLOE's settings is:
# which_forecast = "IST"
# ind_ordering = 'CLOE'ch
# GL_or_LG = "GL" # (Vincenzo uses GL, Luca LG...)

# reminder: original's settings are:
# which_forecast = "IST"
# ind_ordering = 'vincenzo'
# GL_or_LG = "LG" # (Vincenzo uses GL, Luca LG...)

# Santiago's Cls:
XC_is_LG = False
# ind_ordering = TBD
# nbl = 20
# GL_or_LG = "GL"  