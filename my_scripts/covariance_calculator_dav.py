import numpy as np
from matplotlib import pyplot as plt
from tjpcov.covariance_calculator import CovarianceCalculator

# In order to use the config file in tests, you need your working
# directory to be the root folder

ROOT = '/home/davide/Documenti/Lavoro/Programmi/TJPCov'
config_yml = f'{ROOT}/examples/full_3x2pt_cov_example.yml'

cc = CovarianceCalculator(config_yml)

# You can get the total covariance:
cov = cc.get_covariance()

plt.imshow(np.log10(np.abs(cov)))
plt.colorbar()
plt.show()
