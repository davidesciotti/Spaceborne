import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path}/common_config')
import ISTF_fid_params
import mpl_cfg

sys.path.append('/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/bin')
import plots_FM_running as plot_utils

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

for probe in ['WL', 'GC']:
    with open(f'/Users/davide/Downloads/Fisher{probe}_G.json') as json_file:
        Fisher_G = json.load(json_file)
    with open(f'/Users/davide/Downloads/Fisher{probe}_G_cNG_CCL.json') as json_file:
        Fisher_G_cNG_CCL = json.load(json_file)
    with open(f'/Users/davide/Downloads/Fisher{probe}_G_cNG_SSC_CCL.json') as json_file:
        Fisher_G_cNG_SSC_CCL = json.load(json_file)
    with open(f'/Users/davide/Downloads/Fisher{probe}_G_SSC_CCL.json') as json_file:
        Fisher_G_SSC_CCL = json.load(json_file)

    param_list = ['ΩM', 'ΩB', 'w0', 'wa', 'H0', 'ns', 'σ8']

    fishers__list = [Fisher_G, Fisher_G_SSC_CCL, Fisher_G_cNG_SSC_CCL]
    label__list = ['Fisher_G', 'Fisher_G_SSC_CCL', 'Fisher_G_cNG_SSC_CCL', 'G_GS_perc_diff', 'GS_cNG_perc_diff']
    uncert_array = np.asarray([[fisher[param] for param in param_list] for fisher in fishers__list])
    fiducials = (0.32, 0.05, 1, 1, 67, 0.96, 0.816)

    for i in range(len(param_list)):
        uncert_array[:, i] /= fiducials[i]

    G_GS_perc_diff = (uncert_array[1] / uncert_array[0] - 1)
    GS_cNG_perc_diff = (uncert_array[2] / uncert_array[1] - 1)

    # append as last row of uncert_array
    uncert_array = np.vstack((uncert_array, G_GS_perc_diff))
    uncert_array = np.vstack((uncert_array, GS_cNG_perc_diff))
    # uncert_array = np.insert(uncert_array, SSC_perc_diff, axis=0)

    plt.figure()
    plot_utils.bar_plot(uncert_array, probe, label__list, bar_width=0.18, nparams=7,
                        param_names_label=param_list,
                        second_axis=False, no_second_axis_bars=1)
