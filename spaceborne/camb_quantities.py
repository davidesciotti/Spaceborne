"""
Slightly adapted from https://heracles.readthedocs.io/stable/examples/example.html
"""

import numpy as np
import camb
from camb.sources import SplinedSourceWindow


# from https://camb.readthedocs.io/en/latest/CAMBdemo.html
# pars = camb.CAMBparams()
# pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
# pars.InitPower.set_params(As=2e-9, ns=0.965)
# pars.set_for_lmax(lmax, lens_potential_accuracy=1)
# #set Want_CMB to true if you also want CMB spectra or correlations
# pars.Want_CMB = False
# #NonLinear_both or NonLinear_lens will use non-linear corrections
# pars.NonLinear = model.NonLinear_both
# #Set up W(z) window functions, later labelled W1, W2. Gaussian here.
# pars.SourceWindows = [
#     GaussianSourceWindow(redshift=0.17, source_type='counts', bias=1.2, sigma=0.04, dlog10Ndm=-0.2),
#     GaussianSourceWindow(redshift=0.5, source_type='lensing', sigma=0.07)]

# results = camb.get_results(pars)
# cls = results.get_source_cls_dict()


class CAMBQuantities():

    def __init__(self, Oc, Ob, h, w0, ns):
        self.Oc = Oc
        self.Ob = Ob
        self.h = h
        self.w0 = w0
        self.ns = ns

        # set up CAMB parameters for matter angular power spectrum
        self.pars = camb.set_params(H0=100 * h, omch2=Oc * h**2, ombh2=Ob * h**2,
                                    w=-1, ns=0.96, Want_CMB=False,
                                    NonLinear=camb.model.NonLinear_both)

    def compute_cls(self, lmin, lmax):

        pars.min_l = lmin
        pars.set_for_lmax(2 * lmax, lens_potential_accuracy=1)

        nz = np.genfromtxt('/home/davide/Documenti/Lavoro/Programmi/CLOE_benchmarks/nzTabSPV3.dat')
        zgrid = nz[:, 0]
        nz = nz[:, 1:]
        zbins = nz.shape[1]

        sources = []
        for zi in range(zbins):
            sources += [
                SplinedSourceWindow(source_type="counts", z=zgrid, W=nz[:, zi]),
                SplinedSourceWindow(source_type="lensing", z=zgrid, W=nz[:, zi]),
            ]
        pars.SourceWindows = sources

        results = camb.get_results(pars)
        camb_cls = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

        l = np.arange(lmax + 1)
        fl = -np.sqrt((l + 2) * (l + 1) * l * (l - 1))
        fl /= np.clip(l * (l + 1), 1, None)

        # mine
        self.cl_tt = np.zeros((len(l), zbins + 1, zbins + 1))
        self.cl_te = np.zeros((len(l), zbins + 1, zbins + 1))
        self.cl_et = np.zeros((len(l), zbins + 1, zbins + 1))
        self.cl_ee = np.zeros((len(l), zbins + 1, zbins + 1))
        for zi in range(1, zbins + 1):
            for zj in range(zi, zbins + 1):
                # get the full-sky spectra; B-mode is assumed zero
                # P is for CMB lensing!!
                self.cl_tt[:, zi, zj] = camb_cls[f"W{2 * zi - 1}xW{2 * zj - 1}"]
                self.cl_te[:, zi, zj] = fl * camb_cls[f"W{2 * zi - 1}xW{2 * zj}"]
                self.cl_et[:, zi, zj] = fl * camb_cls[f"W{2 * zi}xW{2 * zj - 1}"]
                self.cl_ee[:, zi, zj] = fl**2 * camb_cls[f"W{2 * zi}xW{2 * zj}"]

        # self.cl_tt = self.cl_tt[:, 1:, 1:]
        # self.cl_te = self.cl_te[:, 1:, 1:]
        # self.cl_et = self.cl_et[:, 1:, 1:]
        # self.cl_ee = self.cl_ee[:, 1:, 1:]
        # self.cl_pb = np.zeros_like(cl_te)
        # self.cl_bt = np.zeros_like(cl_et)
        # self.cl_bb = np.zeros_like(cl_ee)
        # self.cl_eb = np.zeros_like(cl_ee)
        # self.cl_be = np.zeros_like(cl_ee)

        # original
        # for i in range(1, zbins + 1):
            # for j in range(i, zbins + 1):
                # get the full-sky spectra; B-mode is assumed zero
                # cl_pp = camb_cls[f"W{2 * i - 1}xW{2 * j - 1}"]
                # cl_pe = fl * camb_cls[f"W{2 * i - 1}xW{2 * j}"]
                # cl_pb = np.zeros_like(cl_pe)
                # cl_ep = fl * camb_cls[f"W{2 * i}xW{2 * j - 1}"]
                # cl_bp = np.zeros_like(cl_ep)
                # cl_ee = fl**2 * camb_cls[f"W{2 * i}xW{2 * j}"]
                # cl_bb = np.zeros_like(cl_ee)
                # cl_eb = np.zeros_like(cl_ee)
                # cl_be = np.zeros_like(cl_ee)
