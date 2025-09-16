"""
python module for calculating integrals with 2 Bessel / spherical Bessel functions.

by Xiao Fang
Feb 22, 2020

MIT License

Copyright (c) 2020 Xiao Fang (方啸)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import sys

import numpy as np
from numpy.fft import irfft2, rfft2
from scipy.special import gamma


class TwoSphBessel:
    def __init__(
        self,
        x1,
        x2,
        fx1x2,
        nu1=1.01,
        nu2=1.01,
        N_extrap_low=0,
        N_extrap_high=0,
        c_window_width=0.25,
        N_pad=0,
    ):
        self.x1_origin = self.x1 = x1  # x is logarithmically spaced
        self.x2_origin = self.x2 = x2

        # self.lnx = np.log(x)
        self.dlnx1 = np.log(x1[1] / x1[0])
        self.dlnx2 = np.log(x2[1] / x2[0])
        self.fx1x2_origin = self.fx1x2 = fx1x2  # f(x1,x2) array
        self.nu1 = nu1
        self.nu2 = nu2
        self.N_extrap_low = N_extrap_low
        self.N_extrap_high = N_extrap_high
        self.c_window_width = c_window_width

        self.N1 = self.x1.size
        self.N2 = self.x2.size
        if (self.N1 + N_extrap_low + N_extrap_high) % 2 == 1 or (
            self.N2 + N_extrap_low + N_extrap_high
        ) % 2 == 1:  # Make sure the array sizes are even
            print('Error: array sizes have to be even!')
            sys.exit()

        # extrapolate x and f(x) linearly in log(x), and log(f(x))
        if N_extrap_low or N_extrap_high:
            self.x1 = log_extrap(x1, N_extrap_low, N_extrap_high)
            self.x2 = log_extrap(x2, N_extrap_low, N_extrap_high)
            self.fx1x2 = bilinear_extra_P(fx1x2, N_extrap_low, N_extrap_high)
            self.N1 += N_extrap_low + N_extrap_high
            self.N2 += N_extrap_low + N_extrap_high

        print(self.N1, self.N2)
        # zero-padding
        self.N_pad = N_pad
        if N_pad:
            pad = np.zeros(N_pad)
            self.x1 = log_extrap(self.x1, N_pad, N_pad)
            self.x2 = log_extrap(self.x2, N_pad, N_pad)
            self.N1 += 2 * N_pad
            self.N2 += 2 * N_pad
            zeros = np.zeros((self.N1, self.N2))
            zeros[N_pad:-N_pad, N_pad:-N_pad] = self.fx1x2
            self.fx1x2 = zeros
            self.N_extrap_high += N_pad
            self.N_extrap_low += N_pad

        self.m, self.n, self.c_mn = self.get_c_mn()
        self.eta_m = 2 * np.pi / self.dlnx1 / float(self.N1) * self.m
        self.eta_n = 2 * np.pi / self.dlnx2 / float(self.N2) * self.n

        self.x10 = self.x1[0]
        self.x20 = self.x2[0]
        self.z1 = self.nu1 + 1j * self.eta_m
        self.z2 = self.nu2 + 1j * self.eta_n

        self.y1 = 1.0 / self.x1[::-1]
        self.y2 = 1.0 / self.x2[::-1]
        self.y10 = self.y1[0]
        self.y20 = self.y2[0]

    def get_c_mn(self):
        """
        return m and c_mn
        c_mn: the smoothed 2D-FFT coefficients of "biased" input function f(x):
        f_b = f(x_1,x_2) / x_1^\nu_1 / x_2^\nu_2

        number of x1, x2 values should be even
        c_window_width: the fraction of any row/column c_mn elements that are smoothed.
        """
        print(self.fx1x2.shape)
        print(self.x2.size, self.x1.size)
        f_b = ((self.fx1x2 * self.x2 ** (-self.nu2)).T * self.x1 ** (-self.nu1)).T
        c_mn = rfft2(f_b)

        m = np.arange(-self.N1 // 2, self.N1 // 2 + 1)
        n = np.arange(-self.N2 // 2, self.N2 // 2 + 1)

        c_mn1 = c_mn[: self.N1 // 2 + 1, :]
        c_mn = np.vstack((c_mn[self.N1 // 2 :, :], c_mn1[:, :]))

        c_mn_left = np.conj(np.flip(np.flip(c_mn, 0), 1))
        c_mn = np.hstack((c_mn_left[:, :-1], c_mn))

        c_window_array1 = c_window(m, int(self.c_window_width * self.N1 // 2.0))
        c_window_array2 = c_window(n, int(self.c_window_width * self.N2 // 2.0))
        c_mn_filter = ((c_mn * c_window_array2).T * c_window_array1).T
        return m, n, c_mn

    def two_sph_bessel(self, ell1, ell2):
        """
        Calculate F(y_1,y_2) = \int_0^\infty dx_1 / x_1 \int_0^\infty dx_2 / x_2 * f(x_1,x_2) * j_{\ell_1}(x_1y_1) * j_{\ell_2}(x_2y_2),
        where j_\ell is the spherical Bessel func of order ell.
        array y is set as y[:] = 1/x[::-1]
        """

        g1 = g_l(ell1, self.z1)
        g2 = g_l(ell2, self.z2)

        mat = np.conj(
            (self.c_mn * (self.x20 * self.y20) ** (-1j * self.eta_n) * g2).T
            * (self.x10 * self.y10) ** (-1j * self.eta_m)
            * g1
        ).T
        mat_right = mat[:, self.N2 // 2 :]
        mat_adjust = np.vstack(
            (mat_right[self.N1 // 2 :, :], mat_right[1 : self.N1 // 2, :])
        )
        # print(mat_adjust[0][1])
        Fy1y2 = (
            (irfft2(mat_adjust) * np.pi / 16.0 / self.y2**self.nu2).T
            / self.y1**self.nu1
        ).T
        # print(Fy1y2)
        return (
            self.y1[self.N_extrap_high : self.N1 - self.N_extrap_low],
            self.y2[self.N_extrap_high : self.N2 - self.N_extrap_low],
            Fy1y2[
                self.N_extrap_high : self.N1 - self.N_extrap_low,
                self.N_extrap_high : self.N2 - self.N_extrap_low,
            ],
        )

    def two_sph_bessel_binave(self, ell1, ell2, binwidth_dlny1, binwidth_dlny2):
        """
        Bin-averaging for 3D statistics: alpha_pow = D = 3
        Calculate F(y_1,y_2) = \int_0^\infty dx_1 / x_1 \int_0^\infty dx_2 / x_2 * f(x_1,x_2) * j_{\ell_1}(x_1y_1) * j_{\ell_2}(x_2y_2),
        where j_\ell is the spherical Bessel func of order ell.
        array y is set as y[:] = 1/x[::-1]
        """
        D = 3
        s_d_lambda1 = (np.exp(D * binwidth_dlny1) - 1.0) / D
        s_d_lambda2 = (np.exp(D * binwidth_dlny2) - 1.0) / D
        g1 = g_l_smooth(ell1, self.z1, binwidth_dlny1, D) / s_d_lambda1
        g2 = g_l_smooth(ell2, self.z2, binwidth_dlny2, D) / s_d_lambda2

        mat = np.conj(
            (self.c_mn * (self.x20 * self.y20) ** (-1j * self.eta_n) * g2).T
            * (self.x10 * self.y10) ** (-1j * self.eta_m)
            * g1
        ).T
        mat_right = mat[:, self.N2 // 2 :]
        mat_adjust = np.vstack(
            (mat_right[self.N1 // 2 :, :], mat_right[1 : self.N1 // 2, :])
        )
        # print(mat_adjust[0][1])
        Fy1y2 = (
            (irfft2(mat_adjust) * np.pi / 16.0 / self.y2 ** (self.nu2)).T
            / self.y1 ** (self.nu1)
        ).T

        return (
            self.y1[self.N_extrap_high : self.N1 - self.N_extrap_low],
            self.y2[self.N_extrap_high : self.N2 - self.N_extrap_low],
            Fy1y2[
                self.N_extrap_high : self.N1 - self.N_extrap_low,
                self.N_extrap_high : self.N2 - self.N_extrap_low,
            ],
        )


class TwoBessel:
    def __init__(
        self,
        x1,
        x2,
        fx1x2,
        nu1=1.01,
        nu2=1.01,
        N_extrap_low=0,
        N_extrap_high=0,
        c_window_width=0.25,
        N_pad=0,
    ):
        self.two_sph = TwoSphBessel(
            x1,
            x2,
            (fx1x2.T * np.sqrt(x1)).T * np.sqrt(x2),
            nu1,
            nu2,
            N_extrap_low,
            N_extrap_high,
            c_window_width,
            N_pad,
        )

    def two_Bessel_binave(self, ell1, ell2, binwidth_dlny1, binwidth_dlny2):
        """
        Bin-averaging for 2D statistics: D = 2, alpha_pow = 2.5
        Calculate F(y_1,y_2) = \int_0^\infty dx_1 / x_1 \int_0^\infty dx_2 / x_2 * f(x_1,x_2) * J_{\ell_1}(x_1y_1) * J_{\ell_2}(x_2y_2),
        where J_\ell is the Bessel func of order ell.
        array y is set as y[:] = 1/x[::-1]
        """
        two_sph = self.two_sph
        D = 2
        s_d_lambda1 = (np.exp(D * binwidth_dlny1) - 1.0) / D
        s_d_lambda2 = (np.exp(D * binwidth_dlny2) - 1.0) / D

        g1 = g_l_smooth(ell1 - 0.5, two_sph.z1, binwidth_dlny1, D + 0.5) / s_d_lambda1
        g2 = g_l_smooth(ell2 - 0.5, two_sph.z2, binwidth_dlny2, D + 0.5) / s_d_lambda2

        mat = np.conj(
            (two_sph.c_mn * (two_sph.x20 * two_sph.y20) ** (-1j * two_sph.eta_n) * g2).T
            * (two_sph.x10 * two_sph.y10) ** (-1j * two_sph.eta_m)
            * g1
        ).T
        mat_right = mat[:, two_sph.N2 // 2 :]
        mat_adjust = np.vstack(
            (mat_right[two_sph.N1 // 2 :, :], mat_right[1 : two_sph.N1 // 2, :])
        )
        # print(mat_adjust[0][1])
        Fy1y2 = (
            (irfft2(mat_adjust) / 8.0 / two_sph.y2 ** (two_sph.nu2 - 0.5)).T
            / two_sph.y1 ** (two_sph.nu1 - 0.5)
        ).T

        return (
            two_sph.y1[two_sph.N_extrap_high : two_sph.N1 - two_sph.N_extrap_low],
            two_sph.y2[two_sph.N_extrap_high : two_sph.N2 - two_sph.N_extrap_low],
            Fy1y2[
                two_sph.N_extrap_high : two_sph.N1 - two_sph.N_extrap_low,
                two_sph.N_extrap_high : two_sph.N2 - two_sph.N_extrap_low,
            ],
        )


### Utility functions ####################


## functions related to gamma functions
def g_m_vals(mu, q):
    """
    g_m_vals function is adapted from FAST-PT
    """
    imag_q = np.imag(q)

    g_m = np.zeros(q.size, dtype=complex)

    cut = 200
    asym_q = q[np.absolute(imag_q) > cut]
    asym_plus = (mu + 1 + asym_q) / 2.0
    asym_minus = (mu + 1 - asym_q) / 2.0

    q_good = q[(np.absolute(imag_q) <= cut) & (q != mu + 1 + 0.0j)]

    alpha_plus = (mu + 1 + q_good) / 2.0
    alpha_minus = (mu + 1 - q_good) / 2.0

    g_m[(np.absolute(imag_q) <= cut) & (q != mu + 1 + 0.0j)] = gamma(
        alpha_plus
    ) / gamma(alpha_minus)

    # asymptotic form
    g_m[np.absolute(imag_q) > cut] = np.exp(
        (asym_plus - 0.5) * np.log(asym_plus)
        - (asym_minus - 0.5) * np.log(asym_minus)
        - asym_q
        + 1.0 / 12 * (1.0 / asym_plus - 1.0 / asym_minus)
        + 1.0 / 360.0 * (1.0 / asym_minus**3 - 1.0 / asym_plus**3)
        + 1.0 / 1260 * (1.0 / asym_plus**5 - 1.0 / asym_minus**5)
    )

    g_m[np.where(q == mu + 1 + 0.0j)[0]] = 0.0 + 0.0j

    return g_m


def g_l(l, z_array):
    """
    gl = 2.**z_array * gamma((l+z_array)/2.) / gamma((3.+l-z_array)/2.)
    """
    gl = 2.0**z_array * g_m_vals(l + 0.5, z_array - 1.5)
    return gl


def g_l_smooth(l, z_array, binwidth_dlny, alpha_pow):
    """
    gl_smooth = 2.**z_array * gamma((l+z_array)/2.) / gamma((3.+l-z_array)/2.) * exp((alpha_pow - z_array)*binwidth_dlny -1. ) / (alpha_pow - z_array)
    """
    gl = 2.0**z_array * g_m_vals(l + 0.5, z_array - 1.5)
    gl *= (np.exp((alpha_pow - z_array) * binwidth_dlny) - 1.0) / (alpha_pow - z_array)
    return gl


## Window function
def c_window(n, n_cut):
    n_right = n[-1] - n_cut
    n_left = n[0] + n_cut

    n_r = n[n[:] > n_right]
    n_l = n[n[:] < n_left]

    theta_right = (n[-1] - n_r) / float(n[-1] - n_right - 1)
    theta_left = (n_l - n[0]) / float(n_left - n[0] - 1)

    W = np.ones(n.size)
    W[n[:] > n_right] = theta_right - 1 / (2 * np.pi) * np.sin(2 * np.pi * theta_right)
    W[n[:] < n_left] = theta_left - 1 / (2 * np.pi) * np.sin(2 * np.pi * theta_left)

    return W


## Extrapolation
def log_extrap(x, N_extrap_low, N_extrap_high):
    low_x = high_x = []
    if N_extrap_low:
        dlnx_low = np.log(x[1] / x[0])
        low_x = x[0] * np.exp(dlnx_low * np.arange(-N_extrap_low, 0))
    if N_extrap_high:
        dlnx_high = np.log(x[-1] / x[-2])
        high_x = x[-1] * np.exp(dlnx_high * np.arange(1, N_extrap_high + 1))
    x_extrap = np.hstack((low_x, x, high_x))
    return x_extrap


def bilinear_extra_P(fk1k2, N_low, N_high):
    """
    2d bilinear extrapolation of the input fk1k2 matrix

    fk1k2: input matrix
    N_low: number of points to extrapolate on the lower sides
    N_high: number of points to extrapolate on the higher sides
    """
    if np.any(fk1k2 <= 0):
        raise ValueError(
            'bilinear_extra_P requires all positive values in fk1k2 '
            'for log extrapolation'
        )
    logfk1k2 = np.log(fk1k2)  # This Extrapolation only works in log space
    h_grad_left = logfk1k2[:, 1] - logfk1k2[:, 0]  # horizontal gradient left side
    h_grad_right = logfk1k2[:, -1] - logfk1k2[:, -2]  # horizontal gradient right side
    add_left = np.arange(-N_low, 0)
    left_matrix = (
        np.matrix(h_grad_left).T.dot(np.matrix(add_left)) + np.matrix(logfk1k2[:, 0]).T
    )
    add_right = np.arange(1, N_high + 1)
    right_matrix = (
        np.matrix(h_grad_right).T.dot(np.matrix(add_right))
        + np.matrix(logfk1k2[:, -1]).T
    )
    new_logfk1k2 = np.hstack((left_matrix, logfk1k2, right_matrix))  ## type: matrix

    v_grad_up = new_logfk1k2[1, :] - new_logfk1k2[0, :]  ## type: matrix
    v_grad_down = new_logfk1k2[-1, :] - new_logfk1k2[-2, :]  ## type: matrix
    up_matrix = np.matrix(add_left).T.dot(v_grad_up) + np.matrix(new_logfk1k2[0, :])
    down_matrix = np.matrix(add_right).T.dot(v_grad_down) + np.matrix(
        new_logfk1k2[-1, :]
    )
    result_matrix = np.vstack((up_matrix, new_logfk1k2, down_matrix))  ## type: matrix
    return np.exp(np.array(result_matrix))  ## type: array
