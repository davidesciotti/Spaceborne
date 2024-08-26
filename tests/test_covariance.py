import unittest
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as npt
from unittest import TestCase
import sys
sys.path.append('/home/davide/Documenti/Lavoro/Programmi/Spaceborne')
import spaceborne.my_module as mm
import matplotlib as mpl

mpl.use('QT5Agg')

import unittest
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as npt
from unittest import TestCase
import spaceborne.my_module as mm


class TestCovarianceMatrix(unittest.TestCase):
    cov_filename = None  # Class attribute to hold the file path

    def setUp(self):
        self.cov = np.genfromtxt(
            '/home/davide/Documenti/Lavoro/Programmi/common_data/Spaceborne/jobs/SPV3/output/Flagship_2/covmat/Sesto/covmat_GSSC_Spaceborne_3x2pt_zbinsEP03_lmax3000_ML245_ZL02_MS245_ZS02_pkHMCodeBar_13245deg2_2D.dat')
        self.length = self.cov.shape[0] if self.cov is not None else None

    def test_plot(self):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        # Plot the covariance matrix
        im1 = ax[0].matshow(self.cov)
        ax[0].set_title('Cov')
        fig.colorbar(im1, ax=ax[0], shrink=.6)

        # Plot the log10 of the Cov
        im2 = ax[1].matshow(np.log10(self.cov))
        ax[1].set_title('log10(Cov)')
        fig.colorbar(im2, ax=ax[1], shrink=.6)

        # Plot the correlation matrix
        im3 = ax[2].matshow(mm.cov2corr(self.cov))
        ax[2].set_title('Corr')
        fig.colorbar(im3, ax=ax[2], shrink=.6)

        plt.tight_layout()
        plt.show()

    def test_symmetry(self):
        npt.assert_allclose(self.cov, self.cov.T, atol=0, rtol=1e-6, err_msg=f"Covariance matrix is not symmetric")

    def test_positive_semi_definiteness(self):
        eigenvalues = np.linalg.eigvalsh(self.cov)
        self.assertTrue(np.all(eigenvalues >= 0), msg=f"Covariance matrix is not positive semi-definite")

    def test_inversion(self):
        inv_cov = np.linalg.inv(self.cov)
        npt.assert_allclose(np.eye(self.cov.shape[0]),
                            np.dot(self.cov, inv_cov), atol=1e-7, rtol=1e-6,
                            err_msg=f"Covariance matrix is not invertible")

    def test_condition_number(self):
        eigenvalues = np.linalg.eigvalsh(self.cov)
        condition_number = np.max(eigenvalues) / np.min(eigenvalues)
        self.assertTrue(condition_number < 1e10,
                        msg=f"Covariance matrix has a large condition number: {condition_number:2e}")

    def test_eigenvalue_analysis(self):
        eigenvalues = np.linalg.eigvalsh(self.cov)
        self.assertGreaterEqual(np.min(eigenvalues), 0, "Negative eigenvalues detected")
        self.assertLess(np.max(eigenvalues) / np.min(eigenvalues), 1e10,
                        f"Large eigenvalue ratio: {np.max(eigenvalues) / np.min(eigenvalues):2e}")

    def test_frobenius_norm(self):
        norm = np.linalg.norm(self.cov, 'fro')
        self.assertGreater(norm, 0, "Frobenius norm should be positive")
        self.assertLess(norm, 1e10, f"Unusually large Frobenius norm: {norm:2e}")

    def test_determinant(self):
        det = np.linalg.det(self.cov)
        self.assertGreater(det, 0, "Determinant should be positive for a positive definite matrix")
        self.assertLess(det, 1e100, f"Unusually large determinant: {det:2e}")

    def test_positive_definiteness(self):
        try:
            np.linalg.cholesky(self.cov)
            self.assertTrue(True, "Matrix is positive definite")
        except np.linalg.LinAlgError:
            self.assertFalse(True, "Matrix is not positive definite")

    def test_sparsity(self):
        total_elements = self.cov.size
        near_zero_elements = np.sum(np.abs(self.cov) < 1e-6)
        sparsity = near_zero_elements / total_elements
        self.assertLess(sparsity, 0.5, f"Matrix is unusually sparse: {sparsity:.2f} of elements are near zero")

    def test_correlation_analysis(self):
        corr = mm.cov2corr(self.cov)
        max_correlation = np.max(np.abs(corr - np.eye(self.cov.shape[0])))
        min_correlation = np.min(np.abs(corr - np.eye(self.cov.shape[0])))
        self.assertLess(max_correlation, 0.9999, f"Unusually high correlation detected: {max_correlation:.4f}")

    def test_rank(self):
        rank = np.linalg.matrix_rank(self.cov)
        self.assertEqual(rank, self.cov.shape[0], f"Matrix is not full rank. Rank: {rank}, Size: {self.cov.shape[0]}")

    def test_trace_to_dimension_ratio(self):
        ratio = np.trace(self.cov) / self.cov.shape[0]
        self.assertGreater(ratio, 0, "Trace to dimension ratio should be positive")
        self.assertLess(ratio, 1e6, f"Unusually large trace to dimension ratio: {ratio:2e}")

    def test_effective_rank(self):
        eigenvalues = np.linalg.eigvalsh(self.cov)
        eff_rank = np.exp(np.sum(np.log(eigenvalues + 1e-10)) / self.cov.shape[0])
        self.assertGreater(eff_rank, 0, "Effective rank should be positive")
        self.assertLessEqual(eff_rank, self.cov.shape[0], f"Effective rank ({
                             eff_rank:.2f}) exceeds matrix size ({self.cov.shape[0]})")


if __name__ == '__main__':
    unittest.main()
