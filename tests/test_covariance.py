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


class TestCovarianceMatrix(unittest.TestCase):

    def setUp(self):
        self.cov = np.load(
            '/home/davide/Documenti/Lavoro/Programmi/CLOE_benchmarks/CovMat-3x2pt-BNT-GaussSSC-32Bins-13245deg2.npy')
        # self.cov = np.genfromtxt(
        # '/home/davide/Documenti/Lavoro/Programmi/common_data/Spaceborne/jobs/SPV3/output/Flagship_2/covmat/Sesto/covmat_GSSC_Spaceborne_3x2pt_zbinsEP03_lmax3000_ML245_ZL02_MS245_ZS02_pkHMCodeBar_13245deg2_2D.dat')
        # self.cov = self.cov[:91, :91]
        self.cov = mm.regularize_covariance(self.cov, lambda_reg=1e-5)
        self.length = self.cov.shape[0] if self.cov is not None else None
        self.eigenvalues = np.linalg.eigvalsh(self.cov)

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
        npt.assert_allclose(self.cov, self.cov.T, atol=0, rtol=1e-6,
                            err_msg="Covariance matrix is not symmetric")

    def test_positive_semi_definiteness(self):
        # A covariance matrix must be positive semi-definite to ensure that no variance is negative
        self.assertTrue(np.all(self.eigenvalues >= 0), msg=f"Covariance matrix is not positive semi-definite: \
                        negative eigenvalues detected")

    @unittest.skip("Skipping this test temporarily")
    def test_positive_definiteness(self):
        """
        Checks that the covariance matrix is positive definite by attempting to compute its Cholesky decomposition. \
        Failure in this decomposition would indicate that the matrix has negative eigenvalues
        """
        try:
            np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            self.fail("Matrix is not positive definite (Cholesky decomposition failed)")

    def test_inversion(self):
        try:
            inv_cov = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            self.fail("Matrix inversion failed with np.linalg.inv. The matrix might be singular or near-singular.")

        npt.assert_allclose(np.eye(self.cov.shape[0]), np.dot(self.cov, inv_cov),
                            atol=1e-7, rtol=1e-6,
                            err_msg="Covariance matrix is not invertible")

    @unittest.skip("Skipping this test temporarily")
    def test_condition_number(self):

        # Checks the condition number of the matrix to assess how sensitive the matrix inversion might be to
        # numerical errors. A high condition number indicates potential issues with numerical stability.
        # For linear systems, a good rule of thumb is that you lose a digit of precision due to conditioning
        # for every power of 10 in your condition number. Since double precision is a 15-16 digit representation,
        # a matrix with a condition number in the range of 10^3 to 10^6 isn't considered problematic from the
        # standpoint of numerical calculation with direct methods (LU decomposition, Cholesky, etc.)

        cond_number = np.linalg.cond(self.cov)
        threshold = 1 / np.finfo(self.cov.dtype).eps  # 1 / machine precision
        self.assertLess(cond_number, threshold,
                        msg=f"Matrix is ill-conditioned with a condition number larger than 1/machine precision: \
                        {cond_number:.2e} > {threshold:.2e}")

        # Check condition number using eigenvalues
        # eigenvalues = np.linalg.eigvalsh(self.cov)
        # min_eigenvalue = np.min(eigenvalues)
        # max_eigenvalue = np.max(eigenvalues)
        # condition_number = max_eigenvalue / min_eigenvalue

    def test_frobenius_norm(self):
        # Ensures that the matrix values are neither too small nor excessively large, which
        # could indicate numerical instability
        norm = np.linalg.norm(self.cov, 'fro')
        self.assertGreater(norm, 0, "Frobenius norm should be positive")
        self.assertLess(norm, 1e10, f"Unusually large Frobenius norm: {norm:.2e}")

    def test_determinant(self):
        det = np.linalg.det(self.cov)
        self.assertGreater(det, 0, "Determinant should be positive for a positive definite matrix")
        self.assertLess(det, 1e100, f"Unusually large determinant: {det:.2e}")

    def test_sparsity(self):
        threshold = 1e-6
        total_elements = self.cov.size
        near_zero_elements = np.sum(np.abs(self.cov) < threshold)
        sparsity = near_zero_elements / total_elements
        self.assertLess(sparsity, 0.5, f"Matrix is sparse: {sparsity:.2f} of elements are near zero (< {threshold})")

    def test_correlation_analysis(self):
        corr = mm.cov2corr(self.cov)
        max_correlation = np.max(corr)
        min_correlation = np.min(corr)
        self.assertLess(max_correlation, 1, f"Unusually high correlation detected: {max_correlation:.4f}")
        self.assertGreater(min_correlation, -1, f"Unusually low correlation detected: {min_correlation:.4f}")

    def test_rank(self):
        """
        Checks that the covariance matrix is of full rank, meaning it does not have any redundant rows or columns.
        A full-rank matrix is essential for the matrix to be invertible and for the linear systems involving the 
        matrix to be solvable.
        """
        rank = np.linalg.matrix_rank(self.cov)
        self.assertEqual(rank, self.cov.shape[0], f"Matrix is not full rank. Rank: {rank}, Size: {self.cov.shape[0]}")

    def test_trace_to_dimension_ratio(self):
        """
        Examines the ratio of the trace (sum of diagonal elements) of the covariance matrix to its dimension. 
        This ratio provides a quick check of the overall scale of the matrix.
        The test ensures that the trace is within a reasonable range, neither too small nor excessively large.
        """
        ratio = np.trace(self.cov) / self.cov.shape[0]
        self.assertGreater(ratio, 0, "Trace to dimension ratio should be positive")
        self.assertLess(ratio, 1e6, f"Unusually large trace to dimension ratio: {ratio:.2e}")

    def test_effective_rank(self):
        """
        This test computes the effective rank of the matrix, which is a more nuanced measure 
        of rank that takes into account the distribution of eigenvalues.
        The effective rank should be positive and less than or equal to the matrix size, 
        indicating that the matrix captures sufficient information without being overly redundant.
        """
        eff_rank = np.exp(np.sum(np.log(self.eigenvalues + 1e-10)) / self.cov.shape[0])
        self.assertGreater(eff_rank, 0, "Effective rank should be positive")
        self.assertLessEqual(eff_rank, self.cov.shape[0], f"Effective rank ({
                             eff_rank:.2f}) exceeds matrix size ({self.cov.shape[0]})")


if __name__ == '__main__':
    unittest.main()
