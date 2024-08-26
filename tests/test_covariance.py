import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/davide/Documenti/Lavoro/Programmi/Spaceborne')
import spaceborne.my_module as mm
import matplotlib as mpl

mpl.use('QT5Agg')



class CovarianceTester:
    def __init__(self, cov=None):
        self.cov = cov
        self.size = cov.shape[0] if cov is not None else None

    def plot(self):
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


    def load_from_npy(self, file_path):
        self.cov = np.load(file_path)
        self.size = self.cov.shape[0]

    def load_from_txt(self, file_path):
        self.cov = np.genfromtxt(file_path)
        self.size = self.cov.shape[0]

    def is_symmetric(self):
        return np.allclose(self.cov, self.cov.T)

    def is_positive_semi_definite(self):
        eigenvalues = np.linalg.eigvalsh(self.cov)
        return np.all(eigenvalues >= 0)

    def check_diagonal_dominance(self):
        for i in range(self.size):
            row_sum = np.sum(np.abs(self.cov[i])) - np.abs(self.cov[i, i])
            if np.abs(self.cov[i, i]) < row_sum:
                return False
        return True

    def check_inversion(self, inv_cov):
        return np.allclose(np.eye(self.cov), np.dot(inv_cov, inv_cov.T), atol=0, rtol=1e-6)

    def check_condition_number(self):
        eigenvalues = np.linalg.eigvalsh(self.cov)
        condition_number = np.max(eigenvalues) / np.min(eigenvalues)
        return condition_number

    def check_trace(self, theoretical_trace):
        trace = np.trace(self.cov)
        return np.isclose(trace, theoretical_trace)

    def check_normalization(self, norm_value):
        norm_matrix = self.cov / norm_value
        return np.allclose(np.sum(norm_matrix), 1.0)
    
    
    def eigenvalue_analysis(self):
        eigenvalues = np.linalg.eigvalsh(self.cov)
        return {
            'min_eigenvalue': np.min(eigenvalues),
            'max_eigenvalue': np.max(eigenvalues),
            'eigenvalue_ratio': np.max(eigenvalues) / np.min(eigenvalues),
            'negative_eigenvalues': np.sum(eigenvalues < 0),
        }

    def frobenius_norm(self):
        return np.linalg.norm(self.cov, 'fro')

    def determinant(self):
        return np.linalg.det(self.cov)

    def is_positive_definite(self):
        try:
            np.linalg.cholesky(self.cov)
            return True
        except np.linalg.LinAlgError:
            return False

    def sparsity_analysis(self, threshold=1e-6):
        total_elements = self.cov.size
        near_zero_elements = np.sum(np.abs(self.cov) < threshold)
        return near_zero_elements / total_elements

    def correlation_analysis(self):
        corr = mm.cov2corr(self.cov)
        return {
            'max_correlation': np.max(np.abs(corr - np.eye(self.size))),
            'min_correlation': np.min(np.abs(corr - np.eye(self.size))),
        }

    def rank_analysis(self):
        return np.linalg.matrix_rank(self.cov)

    def trace_to_dimension_ratio(self):
        return np.trace(self.cov) / self.size

    def effective_rank(self):
        eigenvalues = np.linalg.eigvalsh(self.cov)
        return np.exp(np.sum(np.log(eigenvalues + 1e-10)) / self.size)


class TestCovarianceMatrix(unittest.TestCase):
    cov_filename = None  # Class attribute to hold the file path

    def setUp(self):
        self.tester = CovarianceTester()
        try:
            self.tester.load_from_npy(self.cov_filename)
        except ValueError:
            self.tester.load_from_txt(self.cov_filename)
            
    def test_plot(self):
        self.tester.plot()

    def test_symmetry(self):
        self.assertTrue(self.tester.is_symmetric(), msg=f"Covariance matrix is not symmetric")

    def test_positive_semi_definiteness(self):
        self.assertTrue(self.tester.is_positive_semi_definite(), msg=f"Covariance matrix is not positive semi-definite")

    def test_inversion(self):
        inv_cov = np.linalg.inv(self.tester.cov)
        self.assertTrue(self.tester.check_inversion(inv_cov), msg=f"Covariance matrix is not invertible")

    def test_condition_number(self):
        self.assertTrue(self.tester.check_condition_number() < 1e10,
                        msg=f"Covariance matrix has a large condition number: {self.tester.check_condition_number():2e}")

    def test_eigenvalue_analysis(self):
        result = self.tester.eigenvalue_analysis()
        self.assertGreaterEqual(result['min_eigenvalue'], 0, "Negative eigenvalues detected")
        self.assertLess(result['eigenvalue_ratio'], 1e10, f"Large eigenvalue ratio: {result['eigenvalue_ratio']:2e}")
        self.assertEqual(result['negative_eigenvalues'], 0, "Negative eigenvalues detected")

    def test_frobenius_norm(self):
        norm = self.tester.frobenius_norm()
        self.assertGreater(norm, 0, "Frobenius norm should be positive")
        self.assertLess(norm, 1e10, f"Unusually large Frobenius norm: {norm:2e}")

    def test_determinant(self):
        det = self.tester.determinant()
        self.assertGreater(det, 0, "Determinant should be positive for a positive definite matrix")
        self.assertLess(det, 1e100, f"Unusually large determinant: {det:2e}")

    def test_positive_definiteness(self):
        self.assertTrue(self.tester.is_positive_definite(), "Matrix is not positive definite")

    def test_sparsity(self):
        sparsity = self.tester.sparsity_analysis()
        self.assertLess(sparsity, 0.5, f"Matrix is unusually sparse: {sparsity:.2f} of elements are near zero")

    def test_correlation_analysis(self):
        result = self.tester.correlation_analysis()
        self.assertLess(result['max_correlation'], 0.9999, f"Unusually high correlation detected: {result['max_correlation']:.4f}")

    def test_rank(self):
        rank = self.tester.rank_analysis()
        self.assertEqual(rank, self.tester.size, f"Matrix is not full rank. Rank: {rank}, Size: {self.tester.size}")

    def test_trace_to_dimension_ratio(self):
        ratio = self.tester.trace_to_dimension_ratio()
        self.assertGreater(ratio, 0, "Trace to dimension ratio should be positive")
        self.assertLess(ratio, 1e6, f"Unusually large trace to dimension ratio: {ratio:2e}")

    def test_effective_rank(self):
        eff_rank = self.tester.effective_rank()
        self.assertGreater(eff_rank, 0, "Effective rank should be positive")
        self.assertLessEqual(eff_rank, self.tester.size, f"Effective rank ({eff_rank:.2f}) exceeds matrix size ({self.tester.size})")

    # def test_diagonal_dominance(self):
    #     self.assertTrue(self.tester.check_diagonal_dominance())

    # def test_trace(self):
    #     theoretical_trace = 3.0  # Replace with the correct value for your case
    #     self.assertTrue(self.tester.check_trace(theoretical_trace))

    # def test_normalization(self):
    #     norm_value = 3.5  # Example normalization value
    #     self.assertTrue(self.tester.check_normalization(norm_value))


if __name__ == '__main__':
    TestCovarianceMatrix.cov_filename = '/home/davide/Documenti/Lavoro/Programmi/CLOE_benchmarks/CovMat-3x2pt-BNT-Gauss-32Bins.npy'
    unittest.main()
