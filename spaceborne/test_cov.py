import numpy as np

class CovarianceMatrixTester:
    """
    A class to perform a suite of tests on a covariance matrix.
    """
    def __init__(self, cov_name: str, cov: np.ndarray, cfg: dict):
        """
        Initializes the CovarianceMatrixTester with the covariance matrix, its name,
        and the configuration settings.

        Args:
            cov_name (str): The name of the covariance matrix.
            cov (np.ndarray): The covariance matrix to test.
            cfg (dict): Configuration dictionary controlling which tests to run.
        """
        self.cov_name = cov_name
        self.cov = cov
        self.cfg = cfg
        self.warnings_log = [] # To store any warnings or issues encountered

    def _log_warning(self, message: str):
        """Helper to log warnings and print them."""
        self.warnings_log.append(f"{self.cov_name}: {message}")
        print(message)

    def run_all_tests(self):
        """
        Executes all configured tests on the covariance matrix.
        """
        print(f'Testing {self.cov_name}...')
        print(f'Matrix shape: {self.cov.shape}')
        print(f'Matrix dtype: {self.cov.dtype}')
        print("-" * 30) # Separator for clarity

        # Basic checks always run first and can short-circuit
        if not self._check_basic_issues():
            print("Skipping further tests due to basic issues.")
            print("=" * 30) # End separator
            return

        # Individual tests based on configuration
        if self.cfg['misc'].get('test_diagonal_elements', True): # Assuming true by default if not specified
            self._check_diagonal_elements()

        if self.cfg['misc'].get('test_condition_number', False):
            self._test_condition_number()

        if self.cfg['misc'].get('test_symmetry', False):
            self._test_symmetry()

        if self.cfg['misc'].get('test_cholesky_decomposition', False):
            self._test_cholesky_decomposition()

        if self.cfg['misc'].get('test_numpy_inversion', False):
            self._test_numpy_inversion()

        if self.cfg['misc'].get('test_eigenvalues', False):
            self._test_eigenvalues()

        print("=" * 30) # End separator
        print() # Blank line between tests for different matrices

    def _check_basic_issues(self) -> bool:
        """Checks for empty, NaN, Inf, or all-zero matrices."""
        if self.cov.size == 0:
            self._log_warning('Warning: Matrix is empty!')
            return False

        if np.any(np.isnan(self.cov)) or np.any(np.isinf(self.cov)):
            self._log_warning('Warning: Matrix contains NaN or Inf values!')
            return False

        if np.allclose(self.cov, 0):
            self._log_warning('Warning: Matrix is all zeros!')
            return False
        return True

    def _check_diagonal_elements(self):
        """Checks if diagonal elements are non-positive."""
        diag_elements = np.diag(self.cov)
        if np.any(diag_elements <= 0):
            self._log_warning('Warning: Matrix has non-positive diagonal elements!')
            print(f'Min diagonal element: {np.min(diag_elements)}')
            print(f'Number of non-positive diagonal elements: {np.sum(diag_elements <= 0)}')

    def _test_condition_number(self):
        """Computes and checks the condition number."""
        try:
            cond_number = np.linalg.cond(self.cov)
            print(f'Condition number = {cond_number:.4e}')
            if cond_number > 1e12:
                self._log_warning('Warning: Matrix is poorly conditioned (cond > 1e12)')
        except np.linalg.LinAlgError as e:
            self._log_warning(f'Could not compute condition number: {e}')

    def _test_symmetry(self):
        """Checks if the matrix is symmetric."""
        if not np.allclose(self.cov, self.cov.T, atol=1e-14, rtol=1e-12):
            self._log_warning('Warning: Matrix is not symmetric.')
            max_asymmetry = np.max(np.abs(self.cov - self.cov.T))
            print(f'Maximum asymmetry: {max_asymmetry:.2e}')
        else:
            print('Matrix is symmetric.')

    def _test_cholesky_decomposition(self):
        """Attempts Cholesky decomposition to check for positive definiteness."""
        try:
            L = np.linalg.cholesky(self.cov)
            print('Cholesky decomposition successful')
            if np.allclose(L @ L.T, self.cov, atol=1e-12, rtol=1e-10):
                print('Cholesky decomposition verified (L @ L.T == cov)')
            else:
                self._log_warning('Warning: Cholesky decomposition verification failed')
        except np.linalg.LinAlgError as e:
            self._log_warning(f'Cholesky decomposition failed: {e}. This indicates the matrix is not positive definite.')

    def _test_numpy_inversion(self):
        """Attempts matrix inversion and verifies correctness."""
        try:
            inv_cov = np.linalg.inv(self.cov)
            print('Numpy inversion successful.')
            identity_test = np.dot(self.cov, inv_cov)
            identity_check = np.allclose(identity_test, np.eye(self.cov.shape[0]), atol=1e-12, rtol=1e-10)

            if identity_check:
                print('Inverse test successful (M @ M^{-1} ≈ I)')
            else:
                self._log_warning('Warning: Inverse test failed')
                max_deviation = np.max(np.abs(identity_test - np.eye(self.cov.shape[0])))
                print(f'Maximum deviation from identity: {max_deviation:.2e}')
        except np.linalg.LinAlgError as e:
            self._log_warning(f'Numpy inversion failed: {e}. Matrix is singular or near-singular.')

    def _test_eigenvalues(self):
        """Computes and checks eigenvalues for positive definiteness."""
        try:
            eigenvals = np.linalg.eigvals(self.cov)
            min_eigenval = np.min(eigenvals)
            max_eigenval = np.max(eigenvals)

            print(f'Eigenvalue range: [{min_eigenval:.2e}, {max_eigenval:.2e}]')

            if min_eigenval <= 0:
                self._log_warning(f'Warning: Matrix has {np.sum(eigenvals <= 0)} non-positive eigenvalues. This indicates the matrix is not positive definite.')
            else:
                print('All eigenvalues are positive (matrix is positive definite)')
        except np.linalg.LinAlgError as e:
            self._log_warning(f'Eigenvalue computation failed: {e}')