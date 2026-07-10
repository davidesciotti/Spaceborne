import numpy as np
from scipy.linalg import cholesky, LinAlgError
import warnings


class CovarianceMatrixChecker:
    """
    A class to perform comprehensive sanity checks on covariance matrices.
    """

    def __init__(self, rtol=1e-6, atol=0):
        """
        Initialize the checker with numerical tolerance.

        Args:
            rtol (float): Relative tolerance for checks (default: 1e-6)
            atol (float): Absolute tolerance for checks (default: 0)
        """
        self.rtol = rtol
        self.atol = atol
        self.results = {}
        # Cache expensive computations
        self._eigenvalues = None
        self._inverse_matrix = None
        self._cholesky_L = None

    def check_all(self, matrix, verbose=True):
        """
        Perform all sanity checks on the covariance matrix.

        Args:
            matrix (np.ndarray): Input covariance matrix
            verbose (bool): Print detailed results

        Returns:
            dict: Dictionary containing all check results
        """
        self.results = {}
        # Reset cached computations for new matrix
        self._eigenvalues = None
        self._inverse_matrix = None
        self._cholesky_L = None

        # Basic structure checks
        self._check_shape(matrix)
        self._check_square(matrix)
        self._check_symmetry(matrix)

        # Mathematical property checks
        self._check_positive_definite(matrix)
        self._check_eigenvalues(matrix)
        self._check_diagonal_positive(matrix)

        # Numerical stability checks
        self._check_condition_number(matrix)
        self._check_invertible(matrix)
        self._check_cholesky_decomposition(matrix)

        # Additional checks
        self._check_finite_values(matrix)
        self._check_determinant(matrix)

        if verbose:
            self.print_results()

        return self.results

    def _get_eigenvalues(self, matrix):
        """Get cached eigenvalues or compute them."""
        if self._eigenvalues is None:
            try:
                self._eigenvalues = np.linalg.eigvals(matrix)
            except Exception:
                self._eigenvalues = None
        return self._eigenvalues

    def _get_inverse(self, matrix):
        """Get cached inverse matrix or compute it."""
        if self._inverse_matrix is None:
            try:
                self._inverse_matrix = np.linalg.inv(matrix)
            except Exception:
                self._inverse_matrix = None
        return self._inverse_matrix

    def _get_cholesky(self, matrix):
        """Get cached Cholesky decomposition or compute it."""
        if self._cholesky_L is None:
            try:
                self._cholesky_L = cholesky(matrix, lower=True)
            except Exception:
                self._cholesky_L = None
        return self._cholesky_L

    def _check_shape(self, matrix):
        """Check if matrix is 2D numpy array."""
        print('Checking shape...')
        try:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2:
                self.results['shape'] = {
                    'valid': False,
                    'message': f'Matrix must be 2D, got {matrix.ndim}D',
                }
            else:
                self.results['shape'] = {
                    'valid': True,
                    'message': f'Shape: {matrix.shape}',
                }
        except Exception as e:
            self.results['shape'] = {
                'valid': False,
                'message': f'Shape check failed: {str(e)}',
            }

    def _check_square(self, matrix):
        """Check if matrix is square."""
        print('Checking if matrix is square...')
        try:
            n_rows, n_cols = matrix.shape
            if n_rows != n_cols:
                self.results['square'] = {
                    'valid': False,
                    'message': f'Matrix must be square, got {n_rows}x{n_cols}',
                }
            else:
                self.results['square'] = {
                    'valid': True,
                    'message': f'Square matrix: {n_rows}x{n_cols}',
                }
        except Exception as e:
            self.results['square'] = {
                'valid': False,
                'message': f'Square check failed: {str(e)}',
            }

    def _check_symmetry(self, matrix):
        """Check if matrix is symmetric."""
        print('Checking symmetry...')
        try:
            is_symmetric = np.allclose(matrix, matrix.T, rtol=self.rtol, atol=self.atol)
            max_asymmetry = np.max(np.abs(matrix - matrix.T))

            self.results['symmetry'] = {
                'valid': is_symmetric,
                'max_asymmetry': max_asymmetry,
                'message': f'Symmetric: {is_symmetric}, max asymmetry: {max_asymmetry:.2e}',
            }
        except Exception as e:
            self.results['symmetry'] = {
                'valid': False,
                'message': f'Symmetry check failed: {str(e)}',
            }

    def _check_positive_definite(self, matrix):
        """Check if matrix is positive definite."""
        print('Checking positive definiteness...')
        try:
            eigenvals = self._get_eigenvalues(matrix)
            if eigenvals is None:
                self.results['positive_definite'] = {'valid': False, 'message': 'PD check failed: Cannot compute eigenvalues'}
                return
                
            min_eigenval = np.min(eigenvals)
            # Use relative tolerance: eigenvalue should be > rtol * max_eigenvalue
            max_eigenval = np.max(eigenvals)
            threshold = max(self.rtol * max_eigenval, self.atol)
            is_pos_def = min_eigenval > threshold

            self.results['positive_definite'] = {
                'valid': is_pos_def,
                'min_eigenvalue': min_eigenval,
                'threshold': threshold,
                'message': f'Positive definite: {is_pos_def}, min eigenvalue: {min_eigenval:.2e}, threshold: {threshold:.2e}',
            }
        except Exception as e:
            self.results['positive_definite'] = {
                'valid': False,
                'message': f'PD check failed: {str(e)}',
            }

    def _check_eigenvalues(self, matrix):
        """Analyze eigenvalue spectrum."""
        print('Checking eigenvalue spectrum...')
        try:
            eigenvals = self._get_eigenvalues(matrix)
            if eigenvals is None:
                self.results['eigenvalues'] = {'valid': False, 'message': 'Eigenvalue check failed: Cannot compute eigenvalues'}
                return
                
            eigenvals_sorted = np.sort(eigenvals)[::-1]  # Sort descending
            
            # Use relative tolerance for zero/negative detection
            max_eigenval = np.abs(eigenvals_sorted[0])
            threshold = max(self.rtol * max_eigenval, self.atol)
            
            negative_count = np.sum(eigenvals < -threshold)
            zero_count = np.sum(np.abs(eigenvals) <= threshold)

            self.results['eigenvalues'] = {
                'values': eigenvals_sorted,
                'negative_count': negative_count,
                'zero_count': zero_count,
                'min_eigenvalue': eigenvals_sorted[-1],
                'max_eigenvalue': eigenvals_sorted[0],
                'threshold': threshold,
                'message': f'Eigenvalues: {len(eigenvals)} total, {negative_count} negative, {zero_count} zero (threshold: {threshold:.2e})',
            }
        except Exception as e:
            self.results['eigenvalues'] = {
                'valid': False,
                'message': f'Eigenvalue check failed: {str(e)}',
            }

    def _check_diagonal_positive(self, matrix):
        """Check if diagonal elements are positive (variances)."""
        print('Checking if diagonal elements are positive...')
        try:
            diagonal = np.diag(matrix)
            # Use relative tolerance: diagonal elements should be > rtol * max_diagonal
            max_diagonal = np.max(np.abs(diagonal))
            threshold = max(self.rtol * max_diagonal, self.atol)
            
            negative_diagonal = np.sum(diagonal < -threshold)
            zero_diagonal = np.sum(np.abs(diagonal) <= threshold)

            self.results['diagonal'] = {
                'valid': negative_diagonal == 0,
                'diagonal_elements': diagonal,
                'negative_count': negative_diagonal,
                'zero_count': zero_diagonal,
                'threshold': threshold,
                'message': f'Diagonal: {negative_diagonal} negative, {zero_diagonal} zero elements (threshold: {threshold:.2e})',
            }
        except Exception as e:
            self.results['diagonal'] = {
                'valid': False,
                'message': f'Diagonal check failed: {str(e)}',
            }

    def _check_condition_number(self, matrix):
        """Check condition number for numerical stability."""
        print('Checking condition number...')
        try:
            eigenvals = self._get_eigenvalues(matrix)
            if eigenvals is not None:
                # More efficient: condition number = max_eigenval / min_eigenval
                eigenvals_real = np.real(eigenvals)  # Handle complex eigenvalues
                max_eigenval = np.max(eigenvals_real)
                threshold = max(self.rtol * max_eigenval, self.atol)
                eigenvals_positive = eigenvals_real[eigenvals_real > threshold]
                if len(eigenvals_positive) == 0:
                    cond_num = np.inf
                else:
                    cond_num = np.max(eigenvals_positive) / np.min(eigenvals_positive)
            else:
                # Fallback to direct computation if eigenvalues unavailable
                cond_num = np.linalg.cond(matrix)
            
            well_conditioned = cond_num < 1e12  # Common threshold

            self.results['condition_number'] = {
                'value': cond_num,
                'well_conditioned': well_conditioned,
                'message': f'Condition number: {cond_num:.2e}, well-conditioned: {well_conditioned}',
            }
        except Exception as e:
            self.results['condition_number'] = {
                'valid': False,
                'message': f'Condition number check failed: {str(e)}',
            }

    def _check_invertible(self, matrix):
        """Check if matrix is invertible."""
        print('Checking if matrix is invertible...')
        try:
            inv_matrix = self._get_inverse(matrix)
            if inv_matrix is None:
                self.results['invertible'] = {
                    'valid': False,
                    'message': 'Not invertible: Matrix is singular'
                }
                return
                
            # Verify inversion by checking if A * A^-1 ≈ I
            identity_check = np.allclose(
                matrix @ inv_matrix, np.eye(matrix.shape[0]), rtol=self.rtol, atol=self.atol
            )

            self.results['invertible'] = {
                'valid': True,
                'identity_check': identity_check,
                'message': f'Invertible: True, identity check: {identity_check}',
            }
        except Exception as e:
            self.results['invertible'] = {
                'valid': False,
                'message': f'Inversion check failed: {str(e)}',
            }

    def _check_cholesky_decomposition(self, matrix):
        """Check if Cholesky decomposition is possible."""
        print('Checking Cholesky decomposition...')
        try:
            L = self._get_cholesky(matrix)
            if L is None:
                self.results['cholesky'] = {
                    'valid': False,
                    'message': 'Cholesky decomposition failed: Matrix not positive definite'
                }
                return
                
            # Verify decomposition: L * L^T = A
            reconstruction = L @ L.T
            reconstruction_valid = np.allclose(matrix, reconstruction, rtol=self.rtol, atol=self.atol)
            reconstruction_error = np.max(np.abs(matrix - reconstruction))

            self.results['cholesky'] = {
                'valid': reconstruction_valid,
                'reconstruction_error': reconstruction_error,
                'message': f'Cholesky decomposition: {reconstruction_valid}, reconstruction error: {reconstruction_error:.2e}',
            }
        except Exception as e:
            self.results['cholesky'] = {
                'valid': False,
                'message': f'Cholesky check failed: {str(e)}',
            }

    def _check_finite_values(self, matrix):
        """Check for NaN or infinite values."""
        print('Checking for NaN or infinite values...')
        try:
            has_nan = np.any(np.isnan(matrix))
            has_inf = np.any(np.isinf(matrix))

            self.results['finite_values'] = {
                'valid': not (has_nan or has_inf),
                'has_nan': has_nan,
                'has_inf': has_inf,
                'message': f'Finite values: {not (has_nan or has_inf)}, NaN: {has_nan}, Inf: {has_inf}',
            }
        except Exception as e:
            self.results['finite_values'] = {
                'valid': False,
                'message': f'Finite values check failed: {str(e)}',
            }

    def _check_determinant(self, matrix):
        """Check determinant value."""
        print('Checking determinant...')
        try:
            eigenvals = self._get_eigenvalues(matrix)
            if eigenvals is not None:
                # More numerically stable: det = product of eigenvalues
                det = np.prod(eigenvals)
            else:
                # Fallback to direct computation
                det = np.linalg.det(matrix)

            self.results['determinant'] = {
                'value': det,
                'positive': det > 0,
                'message': f'Determinant: {det:.2e}, positive: {det > 0}',
            }
        except Exception as e:
            self.results['determinant'] = {
                'valid': False,
                'message': f'Determinant check failed: {str(e)}',
            }

    def print_results(self):
        """Print formatted results of all checks."""
        print('=' * 60)
        print('COVARIANCE MATRIX SANITY CHECK RESULTS')
        print('=' * 60)

        for check_name, result in self.results.items():
            status = '✓ PASS' if result.get('valid', True) else '✗ FAIL'
            print(f'{check_name.upper():20} | {status:8} | {result["message"]}')

        print('=' * 60)

        # Summary
        failed_checks = [
            name
            for name, result in self.results.items()
            if not result.get('valid', True)
        ]

        if failed_checks:
            print(
                f'SUMMARY: {len(failed_checks)} checks failed: {", ".join(failed_checks)}'
            )
        else:
            print(
                'SUMMARY: All checks passed! Matrix appears to be a valid covariance matrix.'
            )

        print('=' * 60)

    def is_valid_covariance_matrix(self):
        """
        Return whether the matrix passes all essential covariance matrix tests.

        Returns:
            bool: True if matrix is a valid covariance matrix
        """
        essential_checks = [
            'shape',
            'square',
            'symmetry',
            'positive_definite',
            'finite_values',
        ]

        for check in essential_checks:
            if check in self.results and not self.results[check].get('valid', True):
                return False

        return True

