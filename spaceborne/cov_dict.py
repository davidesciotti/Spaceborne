import numpy as np


def print_cov_dict_info(cov_dict: dict, show_array_info: bool = True):
    """
    Print the structure of a nested covariance dictionary in a tree format.

    Structure: cov_dict[term][probe_ab, probe_cd][dim] = np.ndarray

    Parameters
    ----------
    cov_dict : dict
        Nested covariance dictionary
    show_array_info : bool
        If True, shows array shapes and dtypes
    """

    print('cov_dict info:')
    print('=' * 60)

    # Iterate through terms (level 1)
    for term_idx, (term, cov_probe_dict) in enumerate(cov_dict.items(), 1):
        is_last_term = term_idx == len(cov_dict)
        term_prefix = '└──' if is_last_term else '├──'

        print(
            f"{term_prefix} Term: '{term}' ({len(cov_probe_dict)} probe combinations)"
        )

        # Iterate through probe tuples (level 2)
        for probe_idx, (probe_tpl, cov_dim_dict) in enumerate(
            cov_probe_dict.items(), 1
        ):
            is_last_probe = probe_idx == len(cov_probe_dict)

            # Formatting for tree structure
            if is_last_term:
                probe_prefix = '    └──'
                dim_parent_prefix = '        '
            else:
                probe_prefix = '│   └──' if is_last_probe else '│   ├──'
                dim_parent_prefix = '│       ' if is_last_probe else '│   │   '

            if isinstance(probe_tpl, tuple) and len(probe_tpl) == 2:
                probe_ab, probe_cd = probe_tpl
                probe_str = f'({probe_ab}, {probe_cd})'
            else:
                probe_str = str(probe_tpl)
            print(
                f'{probe_prefix} Probe combination: '
                f'{probe_str} - {len(cov_dim_dict)} dimensions'
            )

            # Iterate through dimensions (level 3)
            for dim_idx, (dim, array) in enumerate(cov_dim_dict.items(), 1):
                is_last_dim = dim_idx == len(cov_dim_dict)
                dim_prefix = '└──' if is_last_dim else '├──'

                if show_array_info and array is not None:
                    array_info = f'shape={array.shape}, dtype={array.dtype}'
                    print(
                        f"{dim_parent_prefix}{dim_prefix} Dim: '{dim}' ({array_info})"
                    )
                elif show_array_info and array is None:
                    print(f"{dim_parent_prefix}{dim_prefix} Dim: '{dim}' (None)")
                elif not show_array_info:
                    print(f"{dim_parent_prefix}{dim_prefix} Dim: '{dim}'")

    print('=' * 60)


def compare_cov_dicts(
    cov_dict1: dict, cov_dict2: dict, rtol: float = 1e-7, atol: float = 0
) -> bool:
    """
    Compare two nested covariance dictionaries with structure:
    cov_dict[term][probe_ab, probe_cd][dim] = np.ndarray

    Parameters
    ----------
    cov_dict1, cov_dict2 : dict
        Nested dictionaries to compare
    rtol : float
        Relative tolerance for numpy array comparison
    atol : float
        Absolute tolerance for numpy array comparison

    Returns
    -------
    bool
        True if dictionaries are identical, False otherwise

    Raises
    ------
    AssertionError
        If dictionaries differ, with detailed error message
    """
    # Check if both dicts have same terms (outermost keys)
    terms1 = set(cov_dict1.keys())
    terms2 = set(cov_dict2.keys())

    if terms1 != terms2:
        missing_in_2 = terms1 - terms2
        missing_in_1 = terms2 - terms1
        raise AssertionError(
            f'Terms mismatch. Missing in dict2: {missing_in_2}. '
            f'Missing in dict1: {missing_in_1}'
        )

    # Iterate through each term
    for term in terms1:
        cov_probe_dict1 = cov_dict1[term]
        cov_probe_dict2 = cov_dict2[term]

        # Check if both cov_probe_dicts have same probe tuples
        probes1 = set(cov_probe_dict1.keys())
        probes2 = set(cov_probe_dict2.keys())

        if probes1 != probes2:
            missing_in_2 = probes1 - probes2
            missing_in_1 = probes2 - probes1
            raise AssertionError(
                f"Probe tuples mismatch in term '{term}'. "
                f'Missing in dict2: {missing_in_2}. '
                f'Missing in dict1: {missing_in_1}'
            )

        # Iterate through each probe tuple
        for probe_tpl in probes1:
            cov_dim_dict1 = cov_probe_dict1[probe_tpl]
            cov_dim_dict2 = cov_probe_dict2[probe_tpl]

            # Check if both cov_dim_dicts have same dimensions
            dims1 = set(cov_dim_dict1.keys())
            dims2 = set(cov_dim_dict2.keys())

            if dims1 != dims2:
                missing_in_2 = dims1 - dims2
                missing_in_1 = dims2 - dims1
                raise AssertionError(
                    f"Dimension keys mismatch in term '{term}', probe {probe_tpl}. "
                    f'Missing in dict2: {missing_in_2}. '
                    f'Missing in dict1: {missing_in_1}'
                )

            # Compare arrays for each dimension
            for dim in dims1:
                arr1 = cov_dim_dict1[dim]
                arr2 = cov_dim_dict2[dim]

                # Check array shapes match
                if arr1.shape != arr2.shape:
                    raise AssertionError(
                        f"Shape mismatch in term '{term}', "
                        f"probe {probe_tpl}, dim '{dim}'. "
                        f'Shape1: {arr1.shape}, Shape2: {arr2.shape}'
                    )

                # Check array values are close
                if not np.allclose(arr1, arr2, rtol=rtol, atol=atol):
                    max_diff = np.max(np.abs(arr1 - arr2))
                    raise AssertionError(
                        f"Array values differ in term '{term}', "
                        f"probe {probe_tpl}, dim '{dim}'. "
                        f'Max absolute difference: {max_diff}'
                    )

    return True


class FrozenDict(dict):
    """Dictionary that prevents adding new keys after initialization."""

    def __init__(self, *args, protect_structure=True, validate_dims=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen = True
        self._protect_structure = protect_structure
        self._validate_dims = validate_dims

    def __setitem__(self, key, value):
        # Prevent adding new keys
        if hasattr(self, '_frozen') and key not in self:
            raise KeyError(
                f"Cannot add new key '{key}'. Allowed keys: {sorted(self.keys())}"
            )

        # Prevent overwriting nested FrozenDict structures
        if hasattr(self, '_protect_structure') and self._protect_structure:
            existing = super().__getitem__(key) if key in self else None
            if isinstance(existing, FrozenDict) and not isinstance(value, FrozenDict):
                raise TypeError(
                    f"Cannot replace nested dictionary structure at key '{key}'. "
                    f'You can only modify values at the leaf level (arrays), '
                    f'not replace intermediate dict levels.'
                )

        # Validate dimension constraints
        if hasattr(self, '_validate_dims') and self._validate_dims:
            if value is not None and not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Key '{key}' can only store None or numpy arrays, "
                    f'got {type(value).__name__}'
                )

            if isinstance(value, np.ndarray) and isinstance(key, str):
                # Extract expected dimension from key (e.g., '4d' -> 4)
                if key.endswith('d') and key[:-1].isdigit():
                    expected_dim = int(key[:-1])
                    actual_dim = value.ndim
                    if actual_dim != expected_dim:
                        raise ValueError(
                            f"Key '{key}' expects numpy array with {expected_dim} dimensions, "
                            f'got array with {actual_dim} dimensions'
                        )

        super().__setitem__(key, value)

    def pop(self, *args):
        raise TypeError('Cannot remove keys from FrozenDict')

    def popitem(self):
        raise TypeError('Cannot remove keys from FrozenDict')

    def clear(self):
        raise TypeError('Cannot clear FrozenDict')


def create_cov_dict(
    required_terms: list[str], probe_pairs: list[tuple[str, str]], dims: list[str]
) -> FrozenDict:
    """
    Create fully structured covariance dict with all levels pre-initialized.

    Parameters
    ----------
    required_terms : list[str]
        Terms like ['g', 'ssc', 'tot']
    probe_pairs : list[tuple]
        All probe combinations, e.g., [('LL','LL'), ('GL','GL'), ('GG','GG')]
    dims : list[str]
        Dimensions to pre-create, e.g. ['2d', '4d', '6d']

    Returns
    -------
    FrozenDict
        Three-level frozen structure. No new keys at ANY level.
        Cannot overwrite intermediate dict structures.

    Examples
    --------
    >>> cov_dict = create_cov_dict(
    ...     required_terms=['g', 'ssc'],
    ...     probe_pairs=[('LL','LL'), ('GG','GG')],
    ...     dims=['2d', '4d', '6d']
    ... )
    >>> # These work:
    >>> cov_dict['g'][('LL','LL')]['6d'] = array  # ✓ Setting array value
    >>>
    >>> # These raise errors:
    >>> cov_dict['cng'] = {}  # ✗ KeyError: 'cng' not in required_terms
    >>> cov_dict['g'][('XX','XX')] = {}  # ✗ KeyError: probe not pre-defined
    >>> cov_dict['g'][('LL','LL')]['7d'] = array  # ✗ KeyError: '7d' not in dims
    >>> cov_dict['g'][('LL','LL')] = {}  # ✗ TypeError: cannot replace dict structure
    """
    cov_dict = {}

    for term in required_terms:
        cov_dict[term] = {}
        for probe_pair in probe_pairs:
            # Initialize with None for each dim (will be replaced by arrays)
            cov_dict[term][probe_pair] = dict.fromkeys(dims)
            # Freeze the dimension level (validate_dims=True: enforces type and dimension checks)
            cov_dict[term][probe_pair] = FrozenDict(
                cov_dict[term][probe_pair], protect_structure=False, validate_dims=True
            )
        # Freeze the probe level (protect_structure=True: prevents replacing FrozenDicts)
        cov_dict[term] = FrozenDict(cov_dict[term], protect_structure=True)

    # Freeze the top level (protect_structure=True: prevents replacing FrozenDicts)
    return FrozenDict(cov_dict, protect_structure=True)
