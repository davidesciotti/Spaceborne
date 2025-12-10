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
                    f"got {type(value).__name__}"
                )
            
            if isinstance(value, np.ndarray) and isinstance(key, str):
                # Extract expected dimension from key (e.g., '4d' -> 4)
                if key.endswith('d') and key[:-1].isdigit():
                    expected_dim = int(key[:-1])
                    actual_dim = value.ndim
                    if actual_dim != expected_dim:
                        raise ValueError(
                            f"Key '{key}' expects numpy array with {expected_dim} dimensions, "
                            f"got array with {actual_dim} dimensions"
                        )

        super().__setitem__(key, value)


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
            dims=['2d', '4d', '6d']
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
            cov_dict[term][probe_pair] = {dim: None for dim in dims}
            # Freeze the dimension level (validate_dims=True: enforces type and dimension checks)
            cov_dict[term][probe_pair] = FrozenDict(
                cov_dict[term][probe_pair], protect_structure=False, validate_dims=True
            )
        # Freeze the probe level (protect_structure=True: prevents replacing FrozenDicts)
        cov_dict[term] = FrozenDict(cov_dict[term], protect_structure=True)

    # Freeze the top level (protect_structure=True: prevents replacing FrozenDicts)
    return FrozenDict(cov_dict, protect_structure=True)
