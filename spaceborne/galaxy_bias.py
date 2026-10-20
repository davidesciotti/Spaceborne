"""Redshift-dependent bias, with one column per tomographic bin.

Sketch of the use in main.py:

    gal_bias = galaxy_bias_from_cfg(cfg, z_grid, zbins)

    gal_bias(z_grid)  # (len(z_grid), zbins), for the tracers
    gal_bias(z_grid_trisp_ssc)  # same object, another grid
    gal_bias.is_bin_independent  # replaces single_b_of_z
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from spaceborne import wf_cl_lib


# ! ============================== 1. the data + operations ===========================
@dataclass(frozen=True)
class RedshiftDependentBias:
    """Bias sampled on its own redshift nodes; evaluate it on any grid with bias(z).

    Only the data that define the bias are stored. Every other form (the CCL tuple,
    the OneCovariance table, ...) is built by its consumer from bias(z).
    """

    z: np.ndarray  # input z values: shape (nz,)
    values: np.ndarray  # bias at the z values: shape (nz, zbins)
    interp_kind: str = 'CubicSpline'  # 'CubicSpline', 'linear' or 'nearest'

    def __post_init__(self):
        # runs right after the generated __init__. Only checks here, no assignments
        if self.values.ndim != 2 or self.values.shape[0] != self.z.size:
            raise ValueError(
                f'values must have shape (len(z), zbins) = ({self.z.size}, zbins), '
                f'got {self.values.shape}'
            )
        if self.interp_kind not in ('CubicSpline', 'linear', 'nearest'):
            raise ValueError(f'Unknown interpolation kind: {self.interp_kind!r}')

    # properties: derived on request, never stored
    @property
    def zbins(self) -> int:
        return self.values.shape[1]

    @property
    def is_bin_independent(self) -> bool:
        """True if all bins share the same b(z)."""
        return np.allclose(self.values, self.values[:, [0]])

    def __call__(self, z_out: np.ndarray) -> np.ndarray:
        """Bias on z_out, always with shape (len(z_out), zbins).

        Same interpolation as sl.check_interpolate_input_tab.
        """
        if self.interp_kind == 'CubicSpline':
            interp_func = CubicSpline(x=self.z, y=self.values, axis=0)
        else:
            interp_func = interp1d(
                self.z,
                self.values,
                axis=0,
                kind=self.interp_kind,
                bounds_error=False,
                fill_value='extrapolate',
            )
        return interp_func(z_out)


# ! ============================== 2. the factory =====================================
# The only place that knows the config keys. RedshiftDependentBias itself never sees cfg.
def galaxy_bias_from_cfg(
    cfg: dict, z_grid: np.ndarray, zbins: int
) -> RedshiftDependentBias:
    cl_cfg = cfg['C_ell']

    if cl_cfg['which_gal_bias'] == 'from_input':
        # keep the table's own nodes: bias(z) then interpolates the input table, as
        # gal_bias_func does today
        table = np.genfromtxt(cl_cfg['gal_bias_table_filename'])
        z_nodes, values = table[:, 0], table[:, 1:]
        interp_kind = cl_cfg['gal_bias_table_interp_method']

    elif cl_cfg['which_gal_bias'] == 'polynomial_fit':
        # sample the fit on z_grid. A cubic spline through a cubic polynomial is the
        # polynomial itself (to ~1e-11), so bias(z) matches the fit on any grid
        b_1d = wf_cl_lib.b_of_z_fs2_fit(
            z_grid,
            magcut_lens=None,
            poly_fit_values=np.array(cl_cfg['gal_bias_fit_coeff']),
        )
        z_nodes = z_grid
        values = np.repeat(b_1d[:, None], zbins, axis=1)  # same b(z) in every bin
        interp_kind = 'CubicSpline'

    else:
        raise ValueError('which_gal_bias should be "from_input" or "polynomial_fit"')

    if values.shape[1] != zbins:
        raise ValueError(
            f'The galaxy bias has {values.shape[1]} bins, but zbins = {zbins}'
        )

    return RedshiftDependentBias(z=z_nodes, values=values, interp_kind=interp_kind)


# ! ============================== 3. toy demo (delete me) ============================
# run from the repo root with: python -m spaceborne.galaxy_bias
if __name__ == '__main__':
    # build directly from arrays: no config needed (this is what makes it testable)
    z_nodes = np.linspace(0.0, 3.0, 50)
    values = np.column_stack([1 + z_nodes, 1 + z_nodes, 1 + z_nodes])  # 3 bins
    bias = RedshiftDependentBias(z=z_nodes, values=values)

    # each consumer passes its own grid; the shape is always (len(z), zbins)
    z_grid_trisp = np.linspace(0.1, 2.5, 7)
    print(bias(z_nodes).shape)  # (50, 3)
    print(bias(z_grid_trisp).shape)  # (7, 3)
    print(bias.zbins, bias.is_bin_independent)  # 3 True

    # the "theme variations": each built in one line, where it's needed
    ccl_bias_tuple_bin0 = (z_nodes, bias(z_nodes)[:, 0])  # set_kernel_obj, bin 0
    trisp_bias_1d = bias(z_grid_trisp)[:, 0]  # build_trisp_dict
    oc_table = np.column_stack([z_nodes, bias(z_nodes)])  # OneCovariance ascii
    print(oc_table.shape)  # (50, 4)

    # frozen: this raises dataclasses.FrozenInstanceError
    try:
        bias.values = np.zeros((50, 3))
    except Exception as err:
        print(type(err).__name__)
