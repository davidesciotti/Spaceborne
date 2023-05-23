# Covariance and data/theory vectors ordering conventions
Davide Sciotti

In the following, we present the conventions used in the ordering of the data/theory vectors and analytical covariance matrices.

## Angular Power Spectra

The angular Power Spectra (PS) have the form $`C^{AB}_{ij}(\ell)`$, with:

* $`A, B`$ the probe index: for the photometric survey, $`A = L`$ (WL) or $`A = G`$ (GCph). We will then have:
  * $`C^{LL}_{ij}(\ell)`$ Lensing-Lensing. This is symmetric in $`(i,j)`$: $`C^{LL}_{ij}(\ell) = C^{LL}_{ji}(\ell)`$
  * $`C^{GL}_{ij}(\ell)`$ Galaxy-Galaxy lensing. This matrix is **not** symmetric under exchange of $`(i,j)`$, but we have that: $`C^{GL}_{ji}(\ell) = C^{LG}_{ij}(\ell)`$
  * $`C^{GG}_{ij}(\ell)`$ Galaxy-Galaxy. Again, $`C^{GG}_{ij}(\ell) = C^{GG}_{ji}(\ell)`$
* $`i, j`$ the photo-$`z`$ bin indices: ($`i = 1, 2, ..., N_{zbins}`$); in this note we will use $`N_{zbins}=10`$
* $`\ell`$ the multipole value

For each probe combination (i.e., one of $`LL, GL, GG`$) we can arrange the angular PS $`C^{AB}_{ij}(\ell)`$ in a 1-, 2- or 3- or 5-dimensional `numpy` array:

* 3D: $`C^{AB}_{ij}(\ell)`$ = `C_AB[ell, i, j]`, of shape `(ell_bins, z_bins, z_bins)`
* 2D: $`C^{AB}_{ij}(\ell)`$ = `C_AB[ell, ij]`, of shape `(ell_bins, z_pairs)`
* 1D: $`C^{AB}_{ij}(\ell)`$ = `C_AB[idx]`, of shape `(ell_bins*z_pairs)`

`z_pairs` being the number of unique redshift bin pairs. Being `C_LL` and `C_GG` symmetric under the exchange of redshift indices, we will have:

- $`LL, GG`$: `z_pairs_auto = (z_bins*(z_bins + 1))//2` (that is, $`\frac{N_{zbins}(N_{zbins}+1)}{2}`$), which is equal to 55 for `z_bins = 10`

while in the case of the cross-spectrum, $`GL`$ (or $`LG`$) we will have

- $`GL`$ (or $`LG`$): `z_pairs_cross = z_bins**2` (that is, $`N_{zbins}^2`$), which is equal to 100 for `z_bins = 10`.

Being built by concatenating `(C_LL, C_GL, C_GG)` (in this order for CLOE) the 3x2pt datavector will have instead

* 3x2pt: `z_pairs = z_pairs_auto + z_pairs_cross + z_pairs_auto`, which equals 210 for `z_bins= 10`.

The important point to keep in mind is that each time we change the dimensionality of the `C_AB` array (from 3D to 2D and from 2D to 1D) an ordering (or "reshaping/flattening", or "packing/unpacking") convention must be chosen. _The 3D representation of the $`C_{ij}(\ell)`$s is the most transparent, as each specific multipole and tomographic bin value can be accessed directly by simply calling the appropriate index of the appropriate axis_. The implementation and the possible ordering conventions can vary, but we present here the most relevant cases.\
We also note that, at least for the auto-spectra, the 3D -> 2D reshaping implies also a reduction in the number of total datavector entries (the symmetric elements are kept only once).

### 3D <-> 2D

Let us focus on the first dimensionality reduction: from `(ell_bins, z_bins, z_bins)` to `(ell_bins, z_pairs)`. The mapping `ij <-> [i, j]` can be described by an array of shape `(z_pairs, 2)`, which we call `ind` (for "indices"). The row index corresponds to `ij`, the redshift pair index, and the values in the first and second columns will correspond to `i` and `j`, respectively. In this way, the ordering of the photometric bins can easily be modified by passing a different `ind` array. For $`LL`$ and $`GG`$, we take `i from 0 to zbins` and `i from j to zbins` for each given `ell`: this corresponds to a row-major, upper triangular ordering. For $`GL/LG`$ we use `i from 0 to zbins` and `j from 0 to zbins`, which again is a row-major ordering:

| Row/column major unpacking ([source](https://en.wikipedia.org/wiki/Row-\_and_column-major_order)) | Upper triangular, row-wise unpacking |
|---------------------------------------------------------------------------------------------------|--------------------------------------|
| ![row-col-major](uploads/b796397071d9115b236c04f902180ade/row-col-major.png) | ![plot_ind_triu_row-wise](uploads/3accb849842b8a57a84000ae784c2c0a/plot_ind_triu_row-wise.png) |

```
import numpy as np
z_bins = 10  
z_pairs_auto = z_bins * (z_bins + 1) // 2 # = 55  
z_pairs_cross = z_bins ** 2 # = 100  
  
# create the ind array for the auto and cross spectra
ind_auto = np.zeros((z_pairs_auto, 2)) # for LL and GG  
ind_cross = np.zeros((z_pairs_cross, 2)) # for GL/LG  
  
ij = 0  
for i in range(z_bins):  
	for j in range(i, z_bins):  
		ind_auto[ij, :] = i, j  
		ij += 1  
  
ij = 0  
for i in range(z_bins):  
	for j in range(z_bins):  
		ind_cross[ij, :] = i, j  
		ij += 1  

ind_auto = ind_auto.astype('int')  
ind_cross = ind_cross.astype('int')
```

In our case, we will then have

```
WL and GCph 		 LG/GL
____________________________________
ij 	 i,j 		 ij 	 i,j
0 	 0 0 		 0 	 0 0
1 	 0 1 		 1 	 0 1
2 	 0 2 		 2 	 0 2
3 	 0 3 		 3 	 0 3
4 	 0 4 		 4 	 0 4
5 	 0 5 		 5 	 0 5
6 	 0 6 		 6 	 0 6
7 	 0 7 		 7 	 0 7
8 	 0 8 		 8 	 0 8
9 	 0 9 		 9 	 0 9
10 	 1 1 		 10 	 1 0
11 	 1 2 		 11 	 1 1
12 	 1 3 		 12 	 1 2
13 	 1 4 		 13 	 1 3
14 	 1 5 		 14 	 1 4
15 	 1 6 		 15 	 1 5
16 	 1 7 		 16 	 1 6
17 	 1 8 		 17 	 1 7
18 	 1 9 		 18 	 1 8
19 	 2 2 		 19 	 1 9
...				 ...
```

It is then easy to switch from 2-dimensional to 3-dimensional (and vice-versa) $`C^{AB}_{ij}(\ell)`$s by using the `ind` array; taking `LL` as an example:

```
ell_bins = 20  
# the cls are initialized to zero, this is just to illustrate the array reshaping
cl_LL_2D = np.zeros((ell_bins, z_pairs_auto))  
cl_GL_2D = np.zeros((ell_bins, z_pairs_cross))  
cl_LL_3D = np.zeros((ell_bins, z_bins, z_bins))  
cl_GL_3D = np.zeros((ell_bins, z_bins, z_bins))  
  
for ell in range(ell_bins):  
	for ij in range(z_pairs_auto):  
		i, j = ind_auto[ij, 0], ind_auto[ij, 1]  
		cl_LL_2D[ell, ij] = cl_LL_3D[ell, i, j]  
  
for ell in range(ell_bins):  
	for ij in range(z_pairs_cross):  
		i, j = ind_cross[ij, 0], ind_cross[ij, 1]  
		cl_GL_2D[ell, ij] = cl_GL_3D[ell, i, j]  
  
# or, in a more pythonic way:  
cl_LL_2D = cl_LL_3D[:, ind_auto[:, 0], ind_auto[:, 1]]  
cl_GL_2D = cl_GL_3D[:, ind_cross[:, 0], ind_cross[:, 1]]  
  
# or, using `numpy.ndarray.flatten`:  
for ell in range(ell_bins):  
	cl_GL_2D[ell, :] = cl_GL_3D[ell, :].flatten(order='C')
```

Where `order='C'` means row-wise, and `order='F'` means column-wise (see the `numpy.ndarray.flatten` [documentation](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html) for further details). Note that to go from 2D to 3D it will suffice to do (for the cross-spectra):

```
# 2D -> 3D  
for ell in range(ell_bins):  
	cl_GL_3D[ell, :, :] = cl_GL_2D[ell, :].reshape((z_bins, z_bins), order='C')
```

This is not the case for the auto-spectra, because the 2D array will only contain the upper or lower triangular elements of the 3D array: one possible way to go back to the 3D array is then

```
for ell in range(ell_bins):  
	for ij in range(z_pairs_auto):  
		i, j = ind_auto[ij, 0], ind_auto[ij, 1]  
		cl_LL_3D[ell, i, j] = cl_LL_2D[ell, ij]  
		cl_LL_3D[ell, j, i] = cl_LL_2D[ell, ij]
```

### 2D <-> 1D

The second dimensionality reduction is simpler because all the elements of the 2D arrays must be kept both for the auto and the cross spectra: using again the row-major unpacking, we have

```
cl_LL_1D = cl_LL_2D.flatten(order='C')  
cl_GL_1D = cl_GL_2D.flatten(order='C')
```

The opposite operation can conveniently be performed with

```
cl_LL_2D = cl_LL_1D.reshape((ell_bins, z_pairs_auto), order='C')  
cl_GL_2D = cl_GL_1D.reshape((ell_bins, z_pairs_cross), order='C')
```

## Covariance Matrix

The covariance matrix must follow the same ordering convention chosen for the data and theory vector, in order to have a consistent computation of the $`\chi^2`$ in the evaluation of the likelihood. The following section will then closely resemble the one above for the ordering of the angular PS.

### 10 (or 6) dimensional covariance

Just as the $`C_{ij}^{AB}(\ell)`$s can be organized _unambiguously_ (i.e., without the need to specify ordering conventions) in a 3-dimensional array of the type `C_AB[ell, i, j]` - once the probes A and B have been chosen -, the covariance matrix can be organized unambiguously in a 10-dimensional array:

`cov[A, B, C, D, ell1, ell2, i, j, k, l]`

with:

* `A, B, C, D` probe indices e.g. `0` or `'L'` for WL, `1` or `'G'` for GCph
* `ell1, ell2` multipole bin indices, from 1 to `ell_bins`
* `i, j, k, l` redshift bin indices, from 1 to `z_bins`

In practice, the first four can be fixed for the single-probe case, so that, for example, `cov_LL_LL` (the lensing-lensing (-lensing-lensing) covariance, with `A = B = C = D = 0` (or `'L'`) is a 6-dimensional array: `cov_LL_LL[ell1, ell2, i, j, k, l]`). Note that if we want `A, B, C, D` to be strings (`L` and `G` in our example) we can use a (numerically slower) dictionary `cov_dict['A', 'B', 'C', 'D'][ell1, ell2, i, j, k, l]`. The same goes for the cls: `cl_dict['A', 'B'][ell, i, j]` or `cl_array[A, B, ell, i, j]`. This is useful for the multi-probe (3x2pt) case.

This arrangement is quite intuitive, but as for the cls one may prefer a lower-dimensional representation:

* For WL and GCph (but not for $`GL/LG`$), there is a symmetry under the exchange of indices `i<->j` and `k<->l`, as seen above. This means that many elements of the array are repeated, taking up storage (and slowing the computation of the 6-dim covariance itself).
* This format is very easy to save and load, for example, in `npy` format, but trickier to save as a `csv`, `txt` or `dat` file.
* Moreover, it's impossible to visualize the whole matrix "at a glance", unlike if it had only two dimensions.
* Imposing scale cuts in the covariance can be more straightforward in 2D

**Note**: since the Gaussian covariance is diagonal in $`(\ell_1, \ell_2)`$, in this case, one could actually use a 5-dim array: `cov_AB_CD[ell, i, j, k, l]`

### 6D <-> 4D

A more compact representation of the covariance is as a 4-dimensional array of the form `cov_AB_CD[ell1, ell2, ij, kl]`, with `ij, kl` from 0 to `z_pairs` (which, as outlined above, is 55 for WL and GCph and 100 for $`LG/GL`$, in the case of 10 tomographic bins). **This ordering depends on the convention chosen to compress `i, j` into `ij` and `k, l` into `kl`, which is again specified by the `ind` array**. This is exactly the same indices compression seen above for the $`C_{ij}^{AB}(\ell)`$s.

```
# "z_pairs" can be z_pairs_auto or z_pairs_cross
cov_4D = np.zeros((ell_bins, ell_bins, z_pairs, z_pairs))  
for ij in range(z_pairs):  
	for kl in range(z_pairs):  
		# rename for better readability  
		i, j, k, l = ind[ij, 0], ind[ij, 1], ind[kl, 0], ind[kl, 1]  
		cov_4D[:, :, ij, kl] = cov_6D[:, :, i, j, k, l]
```

Once again, for the auto-spectra this operation reduces the total number of elements in the covariance matrix.

### 4D <-> 2D

To obtain the final (2-dimensional) of the covariance matrix, we flatten the `ell` and `z_pairs` dimensions: one possible algorithm to do that is

```
cov_2D = np.zeros((ell_bins * z_pairs_AB, ell_bins * z_pairs_CD))
for ell1 in range(ell_bins):  
	for ell2 in range(ell_bins):  
		for ij in range(z_pairs_AB):  
			for kl in range(z_pairs_CD):  
			# block_index * block_size + running_index  
			cov_2D[ell1 * z_pairs_AB + ij, ell2 * z_pairs_CD + kl] = cov_4D[ell1, ell2, ij, kl]
```

where `z_pairs_AB` can be `z_pairs_auto` or `z_pairs_cross`. Note that, when constructing the 3x2pt 2D covariance, some blocks will be non-square, for example for `cov_LL_GL`, `cov_GG_GL`. Again, this is just a different (more general) way of doing the 4D <-> 2D equivalent to the 2D <-> 1D $`C_{ij}(\ell)`$ transformation.

### $`3\times 2{\rm pt}`$

After performing this rearrangement, we can plot the 3x2pt covariance, to highlight three additional ordering conventions:

* whether to use $`LG`$ or $`GL`$
* which ordering to use for the probes: whether $`(LL, LG, GG)`$ or different combinations (e.g. $`(GG, LG, LL)`$)
* whether to unpack the probe index before or after the `ell`/`zpair` ones

| `ell` index unpacked _before_ probe index | Zoom on the first ell block |
|-------------------------------------------|-----------------------------|
| ![cov_2d_3x2pt_ell_probe_zpair](uploads/94130f0aa683e5121041f877204fe26b/cov_2d_3x2pt_ell_probe_zpair.png) | ![cov_2d_3x2pt_ell_probe_zpair_fist_ell_block](uploads/c6dd80413e0300c161b983b8937653ff/cov_2d_3x2pt_ell_probe_zpair_fist_ell_block.png) |

* Being the `ell` index unpacked before the probe and zpair ones (in this example), each block corresponds to a different $`(\ell_1, \ell_2)`$ combination. This is the reason why, with this ordering, the Gaussian covariance is block diagonal.
* Zooming on the first block (plot on the right), it's possible to see the sub-blocks corresponding to the different probes. These will of course change if we rearrange the ordering of the probes in the 3x2pt datavector (e.g., `(GG, GL, LL)` instead of `(LL, GL, GG)`) or choose to use $`LG`$ instead of $`GL`$ for the XC term. Each probe-block will have shape `(z_pairs_AB*ell_bins, z_pairs_CD*ell_bins)`.
* This particular ordering makes it easy to select a given set of `ell` values, but less obvious to select a given probe. A different possibility is to unpack the `probe` index before the `ell` and `z_pair` ones (see plot below): this ordering makes it easy to include additional probes by simply concatenating them to the data/theory vectors and to the covariance matrix.
* In this case, the $LL-LL$ and $GG-GG$ blocks of this matrix correspond respectively to the WL-only and GCph-only covariance matrices (unless different settings - e.g. a different $\ell_{\rm max}$ - are used for the 3x2pt).

![cov_2d_3x2pt_probe_ell_zpair](uploads/a0c075861f8e8ffab43c410bfd8706d6/cov_2d_3x2pt_probe_ell_zpair.png)

## Conclusion

When deciding an ordering convention, the most important thing is to keep consistency between the one chosen for the datavector and the covariance matrix. In particular, a choice must be made every time we change the dimensions of the vectors/covariance matrix.

* The flattening of the 2 tomographic redshift indices: `[i, j] -> ij`. Whether to use a row-wise or column-wise flattening. For the auto-spectra, which are symmetric under the exchange of `i` and `j`, whether we want to keep the upper or lower triangular elements. **CLOE uses an upper triangular, row-wise unpacking**
* The flattening of the multipole index `ell` and the redshift pair index `ij`. Again, whether we want the outermost for loop to run over the former index (which corresponds to a row-wise unpacking if `ell` is the first axis) or the latter. We call the corresponding orderings `ell_zpair` or `zpair_ell`, respectively. **CLOE uses the `ell_zpair` unpacking**
* For the 3x2pt, whether we want to unpack the probe index before or after the multipole and redshift pair indices. **CLOE adopts the `probe_ell_zpair` ordering**, which results in the second covariance matrix plotted above
* Whether we use `GL` or `LG` for the cross-spectra. **CLOE uses GL**
* The probe ordering in the 3x2pt datavector: (LL, XC, GG) or some other permutation. **CLOE uses (LL, GL, GG)**