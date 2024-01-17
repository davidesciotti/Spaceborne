import numpy as np
import sys

sys.path.append('/')
import my_module as mm

"""
Just a small script to test the functions created to generate the ind file - at last!
You can delete this file if needed, tests have passed
"""

# test generate_ind
for triu_tril_square in ['triu', 'tril', 'full_square']:
    for row_col_major in ['row_major', 'col_major']:
        for zbins in range(1, 25):

            rowcol = row_col_major.rstrip('_major')
            zpairs_auto = zbins * (zbins + 1) // 2

            zpairs_cross = zbins ** 2
            if triu_tril_square == 'full_square':
                low = zpairs_auto
                high = zpairs_auto + zpairs_cross
                triu_tril = 'triu'  # ! = con tril?
            else:
                low = 0
                high = zpairs_auto
                triu_tril = triu_tril_square
            ind = np.genfromtxt(f'/home/cosmo/davide.sciotti/data/common_data/ind_files'
                                f'/{triu_tril}_{rowcol}-wise/indices_{triu_tril}_{rowcol}'
                                f'-wise_zbins{zbins:02}.dat', dtype=int)[low:high, 2:]
            ind_new = mm.generate_ind(triu_tril_square, row_col_major, zbins)

            assert np.array_equal(ind, ind_new)

for triu_tril in ['triu', 'tril']:
    for row_col_major in ['row_major', 'col_major']:
        for zbins in range(1, 25):
            rowcol = row_col_major.rstrip('_major')
            ind = np.genfromtxt(f'/home/cosmo/davide.sciotti/data/common_data/ind_files'
                                f'/{triu_tril}_{rowcol}-wise/indices_{triu_tril}_{rowcol}'
                                f'-wise_zbins{zbins:02}.dat', dtype=int)
            ind_new = mm.build_full_ind(triu_tril, row_col_major, zbins)

            assert np.array_equal(ind, ind_new)

print('aaaal good')
