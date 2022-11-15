def create_photo_theory(self, dictionary):
    """Create Photo Theory

    Obtains the photo theory for the likelihood.
    The theory is evaluated only for the probes specified in the masking
    vector. For the probes for which the theory is not evaluated, an array
    of zeros is included in the returned dictionary.

    Parameters
    ----------
    dictionary: dict
        cosmology dictionary from the Cosmology class which is updated at
        each sampling step

    Returns
    -------
    photo_theory_vec: array
        returns an array with the photo theory vector.
        The elements of the array corresponding to probes for which the
        theory is not evaluated, are set to zero.
    """

    # Photo class instance
    phot_ins = Photo(
        dictionary,
        self.data_ins.nz_dict_WL,
        self.data_ins.nz_dict_GC_Phot)

    # Obtain the theory for WL
    if self.data_handler_ins.use_wl:
        wl_array = np.array(
            [phot_ins.Cl_WL_noprefac(ell, element[0], element[1])
             for element in self.indices_diagonal_wl
             for ell in self.data_ins.data_dict['WL']['ells']]
        )
        wl_array = self.prefactor_WL * wl_array
    else:
        wl_array = np.zeros(
            len(self.data_ins.data_dict['WL']['ells']) *
            len(self.indices_diagonal_wl)
        )

    # Obtain the theory for XC-Phot
    if self.data_handler_ins.use_xc_phot:
        xc_phot_array = np.array(
            [phot_ins.Cl_cross_noprefac(ell, element[1], element[0])
             for element in self.indices_all
             for ell in self.data_ins.data_dict['XC-Phot']['ells']]
        )
        xc_phot_array = self.prefactor_XC * xc_phot_array
    else:
        xc_phot_array = np.zeros(
            len(self.data_ins.data_dict['XC-Phot']['ells']) *
            len(self.indices_all)
        )

    # Obtain the theory for GC-Phot
    if self.data_handler_ins.use_gc_phot:
        gc_phot_array = np.array(
            [phot_ins.Cl_GC_phot(ell, element[0], element[1])
             for element in self.indices_diagonal_gcphot
             for ell in self.data_ins.data_dict['GC-Phot']['ells']]
        )
    else:
        gc_phot_array = np.zeros(
            len(self.data_ins.data_dict['GC-Phot']['ells']) *
            len(self.indices_diagonal_gcphot)
        )

    photo_theory_vec = np.concatenate(
        (wl_array, xc_phot_array, gc_phot_array), axis=0)

    return photo_theory_vec


def create_spectro_theory(self, dictionary, dictionary_fiducial):
    """Create Spectro Theory

    Obtains the theory for the likelihood.
    The theory is evaluated only if the GC-Spectro probe is enabled in the
    masking vector.

    Parameters
    ----------
    dictionary: dict
        cosmology dictionary from the Cosmology class
        which is updated at each sampling step

    dictionary_fiducial: dict
        cosmology dictionary from the Cosmology class
        at the fiducial cosmology

    Returns
    -------
    theoryvec: list
        returns the theory array with same indexing/format as the data.
        If the GC-Spectro probe is not enabled in the masking vector,
        an array of zeros of the same size is returned.
    """
    # This is something that Sergio needs to change
    # Now the multipoles are within observables[specifications]
    # In order to pass the tests, I hard code m_inst now
    # Maybe Sergio has a better idea of what to these forloops
    # To include all info of the specifications
    if self.data_handler_ins.use_gc_spectro:
        spec_ins = Spectro(dictionary, dictionary_fiducial)
        # m_ins = [v for k, v in dictionary['nuisance_parameters'].items()
        #         if k.startswith('multipole_')]
        m_ins = [0, 2, 4]
        k_m_matrices = []
        for z_ins in self.zkeys:
            k_m_matrix = []
            for k_ins in (
                    self.data_ins.data_dict['GC-Spectro'][z_ins]['k_pk']):
                k_m_matrix.append(
                    spec_ins.multipole_spectra(
                        float(z_ins),
                        k_ins,
                        ms=m_ins)
                )
            k_m_matrices.append(k_m_matrix)
        theoryvec = np.hstack(k_m_matrices).T.flatten()
        return theoryvec

    else:
        theoryvec = np.zeros(self.data_handler_ins.gc_spectro_size)
        return theoryvec


def create_spectro_data(self):
    """Create Spectro Data

    Arranges the data vector for the likelihood into its final format

    Returns
    -------
    datavec: list
        returns the data as a single array across z, mu, k
    """

    datavec = []
    for z_ins in self.zkeys:
        multipoles = (
            [k for k in
             self.data_ins.data_dict['GC-Spectro'][z_ins].keys()
             if k.startswith('pk')])
        for m_ins in multipoles:
            datavec = np.append(datavec, self.data_ins.data_dict[
                'GC-Spectro'][z_ins][m_ins])
    return datavec