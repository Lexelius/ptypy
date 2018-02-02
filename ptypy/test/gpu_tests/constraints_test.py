'''
The tests for the constraints
'''


import unittest
import numpy as np
import utils as tu
from copy import deepcopy as copy
from ptypy.gpu import data_utils as du
from collections import OrderedDict
from ptypy.engines.utils import basic_fourier_update
from ptypy.gpu.constraints import difference_map_fourier_constraint

class ConstraintsTest(unittest.TestCase):

    def test_difference_map_fourier_constraint(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        diffraction = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']
        view_names = PtychoInstance.diff.views.keys()
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        difference_map_fourier_constraint(mask,
                                          diffraction,
                                          obj,
                                          probe,
                                          exit_wave,
                                          addr,
                                          prefilter=propagator.pre_fft,
                                          postfilter=propagator.post_fft,
                                          pbound=None,
                                          alpha=1.0,
                                          LL_error=True)


    def test_difference_map_fourier_constraint_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        PodPtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')

        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        diffraction = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']
        view_names = PtychoInstance.diff.views.keys()

        error_dict = OrderedDict.fromkeys(view_names)
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        ptypy_ewf, ptypy_error= self.ptypy_difference_map_fourier_constraint(PodPtychoInstance)
        exit_wave, errors = difference_map_fourier_constraint(mask,
                                                              diffraction,
                                                              obj,
                                                              probe,
                                                              exit_wave,
                                                              addr,
                                                              prefilter=propagator.pre_fft,
                                                              postfilter=propagator.post_fft,
                                                              pbound=None,
                                                              alpha=1.0,
                                                              LL_error=True)

        for idx, key in enumerate(ptypy_ewf.keys()):
            print key, idx
            np.testing.assert_allclose(ptypy_ewf[key], exit_wave[idx])

        ptypy_fmag = []
        ptypy_phot = []
        ptypy_exit = []

        for idx, key in enumerate(ptypy_error.keys()):
            err_fmag, err_phot, err_exit = ptypy_error[key]
            ptypy_fmag.append(err_fmag)
            ptypy_phot.append(err_phot)
            ptypy_exit.append(err_exit)

        ptypy_fmag = np.array(ptypy_fmag)
        ptypy_phot = np.array(ptypy_phot)
        ptypy_exit = np.array(ptypy_exit)

        npy_fmag = errors[0, :]
        npy_phot = errors[1, :]
        npy_exit = errors[2, :]
        import pylab as plt
        x = np.arange(92)
        plt.figure('fmag')
        plt.plot(x, npy_fmag, x, ptypy_fmag)
        plt.legend(['npy', 'ptypy'])
        plt.show()
        np.testing.assert_array_equal(npy_fmag, ptypy_fmag)

        np.testing.assert_array_equal(npy_phot, ptypy_phot)
        np.testing.assert_array_equal(npy_exit, ptypy_exit)









    def ptypy_difference_map_fourier_constraint(self, a_ptycho_instance):
        error_dct = OrderedDict()
        exit_wave = OrderedDict()
        for dname, diff_view in a_ptycho_instance.diff.views.iteritems():
            di_view = a_ptycho_instance.diff.V[dname]
            error_dct[dname] = basic_fourier_update(di_view,
                                                   pbound=None,
                                                   alpha=1.0)
            for name, pod in di_view.pods.iteritems():
                exit_wave[name] = pod.exit



        return exit_wave, error_dct


if __name__ == '__main__':
    unittest.main()


