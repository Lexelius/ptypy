'''
tests for the object-probe interactions, including the specific DM, ePIE etc updates

'''

import unittest
import numpy as np
import utils as tu
from ptypy.gpu import data_utils as du
from ptypy.gpu import object_probe_interaction as opi


class ObjectProbeInteractionTest(unittest.TestCase):
    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        self.serialized_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')
        self.addr = self.serialized_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        self.probe = self.serialized_scan['probe']
        self.obj = self.serialized_scan['obj']
        self.exit_wave = self.serialized_scan['exit wave']

    def test_scan_and_multiply(self):
        blank = np.ones_like(self.probe)
        addr_info = self.addr[:, 0]

        po = opi.scan_and_multiply(blank, self.obj, self.exit_wave.shape, addr_info)
        for idx, p in enumerate(self.PtychoInstance.pods.itervalues()):
            np.testing.assert_array_equal(po[idx], p.object)

    def test_exit_wave_calculation(self):
        addr_info = self.addr[:, 0]

        po = opi.scan_and_multiply(self.probe, self.obj, self.exit_wave.shape, addr_info)
        for idx, p in enumerate(self.PtychoInstance.pods.itervalues()):
            np.testing.assert_array_equal(po[idx], p.object * p.probe)




    def test_get_exit_wave(self):
        opi.get_exit_wave(self.obj, self.probe, self.exit_wave, self.addr)
    
    def test_difference_map_realspace_constraint(self):
        opi.difference_map_realspace_constraint(self.obj,
                                                self.probe,
                                                self.exit_wave,
                                                self.addr,
                                                alpha=1.0)

if __name__ == "__main__":
    unittest.main()