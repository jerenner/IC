"""
code: detsim.py
Simulation of sensor responses starting from Nexus output.

An HDF5 file containing Nexus output is given as input, and the simulated
detector response resulting from the Geant4 ionization tracks stored in this
file is produced.
"""

import numpy as np
import h5py

from argparse import Namespace

from .. cities.base_cities           import City
from .. io.mchits_io                 import load_mchits_nexus

from .. io.pmap_io                   import pmap_writer

from .. reco.paolina_functions       import voxelize_hits
from .. detsim.detsim_functions      import diffuse_and_smear_hits
from .. detsim.detsim_functions      import true_voxels_writer
from .. detsim.detsim_functions      import simulate_sensors
from .. detsim.detsim_functions      import sipm_lcone
from .. detsim.detsim_functions      import pmt_lcone

class Detsim(City):
    """Simulates detector response for events produced by Nexus"""

    parameters = tuple("""zmin
      zmax zmax diff_transv diff_long resolution_FWHM
      Qbb write_true_voxels true_voxel_dimensions A_sipm d_sipm
      ze_sipm ze_pmt slice_width_sipm E_to_Q_sipm uniformlight_frac_sipm
      s2_threshold_sipm slice_width_pmt E_to_Q_pmt uniformlight_frac_pmt
      s2_threshold_pmt peak_space""".split())

    def __init__(self, **kwds):
        """actions:
        1. inits base city

        """
        super().__init__(**kwds)

        self.cnt.init(n_events_tot                 = 0)

        self.light_function_sipm = sipm_lcone(self.conf.A_sipm,
                                              self.conf.d_sipm,
                                              self.conf.ze_sipm)
        self.light_function_pmt  = pmt_lcone (self.conf.ze_pmt)

    def file_loop(self):
        """
        The file loop of TDetSim:
        1. read the input Nexus files
        2. pass the hits to the event loop

        """
        for filename in self.input_files:
            mchits_dict = load_mchits_nexus(filename, self.conf.event_range)
            self.event_loop(mchits_dict)

    def event_loop(self, mchits_dict):
        """
        The event loop of TDetSim:
        1. diffuse and apply energy smearing to all hits in each event
        2. create true voxels from the diffused/smeared hits

        """
        write = self.writers

        for evt_number, mchits in mchits_dict.items():

            dmchits = diffuse_and_smear_hits(mchits, self.conf.zmin,
                                                 self.conf.zmax,
                                                 self.conf.diff_transv,
                                                 self.conf.diff_long,
                                                 self.conf.resolution_FWHM,
                                                 self.conf.Qbb)

            voxels = voxelize_hits(dmchits, self.conf.true_voxel_dimensions)

            if(self.conf.write_true_voxels):
                write.true_voxels(evt_number,voxels)

            pmap = simulate_sensors(voxels,
                            self.DataSiPM, self.conf.slice_width_sipm,
                            self.light_function_sipm, self.conf.E_to_Q_sipm,
                            self.conf.uniformlight_frac_sipm,
                            self.conf.s2_threshold_sipm, self.DataPMT,
                            self.conf.slice_width_pmt, self.light_function_pmt,
                            self.conf.E_to_Q_pmt, self.conf.uniformlight_frac_pmt,
                            self.conf.s2_threshold_pmt, self.conf.peak_space)

            write.pmap(evt_number, *pmap)

        self.cnt.n_events_tot += len(mchits_dict)


    def get_writers(self, h5out):
        writers = Namespace(
        true_voxels = true_voxels_writer(h5out),
        pmap        =        pmap_writer(h5out)
        )
        return writers

    def write_parameters(self, h5out):
        pass
