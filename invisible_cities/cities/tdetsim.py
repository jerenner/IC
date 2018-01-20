"""
code: tdetsim.py
Simulation of sensor responses starting from Nexus output.

An HDF5 file containing Nexus output is given as input, and the simulated
detector response resulting from the Geant4 ionization tracks stored in this
file is produced.
"""

import numpy as np
import h5py

from .  base_cities            import City
from .. io.mchits_io           import load_mchits_nexus

from .. tdetsim.tdetsim_functions      import diffuse_and_smear_hits
from .. tdetsim.tdetsim_functions      import true_voxels_writer

class Tdetsim(City):
    """Simulates detector response for events produced by Nexus"""

    parameters = tuple("""zmin
      zmax zmax diff_transv diff_long resolution_FWHM
      Qbb write_true_voxels true_voxel_dimensions""".split())

    def __init__(self, **kwds):
        """actions:
        1. inits base city

        """
        super().__init__(**kwds)

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

        dmchits_dict= tdf.diffuse_and_smear_hits(mchits_dict, self.conf.zmin,
                                                 self.conf.zmax,
                                                 self.conf.diff_transv,
                                                 self.conf.diff_long,
                                                 self.conf.resolution_FWHM,
                                                 self.conf.Qbb)

        voxels_dict = tdf.create_voxels(dmchits_dict,
                                        self.conf.true_voxel_dimensions)

        if(self.conf.write_true_voxels):
            for evt_number,voxels in voxels_dict.items():
                write.true_voxels(evt_number,voxels)

        # cast light on SiPM plane
        #sipm_plane = simulate_tracking_plane(voxels, light_func, cfg.zbins_voxels)

        # save the SiPM waveforms
        #write_waveforms(sipm_plane)

    def get_writers(self, h5out):
        writers = Namespace(
        true_voxels =        true_voxels_writer(h5out)
        )
        return writers

    def write_parameters(self, h5out):
        pass
