"""tdetsim
Simulation of sensor responses starting from Nexus output.

An HDF5 file containing Nexus output is given as input, and the simulated
detector response resulting from the Geant4 ionization tracks stored in this
file is produced.
"""

import numpy as np
import tconfig as cfg

# read nexus information into a collection of MC hits
mchits = read_nexus(cfg.nexus_file, cfg.Nevts, cfg.Nstart)

# create voxels from the MC hits
voxels = create_voxels(mchits,cfg.vox_size,
                       cgf.detector_range, cfg.center_voxels)

# apply diffusion and smearing
dvoxels = diffuse_and_smear_voxels(voxels, cfg.diff_transv,
                                   cfg.diff_long, cfg.resolution_FWHM)

# write voxels to HDF5 if the option is set
if(cfg.write_voxels):
    write_voxels(dvoxels)

# cast light on SiPM plane
sipm_plane = simulate_tracking_plane(dvoxels, light_func, cfg.zbins_voxels)

# save the SiPM waveforms
write_waveforms(sipm_plane)