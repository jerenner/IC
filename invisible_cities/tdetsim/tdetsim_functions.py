"""tdetsim_functions
Defines key functions used in TDetSim.
"""

import numpy as np

from .. evm.event_model        import MCHit
from .. reco.paolina_functions import voxelize_hits

def diffuse_and_smear_hits(mchits_dict, zmin, zmax, diff_transv, diff_long,
                           resolution_FWHM, Qbb):
    """
    Applies diffusion and energy smearing to all MC hits.

    """

    dmchits_dict = {}
    for evt_number,mchits in mchits_dict.items():

        # calculate unscaled variance for energy smearing
        E_evt = sum([hit.E for hh in mchits])
        sigma0 = ((resolution_FWHM/100.) * sqrt(Qbb) * sqrt(E_evt)) / 2.355
        var0 = sigma0**2

        # calculate drift distance
        zdrift = np.random.uniform(zmin,zmax)

        # apply diffusion and energy smearing
        dmchits = []
        for hit in mchits:

            xh = np.random.normal(hit.X,np.sqrt(zdrift/10.)*transv_diff)
            yh = np.random.normal(hit.Y,np.sqrt(zdrift/10.)*transv_diff)
            zh = np.random.normal(hit.Z+zdrift,np.sqrt(zdrift/10.)*long_diff)
            eh = np.random.normal(hit.E,np.sqrt(var0*ee/E_evt))

            dmchits.append(MCHit([xh,yh,zh], hit.T, eh))

        dmchits_dict[evt_number] = dmchits

    return dmchits_dict


def create_voxels(mchits_dict, voxel_dimensions):
    """
    Produce voxels for each list of hits given in the specified dictionary.

    """
    mcvoxels_dict = {}
    for evt_number,mchits in mchits_dict.items():
        voxels = voxelize_hits(mchits, voxel_dimensions)
        mcvoxels_dict[evt_number] = voxels
    return mcvoxels_dict

# writers
def true_voxels_writer(hdf5_file, *, compression='ZLIB4'):
    hdf5_group = hdf5_file.create_group("Run")

    def write_voxels(evt_number,voxels_event):
        vx = [v.X for v in voxels_event]
        vy = [v.Y for v in voxels_event]
        vz = [v.Z for v in voxels_event]
        ve = [v.E for v in voxels_event]
        carr = np.array([vx, vy, vz, ve])
        hdf5_group.create_dataset("truevox{0}".format(evt_number),data=carr)

    return write_voxels
