"""
detsim_functions.py
Defines key functions used in Detsim.
"""

import numpy as np

from .. io.table_io            import make_table
from .. evm.event_model        import MCHit
from .. reco.paolina_functions import voxelize_hits
from .. evm.nh5                import TrueVoxelsTable

def diffuse_and_smear_hits(mchits_dict, zmin, zmax, diff_transv, diff_long,
                           resolution_FWHM, Qbb):
    """
    Applies diffusion and energy smearing to all MC hits.

    """

    dmchits_dict = {}
    for evt_number,mchits in mchits_dict.items():

        # calculate unscaled variance for energy smearing
        E_evt = sum([hit.E for hit in mchits])
        sigma0 = ((resolution_FWHM/100.) * np.sqrt(Qbb) * np.sqrt(E_evt)) / 2.355
        var0 = sigma0**2

        # calculate drift distance
        zdrift = np.random.uniform(zmin,zmax)

        # apply diffusion and energy smearing
        dmchits = []
        for hit in mchits:

            xh = np.random.normal(hit.X,np.sqrt(zdrift/10.)*diff_transv)
            yh = np.random.normal(hit.Y,np.sqrt(zdrift/10.)*diff_transv)
            zh = np.random.normal(hit.Z+zdrift,np.sqrt(zdrift/10.)*diff_long)
            eh = np.random.normal(hit.E,np.sqrt(var0*hit.E/E_evt))

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
    # hdf5_group = hdf5_file.create_group("True")

    voxels_table  = make_table(hdf5_file,
                             group       = 'TrueVoxels',
                             name        = 'Voxels',
                             fformat     = TrueVoxelsTable,
                             description = 'Voxels',
                             compression = compression)
    # Mark column to index after populating table
    voxels_table.set_attr('columns_to_index', ['event'])

    def write_voxels(evt_number,voxels_event):
        row = voxels_table.row
        for voxel in voxels_event:
            row["event"] = evt_number
            row["X"    ] = voxel.X
            row["Y"    ] = voxel.Y
            row["Z"    ] = voxel.Z
            row["E"    ] = voxel.E
            row.append()

    return write_voxels
