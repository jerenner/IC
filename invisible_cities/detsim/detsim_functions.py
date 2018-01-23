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

ulight_frac = 8.0e-7     # scale factor for uniform reflected light: energy E emitted from a single point
                         #   will give rise to a uniform illumination of the SiPM plane  in addition to
                         #   its usual light cone.  The amount of illumination will be a uniform value with
                         #   with min 0 and max E*sipm_par(0,0)*ulight_frac.

E_to_Q = 7.5e3             # energy to Q (pes) conversion factor

def simulate_sensors(voxels_dict, slice_width_sipm, slice_width_pmt,
                     sipm_light_function, data_sipm):
    """
    Simulate sensor responses (SiPMs and PMTs)

    """
    nsipm = len(data_sipm.X)

    for evt_number, voxels in voxels_dict.items():

        zmin = np.min([voxel.Z for voxel in voxels])
        zmax = np.max([voxel.Z for voxel in voxels])
        nslices = int(np.ceil((zmax - zmin)/slice_width_sipm))

        sipm_map      = np.zeros([nslices,nsipm])
        sipm_energies = np.zeros(nslices)

        pmt_energies  = np.zeros(nslices)

        umean = en[ss]*E_to_Q*ulight_frac
        sipm_map = np.maximum(np.random.normal(umean,umean,size=nsipm),np.zeros(nsipm))

        # cast light on sensor planes for each voxel
        for voxel in voxels:

            # sipm plane
            islice_sipm = int((voxel.Z - zmin)/slice_width_sipm)
            rr = np.array([np.sqrt((xi - voxel.X)**2 + (yi - voxel.Y)**2) for xi,yi in zip(data_sipm.X,data_sipm.Y)])
            probs = sipm_light_function(rr)
            sipm_map[islice_sipm,:] += probs*voxel.E*E_to_Q
            sipm_energies[islice_sipm] += voxel.E

            # pmt plane
            islice_pmt = int((voxel.Z - zmin)/slice_width_pmt)
            pmt_energies[islice_pmt] += voxel.E

        # uniform light (based on energy only)

        # Apply the 1-pe threshold.
        sipm_map[sipm_map < 1] = 0.

        # At this point we may want to multiply the SiPM map by a
        #  factor proportional to the slice energy.

def sipm_lcone(A, d, ze):
    """
    Approximate SiPM light cone function.

    A:  the area of a single SiPM
    d:  the length of the EL gap
    ze: the distance from the EL region to the SiPM sipm_plane
    """

    def sipm_lcone_r(r):
        v = (A/(4*np.pi*d*np.sqrt(r**2 + ze**2)))*(1 - np.sqrt((r**2 + ze**2)/(r**2 + (ze+d)**2)))
        return v

    return sipm_lcone_r
