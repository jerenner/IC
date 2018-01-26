"""
detsim_functions.py
Defines key functions used in Detsim.
"""

import numpy as np

from .. io.table_io            import make_table
from .. evm.event_model        import MCHit
from .. evm.nh5                import TrueVoxelsTable

from .. evm.pmaps              import S1
from .. evm.pmaps              import S2
from .. evm.pmaps              import PMTResponses
from .. evm.pmaps              import SiPMResponses
from .. evm.pmaps              import PMap

from .. reco                   import peak_functions           as pkf

def diffuse_and_smear_hits(mchits, zmin, zmax, diff_transv, diff_long,
                           resolution_FWHM, Qbb):
    """
    Applies diffusion and energy smearing to all MC hits.

    """
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

    return dmchits

# writers
def true_voxels_writer(hdf5_file, *, compression='ZLIB4'):

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

def simulate_sensors(voxels,
                     data_sipm, slice_width_sipm, light_function_sipm,
                     E_to_Q_sipm, uniformlight_frac_sipm, s2_threshold_sipm,
                     data_pmt, slice_width_pmt, light_function_pmt,
                     E_to_Q_pmt, uniformlight_frac_pmt,
                     s2_threshold_pmt, peak_space):
    """
    Simulate sensor responses (SiPMs and PMTs)

    """
    nsipm = len(data_sipm.X)
    npmt  = len(data_pmt.X)

    zmin = np.min([voxel.Z for voxel in voxels])
    zmax = np.max([voxel.Z for voxel in voxels])

    #print("Energy is {}".format(np.sum([voxel.E for voxel in voxels])))

    nslices_sipm = int(np.ceil((zmax - zmin)/slice_width_sipm))
    nslices_pmt  = int(np.ceil((zmax - zmin)/slice_width_pmt))

    sipm_map      = np.zeros([nslices_sipm,nsipm])
    sipm_energies = np.zeros(nslices_sipm)
    pmt_map       = np.zeros([nslices_pmt,npmt])
    pmt_energies  = np.zeros(nslices_pmt)

    # cast light on sensor planes for each voxel
    for voxel in voxels:

        # sipm plane
        islice_sipm = int((voxel.Z - zmin)/slice_width_sipm)
        r_sipm = np.array([np.sqrt((xi - voxel.X)**2 + (yi - voxel.Y)**2) for xi,yi in zip(data_sipm.X,data_sipm.Y)])
        probs_sipm = light_function_sipm(r_sipm)
        sipm_map[islice_sipm,:] += probs_sipm*voxel.E*E_to_Q_sipm
        sipm_energies[islice_sipm] += voxel.E

        # pmt plane
        islice_pmt = int((voxel.Z - zmin)/slice_width_pmt)
        r_pmt = np.array([np.sqrt((xi - voxel.X)**2 + (yi - voxel.Y)**2) for xi,yi in zip(data_pmt.X,data_pmt.Y)])
        probs_pmt = light_function_pmt(r_pmt)
        pmt_map[islice_pmt,:] += probs_pmt*voxel.E*E_to_Q_pmt
        pmt_energies[islice_pmt] += voxel.E

    # uniform light (based on energy only)
    for islice,en_slice in enumerate(sipm_energies):
        umean = en_slice*E_to_Q_sipm*uniformlight_frac_sipm
        sipm_map[islice,:] += np.maximum(np.random.normal(umean,umean,size=nsipm),np.zeros(nsipm))
    for islice,en_slice in enumerate(pmt_energies):
        umean = en_slice*E_to_Q_pmt*uniformlight_frac_pmt
        pmt_map[islice,:] += np.maximum(np.random.normal(umean,umean,size=npmt),np.zeros(npmt))

    # apply the SiPM 1-pe threshold
    sipm_map[sipm_map < 1] = 0.

    pmap = get_detsim_pmaps(sipm_map, s2_threshold_sipm, pmt_map,
                                  s2_threshold_pmt, slice_width_pmt,
                                  peak_space)

    return pmap

def get_detsim_pmaps(sipm_map, s2_threshold_sipm,
                     pmt_map, s2_threshold_pmt, slice_width,
                     peak_space):

    ids_pmt = [ipmt for ipmt in range(0,12)]

    # S1: for now, just a single value equal to 0 for each PMT
    s1s = [ S1([0.],
            PMTResponses(ids_pmt, np.zeros([12,1])),
            SiPMResponses.build_empty_instance())]

    # S2
    s2s = []
    islice_lastpk = 0
    for islice in range(len(pmt_map)):

        signals_sum = np.sum(pmt_map[islice,:])
        if(signals_sum > s2_threshold_pmt):

            if((islice - islice_lastpk)*slice_width >= peak_space):

                # create a new S2 peak beginning where the last one left off
                s2s.append(make_s2(pmt_map, sipm_map, s2_threshold_sipm, slice_width,
                                   ids_pmt, islice_lastpk, islice))

                islice_lastpk  = islice

    # create the final S2 peak
    s2s.append(make_s2(pmt_map, sipm_map, s2_threshold_sipm, slice_width,
                       ids_pmt, islice_lastpk, len(pmt_map)))

    return PMap(s1s,s2s)

def make_s2(pmt_map, sipm_map, s2_threshold_sipm, slice_width,
            ids_pmt, islice_lastpk, islice):
    pk_wf_pmt = pmt_map[islice_lastpk:islice,:].transpose()
    ids_sipm, pk_wf_sipm = pkf.select_wfs_above_time_integrated_thr(
            sipm_map[islice_lastpk:islice,:].transpose(),
            s2_threshold_sipm)
    return S2([t*slice_width for t in range(islice_lastpk,islice)],
                  PMTResponses(ids_pmt,pk_wf_pmt),
                  SiPMResponses(ids_sipm,pk_wf_sipm))

def pmt_lcone(ze):
    """
    Approximate PMT light cone function.

    ze: the distance from the EL region to the SiPM sipm_plane
    """

    def pmt_lcone_r(r):
        return np.abs(ze) / (2 * np.pi) / (r**2 + ze**2)**1.5

    return pmt_lcone_r

def sipm_lcone(A, d, ze):
    """
    Approximate SiPM light cone function.

    A:  the area of a single SiPM
    d:  the length of the EL gap
    ze: the distance from the EL region to the SiPM sipm_plane
    """

    def sipm_lcone_r(r):
        return (A/(4*np.pi*d*np.sqrt(r**2 + ze**2)))*(1 - np.sqrt((r**2 + ze**2)/(r**2 + (ze+d)**2)))

    return sipm_lcone_r
