"""
code: detsim.py
Simulation of drift and diffusion starting from Nexus output.
Saves voxelized events.

"""

import numpy  as np
import tables as tb

from argparse import Namespace

from .. evm.event_model              import MCHit

#from .. io.pmaps_io                  import pmap_writer
from .. io.mcinfo_io                 import load_mchits
from .. io.run_and_event_io          import run_and_event_writer
from .. io.voxels_io                 import true_voxels_writer

from .. reco                         import tbl_functions as tbl
from .. reco.paolina_functions       import voxelize_hits

from .. dataflow                     import dataflow as fl
from .. dataflow.dataflow            import push
from .. dataflow.dataflow            import pipe
from .. dataflow.dataflow            import fork

from .  components                   import city
from .  components                   import print_every
from .  components                   import get_run_number

@city
def detsim(files_in, file_out, compression, event_range, print_mod, run_number,
           zmin, zmax, diff_transv, diff_long, resolution_FWHM, Qbb,
           write_true_voxels, voxel_dimensions, A_sipm, d_sipm, ze_sipm,
           ze_pmt, slice_width_sipm, E_to_Q_sipm, uniformlight_frac_sipm,
           s2_threshold_sipm, slice_width_pmt, E_to_Q_pmt, uniformlight_frac_pmt,
           s2_threshold_pmt, peak_space):

    diffuse_and_smear_hits_ = fl.map(diffuse_and_smear_hits(zmin, zmax, diff_transv, diff_long, resolution_FWHM, Qbb),
                                     args = "mchits",
                                     out  = ("dmchits", "zdrift"))
    voxelize_hits_          = fl.map(voxelize_smeared_hits(voxel_dimensions),
                                     args = "dmchits",
                                     out  = "voxels")
    event_count_in        = fl.spy_count()
    event_count_out       = fl.spy_count()

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:

        # Define writers...
        write_true_voxels    = fl.sink(true_voxels_writer(h5out),   args = ("event_number", "voxels"))
        write_event_info     = fl.sink(run_and_event_writer(h5out), args = ("run_number", "event_number", "timestamp"))

        return push(source = nexus_hits_from_files(files_in),
                    pipe   = pipe(
                        fl.slice(*event_range, close_all=True),
                        print_every(print_mod)                ,
                        event_count_in       .spy             ,
                        diffuse_and_smear_hits_               ,
                        voxelize_hits_                        ,
                        event_count_out      .spy             ,
                        fork(write_true_voxels                ,
                             write_event_info)                ),
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future))

def diffuse_and_smear_hits(zmin, zmax, diff_transv, diff_long,
                           resolution_FWHM, Qbb):
    """
    Applies diffusion and energy smearing to all MC hits.

    """
    # calculate drift distance
    zdrift = np.random.uniform(zmin,zmax)

    def diffuse_and_smear_hits(mchits):

        # calculate unscaled variance for energy smearing
        E_evt = sum([hit.E for hit in mchits])
        sigma0 = ((resolution_FWHM/100.) * np.sqrt(Qbb) * np.sqrt(E_evt)) / 2.355
        var0 = sigma0**2

        # apply diffusion and energy smearing
        dmchits = []
        for hit in mchits:
            xh = np.random.normal(hit.X,np.sqrt(zdrift/10.)*diff_transv)
            yh = np.random.normal(hit.Y,np.sqrt(zdrift/10.)*diff_transv)
            zh = np.random.normal(hit.Z+zdrift,np.sqrt(zdrift/10.)*diff_long)
            eh = np.random.normal(hit.E,np.sqrt(var0*hit.E/E_evt))
            dmchits.append(MCHit([xh,yh,zh], hit.T, eh, hit.Label))

        return dmchits, zdrift
    return diffuse_and_smear_hits

def voxelize_smeared_hits(voxel_dimensions):

    def voxelize_smeared_hits(dmchits):
        return voxelize_hits(dmchits, voxel_dimensions)
    return voxelize_smeared_hits

def nexus_hits_from_files(paths):

    for path in paths:
        mchits_dict = load_mchits(path)

        for event_number, mchits in mchits_dict.items():
            yield dict(mchits=mchits, event_number=event_number, run_number=0, timestamp=0)

# ------------------------------------------------------------------------------
# CURRENTLY NOT USED 
# Code for PMap generation
def simulate_sensors(voxels, zdrift,
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
        if(umean >= 0):
            sipm_map[islice,:] += np.maximum(np.random.normal(umean,
                                             umean,size=nsipm),np.zeros(nsipm))
    for islice,en_slice in enumerate(pmt_energies):
        umean = en_slice*E_to_Q_pmt*uniformlight_frac_pmt
        if(umean >= 0):
            pmt_map[islice,:] += np.maximum(np.random.normal(umean,
                                            umean,size=npmt),np.zeros(npmt))

    # apply the SiPM 1-pe threshold
    sipm_map[sipm_map < 1] = 0.

    pmap = get_detsim_pmaps(sipm_map, s2_threshold_sipm, pmt_map,
                                  s2_threshold_pmt, slice_width_pmt,
                                  peak_space, zdrift)

    return pmap


def get_detsim_pmaps(sipm_map, s2_threshold_sipm,
                     pmt_map, s2_threshold_pmt, slice_width,
                     peak_space, zdrift):

    ids_pmt = [ipmt for ipmt in range(0,12)]

    # S1: for now, a default S1
    s1s = [ S1([slice_width*units.mus],
            PMTResponses(ids_pmt, 10*np.ones([12,1])),
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
                                   ids_pmt, islice_lastpk, islice, zdrift))

                islice_lastpk  = islice

    # create the final S2 peak
    s2s.append(make_s2(pmt_map, sipm_map, s2_threshold_sipm, slice_width,
                       ids_pmt, islice_lastpk, len(pmt_map), zdrift))

    return PMap(s1s,s2s)


def make_s2(pmt_map, sipm_map, s2_threshold_sipm, slice_width,
            ids_pmt, islice_lastpk, islice, zdrift):
    pk_wf_pmt = pmt_map[islice_lastpk:islice,:].transpose()
    ids_sipm, pk_wf_sipm = pkf.select_wfs_above_time_integrated_thr(
            sipm_map[islice_lastpk:islice,:].transpose(),
            s2_threshold_sipm)
    return S2([(t+int(zdrift))*slice_width*units.mus for t in range(islice_lastpk,islice)],
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
