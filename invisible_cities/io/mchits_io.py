
import tables
from ..evm.event_model     import MCParticle
from ..evm.event_model     import MCHit
from typing import Mapping

# use Mapping (duck type) rather than dict

def load_mchits(file_name: str, max_events:int =1e+9) -> Mapping[int, MCHit]:

    with tables.open_file(file_name,mode='r') as h5in:
        mctable = h5in.root.MC.MCTracks
        mcevents = read_mctracks (mctable, max_events)
        mchits_dict = compute_mchits_dict(mcevents)
    return mchits_dict


def load_mchits_nexus(file_name: str,
                      max_events:int =1e+9) -> Mapping[int, MCHit]:

    h5f = h5py.File(file_name, 'r')
    mcevents = read_mctracks_nexus(h5f, max_events)
    mchits_dict = compute_mchits_dict(mcevents)

    return mchits_dict


def load_mcparticles(file_name: str, max_events:int =1e+9) -> Mapping[int, MCParticle]:

    with tables.open_file(file_name,mode='r') as h5in:
        mctable = h5in.root.MC.MCTracks
        return read_mctracks (mctable, max_events)


def load_mcparticles_nexus(file_name: str, max_events:int =1e+9) -> Mapping[int, MCParticle]:

    h5f = h5py.File(file_name, 'r')
    return read_mctracks_nexus(h5f, max_events)


def read_mctracks (mc_table: tables.table.Table,
                   max_events:int =1e+9) ->Mapping[int, Mapping[int, MCParticle]]:

    all_events = {}
    current_event = {}
#    convert table to numpy.ndarray
    data       = mc_table[:]
    data_size  = len(data)

    event            =  data["event_indx"]
    particle         =  data["mctrk_indx"]
    particle_name    =  data["particle_name"]
    pdg_code         =  data["pdg_code"]
    initial_vertex   =  data["initial_vertex"]
    final_vertex     =  data["final_vertex"]
    momentum         =  data["momentum"]
    energy           =  data["energy"]
    nof_hits         =  data["nof_hits"]
    hit              =  data["hit_indx"]
    hit_position     =  data["hit_position"]
    hit_time         =  data["hit_time"]
    hit_energy       =  data["hit_energy"]

    for i in range(data_size):
        if event[i] >= max_events:
            break

        current_event = all_events.setdefault(event[i], {})

        current_particle = current_event.setdefault( particle[i],
                                                    MCParticle(particle_name[i],
                                                               pdg_code[i],
                                                               initial_vertex[i],
                                                               final_vertex[i],
                                                               momentum[i],
                                                               energy[i]))
        hit = MCHit(hit_position[i], hit_time[i], hit_energy[i])
        current_particle.hits.append(hit)

    return all_events


def read_mctracks_nexus (h5f, max_events:int =1e+9) ->Mapping[int, Mapping[int, MCParticle]]:

    h5extents = h5f.get('Run/extents')
    h5hits = h5f.get('Run/hits')
    h5particles = h5f.get('Run/particles')

    all_events = {}
    particles = {}
    current_event = {}

    ihit = 0
    ipart = 0

    for iext in range(h5extents):
        if(iext >= max_events):
            break

        current_event = {}

        ipart_end = h5extents[iext]['last_particle']
        while(ipart < ipart_end):
            h5particle = h5particles[ipart]
            itrack = h5particle['track_indx']

            current_event[itrack] = MCParticle(h5particle['particle_name'],
                                               0, # PDG code not currently stored
                                               h5particle['initial_vertex'],
                                               h5particle['final_vertex'],
                                               h5particle['momentum'],
                                               h5particle['energy'])

            ipart += 1

        ihit_end = h5extents[iext]['last_hit']
        while(ihit < ihit_end):
            h5hit = h5hits[ihit]
            itrack = h5hit['track_indx']

            current_particle = current_event[itrack]

            hit = MCHit(h5hit['hit_position'],h5hit['hit_time'],
                          h5hit['hit_energy'])
            current_particle.hits.append(hit)
            ihit += 1

        evt_number = h5extents[iext]['evt_number']
        all_events[evt_number] = current_event

    return all_events


def compute_mchits_dict(mcevents:Mapping[int, Mapping[int, MCParticle]])->Mapping[int, MCHit]:
    """Returns all hits in the event"""
    mchits_dict = {}
    for event_no, particle_dict in mcevents.items():
        hits = []
        for particle_no in particle_dict.keys():
            particle = particle_dict[particle_no]
            hits.extend(particle.hits)
        mchits_dict[event_no] = hits
    return mchits_dict
