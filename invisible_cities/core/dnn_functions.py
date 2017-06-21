"""Helper functions for DNN-based analysis
JR March 2017
"""
from __future__ import print_function, division, absolute_import

import numpy as np
import tables as tb

def read_xyz_labels(files_in, nmax, evt_numbers):
    labels = []; levt_numbers = []
    tot_ev = 0
    for fin in files_in:

        fxy = tb.open_file(fin,'r')
        tracks = fxy.root.MC.MCTracks

        # get the arrays containing the true information
        event_indx = np.array(tracks[:]['event_indx'],dtype=np.int32)
        hit_energy = np.array(tracks[:]['hit_energy'],dtype=np.float32)
        hit_pos    = np.array(tracks[:]['hit_position'],dtype=np.float32)

        label = np.zeros(3)
        tot_energy = 0; i = 0
        rd_evt = 0

        # align the events properly and loop over table rows
        while(i < len(event_indx) and (event_indx[i] != evt_numbers[rd_evt])):
            i += 1
        while((nmax < 0 or tot_ev < nmax) and i < len(event_indx)):
            ev = event_indx[i]
            
            # ensure we stay synchronized with the specified event numbers list
            if(ev == evt_numbers[rd_evt]):
                label = label + hit_energy[i] * hit_pos[i]
                tot_energy += hit_energy[i]

                # save and reset the label if we have reached the end of this event
                if(i >= len(event_indx)-1 or
                  (i < len(event_indx)-1 and event_indx[i+1] > ev)):
                    label /= tot_energy
                    tot_energy = 0
                    labels.append(label[0:2])
                    levt_numbers.append(ev)
                    label = np.zeros(3)
                    tot_ev += 1
                    rd_evt += 1
            i += 1
        fxy.close()

    labels = np.array(labels,dtype=np.float32)
    levt_numbers = np.array(levt_numbers)
    return labels, levt_numbers

def read_dnn_datafile(datafile,nmax,mode):
        indata = tb.open_file(datafile, 'r')
        
        labels = None
        if(nmax > 0):
            in_maps = indata.root.maps[0:nmax]            
            if(mode == 'train' or mode == 'retrain' or mode == 'test'):
                in_coords = indata.root.coords[0:nmax]
        else:
            in_maps = indata.root.maps
            if(mode == 'train' or mode == 'retrain' or mode == 'test'):
                in_coords = indata.root.coords
                
        sum_maps = np.reshape(in_maps,(len(in_maps), 48, 48, 1))
        if(mode == 'train' or mode == 'retrain' or mode == 'test'):
            labels = np.array(in_coords,dtype=np.float32)
            
        return sum_maps,labels
