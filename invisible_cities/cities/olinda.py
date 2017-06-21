from __future__ import print_function
import sys

from glob import glob
from time import time
import numpy as np
import tables as tb
import os
import matplotlib.pyplot as plt
from matplotlib.patches         import Ellipse
import textwrap

from keras.models               import Model
from keras.models               import load_model
from keras.models               import Sequential
from keras.layers               import Input
from keras.layers               import Dense
from keras.layers               import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core          import Flatten

from   invisible_cities.core.log_config       import logger
from   invisible_cities.core.configure        import configure
from   invisible_cities.core.dnn_functions    import read_xyz_labels
from   invisible_cities.core.dnn_functions    import read_dnn_datafile
from   invisible_cities.cities.base_cities    import KerasDNNCity

from   invisible_cities.io.dnn_io              import dnn_writer
from   invisible_cities.reco.event_model       import NNEvent

from   invisible_cities.reco                   import tbl_functions   as tbl
from   invisible_cities.reco.pmaps_functions   import load_pmaps
from   invisible_cities.reco.tbl_functions     import get_event_numbers_and_timestamps_from_file_name

from   invisible_cities.filters.s1s2_filter    import s1s2_filter
from   invisible_cities.filters.s1s2_filter    import S12Selector

#from   .. core                  import fit_functions        as fitf
#from   .. core.log_config       import logger
#from   .. core.configure        import configure
#from   .. core.dnn_functions    import read_xyz_labels
#from   .. core.dnn_functions    import read_pmaps
#from   .  base_cities           import KerasDNNCity
#from   .. reco.dst_io           import XYcorr_writer
#
#from   .. reco.corrections      import Correction

class Olinda(KerasDNNCity):
    """
    The city of OLINDA performs the DNN analysis for point reconstruction.

    This city takes a set of input files (HDF5 files containing PMAPS, and
    MCTrack data if it is to be used for training).  It then reads the PMAPS
    and MCTrack data and performs a pre-processing step which prepares HDF5
    files containing:

        maps: [48, 48] matrices containing the SiPM responses
        coords: length-2 arrays containing true (x,y) points, if available

    This file is saved with the name specified in the configuration file
    DNN_DATAFILE.  Once this file is created, this process or creating
    "maps" and "coords" objects does not need to be run again until new data
    is to be input.

    The city can be run in 4 modes:
        - 'train': trains a DNN with the input events
        - 'retrain': same as 'train' but ensures that the DNN is trained
        from a new initialization
        - 'test': predicts (x,y) values for a given set of input events (PMAPS)
        for comparison with true values, which must also be given in the inputs
        - 'eval': predicts (x,y) values for a given set of input events (PMAPS)
        but unlike 'test' does not require that the true values are given
        (this is the mode that would be applied to detector data)

    Thus in general the city will be used as follows:
        - a large MC dataset will be input in 'train' mode, and the weights
        of the trained net will be saved
        - a subset of this MC dataset can be used in 'test' mode to verify
        that the net has been properly trained
        - real data can be input in 'eval' mode and the (x,y) predictions
        saved

    A summary of the key inputs to the configuration file:
        - FILE_IN: a list of input files
        - RUN_NUMBER: the run number
        - TEMP_DIR: a temporary directory to which network weights are written
        - MODE: the operating mode 'train', 'retrain', 'test', or 'eval'
        - WEIGHTS_FILE: the name of the file containing the weights of the
        neural network to be employed.  If this file does not exist, new files
        will be saved.
        - DNN_DATAFILE: the name of the datafile to which the pre-processed
        datasets (maps and coords) will be saved.  If this file already exists,
        the pre-processed data will be read from the file directly and the
        input files will be ignored.
        - OPT: the optimizer ('nadam' or 'sgd')
        - LRATE: the learning rate
        - DECAY: the learning rate decay rate
        - LOSS: the loss function (see Keras loss function names)
        - FILE_OUT: the name of the output file containing (x,y) predictions
        - NEVENTS: the number of events to be read

    """

    def __init__(self,
                 run_number   = 0,
                 files_in     = None,
                 file_out     = None,
                 temp_dir     = 'database/test_data',
                 weights_file = 'weights.h5',
                 dnn_datafile = 'dnn_datafile.h5',
                 nprint      = 10000,
                 lrate       = 0.01,
                 sch_decay   = 0.01,
                 loss_type   = 'mse',
                 opt         = 'nadam',
                 mode        = 'eval',
                 lifetime    = 1000,
                 max_slices  = 3,
                 tbin_slice  = 10000,
                 
                 S1_Emin     = 0,
                 S1_Emax     = 10000,
                 S1_Lmin     = 4,
                 S1_Lmax     = 20,
                 S1_Hmin     = 0,
                 S1_Hmax     = 1000,
                 S1_Ethr     = 0.5,

                 S2_Nmin     = 1,
                 S2_Nmax     = 1,
                 S2_Emin     = 1000,
                 S2_Emax     = 1000000,
                 S2_Lmin     = 1,
                 S2_Lmax     = 1000,
                 S2_Hmin     = 0,
                 S2_Hmax     = 100000,
                 S2_NSIPMmin = 0,
                 S2_NSIPMmax = 1000,
                 S2_Ethr     = 1):
        """
        Init the machine with the run number.
        Load the data base to access calibration and geometry.
        Sets all switches to default value.
        """
        KerasDNNCity.__init__(self,
                                   run_number      = run_number,
                                   files_in        = files_in,
                                   file_out        = file_out,
                                   temp_dir        = temp_dir,
                                   weights_file    = weights_file,
                                   dnn_datafile    = dnn_datafile,
                                   nprint          = nprint,
                                   lrate           = lrate,
                                   sch_decay       = sch_decay,
                                   loss_type       = loss_type,
                                   opt             = opt,
                                   mode            = mode)
        self.lifetime = lifetime   
        self._s1s2_selector = S12Selector(S1_Nmin     = 1,
                                          S1_Nmax     = 1,
                                          S1_Emin     = S1_Emin,
                                          S1_Emax     = S1_Emax,
                                          S1_Lmin     = S1_Lmin,
                                          S1_Lmax     = S1_Lmax,
                                          S1_Hmin     = S1_Hmin,
                                          S1_Hmax     = S1_Hmax,
                                          S1_Ethr     = S1_Ethr,

                                          S2_Nmin     = 1,
                                          S2_Nmax     = S2_Nmax,
                                          S2_Emin     = S2_Emin,
                                          S2_Emax     = S2_Emax,
                                          S2_Lmin     = S2_Lmin,
                                          S2_Lmax     = S2_Lmax,
                                          S2_Hmin     = S2_Hmin,
                                          S2_Hmax     = S2_Hmax,
                                          S2_NSIPMmin = S2_NSIPMmin,
                                          S2_NSIPMmax = S2_NSIPMmax,
                                          S2_Ethr     = S2_Ethr)

    def run(self, nmax):
        self.display_IO_info(nmax)
        
        nevt_in = -1; nevt_out = -1
        if(not os.path.isfile(self.dnn_datafile)):
            with tb.open_file(self.output_file, "w",
                              filters = tbl.filters(self.compression)) as h5out:
    
                write_dnn = dnn_writer(h5out)
    
                nevt_in, nevt_out = self._file_loop(write_dnn, nmax)
                print(textwrap.dedent("""
                                  Number of events in : {}
                                  Number of events out: {}
                                  Ratio               : {}
                                  """.format(nevt_in, nevt_out, nevt_out / nevt_in)))

        self.X_in, self.Y_in = read_dnn_datafile(self.dnn_datafile)
        if(nevt_in == -1):
            nevt_in = len(self.X_in)
        
        print("-- X_in shape is {0}".format(self.X_in.shape))
        print("-- Max X is {0}".format(np.max(self.X_in)))
        print("-- Min X is {0}".format(np.min(self.X_in)))
        
        self.build_model()
        if(self.mode == 'train' or self.mode == 'retrain'):
            self.train(nbatch=40,nepochs=100)
        else:
            prediction = self.evaluate()
            
        return nevt_in, nevt_out

    def _file_loop(self, write_kr, nmax):
        nevt_in = nevt_out = 0

        for filename in self.input_files:
            print("Opening {filename}".format(**locals()), end="... ")

            try:
                S1s, S2s, S2Sis = load_pmaps(filename)
            except (ValueError, tb.exceptions.NoSuchNodeError):
                print("Empty file. Skipping.")
                continue

            event_numbers, timestamps = get_event_numbers_and_timestamps_from_file_name(filename)
            labels = None
            if(self.mode == 'test' or self.mode == 'train' or self.mode == 'retrain'):

                print("Reading labels...")
                labels, levt_numbers = read_xyz_labels(self.input_files,nmax,event_numbers)
                if(len(event_numbers) != len(levt_numbers)):
                    print("ERROR: number of labels does not match number of events")
                    exit()
                    
                for e1,e2 in zip(event_numbers,levt_numbers):
                    if(e1 != e2):
                        print("ERROR: Mismatch in event numbers e1 = {0}, e2 = {1}.".format(e1,e2))
                        exit()
                print("Found {0} labels for {1} maps".format(len(levt_numbers),len(event_numbers)))

            nevt_in, nevt_out, max_events_reached = self._event_loop(
                event_numbers, labels, nmax, nevt_in, nevt_out, write_kr, S1s, S2s, S2Sis)

            if max_events_reached:
                print('Max events reached')
                break
            else:
                print("OK")

        return nevt_in, nevt_out

    def _event_loop(self, event_numbers, labels, nmax, nevt_in, nevt_out, write_dnn, S1s, S2s, S2Sis):
        max_events_reached = False
        for evt_number, evt_label in zip(event_numbers, labels):
            nevt_in += 1
            if self.max_events_reached(nmax, nevt_in):
                max_events_reached = True
                break
            S1 = S1s  .get(evt_number, {})
            S2 = S2s  .get(evt_number, {})
            Si = S2Sis.get(evt_number, {})

            if not s1s2_filter(self._s1s2_selector, S1, S2, Si):
                continue
            nevt_out += 1

            evt = self._create_NN_event(evt_number, evt_label, S2, Si)
            write_dnn(evt)

            self.conditional_print(evt, nevt_in)

        return nevt_in, nevt_out, max_events_reached

    def _create_NN_event(self, evt_number, evt_label, S2, Si):

        evt = NNEvent()
        evt.event = evt_number
        for peak_no, (t, e) in sorted(S2.items()):
            
            si = Si[peak_no]
            for sipm_no,sipm_q in si.items():
                [i, j] = (self.id_to_coords[sipm_no] + 235) / 10
                evt.map48x48[np.int8(i),np.int8(j)] += np.sum(sipm_q)
                
        evt.map48x48 /= np.sum(evt.map48x48)
        evt.label[:] = evt_label
        return evt

    def build_DNN_FC(self):
        """Builds a fully-connected neural network.
        """
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(48,48,1)))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=32,  activation='relu'))
        self.model.add(Dense(units=16,  activation='relu'))
        self.model.add(Dense(units=8,  activation='relu'))
        self.model.add(Dense(units=2,    activation='relu'))

    def build_DNN_conv2D(self):
        """Builds a 2D-convolutional neural network.
        """
        inputs = Input(shape=(48, 48, 1))
        cinputs = Conv2D(32, (4, 4), padding='same', strides=(4, 4), activation='relu', kernel_initializer='normal')(inputs)
        cinputs = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(cinputs)
        cinputs = BatchNormalization(epsilon=1e-05, axis=3, momentum=0.99, weights=None, beta_initializer='zero', gamma_initializer='one', gamma_regularizer=None, beta_regularizer=None)(cinputs)
        cinputs = Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu', kernel_initializer='normal')(cinputs)
        cinputs = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same', data_format=None)(cinputs)
        cinputs = BatchNormalization(epsilon=1e-05, axis=3, momentum=0.99, weights=None, beta_initializer='zero', gamma_initializer='one', gamma_regularizer=None, beta_regularizer=None)(cinputs)
        cinputs = Conv2D(256, (2, 2), padding='same', strides=(1, 1), activation='relu', kernel_initializer='normal')(cinputs)
        cinputs = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(cinputs)
        cinputs = BatchNormalization(epsilon=1e-05, axis=3, momentum=0.99, weights=None, beta_initializer='zero', gamma_initializer='one', gamma_regularizer=None, beta_regularizer=None)(cinputs)
        f1 = Flatten()(cinputs)
        f1 = Dense(units=1024, activation='relu', kernel_initializer='normal')(f1)
        f1 = Dropout(.6)(f1)
        coutput = Dense(units=2, activation='relu', kernel_initializer='normal')(f1)
        self.model = Model(inputs,coutput)
        
    def build_model(self):
        """Constructs or reads in the DNN model to be trained.

        The model will be build from scratch if:
            - the city is initialized in 'retrain' mode
            - the city is initialized in 'train' mode and previously trained
            weights do not already exist

        Otherwise, the weights will be read from weights_file.
        """

        weights_exist = os.path.isfile(self.weights_file)

        # build the model if no weights exist or mode is 'retrain'
        if(self.mode == 'retrain' or (self.mode == 'train' and not weights_exist)):
            self.build_DNN_FC()
            self.model.compile(loss=self.loss_type, optimizer=self.optimizer,
                               metrics=['accuracy'])
            self.model.summary()

        # otherwise read in the existing weights
        elif(weights_exist):
            logger.info("Loading model from {0}".format(self.weights_file))
            self.model = load_model(self.weights_file)
        else:
            logger.error("ERROR: invalid state in function build_model")

    def check_evt(self,evt_num,ept=None):
        """Plots the event with number evt_num

        If ept is specified, it must be a length 2 array containing the
        x and y coordinates of the reconstructed point.
        """

        logger.info("Checking event {0}".format(evt_num))
        logger.info("-- Shape of X is {0}".format(self.X_in.shape))

        # set up the figure
        fig = plt.figure();
        ax1 = fig.add_subplot(111);
        fig.set_figheight(15.0)
        fig.set_figwidth(15.0)
        ax1.axis([-250, 250, -250, 250]);

        # get the SiPM map and label
        xarr = self.X_in[evt_num]
        yarr = self.Y_in[evt_num]*400. - 200.

        # convert it to a normalized map
        probs = (xarr - np.min(xarr))
        probs /= np.max(probs)

        # draw the map
        for i in range(48):
            for j in range(48):
                r = Ellipse(xy=(i * 10 - 235, j * 10 - 235), width=2., height=2.);
                r.set_facecolor('0');
                r.set_alpha(probs[i, j]);
                ax1.add_artist(r);

        # place a large blue circle for the true EL points
        xpt = yarr[0]
        ypt = yarr[1]
        mrk = Ellipse(xy=(xpt,ypt), width=4., height=4.);
        mrk.set_facecolor('b');
        ax1.add_artist(mrk);

        # place a large red circle for reconstructed points
        if(ept != None):
            xpt = ept[0]*400. - 200.
            ypt = ept[1]*400. - 200.
            mrk = Ellipse(xy=(xpt,ypt), width=4., height=4.);
            mrk.set_facecolor('r');
            ax1.add_artist(mrk);

        plt.savefig("{0}/evt_{1}.png".format(self.temp_dir,evt_num))

def OLINDA(argv = sys.argv):
    """OLINDA DRIVER"""
    CFP = configure(argv)

    files_in    = glob(CFP.FILE_IN)
    files_in.sort()
    print("input files = {0}".format(files_in))

    fpp = Olinda(run_number  = CFP.RUN_NUMBER,
                 files_in    = files_in,
                 temp_dir    = CFP.TEMP_DIR,
                 mode        = CFP.MODE,
                 weights_file = CFP.WEIGHTS_FILE,
                 dnn_datafile = CFP.DNN_DATAFILE,
                 opt             = CFP.OPT,
                 lrate           = CFP.LRATE,
                 sch_decay       = CFP.DECAY,
                 loss_type       = CFP.LOSS,
                 lifetime        = CFP.LIFETIME,
                 )

    fpp.set_output_file(CFP.FILE_OUT)
    fpp.set_compression(CFP.COMPRESSION)
    fpp.set_print(nprint = CFP.NPRINT)

    t0 = time()
    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    nevt = fpp.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt / nevt))

    return nevts, nevt

if __name__ == "__main__":
    OLINDA(sys.argv)
