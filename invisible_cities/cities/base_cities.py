"""
This module defines base classes for the IC cities. The classes are:
City: Handles input and output files, compression, and access to data base
DeconvolutionCity: A City that performs deconvolution of the PMT RWFs
CalibratedCity: A DeconvolutionCity that perform the calibrated sum of the
                PMTs and computes the calibrated signals in the SiPM plane.
PmapCity: A CalibratedCity that computes S1, S2 and S2Si that togehter
          constitute a PMAP.
SensorResponseCity: A city that describes sensor response
DNNCity: A city equipped to construct and train and/or evaluate a DNN

Authors: J.J. Gomez-Cadenas and J. Generowicz.
Feburary, 2017.
"""

import sys
from argparse import Namespace
from operator import attrgetter
from glob     import glob
from time     import time
from os.path  import expandvars

import numpy as np
import keras
import tables as tb

from .. core.configure         import configure
from .. core.exceptions        import NoInputFiles
from .. core.exceptions        import NoOutputFile
from .. core.ic_types          import minmax
from .. core.system_of_units_c import units
from .. core                   import fit_functions        as fitf

from .. database import load_db

from ..io                 import pmap_io          as pio

from ..reco               import peak_functions_c as cpf
from ..reco               import peak_functions   as pf
from ..reco               import pmaps_functions  as pmp
from ..reco               import dst_functions    as dstf
from ..reco               import tbl_functions    as tbf
from ..reco               import wfm_functions    as wfm
from ..reco               import tbl_functions    as tbl
from ..reco.params        import S12Params
from ..reco.event_model   import SensorParams
from ..reco.nh5           import DECONV_PARAM
from ..reco.corrections   import Correction
from ..reco.corrections   import Fcorrection
from ..reco.corrections   import LifetimeCorrection
from ..reco.xy_algorithms import find_algorithm

from ..sierpe             import blr
from ..sierpe             import fee as FE


def merge_two_dicts(a,b):
    return {**a, **b}


class City:
    """Base class for all cities.
       An IC city consumes data stored in the input_files and produce new data
       which is stored in the output_file. In addition to setting input and
       output files, the base class sets the print frequency and accesses
       the data base, storing as attributed several calibration coefficients

     """

    def __init__(self, **kwds):
        conf = Namespace(**kwds)

        self.conf = conf

        if not hasattr(conf, 'files_in'):
            raise NoInputFiles

        if not hasattr(conf, 'file_out'):
            raise NoOutputFile


        self.input_files = sorted(glob(expandvars(conf.files_in)))
        self.output_file =             expandvars(conf.file_out)
        self.compression = conf.compression
        self.run_number  = conf.run_number
        self.nprint      = conf.nprint  # default print frequency
        self.nmax        = conf.nmax

        self.set_up_database()

    @classmethod
    def drive(cls, argv):
        conf = configure(argv)
        opts = conf.as_namespace
        if not opts.hide_config:
            conf.display()
        if opts.print_config_only:
            return
        instance = cls(**conf.as_dict)
        instance.go()

    def go(self):
        t0 = time()
        nevt_in, nevt_out = self.run()
        t1 = time()
        dt = t1 - t0
        print("run {} evts in {} s, time/event = {}".format(nevt_in, dt, dt/nevt_in))

    def set_up_database(self):
        DataPMT       = load_db.DataPMT (self.run_number)
        DataSiPM      = load_db.DataSiPM(self.run_number)
        self.det_geo  = load_db.DetectorGeo()
        self.DataPMT  = DataPMT
        self.DataSiPM = DataSiPM

        self.xs              = DataSiPM.X.values
        self.ys              = DataSiPM.Y.values
        self.pmt_active      = np.nonzero(DataPMT.Active.values)[0].tolist()
        self.adc_to_pes      = abs(DataPMT.adc_to_pes.values).astype(np.double)
        self.sipm_adc_to_pes = DataSiPM.adc_to_pes.values    .astype(np.double)
        self.coeff_c         = DataPMT.coeff_c.values        .astype(np.double)
        self.coeff_blr       = DataPMT.coeff_blr.values      .astype(np.double)
        self.noise_rms       = DataPMT.noise_rms.values      .astype(np.double)

    @property
    def monte_carlo(self):
        return self.run_number <= 0

    def conditional_print(self, evt, n_events_tot):
        if n_events_tot % self.nprint == 0:
            print('event in file = {}, total = {}'
                  .format(evt, n_events_tot))

    def max_events_reached(self, n_events_in):
        if self.nmax < 0:
            return False
        if n_events_in == self.nmax:
            print('reached max nof of events (= {})'
                  .format(self.nmax))
            return True
        return False


    def display_IO_info(self):
        print("""
                 {} will run a max of {} events
                 Input Files = {}
                 Output File = {}
                          """.format(self.__class__.__name__,
                                     self.nmax, self.input_files, self.output_file))

    @staticmethod
    def get_rwf_vectors(h5in):
        "Return RWF vectors and sensor data."
        return tbl.get_rwf_vectors(h5in)

    @staticmethod
    def get_rd_vectors(h5in):
        "Return MC RD vectors and sensor data."
        return tbl.get_rd_vectors(h5in)

    def get_sensor_rd_params(self, filename):
        """Return MCRD sensors.
           pmtrd.shape returns the length of the RD PMT vector
           (1 ns bins). PMTWL_FEE is the length of the RWF vector
           obtained by divinding the RD PMT vector and the sample
           time of the electronics (25 ns). """
        with tb.open_file(filename, "r") as h5in:
            #pmtrd, sipmrd = self._get_rd(h5in)
            _, pmtrd, sipmrd = tbl.get_rd_vectors(h5in)
            _, NPMT,   PMTWL = pmtrd .shape
            _, NSIPM, SIPMWL = sipmrd.shape
            PMTWL_FEE = int(PMTWL // self.FE_t_sample)
            return SensorParams(NPMT, PMTWL_FEE, NSIPM, SIPMWL)

    @staticmethod
    def get_sensor_params(filename):
        return tbl.get_sensor_params(filename)

    @staticmethod
    def get_run_and_event_info(h5in):
        return h5in.root.Run.events


    @staticmethod
    def event_and_timestamp(evt, events_info):
        return events_info[evt]

    @staticmethod
    def event_number_from_input_file_name(filename):
        return tbf.event_number_from_input_file_name(filename)

    def _get_rwf(self, h5in):
        "Return raw waveforms for SIPM and PMT data"
        return (h5in.root.RD.pmtrwf,
                h5in.root.RD.sipmrwf,
                h5in.root.RD.pmtblr)

    def _get_rd(self, h5in):
        "Return (MC) raw data waveforms for SIPM and PMT data"
        return (h5in.root.pmtrd,
                h5in.root.sipmrd)


class SensorResponseCity(City):
    """A SensorResponseCity city extends the City base class adding the
       response (Monte Carlo simulation) of the energy plane and
       tracking plane sensors (PMTs and SiPMs).
    """

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.sipm_noise_cut = self.conf.sipm_noise_cut

    def simulate_sipm_response(self, event, sipmrd,
                               sipms_noise_sampler):
        """Add noise with the NoiseSampler class and return
        the noisy waveform (in pes)."""
        # add noise (in PES) to true waveform
        dataSiPM = sipmrd[event] + sipms_noise_sampler.Sample()
        # return total signal in adc counts
        return wfm.to_adc(dataSiPM, self.sipm_adc_to_pes)

    def simulate_pmt_response(self, event, pmtrd):
        """ Full simulation of the energy plane response
        Input:
         1) extensible array pmtrd
         2) event_number

        returns:
        array of raw waveforms (RWF) obtained by convoluting pmtrd with the PMT
        front end electronics (LPF, HPF filters)
        array of BLR waveforms (only decimation)
        """
        # Single Photoelectron class
        spe = FE.SPE()
        # FEE, with noise PMT
        fee  = FE.FEE(noise_FEEPMB_rms=FE.NOISE_I, noise_DAQ_rms=FE.NOISE_DAQ)
        NPMT = pmtrd.shape[1]
        RWF  = []
        BLRX = []

        for pmt in range(NPMT):
            # normalize calibration constants from DB to MC value
            cc = self.adc_to_pes[pmt] / FE.ADC_TO_PES
            # signal_i in current units
            signal_i = FE.spe_pulse_from_vector(spe, pmtrd[event, pmt])
            # Decimate (DAQ decimation)
            signal_d = FE.daq_decimator(FE.f_mc, FE.f_sample, signal_i)
            # Effect of FEE and transform to adc counts
            signal_fee = FE.signal_v_fee(fee, signal_d, pmt) * FE.v_to_adc()
            # add noise daq
            signal_daq = cc * FE.noise_adc(fee, signal_fee)
            # signal blr is just pure MC decimated by adc in adc counts
            signal_blr = cc * FE.signal_v_lpf(fee, signal_d) * FE.v_to_adc()
            # raw waveform stored with negative sign and offset
            RWF.append(FE.OFFSET - signal_daq)
            # blr waveform stored with positive sign and no offset
            BLRX.append(signal_blr)
        return np.array(RWF), np.array(BLRX)

    @property
    def FE_t_sample(self):
        return FE.t_sample


class DeconvolutionCity(City):
    """A Deconvolution city extends the City base class adding the
       deconvolution step, which transforms RWF into CWF.
       The parameters of the deconvolution are the number of samples
       used to compute the baseline (n_baseline) and the threshold to
       thr_trigger in the rising signal (thr_trigger)
    """

    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf

        # BLR parameters
        self.n_baseline            = conf.n_baseline
        self.thr_trigger           = conf.thr_trigger
        self.acum_discharge_length = conf.acum_discharge_length

    def write_deconv_params(self, ofile):
        group = ofile.create_group(ofile.root, "DeconvParams")

        table = ofile.create_table(group,
                                   "DeconvParams",
                                   DECONV_PARAM,
                                   "deconvolution parameters",
                                   tbf.filters(self.compression))

        row = table.row
        row["N_BASELINE"]            = self.n_baseline
        row["THR_TRIGGER"]           = self.thr_trigger
        row["ACUM_DISCHARGE_LENGTH"] = self.acum_discharge_length
        table.flush()

    def deconv_pmt(self, RWF):
        """Deconvolve the RWF of the PMTs"""
        return blr.deconv_pmt(RWF,
                              self.coeff_c,
                              self.coeff_blr,
                              pmt_active            = self.pmt_active,
                              n_baseline            = self.n_baseline,
                              thr_trigger           = self.thr_trigger,
                              acum_discharge_length = self.acum_discharge_length)


class CalibratedCity(DeconvolutionCity):
    """A calibrated city extends a DeconvCity, performing two actions.
       1. Compute the calibrated sum of PMTs, in two flavours:
          a) csum: PMTs waveforms are equalized to photoelectrons (pes) and
             added
          b) csum_mau: waveforms are equalized to photoelectrons;
             compute a MAU that follows baseline and add PMT samples above
             MAU + threshold
       2. Compute the calibrated signal in the SiPMs:
          a) equalize to pes;
          b) compute a MAU that follows baseline and keep samples above
             MAU + threshold.
       """

    def __init__(self, **kwds):

        super().__init__(**kwds)
        conf = self.conf
        # Parameters of the PMT csum.
        self.n_MAU       = conf.n_mau
        self.thr_MAU     = conf.thr_mau
        self.thr_csum_s1 = conf.thr_csum_s1
        self.thr_csum_s2 = conf.thr_csum_s2

        # Parameters of the SiPM signal
        self.n_MAU_sipm = conf.n_mau_sipm
        self.  thr_sipm = conf.  thr_sipm

    def calibrated_pmt_sum(self, CWF):
        """Return the csum and csum_mau calibrated sums."""
        return cpf.calibrated_pmt_sum(CWF,
                                      self.adc_to_pes,
                                      pmt_active = self.pmt_active,
                                           n_MAU = self.  n_MAU   ,
                                         thr_MAU = self.thr_MAU   )

    def csum_zs(self, csum, threshold):
        """Zero Suppression over csum"""
        return cpf.wfzs(csum, threshold=threshold)

    def calibrated_signal_sipm(self, SiRWF):
        """Return the calibrated signal in the SiPMs."""
        return cpf.signal_sipm(SiRWF,
                               self.sipm_adc_to_pes,
                               thr   = self.  thr_sipm,
                               n_MAU = self.n_MAU_sipm)


class PmapCity(CalibratedCity):
    """A PMAP city extends a CalibratedCity, computing the S1, S2 and S2Si
       objects that togehter constitute a PMAP.

    """

    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf
        self.s1_params = S12Params(time = minmax(min   = conf.s1_tmin,
                                                 max   = conf.s1_tmax),
                                   stride              = conf.s1_stride,
                                   length = minmax(min = conf.s1_lmin,
                                                   max = conf.s1_lmax),
                                   rebin               = False)

        self.s2_params = S12Params(time = minmax(min   = conf.s2_tmin,
                                                 max   = conf.s2_tmax),
                                   stride              = conf.s2_stride,
                                   length = minmax(min = conf.s2_lmin,
                                                   max = conf.s2_lmax),
                                   rebin               = True)

        self.thr_sipm_s2 = conf.thr_sipm_s2

    def pmaps(self, s1_indx, s2_indx, csum, sipmzs):
        S1, S2 = self.find_S12(csum, s1_indx, s2_indx)
        S1     = self.correct_S1_ene(S1, csum)
        Si     = self.find_S2Si(S2, sipmzs)
        return S1, S2, Si

    def find_S12(self, csum, s1_indx, s2_indx):
        """Return S1 and S2."""
        S1 = cpf.find_S12(csum,
                          s1_indx,
                          **self.s1_params._asdict())

        S2 = cpf.find_S12(csum,
                          s2_indx,
                          **self.s2_params._asdict())
        return S1, S2

    def correct_S1_ene(self, S1, csum):
        return cpf.correct_S1_ene(S1, csum)

    def find_S2Si(self, S2, sipmzs):
        """Return S2Si."""
        SIPM = cpf.select_sipm(sipmzs)
        S2Si = pf.sipm_s2_dict(SIPM, S2, thr = self.thr_sipm_s2)
        return pio.S2Si(S2Si)

    def check_s1s2_params(self):
        if (not self.s1_params) or (not self.s2_params):
            raise IOError('must set S1/S2 parameters before running')


class MapCity(City):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        conf = self.conf
        required_names = 'lifetime u_lifetime xmin xmax ymin ymax xbins ybins'.split()
        lifetime, u_lifetime, xmin, xmax, ymin, ymax, xbins, ybins = attrgetter(*required_names)(conf)

        self.  _lifetimes = [lifetime]   if not np.shape(  lifetime) else   lifetime
        self._u_lifetimes = [u_lifetime] if not np.shape(u_lifetime) else u_lifetime
        self._lifetime_corrections = tuple(map(LifetimeCorrection, self._lifetimes, self._u_lifetimes))

        xmin = self.det_geo.XMIN[0] if xmin is None else xmin
        xmax = self.det_geo.XMAX[0] if xmax is None else xmax
        ymin = self.det_geo.YMIN[0] if ymin is None else ymin
        ymax = self.det_geo.YMAX[0] if ymax is None else ymax

        self._xbins  = xbins
        self._ybins  = ybins
        self._xrange = xmin, xmax
        self._yrange = ymin, ymax

    def xy_correction(self, X, Y, E):
        xs, ys, es, us = \
        fitf.profileXY(X, Y, E, self._xbins, self._ybins, self._xrange, self._yrange)

        norm_index = xs.size//2, ys.size//2
        return Correction((xs, ys), es, us, norm_strategy="index", index=norm_index)

    def xy_statistics(self, X, Y):
        return np.histogram2d(X, Y, (self._xbins, self._ybins), (self._xrange, self._yrange))

class HitCollectionCity(City):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf  = self.conf
        self.rebin          = conf.rebin
        self.reco_algorithm = find_algorithm(conf.reco_algorithm)

    def rebin_s2(self, S2, Si):
        if self.rebin <= 1:
            return S2, Si

        S2_rebin = {}
        Si_rebin = {}
        for peak in S2:
            t, e, sipms = cpf.rebin_S2(S2[peak][0], S2[peak][1], Si[peak], self.rebin)
            S2_rebin[peak] = Peak(t, e)
            Si_rebin[peak] = sipms
        return S2_rebin, Si_rebin

    def split_energy(self, e, clusters):
        if len(clusters) == 1:
            return [e]
        qs = np.array([c.Q for c in clusters])
        return e * qs / np.sum(qs)

    def compute_xy_position(self, si, slice_no):
        si_slice = pmp.select_si_slice(si, slice_no)
        IDs, Qs  = pmp.integrate_sipm_charges_in_peak(si)
        xs, ys   = self.xs[IDs], self.ys[IDs]
        return self.reco_algorithm(np.stack((xs, ys)).T, Qs)

class DNNCity(City):
    """A DNN city extends the City base class, adding functionality
        common to all cities that will perform DNN-based analysis.
        lrate     - the learning rate
        sch_decay - the decay per iteration
        model     - the Keras model describing the network
        opt       - the optimizer type to be used in training
    """

    def __init__(self,
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 temp_dir     = 'database/test_data',
                 compression = 'ZLIB4',
                 nprint      = 10000,
                 # Parameters added at this level
                 weights_file = '',
                 dnn_datafile = '',
                 lrate       = 0.01,
                 sch_decay   = 0.001,
                 loss_type   = 'mse',
                 opt         = 'nadam',
                 mode        = 'eval'):

        City.__init__(self,
                      run_number  = run_number,
                      files_in    = files_in,
                      file_out    = file_out,
                      compression = compression,
                      nprint      = nprint)

        self.temp_dir     = temp_dir
        self.weights_file = weights_file
        self.dnn_datafile = dnn_datafile
        self.lrate        = lrate
        self.sch_decay    = sch_decay
        self.loss_type    = loss_type
        self.mode         = mode
        self.opt          = opt

        self.model        = None

        # load the SiPM (x,y) values
        DataSensor = load_db.DataSiPM(0)
        xs = DataSensor.X.values
        ys = DataSensor.Y.values

        self.id_to_coords = {}
        for ID, x, y in zip(range(1792), xs, ys):
            self.id_to_coords[np.int32(ID)] = np.array([x, y])

    def build_XY(self):
        """Builds the arrays X_in and Y_in. To be implemented in a subclass."""
        raise NotImplementedError

    def build_model(self):
        """Builds the Keras model. To be implemented in a subclass."""
        raise NotImplementedError

    def train(self,nepochs=10,nbatch=64,fval=0.05):
        """Run the training step for the model that was setup"""
        raise NotImplementedError

    def evaluate(self):
        """Evaluates the input data using the trained DNN"""
        raise NotImplementedError

    config_file_format = City.config_file_format + """
    # paths and input/output
    TEMP_DIR {TEMP_DIR}
    WEIGHTS_FILE {WEIGHTS_FILE}
    DNN_DATAFILE {DNN_DATAFILE}

    # DNNCity
    RUN_NUMBER {RUN_NUMBER}
    MODE {MODE}
    OPT {OPT}
    LRATE {LRATE}
    DECAY {DECAY}
    LOSS {LOSS}

    # run
    NEVENTS {NEVENTS}
    RUN_ALL {RUN_ALL}"""

    config_file_format = dedent(config_file_format)

    default_config = merge_two_dicts(
        City.default_config,
        dict(RUN_NUMBER   = 0,
             TEMP_DIR     = '$ICDIR/database/test_data',
             WEIGHTS_FILE = None,
             DNN_DATAFILE = None,
             MODE         = 'test',
             OPT          = 'nadam',
             LRATE        = 0.001,
             DECAY        = 0.001,
             LOSS         = 'mse',
             NEVENTS      = 5,
             RUN_ALL      = False))


class KerasDNNCity(DNNCity):
    """A KerasDNNCity extends the DNNCity base class and implements key
       functions using the Keras interface.
    """
    def __init__(self,
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 temp_dir     = 'database/test_data',
                 compression = 'ZLIB4',
                 nprint      = 10000,
                 weights_file = '',
                 dnn_datafile = '',
                 lrate       = 0.01,
                 sch_decay   = 0.001,
                 loss_type   = 'mse',
                 opt         = 'nadam',
                 mode        = 'eval'):

        DNNCity.__init__(self,
                      run_number      = run_number,
                      files_in        = files_in,
                      file_out        = file_out,
                      temp_dir        = temp_dir,
                      compression     = compression,
                      nprint          = nprint,
                      weights_file    = weights_file,
                      dnn_datafile    = dnn_datafile,
                      lrate           = lrate,
                      sch_decay       = sch_decay,
                      loss_type       = loss_type,
                      opt             = opt,
                      mode            = mode)

        if(opt == 'nadam'):
            self.optimizer = keras.optimizers.Nadam(lr=self.lrate, beta_1=0.9,
                                              beta_2=0.999, epsilon=1e-08,
                                              schedule_decay=self.sch_decay)
        else:
            print("Setting SGD opt")
            self.optimizer = keras.optimizers.SGD(lr=self.lrate,
                                            decay=self.sch_decay)

    def build_XY(self):
        """Builds the arrays X_in and Y_in. To be implemented in a subclass."""
        raise NotImplementedError

    def build_model(self):
        """Builds the Keras model. To be implemented in a subclass."""
        raise NotImplementedError

    def train(self,nepochs=10,nbatch=64,fval=0.05):
        """Run the training step for the model that was setup"""

        # set up the callbacks
        file_lbl = "{epoch:02d}-{loss:.4f}"
        filepath="{0}/weights-{1}.h5".format(self.temp_dir,file_lbl)
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        # train the model
        self.model.fit(self.X_in, self.Y_in, nb_epoch=nepochs, batch_size=nbatch, callbacks=callbacks_list, validation_split=fval, shuffle=True)

    def evaluate(self):
        """Evaluates the input data using the trained DNN"""

        prediction = self.model.predict(self.X_in,verbose=2)

        # print test results if true information is given
        if(len(self.Y_in) == len(self.X_in)):
            loss_and_metrics = self.model.evaluate(self.X_in, self.Y_in)

        return prediction
