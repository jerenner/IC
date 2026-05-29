"""
Waveform Generator

Generates toy PMT and SiPM waveforms
"""

import numpy as np

from .       import fee as FE
from .  fee  import FEE

from typing import NamedTuple


class Point(NamedTuple):
    x : float
    y : float


class WfmPar(NamedTuple):
    w_type    : str
    t_tot     : int
    t_pre     : int
    t_rise_s2 : int
    t_flat_s2 : int
    noise     : int
    q_s2      : int


class WaveformPmt(NamedTuple):
    blr : np.ndarray
    fee : np.ndarray


def square_waveform(wp : WfmPar) -> np.ndarray:
    """Generate a trapezoidal S2-like waveform with noise.

    Parameters
    ----------
    wp : WfmPar
        Waveform parameters (total length, pre-signal time, rise time, flat time, noise, charge).

    Returns
    -------
    np.ndarray
        1-D array of waveform samples with trapezoidal signal and Gaussian noise.
    """
    t_s2  = 2 * wp.t_rise_s2 + wp.t_flat_s2
    t_pos = wp.t_tot - wp.t_pre - t_s2
    assert t_s2 + wp.t_pre < wp.t_tot

    def wf_noise(length : int) -> np.ndarray:
        """Generate Gaussian noise samples."""
        return wp.noise * np.random.randn(length)

    def line(x : float, p1 : Point, p2 : Point) -> float:
        """Evaluate linear interpolation between two points."""
        def coef(p1, p2):
            """Compute slope and intercept for line through two points."""
            b = (p2.y - p1.y) / (p2.x - p1.x)
            a = (p1.y * p2.x - p2.y * p1.x) / (p2.x - p1.x)
            return a, b
        a, b = coef(p1, p2)
        return a + b * x

    t        = np.arange(wp.t_tot)
    sgn_pre  = np.zeros (wp.t_tot)
    sgn_s2   = np.zeros (wp.t_tot)
    sgn_post = np.zeros (wp.t_tot)

    # waveform pre (before signal raises)
    f = 0
    l = wp.t_pre
    sgn_pre[f:l] = wf_noise(wp.t_pre)

    # signal raise
    f = l
    l = f + wp.t_rise_s2
    p1 = Point(t[f], 0)
    p2 = Point(t[l], wp.q_s2)
    sgn_s2[f:l+1] = line(t[f:l+1], p1, p2) + wf_noise(l+1-f)

    # signal flattens
    f = l
    l = f + wp.t_flat_s2
    sgn_s2[f:l] = wp.q_s2  + wf_noise(l-f)


    # signal decreases
    f = l
    l = f + wp.t_rise_s2
    p1 = Point(t[f], wp.q_s2)
    p2 = Point(t[l], 0)
    sgn_s2[f:l+1] = line(t[f:l+1], p1, p2)  + wf_noise(l+1-f)

    # signal post (after signal has decreases)
    f = l
    l = wp.t_tot
    sgn_post[f:l] = wp.noise * np.random.randn(t_pos)

    return sgn_pre + sgn_s2 + sgn_post


def sawteeth_waveform(wfp: WfmPar) -> np.ndarray:
    """Generate a multi-peak sawtooth waveform for testing.

    Parameters
    ----------
    wfp : WfmPar
        Waveform parameters defining timing and amplitude.

    Returns
    -------
    np.ndarray
        1-D array of concatenated linear ramps forming three trapezoidal peaks.
    """
    t_tot     = wfp.t_tot
    t_pre     = wfp.t_pre
    t_rise_s2 = wfp.t_rise_s2
    t_flat_s2 = wfp.t_flat_s2
    q_s2      = wfp.q_s2

    signal_i = np.concatenate((np.zeros(t_pre),
                               np.linspace(0,q_s2,t_rise_s2),
                               np.linspace(q_s2,q_s2,t_flat_s2),
                               np.linspace(q_s2,0,t_rise_s2),
                               np.linspace(0,q_s2,t_rise_s2),
                               np.linspace(q_s2,q_s2,t_flat_s2),
                               np.linspace(q_s2,0,t_rise_s2),
                               np.linspace(0,q_s2,t_rise_s2),
                               np.linspace(q_s2,q_s2,t_flat_s2),
                               np.linspace(q_s2,0,t_rise_s2),
                               np.zeros(t_tot - t_pre - 3* (t_rise_s2 + 2 * t_flat_s2))),
                                        axis=0)
    return signal_i


def simulate_pmt_response(fee : FEE, wf : np.ndarray) -> WaveformPmt:
    """Simulate PMT response by processing a waveform through FEE and decimation.

    Parameters
    ----------
    fee : FEE
        Front-end electronics configuration object.
    wf : np.ndarray
        Input waveform in photoelectrons.

    Returns
    -------
    WaveformPmt
        Named tuple with (blr, fee) waveform arrays in ADC counts.
    """
    spe        = FE.SPE()                                           # FEE, with noise PMT
    signal_i   = FE.spe_pulse_from_vector(spe, wf)                  # signal_i in current units
    signal_d   = FE.daq_decimator(FE.f_mc, FE.f_sample, signal_i)   # Decimate (DAQ decimation)
    signal_fee = FE.signal_v_fee(fee, signal_d, -1) * FE.v_to_adc() # Effect of FEE and transform to adc counts
    signal_blr = FE.signal_v_lpf(fee, signal_d) * FE.v_to_adc()     # signal blr is just pure MC decimated by adc in adc counts

    return WaveformPmt(signal_blr.astype(int), signal_fee.astype(int))


def waveform_generator(fee : FEE, wfp : WfmPar, nsensors=5, pedestal=1024,
                       random_t0=True) -> WaveformPmt:
    """
        Generate PMT-like (e.g, negative swing) and SiPM-like (positive)
        waveforms.

    """
    wfm_blr  = np.zeros((nsensors,  int(wfp.t_tot / FE.t_sample)), dtype=int)
    wfm_fee  = np.zeros((nsensors,  int(wfp.t_tot / FE.t_sample)), dtype=int)
    ped  = np.ones( int(wfp.t_tot / FE.t_sample)) * pedestal

    for i in range(nsensors):
        if random_t0:
            tpre         = np.random.randint(int(wfp.t_tot/3))
            wfp          = WfmPar(w_type    = wfp.w_type,
                             t_tot     = wfp.t_tot,
                             t_pre     = tpre,
                             t_rise_s2 = wfp.t_rise_s2,
                             t_flat_s2 = wfp.t_flat_s2,
                             noise     = wfp.noise,
                             q_s2      = wfp.q_s2)

        if wfp.w_type == 'square' :
            pmtwf        =  simulate_pmt_response(fee, square_waveform(wfp))
        else:
            pmtwf        =  simulate_pmt_response(fee, sawteeth_waveform(wfp))
        wfm_fee [i, :] = pmtwf.fee  + ped
        wfm_blr [i, :] = pmtwf.blr  + ped

    return WaveformPmt(wfm_blr, wfm_fee)


def deconv_simple(wfm, coef):
    """
    Deconvolution of the fine-grained fee signal (no DAQ)
    no noise
    using true start and end of signals
    """

    signal = wfm - np.mean(wfm)
    acum = np.zeros(len(signal))

    acum[0] = coef * signal[0]
    for n in np.arange(1, len(signal)):
        acum[n] = acum[n-1] + signal[n]

    signal_r = signal + coef*acum

    return signal_r


def deconv_pmts(wfms, coef):
    """Apply simple deconvolution to a batch of PMT waveforms.

    Parameters
    ----------
    wfms : array-like
        2-D array of waveforms (nsensors, nsamples).
    coef : float
        Deconvolution coefficient.

    Returns
    -------
    np.ndarray
        Deconvolved waveforms with the same shape as input.
    """
    return np.array([deconv_simple(wfm, coef) for wfm in wfms])
