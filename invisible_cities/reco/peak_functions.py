"""
code: peak_functions.py

description: functions related to the pmap creation.

credits: see ic_authors_and_legal.rst in /doc

last revised: @abotas & @gonzaponte. Dec 1st 2017
"""

import numpy        as np

from .. core               import system_of_units as units
from .. evm .ic_containers import ZsWf
from .. evm .pmaps         import S1
from .. evm .pmaps         import S2
from .. evm .pmaps         import PMap
from .. evm .pmaps         import PMTResponses
from .. evm .pmaps         import SiPMResponses


def indices_and_wf_above_threshold(wf, thr):
    """Find waveform samples above a threshold.

    Parameters
    ----------
    wf : np.ndarray
        Input waveform.
    thr : float
        Threshold value.

    Returns
    -------
    ZsWf
        Container with indices and values above threshold.
    """
    indices_above_thr = np.where(wf > thr)[0]
    wf_above_thr      = wf[indices_above_thr]
    return ZsWf(indices_above_thr, wf_above_thr)


def select_wfs_above_time_integrated_thr(wfs, thr):
    """Select waveforms whose time-integrated charge exceeds a threshold.

    Parameters
    ----------
    wfs : np.ndarray
        Waveform array of shape ``(n_sensors, n_samples)``.
    thr : float
        Minimum integrated charge.

    Returns
    -------
    tuple
        ``(selected_ids, selected_wfs)`` with indices and filtered waveforms.
    """
    selected_ids = np.where(np.sum(wfs, axis=1) >= thr)[0]
    selected_wfs = wfs[selected_ids]
    return selected_ids, selected_wfs


def split_in_peaks(indices, stride):
    """Split contiguous index array into separate peaks.

    Splits whenever the gap between consecutive indices exceeds ``stride``.

    Parameters
    ----------
    indices : np.ndarray
        Sorted array of sample indices.
    stride : int
        Maximum gap between consecutive indices to remain in the same peak.

    Returns
    -------
    list of np.ndarray
        Index arrays for each peak.
    """
    where = np.where(np.diff(indices) > stride)[0]
    return np.split(indices, where + 1)


def select_peaks(peaks, time, length, pmt_samp_wid=25*units.ns):
    """Filter peaks to keep only those within valid time and length ranges.

    Parameters
    ----------
    peaks : sequence of np.ndarray
        Peak index arrays.
    time : Interval
        Valid time range.
    length : Interval
        Valid peak length range.
    pmt_samp_wid : float
        PMT sampling width.

    Returns
    -------
    tuple of np.ndarray
        Peaks that satisfy time and length constraints.
    """
    def is_valid(indices):
        return (time  .contains(indices[ 0] * pmt_samp_wid) and
                time  .contains(indices[-1] * pmt_samp_wid) and
                length.contains(indices[-1] + 1 - indices[0]))
    return tuple(filter(is_valid, peaks))


def pick_slice_and_rebin(indices, times, widths,
                         wfs, rebin_stride, pad_zeros=False,
                         sipm_pmt_bin_ratio=40):
    """Extract waveform slice for a peak and rebin.

    Parameters
    ----------
    indices : np.ndarray
        Sample indices belonging to the peak.
    times : np.ndarray
        Time array for all samples.
    widths : np.ndarray
        Bin width array for all samples.
    wfs : np.ndarray
        Waveform array of shape ``(n_sensors, n_samples)``.
    rebin_stride : int
        Number of bins to combine per rebin.
    pad_zeros : bool
        If True, prepend zero samples to align with PMT bins.
    sipm_pmt_bin_ratio : int
        Ratio of SiPM to PMT bin widths.

    Returns
    -------
    tuple
        ``(times, widths, wfs)`` for the rebinned peak.
    """
    slice_ = slice(indices[0], indices[-1] + 1)
    times_  = times [   slice_]
    widths_ = widths[   slice_]
    wfs_    = wfs   [:, slice_]
    if pad_zeros:
        n_miss = indices[0] % sipm_pmt_bin_ratio
        n_wfs  = wfs.shape[0]
        times_  = np.concatenate([np.zeros(        n_miss) ,  times_])
        widths_ = np.concatenate([np.zeros(        n_miss) , widths_])
        wfs_    = np.concatenate([np.zeros((n_wfs, n_miss)),    wfs_], axis=1)
    (times ,
     widths,
     wfs   ) = rebin_times_and_waveforms(times_, widths_, wfs_, rebin_stride)
    return times, widths, wfs


def build_pmt_responses(indices, times, widths, ccwf,
                        pmt_ids, rebin_stride, pad_zeros,
                        sipm_pmt_bin_ratio):
    """Build PMT responses for a single peak.

    Parameters
    ----------
    indices : np.ndarray
        Sample indices for the peak.
    times : np.ndarray
        Time array.
    widths : np.ndarray
        Bin width array.
    ccwf : np.ndarray
        Calibrated, corrected PMT waveforms.
    pmt_ids : np.ndarray
        PMT channel IDs.
    rebin_stride : int
        Rebinning factor.
    pad_zeros : bool
        Whether to pad leading zeros.
    sipm_pmt_bin_ratio : int
        SiPM-to-PMT bin ratio.

    Returns
    -------
    tuple
        ``(times, widths, PMTResponses)`` for the peak.
    """
    (pk_times ,
     pk_widths,
     pmt_wfs  ) = pick_slice_and_rebin(indices, times, widths,
                                       ccwf   , rebin_stride,
                                       pad_zeros = pad_zeros,
                                       sipm_pmt_bin_ratio = sipm_pmt_bin_ratio)
    return pk_times, pk_widths, PMTResponses(pmt_ids, pmt_wfs)


def build_sipm_responses(indices, times, widths,
                         sipm_wfs, rebin_stride, thr_sipm_s2):
    """Build SiPM responses for a single peak.

    Parameters
    ----------
    indices : np.ndarray
        Sample indices for the peak.
    times : np.ndarray
        Time array.
    widths : np.ndarray
        Bin width array.
    sipm_wfs : np.ndarray
        SiPM waveforms.
    rebin_stride : int
        Rebinning factor.
    thr_sipm_s2 : float
        Minimum integrated charge threshold.

    Returns
    -------
    SiPMResponses
        Container with SiPM IDs and filtered waveforms.
    """
    _, _, sipm_wfs_ = pick_slice_and_rebin(indices , times, widths,
                                           sipm_wfs, rebin_stride,
                                           pad_zeros = False)
    (sipm_ids,
     sipm_wfs)   = select_wfs_above_time_integrated_thr(sipm_wfs_,
                                                        thr_sipm_s2)
    return SiPMResponses(sipm_ids, sipm_wfs)


def build_peak(indices, times,
               widths, ccwf, pmt_ids,
               rebin_stride,
               with_sipms, Pk,
               pmt_samp_wid  = 25 * units.ns,
               sipm_samp_wid =  1 * units.mus,
               sipm_wfs      = None,
               thr_sipm_s2   = 0):
    """Build an S1 or S2 peak object from waveform data.

    Parameters
    ----------
    indices : np.ndarray
        Sample indices for the peak.
    times : np.ndarray
        Time array.
    widths : np.ndarray
        Bin width array.
    ccwf : np.ndarray
        Calibrated PMT waveforms.
    pmt_ids : np.ndarray
        PMT channel IDs.
    rebin_stride : int
        Rebinning factor.
    with_sipms : bool
        Whether to include SiPM data.
    Pk : type
        Peak class (``S1`` or ``S2``).
    pmt_samp_wid : float
        PMT sampling width.
    sipm_samp_wid : float
        SiPM sampling width.
    sipm_wfs : np.ndarray or None
        SiPM waveforms.
    thr_sipm_s2 : float
        SiPM charge threshold.

    Returns
    -------
    S1 or S2
        Peak object with PMT and SiPM responses.
    """
    sipm_pmt_bin_ratio = int(sipm_samp_wid/pmt_samp_wid)
    (pk_times ,
     pk_widths,
     pmt_r    ) = build_pmt_responses(indices, times, widths,
                                      ccwf, pmt_ids,
                                      rebin_stride, pad_zeros = with_sipms,
                                      sipm_pmt_bin_ratio = sipm_pmt_bin_ratio)
    if with_sipms:
        sipm_r = build_sipm_responses(indices // sipm_pmt_bin_ratio,
                                       times // sipm_pmt_bin_ratio,
                                       widths * sipm_pmt_bin_ratio,
                                       sipm_wfs,
                                       rebin_stride // sipm_pmt_bin_ratio,
                                       thr_sipm_s2)
    else:
        sipm_r = SiPMResponses.build_empty_instance()

    return Pk(pk_times, pk_widths, pmt_r, sipm_r)


def find_peaks(ccwfs, index,
               time, length,
               stride, rebin_stride,
               Pk, pmt_ids,
               pmt_samp_wid = 25*units.ns,
               sipm_samp_wid = 1*units.mus,
               sipm_wfs=None, thr_sipm_s2=0):
    """Find and build peaks from calibrated waveforms.

    Parameters
    ----------
    ccwfs : array-like
        Calibrated PMT waveforms.
    index : np.ndarray
        Indices of samples above threshold.
    time : Interval
        Valid time range.
    length : Interval
        Valid peak length range.
    stride : int
        Maximum gap between indices for same peak.
    rebin_stride : int
        Rebinning factor.
    Pk : type
        Peak class (``S1`` or ``S2``).
    pmt_ids : np.ndarray
        PMT channel IDs.
    pmt_samp_wid : float
        PMT sampling width.
    sipm_samp_wid : float
        SiPM sampling width.
    sipm_wfs : np.ndarray or None
        SiPM waveforms.
    thr_sipm_s2 : float
        SiPM charge threshold.

    Returns
    -------
    list
        List of peak objects.
    """
    ccwfs = np.array(ccwfs, ndmin=2)

    peaks           = []
    times           = np.arange     (ccwfs.shape[1]) * pmt_samp_wid
    widths          = np.full       (ccwfs.shape[1],   pmt_samp_wid)
    indices_split   = split_in_peaks(index, stride)
    selected_splits = select_peaks  (indices_split, time, length, pmt_samp_wid)
    with_sipms      = Pk is S2 and sipm_wfs is not None

    for indices in selected_splits:
        pk = build_peak(indices, times,
                        widths, ccwfs, pmt_ids,
                        rebin_stride,
                        with_sipms, Pk,
                        pmt_samp_wid, sipm_samp_wid,
                        sipm_wfs, thr_sipm_s2)
        peaks.append(pk)
    return peaks


def get_pmap(ccwf, s1_indx, s2_indx, sipm_zs_wf,
             s1_params, s2_params, thr_sipm_s2, pmt_ids,
             pmt_samp_wid, sipm_samp_wid):
    """Build a PMap containing S1 and S2 peaks from waveform data.

    Parameters
    ----------
    ccwf : np.ndarray
        Calibrated PMT waveforms.
    s1_indx : np.ndarray
        Indices of S1 samples above threshold.
    s2_indx : np.ndarray
        Indices of S2 samples above threshold.
    sipm_zs_wf : np.ndarray
        SiPM waveforms.
    s1_params : dict
        Parameters for S1 peak finding.
    s2_params : dict
        Parameters for S2 peak finding.
    thr_sipm_s2 : float
        SiPM charge threshold for S2.
    pmt_ids : np.ndarray
        PMT channel IDs.
    pmt_samp_wid : float
        PMT sampling width.
    sipm_samp_wid : float
        SiPM sampling width.

    Returns
    -------
    PMap
        Container with S1 and S2 peaks.
    """
    return PMap(find_peaks(ccwf, s1_indx, Pk=S1, pmt_ids=pmt_ids,
                           pmt_samp_wid=pmt_samp_wid,
                           **s1_params),
                 find_peaks(ccwf, s2_indx, Pk=S2, pmt_ids=pmt_ids,
                            sipm_wfs      = sipm_zs_wf,
                            thr_sipm_s2   = thr_sipm_s2,
                            pmt_samp_wid  = pmt_samp_wid,
                            sipm_samp_wid = sipm_samp_wid,
                            **s2_params))


def rebin_times_and_waveforms(times, widths, waveforms,
                              rebin_stride=2, slices=None):
    """Rebin time samples and waveforms by combining adjacent bins.

    Combines ``rebin_stride`` consecutive bins, summing waveforms and
    computing weighted average times.

    Parameters
    ----------
    times : np.ndarray
        Time array.
    widths : np.ndarray
        Bin width array.
    waveforms : np.ndarray
        Waveform array of shape ``(n_sensors, n_samples)``.
    rebin_stride : int
        Number of bins to combine.
    slices : list of slice or None
        Pre-computed slices, or None to compute automatically.

    Returns
    -------
    tuple
        ``(rebinned_times, rebinned_widths, rebinned_waveforms)``.
    """
    if rebin_stride < 2: return times, widths, waveforms

    if slices is None:
        n_bins = int(np.ceil(len(times) / rebin_stride))
        reb    = rebin_stride
        slices = [slice(reb * i, reb * (i + 1)) for i in range(n_bins)]
    n_sensors = waveforms.shape[0]

    rebinned_times  = np.zeros(            len(slices) )
    rebinned_widths = np.zeros(            len(slices) )
    rebinned_wfs    = np.zeros((n_sensors, len(slices)))

    for i, sl in enumerate(slices):
        t = times    [   sl]
        e = waveforms[:, sl]
        ## Weight with the charge sum per slice
        ## if positive and unweighted if all
        ## negative.
        s = np.sum(e, axis=0).clip(0)
        w = s if np.any(s) else None
        rebinned_times [   i] = np.average(t, weights=w)
        rebinned_widths[   i] = np.sum    (  widths[sl])
        rebinned_wfs   [:, i] = np.sum    (e,    axis=1)
    return rebinned_times, rebinned_widths, rebinned_wfs
