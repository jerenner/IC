import numpy as np

from typing import List, Union

from ..evm.pmaps           import                     _Peak
from ..evm.pmaps           import              PMTResponses
from ..evm.pmaps           import             SiPMResponses
from . peak_functions      import rebin_times_and_waveforms
from ..core.core_functions import        dict_filter_by_key
from ..types.symbols       import               RebinMethod


def get_even_slices(bins : int, stride : int) -> List[slice]:
    """Create evenly-spaced slices for rebinning.

    Divides a range of ``bins`` into contiguous slices of width ``stride``.

    Parameters
    ----------
    bins : int
        Total number of bins.
    stride : int
        Number of bins per slice.

    Returns
    -------
    List[slice]
        List of slice objects covering the full range.
    """
    new_bins = int(np.ceil(bins / stride))
    return [slice(stride * i, stride * (i + 1)) for i in range(new_bins)]


def get_threshold_slices(pmt_sum   : np.array,
                          threshold :    float) -> List[slice]:
    """Create slices where cumulative PMT charge exceeds a threshold.

    Splits the PMT array into contiguous regions such that each region
    (except possibly the last) accumulates at least ``threshold`` charge.
    A trailing region with charge below the threshold is merged into the
    preceding slice.

    Parameters
    ----------
    pmt_sum : np.array
        Array of PMT charge values.
    threshold : float
        Minimum cumulative charge per slice.

    Returns
    -------
    List[slice]
        List of slice objects defining the regions.
    """
    slices     = []
    last_index = 0
    last_sum   = 0

    def slice_condition(indx, charge, prev_charge):
        if i == len(pmt_sum) - 1 or charge - prev_charge >= threshold:
            return True
        return False

    for i, sum_val in enumerate(np.cumsum(pmt_sum)):
        if slice_condition(i, sum_val, last_sum):
            slices.append(slice(last_index, i + 1))

            last_index = i + 1
            last_sum   = sum_val

    if len(slices) > 1 and pmt_sum[slices[-1]].sum() < threshold:
        slices[-2:] = [slice(slices[-2].start, slices[-1].stop)]
    return slices


def rebin_peak(peak : _Peak, rebin_factor : Union[int, float],
               model=RebinMethod.stride) -> _Peak:
    """
    Rebin a peak either using a set stride of
    a set number of existing bins or a charge
    threshold.

    Parameters
    ----------
    peak : _Peak object
        The Peak to be rebinned
    rebin_factor : int or float
        The rebin stride for RebinMethod.stride (int)
        or the rebin threshold for RebinMethod.threshold (float)
    model : RebinMethod
        The model to be used, stride for set bin width
        threshold for minimum pmt sum charge.

    Returns
    -------
    The rebinned version of the peak
    """
    if model is RebinMethod.threshold:
        slices = get_threshold_slices(peak.pmts.sum_over_sensors,
                                      rebin_factor)
    else:
        if rebin_factor <= 1: return peak
        slices = get_even_slices(peak.times.shape[0], rebin_factor)

    return rebin_peak_to_slices(peak, slices)


def rebin_peak_to_slices(peak : _Peak, slices : List[slice]) -> _Peak:
    """Rebin a peak using custom slices.

    Rebins PMT and SiPM waveforms according to the given slices,
    preserving the peak type (S1 or S2).

    Parameters
    ----------
    peak : _Peak
        Input peak (S1 or S2).
    slices : List[slice]
        Slice objects defining how to rebin the waveforms.

    Returns
    -------
    _Peak
        Rebinned peak of the same type as the input.
    """
    (times,
     widths,
     pmt_wfs) = rebin_times_and_waveforms(peak.times,
                                          peak.bin_widths,
                                          peak.pmts.all_waveforms,
                                          slices = slices)

    pmt_r  = PMTResponses(peak.pmts.ids, pmt_wfs)

    sipm_r = SiPMResponses.build_empty_instance()
    if peak.sipms.ids.size:
        *_, sipms = rebin_times_and_waveforms(peak.times,
                                              peak.bin_widths,
                                              peak.sipms.all_waveforms,
                                              slices = slices)

        sipm_r = SiPMResponses(peak.sipms.ids, sipms)

    return type(peak)(times, widths, pmt_r, sipm_r)


def pmap_event_id_selection(data, event_ids):
    """Filter a pmap dictionary by a list of event IDs.
    Parameters
    ----------
    data      : dict
        pmaps to be filtered.
    event_ids : list
        List of event ids that will be kept.
    Returns
    -------
    filterdata : dict
        Filtered pmaps
    """
    return dict_filter_by_key(lambda x: x in event_ids, data)
