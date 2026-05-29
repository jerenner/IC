import numpy        as np
import scipy.signal as signal
import scipy.stats  as stats

from functools import wraps

from .. core.core_functions import to_col_vector

from ..types.symbols import BlsMode
from ..types.symbols import SiPMCalibMode


def mask_sensors(wfs, active):
    """Zero out waveforms for inactive sensors.

    Parameters
    ----------
    wfs : np.ndarray
        Waveform array of shape ``(n_sensors, n_samples)``.
    active : np.ndarray
        Boolean array of shape ``(n_sensors,)`` indicating active sensors.

    Returns
    -------
    np.ndarray
        Waveforms with inactive sensors zeroed.
    """
    return wfs * active.astype(wfs.dtype).reshape(active.size, 1)


def scipy_mode(x, axis=0):
    """
    Scipy implementation of the mode (runs very slow).
    Returns a column vector.
    """
    m, c = stats.mode(x, axis=axis, keepdims=True)
    return m


def mode(wfs, axis=0):
    """
    A fast calculation of the mode: it runs 10 times
    faster than the SciPy version but only applies to
    positive waveforms.
    """
    def wf_mode(wf):
        positive = wf > 0
        return np.bincount(wf[positive]).argmax() if np.count_nonzero(positive) else 0
    return np.apply_along_axis(wf_mode, axis, wfs).astype(float)


def zero_masked(fn):
    """
    protection for mean and median
    so that we get the correct answer in case of
    zero suppressed data
    """
    @wraps(fn)
    def proxy(wfs, *args, **kwds):
        mask_wfs = np.ma.masked_where(wfs == 0, wfs)
        return fn(mask_wfs, *args, **kwds).filled(0)
    proxy.__doc__ = "Masked version to protect ZS mode \n\n" + proxy.__doc__
    return proxy

median = zero_masked(np.ma.median)
mean   = zero_masked(np.ma.mean)


def means  (wfs):
    """Compute per-sensor mean baseline, masked for zero-suppressed data."""
    return to_col_vector(mean  (wfs, axis=1))

def medians(wfs):
    """Compute per-sensor median baseline, masked for zero-suppressed data."""
    return to_col_vector(median(wfs, axis=1))

def modes  (wfs):
    """Compute per-sensor mode baseline."""
    return to_col_vector(mode  (wfs, axis=1))


def subtract_baseline(wfs, *, bls_mode=BlsMode.mean):
    """
    Subtract the baseline to all waveforms in the input
    with a specific algorithm.

    Parameters
    ----------
    wfs: np.ndarray with shape (n, m)
        Waveforms with baseline.

    Keyword-only parameters
    -----------------------
    bls_mode: BlsMode
        Algorithm to be used.

    Returns
    -------
    bls: np.ndarray with shape (n, m)
        Baseline-subtracted waveforms.
    """

    if   bls_mode is BlsMode.mean     : return wfs - means     (wfs)
    elif bls_mode is BlsMode.median   : return wfs - medians   (wfs)
    elif bls_mode is BlsMode.mode     : return wfs - modes     (wfs)
    elif bls_mode is BlsMode.scipymode: return wfs - scipy_mode(wfs, axis=1)
    else:
        raise TypeError(f"Unrecognized baseline subtraction option: {bls_mode}")


def calibrate_wfs(wfs, adc_to_pes):
    """
    Convert waveforms in adc to pes. Masked channels
    are ignored.
    """
    adc_to_pes = to_col_vector(adc_to_pes)
    ok         = adc_to_pes > 0
    out        = np.zeros(wfs.shape, dtype=float)
    return np.divide(wfs, adc_to_pes, out=out, where=ok)


def subtract_baseline_and_calibrate(sipm_wfs, adc_to_pes, *, bls_mode=BlsMode.mean):
    """Subtract baseline and convert SiPM waveforms from ADC to photoelectrons.

    Parameters
    ----------
    sipm_wfs : np.ndarray
        SiPM waveforms of shape ``(n_sensors, n_samples)``.
    adc_to_pes : np.ndarray
        ADC-to-photoelectron conversion factors per sensor.
    bls_mode : BlsMode
        Baseline subtraction method.

    Returns
    -------
    np.ndarray
        Calibrated waveforms in photoelectrons.
    """
    bls = subtract_baseline(sipm_wfs, bls_mode=bls_mode)
    return calibrate_wfs(bls, adc_to_pes)


def calibrate_pmts(cwfs, adc_to_pes, n_maw=100, thr_maw=3):
    """
    This function is called for PMT waveforms that have
    already been baseline restored and pedestal subtracted.
    It computes the calibrated waveforms and its sensor sum.
    It also computes the calibrated waveforms and sensor
    sum for elements of the waveforms above some value
    (thr_maw) over a MAW that follows the waveform. These
    are useful to suppress oscillatory noise and thus can
    be applied for S1 searches (the calibrated version
    without the MAW should be applied for S2 searches).
    """
    window      = np.full(n_maw, 1 / n_maw)
    maw         = signal.lfilter(window, 1, cwfs, axis=1)

    # ccwfs stands for calibrated corrected waveforms
    ccwfs       = calibrate_wfs(cwfs, adc_to_pes)
    ccwfs_maw   = np.where(cwfs >= maw + thr_maw, ccwfs, 0)

    cwf_sum     = np.sum(ccwfs    , axis=0)
    cwf_sum_maw = np.sum(ccwfs_maw, axis=0)
    return ccwfs, ccwfs_maw, cwf_sum, cwf_sum_maw


def pmt_subtract_maw(cwfs, n_maw=100):
    """
    Subtract a MAW from the input waveforms.
    """
    window = np.full(n_maw, 1 / n_maw)
    maw    = signal.lfilter(window, 1, cwfs, axis=1)

    return cwfs - maw


def calibrate_sipms(sipm_wfs, adc_to_pes, thr, *, bls_mode=BlsMode.mode):
    """
    Subtracts the baseline, calibrates waveforms to pes
    and suppresses values below `thr` (in pes).
    """
    thr  = to_col_vector(np.full(sipm_wfs.shape[0], thr))
    bls  = subtract_baseline(sipm_wfs, bls_mode=bls_mode)
    cwfs = calibrate_wfs(bls, adc_to_pes)
    return np.where(cwfs > thr, cwfs, 0)


def subtract_mean  (wfs):
    """Subtract per-sensor mean baseline from waveforms."""
    return subtract_baseline(wfs, bls_mode=BlsMode.mean  )

def subtract_median(wfs):
    """Subtract per-sensor median baseline from waveforms."""
    return subtract_baseline(wfs, bls_mode=BlsMode.median)

def subtract_mode  (wfs):
    """Subtract per-sensor mode baseline from waveforms."""
    return subtract_baseline(wfs, bls_mode=BlsMode.mode  )


def sipm_subtract_mode_and_calibrate  (sipm_wfs, adc_to_pes):
    """Subtract mode baseline and calibrate SiPM waveforms to photoelectrons."""
    return calibrate_wfs(subtract_mode  (sipm_wfs), adc_to_pes)

def sipm_subtract_mean_and_calibrate  (sipm_wfs, adc_to_pes):
    """Subtract mean baseline and calibrate SiPM waveforms to photoelectrons."""
    return calibrate_wfs(subtract_mean  (sipm_wfs), adc_to_pes)

def sipm_subtract_median_and_calibrate(sipm_wfs, adc_to_pes):
    """Subtract median baseline and calibrate SiPM waveforms to photoelectrons."""
    return calibrate_wfs(subtract_median(sipm_wfs), adc_to_pes)


# Dict of functions for SiPM processing
sipm_processing = {
    SiPMCalibMode.subtract_mode            :      subtract_mode                ,# For gain extraction
    SiPMCalibMode.subtract_median          :      subtract_median              ,# For gain extraction
    SiPMCalibMode.subtract_mode_calibrate  : sipm_subtract_mode_and_calibrate  ,# For PDF calculation
    SiPMCalibMode.subtract_mean_calibrate  : sipm_subtract_mean_and_calibrate  ,# For PDF calculation
    SiPMCalibMode.subtract_median_calibrate: sipm_subtract_median_and_calibrate,# For PDF calculation
    SiPMCalibMode.subtract_mode_zs         : calibrate_sipms                    # For data processing
}
