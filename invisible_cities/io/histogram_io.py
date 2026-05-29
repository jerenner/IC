import numpy  as np
import tables as tb

from .. core import tbl_functions as tbl


def hist_writer(file,
                *,
                group_name  : str,
                table_name  : str,
                n_sensors   : int,
                bin_centres : np.ndarray,
                compression = None):
    """Create an HDF5 writer for sensor histograms.

    Creates a group and extendable array for storing histograms, plus a
    companion array for bin centers.

    Parameters
    ----------
    file : tb.File
        Open HDF5 file.
    group_name : str
        HDF5 group name, e.g. ``HIST`` or ``HIST2D``.
    table_name : str
        Table name, e.g. ``pmt``, ``pmtMAW``, ``sipm``, ``sipmMAW``.
    n_sensors : int
        Number of sensors (PMTs or SiPMs).
    bin_centres : np.ndarray
        Array of histogram bin centers.
    compression : str, optional
        Compression mode, passed to ``tbl.filters``.

    Returns
    -------
    Callable
        Function that appends a histogram array of shape ``(n_sensors, n_bins)``.
    """
    try:                       hist_group = getattr          (file.root, group_name)
    except tb.NoSuchNodeError: hist_group = file.create_group(file.root, group_name)

    n_bins = len(bin_centres)

    hist_table = file.create_earray(hist_group,
                                    table_name,
                                    atom    = tb.Int32Atom(),
                                    shape   = (0, n_sensors, n_bins),
                                    filters = tbl.filters(compression))

    file.create_carray( hist_group
                      , table_name + '_bins'
                      , tb.Float64Atom()
                      , filters = tbl.filters(compression)
                      , obj     = bin_centres)

    def write_hist(histo : np.ndarray):
        hist_table.append(histo.reshape(1, n_sensors, n_bins))

    return write_hist
