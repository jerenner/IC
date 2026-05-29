import pandas as pd
import tables as tb

from .. evm  import nh5           as table_formats
from .. core import tbl_functions as tbl


def _make_run_event_tables(hdf5_file, compression):

    c = tbl.filters(compression)
    rungroup = hdf5_file.create_group(hdf5_file.root, "Run")

    RunInfo, EventInfo = table_formats.RunInfo, table_formats.EventInfo
    MKT = hdf5_file.create_table

    run_info   = MKT(rungroup, "runInfo",   RunInfo,   "run info table", c)
    event_info = MKT(rungroup,  "events", EventInfo, "event info table", c)
    run_tables = (run_info, event_info)

    return run_tables


def run_and_event_writer(file, *, compression=None):
    """Create an HDF5 writer for run and event metadata.

    Creates tables in the ``Run`` group for run info and event info.

    Parameters
    ----------
    file : tb.File
        Open HDF5 file.
    compression : str, optional
        Compression mode, passed to ``tbl.filters``.

    Returns
    -------
    Callable
        Function that writes ``(run_number, event_number, timestamp)`` tuples.
    """
    run_tables = _make_run_event_tables(file, compression)
    def write_run_and_event(run_number, event_number, timestamp):
        run_table_dumper  (run_tables[0],   run_number)
        event_table_dumper(run_tables[1], event_number, timestamp)
    return write_run_and_event


def run_table_dumper(table, run_number):
    """Write a run number to a run info table.

    Parameters
    ----------
    table : tb.Table
        Open run info table.
    run_number : int
        Run number to store.
    """
    row = table.row
    row['run_number'] = run_number
    row.append()


def event_table_dumper(table, event_number, timestamp):
    """Write event metadata to an event info table.

    Parameters
    ----------
    table : tb.Table
        Open event info table.
    event_number : int
        Event number to store.
    timestamp : float
        Event timestamp.
    """
    row = table.row
    row["evt_number"] = event_number
    row["timestamp"] = timestamp
    row.append()


def read_run_and_event(filename):
    """Read run and event metadata from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    tuple of pd.DataFrame
        ``(run_info, event_info)`` DataFrames.
    """
    with tb.open_file(filename) as h5f:
        return (pd.DataFrame.from_records(h5f.root.Run.runInfo.read()),
                pd.DataFrame.from_records(h5f.root.Run.events .read()))
