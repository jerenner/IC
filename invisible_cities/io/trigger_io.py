from functools import partial

import tables as tb

from .. evm                import nh5     as table_formats
from .. core.tbl_functions import filters as tbl_filters


def store_trigger(tables, trg_type, trg_channels):
    """Store trigger type and channels to HDF5 tables.

    Parameters
    ----------
    tables : tuple
        ``(trg_type_table, trg_channels_array)`` from ``_make_tables``.
    trg_type : int or None
        Trigger type value, or None to skip.
    trg_channels : np.ndarray or None
        Array of triggered sensor channels, or None to skip.
    """
    trg_type_table, trg_channels_array = tables

    if trg_type:
        trg_type_row   = trg_type_table.row
        trg_type_row['trigger_type'] = trg_type
        trg_type_row.append()

    if trg_channels is not None:
        new_shape = 1, trg_channels.shape[0]
        trg_channels_array.append(trg_channels.reshape(new_shape))


def trigger_writer(file, n_sensors, compression=None):
    """Create an HDF5 writer for trigger data.

    Creates tables in the ``Trigger`` group for trigger type and channels.

    Parameters
    ----------
    file : tb.File
        Open HDF5 file.
    n_sensors : int
        Number of sensor channels.
    compression : str, optional
        Compression mode, passed to ``tbl.filters``.

    Returns
    -------
    Callable
        Function that writes trigger type and channels for an event.
    """
    tables = _make_tables(file, n_sensors, compression)
    return partial(store_trigger, tables)


def _make_tables(hdf5_file, n_sensors, compression):
    compr         = tbl_filters(compression)
    trigger_group = hdf5_file.create_group(hdf5_file.root, 'Trigger')
    make_table    = partial(hdf5_file.create_table, trigger_group, filters=compr)

    trg_type    = make_table('trigger', table_formats.TriggerType, "Trigger Type")

    array_name = "events"
    trg_channels = hdf5_file.create_earray(trigger_group,
                                   array_name,
                                   atom    = tb.Int16Atom(),
                                   shape   = (0, n_sensors),
                                   filters = compr)

    trg_tables = trg_type, trg_channels

    return trg_tables
