from .  table_io import make_table
from .. evm.nh5  import KrTable

from .. io.dst_io import df_writer


def kr_writer(hdf5_file, *, compression=None):
    """Create an HDF5 writer for krypton DST events.

    Creates a table in the ``DST/Events`` group with the KrTable format,
    indexed on the ``event`` column.

    Parameters
    ----------
    hdf5_file : tb.File
        Open HDF5 file.
    compression : str, optional
        Compression mode, passed to ``tbl.filters``.

    Returns
    -------
    Callable
        Function that writes a krypton event to the HDF5 table.
    """
    kr_table = make_table(hdf5_file,
                          group       = 'DST',
                          name        = 'Events',
                          fformat     = KrTable,
                          description = 'KDST Events',
                          compression = compression)
    kr_table.set_attr('columns_to_index', ['event'])

    def write_kr(kr_event):
        kr_event.store(kr_table)
    return write_kr


def kdst_from_df_writer(h5out, compression=None):
    """
    For a given open table returns a writer for KDST dataframe info
    """
    def write_kdst(df):
        return df_writer(h5out              = h5out        ,
                         df                 = df           ,
                         compression        = compression  ,
                         group_name         = 'DST'        ,
                         table_name         = 'Events'     ,
                         descriptive_string = 'KDST Events',
                         columns_to_index   = ['event']    )
    return write_kdst
