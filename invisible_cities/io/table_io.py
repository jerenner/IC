
from .. core import tbl_functions as tbl

def make_table(hdf5_file,
               group, name, fformat, description, compression = None):
    """Create an HDF5 table in the specified group.

    Creates the group if it does not already exist.

    Parameters
    ----------
    hdf5_file : tb.File
        Open HDF5 file.
    group : str
        HDF5 group name.
    name : str
        Table name within the group.
    fformat : type
        PyTables table description class.
    description : str
        Human-readable description stored in table metadata.
    compression : str, optional
        Compression mode, passed to ``tbl.filters``.

    Returns
    -------
    tb.Table
        Created PyTables table object.
    """
    if group not in hdf5_file.root:
        hdf5_file.create_group(hdf5_file.root, group)

    table = hdf5_file.create_table(getattr(hdf5_file.root, group),
                                   name,
                                   fformat,
                                   description,
                                   tbl.filters(compression))
    return table
