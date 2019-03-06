import tables as tb
import pandas as pd
import numpy  as np
from tables import NoSuchNodeError
from tables import HDF5ExtError
import warnings

from .. reco import tbl_functions as tbl

def load_dst(filename, group, node):
    """load a kdst if filename, group and node correctly found"""
    try:
        with tb.open_file(filename) as h5in:
            try:
                table = getattr(getattr(h5in.root, group), node).read()
                return pd.DataFrame.from_records(table)
            except NoSuchNodeError:
                warnings.warn(f' not of kdst type: file= {filename} ', UserWarning)
    except HDF5ExtError:
        warnings.warn(f' corrupted: file = {filename} ', UserWarning)
    except IOError:
        warnings.warn(f' does not exist: file = {filename} ', UserWarning)


def load_dsts(dst_list, group, node):
    dsts = [load_dst(filename, group, node) for filename in dst_list]
    return pd.concat(dsts)

class TableMismatch(Exception):
    def __init__(self):
        s  = 'The table and dataframe dont match! '
        Exception.__init__(self, s)

def _store_pandas_as_tables(h5out, df, group_name, table_name, compression, descriptive_string, str_col_length=32):
    if '/'+group_name not in h5out:
        group = h5out.create_group(h5out.root,group_name)
    else:
        group = getattr(h5out.root,group_name)
    if table_name in group:
        table=getattr(group,table_name)
    else:
        c = tbl.filters(compression)
        tabledef =df.dtypes.apply(lambda x : tb.Col.from_type(x.name) if x.name !='object' else tb.StringCol(str_col_length)).to_dict()
        table = h5out.create_table(group,table_name,tabledef,c)
    if not np.array_equal(df.columns,table.colnames):
        raise TableMismatch
    for indx in df.index:
        tablerow = table.row
        for colname in table.colnames:
            tablerow[colname] = df.at[indx,colname]
        tablerow.append()
    table.flush()
