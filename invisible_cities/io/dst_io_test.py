import os
import pandas as pd
import tables as tb
from ..core.testing_utils import assert_dataframes_close
from ..core.exceptions    import TableMismatch
from . dst_io             import load_dst
from . dst_io             import load_dsts
from . dst_io             import _store_pandas_as_tables

import warnings
import pytest

from numpy     .testing      import assert_raises
from hypothesis              import given
from hypothesis.extra.pandas import columns
from hypothesis.extra.pandas import data_frames
from hypothesis.extra.pandas import column
from hypothesis.extra.pandas import range_indexes


def test_load_dst(KrMC_kdst):
    df_read = load_dst(*KrMC_kdst[0].file_info)
    assert_dataframes_close(df_read, KrMC_kdst[0].true,
                            False  , rtol=1e-5)


def test_load_dsts_single_file(KrMC_kdst):
    tbl     = KrMC_kdst[0].file_info
    df_read = load_dsts([tbl.filename], tbl.group, tbl.node)

    assert_dataframes_close(df_read, KrMC_kdst[0].true,
                            False  , rtol=1e-5)


def test_load_dsts_double_file(KrMC_kdst):
    tbl     = KrMC_kdst[0].file_info
    df_true = KrMC_kdst[0].true
    df_read = load_dsts([tbl.filename]*2, tbl.group, tbl.node)
    df_true = pd.concat([df_true, df_true])

    assert_dataframes_close(df_read, df_true  ,
                            False  , rtol=1e-5)

def test_load_dsts_reads_good_kdst(ICDATADIR):
    good_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_KDST.h5"
    good_file = os.path.join(ICDATADIR, good_file)
    group = "DST"
    node  = "Events"
    load_dsts([good_file], group, node)

def test_load_dsts_warns_not_of_kdst_type(ICDATADIR):
    good_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_KDST.h5"
    good_file = os.path.join(ICDATADIR, good_file)

    wrong_file = "kdst_5881_map_lt.h5"
    wrong_file = os.path.join(ICDATADIR, wrong_file)
    group = "DST"
    node  = "Events"
    with pytest.warns(UserWarning, match='not of kdst type'):
        load_dsts([good_file, wrong_file], group, node)

def test_load_dsts_warns_if_not_existing_file(ICDATADIR):
    good_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_KDST.h5"
    good_file = os.path.join(ICDATADIR, good_file)

    wrong_file = "non_existing_file.h5"

    group = "DST"
    node  = "Events"
    with pytest.warns(UserWarning, match='does not exist'):
        load_dsts([good_file, wrong_file], group, node)


dataframe = data_frames(index=range_indexes(min_size=1),columns=[column('int_value', dtype=int),column('float_val',dtype='float'),column('bool_value',dtype=bool)])
@given(df = dataframe)
def  test_store_pandas_as_tables_exact(config_tmpdir, df):
    filename = config_tmpdir+'dataframe_to_table_exact.h5'
    group_name ='test_group'
    table_name = 'table_name_1'
    with tb.open_file(filename,'w') as h5out:
        _store_pandas_as_tables(h5out, df, group_name,table_name)
    df_read = load_dst(filename, group_name, table_name)
    assert_dataframes_close(df_read, df, False, rtol=1e-5)

@given(df1 = dataframe, df2=dataframe)
def  test_store_pandas_as_tables_2df(config_tmpdir, df1, df2):
    filename = config_tmpdir+'dataframe_to_table_exact.h5'
    group_name ='test_group'
    table_name = 'table_name_2'
    with tb.open_file(filename,'w') as h5out:
        _store_pandas_as_tables(h5out, df1, group_name,table_name)
        _store_pandas_as_tables(h5out, df2, group_name,table_name)
    df_read = load_dst(filename, group_name, table_name)
    assert_dataframes_close(df_read, pd.concat([df1, df2]).reset_index(drop=True),False, rtol=1e-5)

dataframe_diff = data_frames(index=range_indexes(min_size=1),columns=[column('int_value', dtype=int),column('float_val',dtype='float')])
@given(df1 = dataframe, df2=dataframe_diff)
def  test_store_pandas_as_tables_raises_exception(config_tmpdir, df1, df2):
    filename = config_tmpdir+'dataframe_to_table_exact.h5'
    group_name ='test_group'
    table_name = 'table_name_2'
    with tb.open_file(filename,'w') as h5out:
        _store_pandas_as_tables(h5out, df1, group_name,table_name)
        assert_raises(TableMismatch, _store_pandas_as_tables,h5out, df2, group_name,table_name)
