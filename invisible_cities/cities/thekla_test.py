import os
import numpy  as np
import tables as tb

from pytest import mark

from .. core.core_functions    import in_range
from .. core.system_of_units_c import units
from .. core.testing_utils     import assert_dataframes_close
from .  thekla                 import Thekla
from .. core.configure         import configure
from .. io                     import dst_io as dio
from .. io.mcinfo_io           import load_mchits


def test_thekla_Kr(KrMC_hdst_filename, config_tmpdir):

    PATH_IN   = KrMC_hdst_filename
    PATH_OUT  = os.path.join(config_tmpdir,'Kr_NDST.h5')
    conf      = configure('dummy invisible_cities/config/thekla.conf'.split())
    nevt_req  = 4

    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     event_range   = (nevt_req,)))

    thekla = Thekla(**conf)
    thekla.run()
    cnt         = thekla.end()
    assert cnt.n_events_tot      == nevt_req
