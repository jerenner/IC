import pandas as pd
import tables as tb

from .. evm.nh5  import RunSummaryTable
from .. reco import tbl_functions as tbl


def event_summary_writer(file, *, compression='ZLIB4'):
    event_summary_table = make_table(hdf5_file,
                          group       = 'DST',
                          name        = 'EventSummary',
                          fformat     = EventSummaryTable,
                          description = 'Event summary',
                          compression = compression)

    def write_event_summary(event_summary):
        event_summary.store(event_summary_table)
    return write_event_summary


#def read_event_summary(filename):
#    with tb.open_file(filename) as h5f:
#        return (pd.DataFrame.from_records(h5f.root.Run.runInfo.read()),
#                pd.DataFrame.from_records(h5f.root.Run.events .read()))
