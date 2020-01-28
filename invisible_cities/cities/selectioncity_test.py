import os
import glob
import numpy  as np
import tables as tb

from pytest import fixture, mark

from invisible_cities.core.configure import configure

from .selectioncity import selectioncity


@fixture
def testfile(filetype):
    return glob.glob(f"/Users/gonzalo/Documents/NEXT/DATA/{filetype}/*")[0]


@fixture
def selectionfile(testfile, filetype, n_evts=2):
    with tb.open_file(testfile, "r") as h5file:
        events_node = h5file.get_node("/Run/events")
        events = events_node.read()["evt_number"]
    selevents = np.sort(np.random.choice(events, n_evts))

    selfile = "/Users/gonzalo/Documents/NEXT/selectioncity" + f"/{filetype}" + "/selected_events.txt"
    with open(selfile, "w") as file:
        for event in selevents:
            file.write(f"{event}\n")
    return selfile


@mark.parametrize("filetype", ("pmaps", "rwfs"))
def test_selectioncity_contain_structure(testfile, selectionfile, output_tmpdir):
    ## run selection city ###
    PATH_IN  = testfile
    PATH_OUT = os.path.join(output_tmpdir, "selected.h5")

    conf = configure("dummy /Users/gonzalo/Documents/NEXT/selectioncity/selectioncity.conf".split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT,
                     selected_events_filename = selectionfile,
                     event_range = all ))
    selectioncity(**conf)

    ## test structure ##
    with tb.open_file(PATH_IN , "r") as h5in, \
         tb.open_file(PATH_OUT, "r") as h5out:

         for node in h5in.walk_nodes():
             assert node in h5out

@mark.parametrize("filetype", ("pmaps", "rwfs"))
def test_selectioncity_contain_selected_events_uniquely(testfile, selectionfile, output_tmpdir):
    ## run selection city ###
    PATH_IN  = testfile
    PATH_OUT = os.path.join(output_tmpdir, "selected.h5")

    conf = configure("dummy /Users/gonzalo/Documents/NEXT/selectioncity/selectioncity.conf".split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT,
                     selected_events_filename = selectionfile,
                     event_range = all ))
    selectioncity(**conf)

    ## test ##
    with tb.open_file(PATH_IN , "r") as h5in, \
         tb.open_file(PATH_OUT, "r") as h5out:

         inevents = h5in .get_node("/Run/events").read()["evt_number"]
         outevents= h5out.get_node("/Run/events").read()["evt_number"]

         assert (np.sort( np.loadtxt(selectionfile, dtype=int) ) == np.sort(outevents)).all()
         assert np.isin(outevents                             , inevents).all()


@mark.parametrize("filetype", ("pmaps", "rwfs"))
def test_selectioncity_contain_event_data(testfile, selectionfile, output_tmpdir):
    ## run selection city ###
    PATH_IN  = testfile
    PATH_OUT = os.path.join(output_tmpdir, "selected.h5")

    conf = configure("dummy /Users/gonzalo/Documents/NEXT/selectioncity/selectioncity.conf".split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT,
                     selected_events_filename = selectionfile,
                     event_range = all ))
    selectioncity(**conf)

    #### test ###
    with tb.open_file(PATH_IN , "r") as h5in, \
         tb.open_file(PATH_OUT, "r") as h5out:

         events = np.loadtxt(selectionfile, dtype=int)
         events    = h5out.get_node("/Run/events").read()["evt_number"]
         events_in = h5in.get_node("/Run/events").read()["evt_number"]

         for node in h5out.walk_nodes():
             if isinstance(node, tb.Table):
                 tableout = node.read()
                 tablein  = h5in.get_node(node._v_pathname).read()

                 for i, event in enumerate(events):
                     try:
                         selout = tableout["event"] == event
                         selin  = tablein ["event"] == event
                         assert (tablein[selin] == tabeout[selout]).all()
                     except:
                         pass

             if isinstance(node, tb.EArray):
                 earrayout = node
                 earrayin  = h5in.get_node(node._v_pathname)

                 for i, event in enumerate(events):
                     ii = np.argwhere(events_in == event).flatten()[0]
                     assert (earrayin[ii] == earrayout[i]).all()
