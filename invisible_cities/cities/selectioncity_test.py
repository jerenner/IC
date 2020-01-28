import os
import glob
import numpy  as np
import tables as tb

from pytest import fixture, mark

from invisible_cities.core.configure import configure

from .selectioncity import selectioncity


def generate_textfile_with_random_events(filein, fileout, n_evts=2):
    with tb.open_file(filein, "r") as h5file:
        events_node = h5file.get_node("/Run/events")
        events = events_node.read()["evt_number"]
    selevents = np.sort(np.random.choice(events, n_evts))
    with open(fileout, "w") as file:
        for event in selevents:
            file.write(f"{event}\n")


@mark.parametrize("filetype", ("pmaps", "rwfs"))
def test_selectioncity_contain_structure(output_tmpdir, filetype):
    ## run selection city ###
    PATH_IN  = glob.glob(f"/Users/gonzalo/Documents/NEXT/DATA/{filetype}/*")[0]
    PATH_OUT = os.path.join(output_tmpdir, "selected.h5")

    selected_events = "/Users/gonzalo/Documents/NEXT/selectioncity" + "/selected_events.txt"
    generate_textfile_with_random_events(PATH_IN, selected_events, n_evts=2)

    conf = configure("dummy /Users/gonzalo/Documents/NEXT/selectioncity/selectioncity.conf".split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT,
                     selected_events_filename = selected_events,  #f"/Users/gonzalo/Documents/NEXT/selectioncity/{filetype}/selected_events.txt",
                     event_range = all ))
    selectioncity(**conf)

    ## test structure ##
    with tb.open_file(PATH_IN , "r") as h5in, \
         tb.open_file(PATH_OUT, "r") as h5out:

         for node in h5in.walk_nodes():
             assert node in h5out


@mark.parametrize("filetype", ("pmaps", "rwfs"))
def test_selectioncity_contain_selected_events_uniquely(output_tmpdir, filetype):
    ## run selection city ###
    PATH_IN  = glob.glob(f"/Users/gonzalo/Documents/NEXT/DATA/{filetype}/*")[0]
    PATH_OUT = os.path.join(output_tmpdir, "selected.h5")

    selected_events = "/Users/gonzalo/Documents/NEXT/selectioncity" + "/selected_events.txt"
    generate_textfile_with_random_events(PATH_IN, selected_events, n_evts=2)

    conf = configure("dummy /Users/gonzalo/Documents/NEXT/selectioncity/selectioncity.conf".split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT,
                     selected_events_filename = selected_events,
                     event_range = all ))
    selectioncity(**conf)

    ## test ##
    with tb.open_file(PATH_IN , "r") as h5in, \
         tb.open_file(PATH_OUT, "r") as h5out:

         inevents = h5in .get_node("/Run/events").read()["evt_number"]
         outevents= h5out.get_node("/Run/events").read()["evt_number"]

         assert (np.sort( np.loadtxt(selected_events, dtype=int) ) == np.sort(outevents)).all()
         assert np.isin(outevents                             , inevents).all()


@mark.parametrize("filetype", ("pmaps", "rwfs"))
def test_selectioncity_contain_event_data(output_tmpdir, filetype):
    ## run selection city ###
    PATH_IN  = glob.glob(f"/Users/gonzalo/Documents/NEXT/DATA/{filetype}/*")[0]
    PATH_OUT = os.path.join(output_tmpdir, "selected.h5")

    selected_events = "/Users/gonzalo/Documents/NEXT/selectioncity" + "/selected_events.txt"
    generate_textfile_with_random_events(PATH_IN, selected_events, n_evts=2)

    conf = configure("dummy /Users/gonzalo/Documents/NEXT/selectioncity/selectioncity.conf".split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT,
                     selected_events_filename = selected_events,
                     event_range = all ))
    selectioncity(**conf)

    #### test ###
    with tb.open_file(PATH_IN , "r") as h5in, \
         tb.open_file(PATH_OUT, "r") as h5out:

         events = np.loadtxt(selected_events, dtype=int)
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


@mark.parametrize("filetype", ("pmaps", "rwfs"))
def test_selectioncity_globalnodes(output_tmpdir, filetype):
    ## run selection city ###
    PATH_IN  = glob.glob(f"/Users/gonzalo/Documents/NEXT/DATA/{filetype}/*")[0]
    PATH_OUT = os.path.join(output_tmpdir, "selected.h5")

    selected_events = "/Users/gonzalo/Documents/NEXT/selectioncity" + "/selected_events.txt"
    generate_textfile_with_random_events(PATH_IN, selected_events, n_evts=2)

    conf = configure("dummy /Users/gonzalo/Documents/NEXT/selectioncity/selectioncity.conf".split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT,
                     selected_events_filename = selected_events,
                     event_range = all ))
    selectioncity(**conf)

    #### test ###
    with tb.open_file(PATH_IN , "r") as h5in, \
         tb.open_file(PATH_OUT, "r") as h5out:
         pass
