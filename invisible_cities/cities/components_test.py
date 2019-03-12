import os

import numpy as np
import tables as tb

from argparse  import Namespace
from functools import partial

from pytest import mark
from pytest import raises

from .. core.configure  import EventRange as ER
from .. core.exceptions import InvalidInputFileStructure
from .. core.exceptions import ClusterEmptyList

from .. core.system_of_units_c import units

from .  components import event_range
from .  components import WfType
from .  components import wf_from_files
from .  components import pmap_from_files
from .  components import compute_xy_position
from .  components import city
from .  components import hits_and_kdst_from_files
from .. database   import load_db



def _create_dummy_conf_with_event_range(value):
    return Namespace(event_range = value)


@mark.parametrize("given expected".split(),
                  ((       9          , (   9,     )),
                   ( (     9,        ), (   9,     )),
                   ( (     5,       9), (   5,    9)),
                   ( (     5, ER.last), (   5, None)),
                   (  ER.all          , (None,     )),
                   ( (ER.all,        ), (None,     ))))
def test_event_range_valid_options(given, expected):
    conf = _create_dummy_conf_with_event_range(given)
    assert event_range(conf) == expected


@mark.parametrize("given",
                  ( ER.last    ,
                   (ER.last,)  ,
                   (ER.last, 4),
                   (ER.all , 4),
                   ( 1,  2,  3)))

def test_event_range_invalid_options_raises_ValueError(given):
    conf = _create_dummy_conf_with_event_range(given)
    with raises(ValueError):
        event_range(conf)


_rwf_from_files = partial(wf_from_files, wf_type=WfType.rwf)
@mark.parametrize("source filename".split(),
                  ((_rwf_from_files, "defective_rwf_rd_pmtrwf.h5"      ),
                   (_rwf_from_files, "defective_rwf_rd_sipmrwf.h5"     ),
                   (_rwf_from_files, "defective_rwf_run_events.h5"     ),
                   (_rwf_from_files, "defective_rwf_trigger_events.h5" ),
                   (_rwf_from_files, "defective_rwf_trigger_trigger.h5"),
                   (pmap_from_files, "defective_pmp_pmap_all.h5"       ),
                   (pmap_from_files, "defective_pmp_run_events.h5"     )))
def test_sources_invalid_input_raises_InvalidInputFileStructure(ICDATADIR, source, filename):
    full_filename = os.path.join(ICDATADIR, "defective_files", filename)
    s = source((full_filename,))
    with raises(InvalidInputFileStructure):
        next(s)


def test_compute_xy_position_depends_on_actual_run_number():
    """
    The channels entering the reco algorithm are the ones in a square of 3x3
    that includes the masked channel.
    Scheme of SiPM positions (the numbers are the SiPM charges):
    x - - - >
    y | 5 5 5
      | X 7 5
      v 5 5 5

    This test is meant to fail if them compute_xy_position function
    doesn't use the run_number parameter.
    """
    minimum_seed_charge = 6*units.pes
    reco_parameters = {'Qthr': 2*units.pes,
                       'Qlm': minimum_seed_charge,
                       'lm_radius': 0*units.mm,
                       'new_lm_radius': 15 * units.mm,
                       'msipm': 9,
                       'consider_masked': True}
    run_number = 6977
    find_xy_pos = compute_xy_position('new', run_number, **reco_parameters)

    xs_to_test  = np.array([-65, -65, -55, -55, -55, -45, -45, -45])
    ys_to_test  = np.array([  5,  25,   5,  15,  25,   5,  15,  25])
    xys_to_test = np.stack((xs_to_test, ys_to_test), axis=1)

    charge         = minimum_seed_charge - 1
    seed_charge    = minimum_seed_charge + 1
    charge_to_test = np.array([charge, charge, charge, seed_charge, charge, charge, charge, charge])

    try:
        find_xy_pos(xys_to_test, charge_to_test)
    except(ClusterEmptyList):
        assert False


def test_city_adds_default_detector_db(config_tmpdir):
    default_detector_db = 'new'
    args = {'files_in'    : 'dummy_in',
            'file_out'    : os.path.join(config_tmpdir, 'dummy_out')}
    @city
    def dummy_city(files_in, file_out, event_range, detector_db):
        with tb.open_file(file_out, 'w') as h5out:
            pass
        return detector_db

    db = dummy_city(**args)
    assert db == default_detector_db


def test_city_does_not_overwrite_detector_db(config_tmpdir):
    args = {'detector_db' : 'some_detector',
            'files_in'    : 'dummy_in',
            'file_out'    : os.path.join(config_tmpdir, 'dummy_out')}
    @city
    def dummy_city(files_in, file_out, event_range, detector_db):
        with tb.open_file(file_out, 'w') as h5out:
            pass
        return detector_db

    db = dummy_city(**args)
    assert db == args['detector_db']


def test_city_only_pass_default_detector_db_when_expected(config_tmpdir):
    args = {'files_in'    : 'dummy_in',
            'file_out'    : os.path.join(config_tmpdir, 'dummy_out')}
    @city
    def dummy_city(files_in, file_out, event_range):
        with tb.open_file(file_out, 'w') as h5out:
            pass


def test_hits_and_kdst_from_files(ICDATADIR):
    event_number = 1
    timestamp    = 0.
    num_hits     = 13
    keys = ['hits', 'mc', 'kdst', 'run_number', 'event_number', 'timestamp']
    file_in     = os.path.join(ICDATADIR    ,  'Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.HDST.h5')
    generator = hits_and_kdst_from_files([file_in])
    output = next(generator)
    assert set(keys) == set(output.keys())
    assert output['event_number']   == event_number
    assert output['timestamp']      == timestamp
    assert len(output['hits'].hits) == num_hits
