Core Modules
============

The ``invisible_cities.core`` package provides shared utility functions
used across all cities and submodules. These are the building blocks
that every other part of IC depends on.

Configuration System
--------------------

The :mod:`~invisible_cities.core.configure` module powers the city
configuration system described in :doc:`getting_started`. Key features:

- **Config file parsing** — ``.conf`` files are plain Python with
  environment variable expansion and Python expressions
- **CLI overrides** — command-line flags override config file values
- **Type checking** — config values are validated against city function
  type annotations on first run
- **Include hierarchy** — configs can include other configs via the
  built-in ``include()`` function

.. code-block:: python

   from invisible_cities.core.configure import configure
   options = configure(sys.argv)
   # Use options.files_in, options.file_out, etc.

Units
-----

The :mod:`~invisible_cities.core.system_of_units` module provides a
coherent system of units derived from Geant4, using the HEP convention
where the base units are millimeters, nanoseconds, MeV, and positron
charge. Units are available as constants in config files and Python code:

.. code-block:: python

   from invisible_cities.core import system_of_units as units
   distance = 25 * units.centimeter
   energy   = 2039 * units.keV

See :doc:`api/core` for the full list of available units.

Utility Functions
-----------------

The :mod:`~invisible_cities.core.core_functions` module provides general
purpose functions used throughout the framework:

- **Array operations** — flattening, binning, range selection, vector transformations
- **Math helpers** — relative difference with configurable normalization, weighted mean/std
- **Timing** — ``@timefunc`` decorator for profiling functions
- **Dict/list utilities** — mapping and filtering over collections

See :doc:`api/core` for the complete function list.

Fitting and Statistics
----------------------

The :mod:`~invisible_cities.core.fit_functions` module provides tools for
curve fitting, including parameter fixing, covariance analysis, and
fit function management.

The :mod:`~invisible_cities.core.stat_functions` module provides
statistical helpers, primarily Poisson probability and uncertainty
calculations.

Random Sampling
---------------

The :mod:`~invisible_cities.core.random_sampling` module provides
samplers for SiPM probability distributions, used by simulation
cities like Diomira. Includes discrete distribution sampling,
uniform smearing, inverse CDF methods, and PDF padding.

HDF5 / PyTables Utilities
-------------------------

The :mod:`~invisible_cities.core.tbl_functions` module provides helpers
for working with HDF5 files via PyTables:

- **Compression filters** — ``filters("ZLIB4")``, ``filters("BLOSC5")``, etc.
- **Table reading** — utilities for extracting vectors and tables from data files

Exceptions
----------

The :mod:`~invisible_cities.core.exceptions` module defines the IC
exception hierarchy. All IC-specific exceptions inherit from
``ICException``. Key exceptions include:

- ``NoInputFiles``, ``InvalidInputFileStructure`` — input validation
- ``NoHits``, ``NoVoxels`` — missing reconstructed objects
- ``XYRecoFail`` and subclasses — position reconstruction failures
- ``SensorIDMismatch``, ``TableMismatch`` — data consistency errors

Logging
-------

The :mod:`~invisible_cities.core.log_config` module configures the
application logger. Verbosity can be controlled via the ``-v`` flag
on the command line.

Testing Utilities
-----------------

The :mod:`~invisible_cities.core.testing_utils` module provides helpers
for writing tests, including Hypothesis strategies for float arrays,
equality assertions for IC data structures (PMAPs, Hits, Clusters),
and PyTables table comparison utilities.

API Reference
-------------

For the complete API documentation of all core modules, see :doc:`api/core`.
