"""Event model and data containers.

Defines the core data structures used throughout IC:

- ``KrEvent`` : top-level event container
- ``PMap``    : pulse map matching PMT and SiPM signals
- ``_Peak``   : individual signal pulse (S1 or S2)
- ``Hit``     : reconstructed 3D hit position
- ``Cluster`` : grouped hits representing an energy deposit
- ``FitFunction`` : curve fitting result container

Also provides ``nh5`` for HDF5 event I/O.
"""
