"""HDF5 readers and writers for all IC data formats.

Provides I/O functions for reading and writing persistent storage files:

- ``rwf_io`` : raw waveforms (PMT and SiPM)
- ``dst_io`` : data storage format
- ``pmaps_io`` : pulse maps (PMAPs)
- ``hits_io`` : reconstructed hits
- ``kdst_io`` : kinematic data
- ``voxels_io`` : voxelized detector geometry
- ``mcinfo_io`` : Monte Carlo truth information
- ``event_filter_io`` : event filter results
- ``channel_param_io`` : channel parameters
- ``histogram_io`` : histogram data
- ``table_io`` : generic table I/O
- ``trigger_io`` : trigger information
- ``fee_io`` : front-end electronics data
- ``run_and_event_io`` : run/event metadata
"""
