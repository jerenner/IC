"""Invisible Cities — data processing and reconstruction framework for the NEXT experiment.

IC organizes data processing into *cities*, named pipelines that read raw detector
data, apply transformations, and write processed results. Cities chain together
modular functions from the subpackages below.

Subpackages
-----------
cities : Processing pipelines (Isidora, Irene, Diomira, etc.)
core   : Shared utilities, configuration, units, exceptions
io     : HDF5 readers and writers for all data formats
reco   : Reconstruction algorithms (peaks, hits, clusters, corrections)
calib  : Calibration functions for PMTs and SiPMs
detsim : Detector simulation (S1, electron drift, sensor response)
evm    : Event model and data containers (PMAP, Hit, Cluster, etc.)
icaros : ICAROS detector correction and selection functions
dataflow : Coroutine-based pipeline primitives (pipe, fork, sink, etc.)
filters : Event and peak filtering (S1/S2 selection, triggers)
database : Database access, download, and connection management
sierpe : Waveform processing (BLR deconvolution, FEE simulation)
types  : Type definitions and symbolic constants
"""
