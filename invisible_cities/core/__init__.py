"""Core utilities shared across all IC modules.

Submodules
----------
core_functions : General purpose helpers (array ops, math, timing)
configure      : Config file parsing, CLI argument handling, type checking
system_of_units : Geant4-derived HEP units (mm, ns, MeV, e+)
random_sampling : SiPM distribution samplers for simulation
fit_functions   : Curve fitting, parameter fixing, covariance analysis
stat_functions  : Poisson statistics and uncertainties
tbl_functions   : HDF5/PyTables compression filters and table readers
exceptions      : IC-specific exception hierarchy
log_config      : Application logger configuration
testing_utils   : Hypothesis strategies and data structure assertions
"""
