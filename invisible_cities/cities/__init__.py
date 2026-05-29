"""Cities — named data processing pipelines.

Each city reads data from disk, applies a sequence of transformations, and writes
new data. Cities are decorated with ``@city`` and configured via ``.conf`` files.

Available cities
----------------
Isidora  : Deconvolve PMT waveforms (RWF → CWF)
Irene    : Find pulses, match PMT+SiPM signals (RWF → PMAPs)
Diomira  : Monte Carlo waveform simulation
Berenice : SiPM dark noise spectrum calibration
Buffy    : MC sensor info sorting into data-like buffers
Dorothea : Pointlike energy deposition reconstruction from S1/S2
Esmeralda : Hit thresholding, energy corrections, track finding
Eutropia : Point Spread Function (PSF) computation
Hypathia : PMAP computation from true waveforms (no electronics sim)
Phyllis  : PMT light/dark spectrum for calibration
Sophronia : Hit reconstruction from S2 signals
Trude    : SiPM light/dark spectrum for calibration
Zemrude  : ICAROS krypton map computation
Beersheba : Lucy-Richardson deconvolution for electron cloud density
Isaura   : Track computation from deconvolved hits

Shared components
-----------------
The ``components`` module provides shared functions used by multiple cities:
waveform I/O, deconvolution, calibration, zero-suppression, PMAP building,
and dataflow helpers.
"""
