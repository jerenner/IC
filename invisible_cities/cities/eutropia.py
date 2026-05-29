"""
-----------------------------------------------------------------------
                                Eutropia
-----------------------------------------------------------------------

From ancient Greek, ευτροπία: docile, of good manners, versatile, variable.

This city processes hits to produce a Point Spread Function (PSF).
The PSF is obtained by mapping the spatial charge distribution
relative to the barycenter (center of gravity of SiPM response)
for pointlike events collapsed along the drift coordinate. This
charge distribution is averaged over a large number of events.
The PSF can be obtained for different sections of the active
volume independently.

The tasks performed are:

    - Load hits processed for psf configuration
    - Split data in z-sections
    - Process each slice (charge normalization)
    - Split each slice in xy-sectors
    - Create PSF for each sector (binned charge average)
    - Accumulate the PSFs to produce an overall PSF
    - Write the PSF

The overall PSF is table that describes a collection of discrete
matrices organized according to these variables:

    - x, y, z: coordinates of the center of each xy-sector and slice

Each triplet of values is a PSF on its own. A PSF is a table containing:

    - xr, yr, zr: relative coordinates of the center of the bin
    - factor    : fraction of charge at this bin
    - nevt      : number of entries at this bin
"""

import numpy  as np
import tables as tb
import pandas as pd

from .. dataflow                  import dataflow        as fl

from .. cities  .components       import city
from .. cities  .components       import dst_from_files
from .. core    .core_functions   import in_range
from .. core    .configure        import EventRangeType
from .. core    .configure        import OneOrManyFiles
from .. core                      import tbl_functions as tbl
from .. database.load_db          import DataSiPM
from .. io      .dst_io           import df_writer
from .. io      .run_and_event_io import run_and_event_writer

from .. reco    .psf_functions    import create_psf
from .. reco    .psf_functions    import hdst_psf_processing

from typing import Sequence
from typing import Tuple


@city
def eutropia(files_in    : OneOrManyFiles,
             file_out    : str,
             compression : str,
             event_range : EventRangeType,
             detector_db : str,
             run_number  : int,
             xrange      : Sequence[float],
             yrange      : Sequence[float],
             zbins       : Sequence[float],
             xsectors    : Sequence[float],
             ysectors    : Sequence[float],
             bin_size_xy : float):
    """Compute the Point Spread Function (PSF) from reconstructed hits.

    Averages the spatial charge distribution relative to the barycenter
    for pointlike events, organized by XY sectors and Z slices.

    Parameters
    ----------
    files_in : OneOrManyFiles
        Input hits files.
    file_out : str
        Output file path.
    compression : str
        HDF5 compression filter.
    event_range : EventRangeType
        Events to process.
    detector_db : str
        Detector database identifier.
    run_number : int
        Run number.
    xrange, yrange : Sequence[float]
        XY coordinate ranges for the PSF grid.
    zbins : Sequence[float]
        Z-axis bin edges for slicing.
    xsectors, ysectors : Sequence[float]
        XY sector boundaries for PSF averaging.
    bin_size_xy : float
        PSF grid bin size in mm.
    """
    sipms   = DataSiPM(detector_db, run_number)
    ranges  = xrange  , yrange
    sectors = xsectors, ysectors

    nbin      = (np.diff(ranges, axis=1) / bin_size_xy).astype(int).flatten()
    bin_edges = [np.linspace(*limits, n+1) for limits, n in zip(ranges, nbin)]


    create_slices_z   = fl.flatmap( z_splitter(zbins)
                                  , args = "dst"
                                  , out  = ("z", "dst_z")
                                  )

    process_slice     = fl.    map( hdst_processor(ranges, sipms)
                                  , item = "dst_z"
                                  )

    create_sectors_xy = fl.flatmap( xy_splitter(sectors, bin_edges)
                                  , args = "dst_z"
                                  , out  = ("x", "y", "dst_xy")
                                  )

    compute_psf = fl.map( psf_builder(bin_edges)
                        , args = "dst_xy"
                        , out  = ("factors", "entries", "centers")
                        )

    build_dataframe = fl.map( df_builder
                            , args = ("x", "y", "z", "factors", "entries", "centers")
                            , out  = "psf"
                            )

    flatten_event_info = fl.flatmap( zip
                                   , args = ("event_numbers", "timestamps")
                                   , out  = ("event_number" , "timestamp" )
                                   )

    accumulate_psf = fl.reduce(combine_psfs, None)()

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        write_event_info = fl.sink( run_and_event_writer(h5out)
                                  , args = ("run_number", "event_number", "timestamp")
                                  )

        result = fl.push( source = dst_from_files(files_in, "RECO", "Events")
                        , pipe   = fl.pipe( fl.branch( flatten_event_info
                                                     ,   write_event_info
                                                     )
                                          , create_slices_z
                                          , process_slice
                                          , create_sectors_xy
                                          , compute_psf
                                          , build_dataframe
                                          , "psf"
                                          , accumulate_psf.sink
                                          )
                        , result = accumulate_psf.future
                        )

        df_writer(h5out, result
                 , "PSF", "PSFs"
                 , descriptive_string = f"PSF with {bin_size_xy} mm bin size"
                 )

    return result


def z_splitter(zbins : Sequence[float]):
    """Create a function that splits a hits DataFrame into z-slices by bin edges.

    Parameters
    ----------
    zbins : Sequence[float]
        Z-axis bin edges defining slices.

    Returns
    -------
    Callable
        Generator function yielding (z_centre, sliced_df) for each non-empty z slice.
    """
    def split_in_z(dst):
        """Yield (z_centre, hits) for each non-empty z bin."""
        for zmin, zmax in zip(zbins, zbins[1:]):
            zpsf = (zmin + zmax) / 2

            sel  = in_range(dst.Z, zmin, zmax)
            if not np.any(sel): continue

            yield zpsf, dst[sel]

    return split_in_z

def hdst_processor(ranges : Sequence[float], sipms : pd.DataFrame):
    """Create a processor that applies PSF-specific transformations to a slice.

    Parameters
    ----------
    ranges : Sequence[float]
        XY coordinate ranges for normalization.
    sipms : pd.DataFrame
        SiPM database with positions and calibration data.

    Returns
    -------
    Callable
        Function that processes a slice DataFrame for PSF computation.
    """
    def process_hdst(dst):
        """Apply PSF processing (charge normalization) to a z slice."""
        return hdst_psf_processing(dst, ranges, sipms)
    return process_hdst


def xy_splitter(sectors : Sequence[float], bin_edges : Sequence[float]):
    """Create a function that splits a slice into xy-sector sub-slices.

    Parameters
    ----------
    sectors : Sequence[float]
        X and Y sector bin edges.
    bin_edges : Sequence[float]
        PSF bin edges for relative coordinate computation.

    Returns
    -------
    Callable
        Generator function yielding (x_centre, y_centre, sliced_df) per sector.
    """
    def split_in_xy(dst):
        """Yield (x_centre, y_centre, hits) for each non-empty xy sector."""
        xsectors, ysectors = sectors
        for xmin, xmax in zip(xsectors, xsectors[1:]):
            xpsf = np.round((xmin + xmax) / 2, 2)
            selx = in_range(dst.Xpeak, xmin, xmax)

            for ymin, ymax in zip(ysectors, ysectors[1:]):
                ypsf = np.round((ymin + ymax) / 2, 2)
                sely = in_range(dst.Ypeak, ymin, ymax)

                sel  = selx & sely
                if not np.any(sel): continue

                yield xpsf, ypsf, dst[sel]
    return split_in_xy


def psf_builder(bin_edges : Sequence[float]):
    """Create a function that computes a PSF from a sector's hits.

    Parameters
    ----------
    bin_edges : Sequence[float]
        XY bin edges for the PSF grid.

    Returns
    -------
    Callable
        Function that returns (factors, entries, bin_centers) for a sector.
    """
    def build(df):
        """Compute PSF factors, entries, and bin centers from sector hits."""
        psf, entries, bin_centers = create_psf( (df.RelX, df.RelY)
                                              ,  df.NormQ
                                              , bin_edges
                                              )
        return psf, entries, bin_centers
    return build


def df_builder(x       : Sequence[float]
              , y       : Sequence[float]
              , z       : Sequence[float]
              , factors : Sequence[float]
              , entries : Sequence[float]
              , centers : Tuple[Sequence[float], Sequence[float]]
              ):
    """Assemble PSF data from sector results into a flat DataFrame.

    Parameters
    ----------
    x, y, z : Sequence[float]
        Sector centre coordinates.
    factors : Sequence[float]
        PSF charge fractions per bin.
    entries : Sequence[float]
        Number of entries contributing to each bin.
    centers : Tuple[Sequence[float], Sequence[float]]
        Relative bin center coordinates (xr, yr).

    Returns
    -------
    pd.DataFrame
        PSF DataFrame with xr, yr, zr, x, y, z, factor, nevt columns.
    """
    xr, yr = centers
    xr, yr = np.meshgrid(xr, yr, indexing="ij")
    xr, yr = np.ravel(xr), np.ravel(yr)

    return pd.DataFrame(dict( xr=xr, yr=yr, zr=0.
                            , x =x , y =y , z =z
                            , factor = factors.flatten()
                            , nevt   = entries.flatten().astype(np.uint)
                            ))


def combine_psfs(acc : pd.DataFrame, new : pd.DataFrame):
    """Merge two PSF DataFrames by averaging charge factors weighted by entries.

    Parameters
    ----------
    acc : pd.DataFrame or None
        Accumulated PSF DataFrame. None on first call.
    new : pd.DataFrame
        New PSF DataFrame to merge.

    Returns
    -------
    pd.DataFrame
        Combined PSF with averaged factors and summed entries.
    """
    if acc is None:
        return new

    columns  = ["xr", "yr", "zr", "x", "y", "z"]
    acc      = acc.assign(factor=acc.factor * acc.nevt)
    new      = new.assign(factor=new.factor * new.nevt)
    combined = pd.concat( [acc, new]
                        , ignore_index = True
                        , sort         = False
                        )
    combined = combined.groupby(columns, as_index=False).agg("sum")
    average  = combined.factor / combined.nevt
    acc      = combined.assign(factor = np.nan_to_num(average))
    return acc
