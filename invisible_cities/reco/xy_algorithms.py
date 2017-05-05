import sys

import numpy as np

from invisible_cities.reco.params import Cluster
from invisible_cities.core.system_of_units_c import units

def find_algorithm(algoname):
    if algoname in sys.modules[__name__].__dict__:
        return getattr(sys.modules[__name__], algoname)
    else:
        raise ValueError("The algorithm <{}> does not exist".format(algoname))


def barycenter(xs, ys, qs, default=np.nan):
    q    = np.sum(qs)
    n    = len(qs)
    x    = np.average(xs, weights=qs)         if n and q>0 else default
    y    = np.average(ys, weights=qs)         if n and q>0 else default
    xvar = np.sum(qs * (xs - x)**2) / (q - 1) if n and q>0 else default
    yvar = np.sum(qs * (ys - y)**2) / (q - 1) if n and q>0 else default

    c    = Cluster(q, x, y, xvar**0.5, yvar**0.5, n)
    return c


def corona(xs, ys, qs, rmax=30*units.mm, T=10.0*units.pes):
    """
    rmax is the maximum radius of a cluster
    T is the threshold for local maxima (this kwarg may be unnecessary)
    returns a list of Clusters
    """
    c = []
    xs = np.copy(xs)
    ys = np.copy(ys)
    qs = np.copy(qs)

    # While there are more local maxima
    while len(qs) > 0:
        i_max = np.argmax(qs)    # SiPM with largest Q
        if qs[i_max] < T: break  # largest Q remaining is negligible

        # get SiPMs within rmax of SiPM with largest Q
        dists = np.sqrt((xs - xs[i_max]) ** 2 + (ys - ys[i_max]) ** 2)
        cluster = np.where(dists < rmax)[0]

        # get barycenter of this cluster
        c.append(barycenter(xs[cluster], ys[cluster], qs[cluster]))

        xs = np.delete(xs, cluster) # delete the SiPMs
        ys = np.delete(ys, cluster) # contributing to
        qs = np.delete(qs, cluster) # this cluster

    return c