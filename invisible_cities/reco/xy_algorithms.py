import sys

import numpy as np

from invisible_cities.reco.params import Cluster

def find_algorithm(algoname):
    if algoname in sys.modules[__name__].__dict__:
        return getattr(sys.modules[__name__], algoname)
    else:
        raise ValueError("The algorithm <{}> does not exist".format(algoname))


def barycenter(xs, ys, qs):
    q    = np.sum(qs)
    n    = len(qs)
    x    = np.average(xs, weights=qs)
    y    = np.average(qs, weights=qs)
    xvar = np.sum(qs * (xs - x)**2) / (q - 1) if n else 0
    yvar = np.sum(qs * (ys - y)**2) / (q - 1) if n else 0

    c    = Cluster(q, x, y, xvar**0.5, yvar**0.5, n)
    return [c]

