import sys

import numpy as np


def find_algorithm(algoname):
    if algoname in sys.modules[__name__].__dict__:
        return getattr(sys.modules[__name__], algoname)
    else:
        raise ValueError("The algorithm <{}> does not exist".format(algoname))


def barycenter(xs, ys, qs):
    x = np.average(xs, weights=qs)
    y = np.average(qs, weights=qs)
    q = np.sum(qs)
    n = len(qs)
    return [x], [y], [q], [n]

