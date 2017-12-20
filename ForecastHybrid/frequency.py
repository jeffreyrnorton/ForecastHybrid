import rpy2.robjects as ro
from rpy2.robjects import pandas2ri


def frequency(ts):
    pandas2ri.activate()
    ro.r('library(stats)')
    rdf = pandas2ri.py2ri(ts)
    ro.globalenv['ts'] = rdf
    return int(ro.r('frequency(ts)').ravel()[0])


# import pandas as pd
# import numpy as np
# rng = pd.date_range('1/1/2011', periods=72, freq='H')
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# out = frequency(ts)
# print(out)