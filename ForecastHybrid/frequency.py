import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

def frequency(ts):
    pandas2ri.activate()
    ro.r('library(stats)')
    rdf = pandas2ri.py2ri(ts)
    ro.globalenv['ts'] = rdf
    return int(ro.r('frequency(ts)').ravel()[0])