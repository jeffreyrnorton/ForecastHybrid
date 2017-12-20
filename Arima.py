import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
import frequency


class Arima:
    def __init__(self, ts=None):
        p = d = q = 0
        r_arima = None
        fitted = None
        forecast = None
        pandas2ri.activate()
        ro.r('library(forecast)')
        if ts is not None:
            self.fit(ts)


    def __del__(self):
        pandas2ri.deactivate()


    def fit(self, ts, d = None, D = None, maxp = 5, maxq = 5, maxP = 2,
        maxQ = 2, maxorder = 5, maxd = 2, maxD = 1, startp = 2,
        startq = 2, startP = 1, startQ = 1, stationary = False,
        seasonal = True, ic = ["aicc", "aic", "bic"], stepwise = True,
        trace = False, approximation = (len(ts) > 150 | frequency(x) > 12),
        truncate = None, xreg = None, test = ["kpss", "adf", "pp"],
        seasonaltest = ["ocsb", "ch"], allowdrift = True, allowmean = True,
        lambdav = None, biasadj = False, parallel = False, numcores = 2, x = ts):

        # Convert the Python time series to an R time series
        rdf = pandas2ri.py2ri(ts)
        ro.globalenv['r_timeseries'] = rdf
        # Fit the time series
        self.r_arima = ro.r('auto.arima(r_timeseries)')
        ro.globalenv['r_arima'] = self.r_arima
        # Fitted points
        fitted = ro.r('fitted(r_arima)').ravel() # numpy.ndarray (unraveled to 1D)
        # Orders
        [self.p, self.d, self.q] = ro.r('arimaorder(r_arima)').ravel() # numpy.ndarray
        return fitted

# d = NA, D = NA, max.p = 5, max.q = 5, max.P = 2,
#   max.Q = 2, max.order = 5, max.d = 2, max.D = 1, start.p = 2,
#   start.q = 2, start.P = 1, start.Q = 1, stationary = FALSE,
#   seasonal = TRUE, ic = c("aicc", "aic", "bic"), stepwise = TRUE,
#   trace = FALSE, approximation = (length(x) > 150 | frequency(x) > 12),
#   truncate = NULL, xreg = NULL, test = c("kpss", "adf", "pp"),
#   seasonal.test = c("ocsb", "ch"), allowdrift = TRUE, allowmean = TRUE,
#   lambda = NULL, biasadj = FALSE, parallel = FALSE, num.cores = 2,
#   x = y


    def forecast(self, h=5):
        # Forecast some h ahead
        sr_forecast = 'forecast(r_arima, h={})'.format(h)
        r_forecast = ro.r(sr_forecast)
        ro.globalenv['r_forecast'] = r_forecast
#pred = ro.r('as.data.frame(r_forecast)')
#


rng = pd.date_range('1/1/2011', periods=72, freq='H')
ts = pd.Series(np.random.randn(len(rng)), index=rng)

ar = Arima()
ar.fit(ts)
ar.forecast()