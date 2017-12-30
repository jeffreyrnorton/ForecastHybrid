import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.frequency
import numpy as np


class Arima:
    def __init__(self, timeseries):
        self.ts = timeseries
        self.p = self.d = self.q = None
        self.r_arima = None
        self.fitted = None
        self.forecasted = None
        pandas2ri.activate()
        ro.r('library(forecast)')

    def __del__(self):
        pandas2ri.deactivate()

    def fit(self, d = None, D = None, maxp = 5, maxq = 5, maxP = 2,
            maxQ = 2, maxorder = 5, maxd = 2, maxD = 1, startp = 2,
            startq = 2, startP = 1, startQ = 1, stationary = False,
            seasonal = True, ic = ["aicc", "aic", "bic"], stepwise = True,
            trace = False, approximation = None,
            truncate = None, xreg = None, test = ["kpss", "adf", "pp"],
            seasonaltest = ["ocsb", "ch"], allowdrift = True, allowmean = True,
            lambdav = None, biasadj = False, parallel = False, numcores = 2):

        if approximation is None:
            approximation = len(self.ts) > 150 | ForecastHybrid.frequency.frequency(self.ts) > 12

        # Convert the Python time series to an R time series
        rdf = pandas2ri.py2ri(self.ts)
        # Create a call string setting variables as necessary
        ro.globalenv['r_timeseries'] = rdf
        command = 'auto.arima(r_timeseries'
        if d is not None:
            ro.globalenv['d'] = d
            command += ', d=d'
        if D is not None:
            ro.globalenv['D'] = D
            command += ', D=D'

        ro.globalenv['max.p'] = maxp
        command += ', max.p=max.p'

        ro.globalenv['max.q'] = maxq
        command += ', max.q=max.q'

        ro.globalenv['max.P'] = maxP
        command += ', max.P=max.P'

        ro.globalenv['max.Q'] = maxQ
        command += ', max.Q=max.Q'

        ro.globalenv['max.order'] = maxorder
        command += ', max.order=max.order'

        ro.globalenv['max.d'] = maxd
        command += ', max.d=max.d'

        ro.globalenv['max.D'] = maxD
        command += ', max.D=max.D'

        ro.globalenv['start.p'] = startp
        command += ', start.p=start.p'

        ro.globalenv['start.q'] = startq
        command += ', start.q=start.q'

        ro.globalenv['start.P'] = startP
        command += ', start.P=start.P'

        ro.globalenv['start.Q'] = startQ
        command += ', start.Q=start.Q'

        ro.globalenv['stationary'] = 'TRUE' if stationary else 'FALSE'
        command += ', stationary=stationary'

        ro.globalenv['seasonal'] = 'TRUE' if seasonal else 'FALSE'
        command += ', seasonal=seasonal'

        command += ')'

        # Fit the time series
        self.r_arima = ro.r(command)
        ro.globalenv['r_arima'] = self.r_arima
        # Fitted points
        self.fitted = ro.r('fitted(r_arima)').ravel() # numpy.ndarray (unraveled to 1D)
        # Orders
        [self.p, self.d, self.q] = ro.r('arimaorder(r_arima)').ravel() # numpy.ndarray
        return self.fitted

    def forecast(self, h=5):
        # Forecast some h ahead
        sr_forecast = 'forecast(r_arima, h={})'.format(h)
        self.forecasted = np.array(ro.r(sr_forecast).rx)
        #ro.globalenv['r_forecast'] = r_forecast
        return self.forecasted

