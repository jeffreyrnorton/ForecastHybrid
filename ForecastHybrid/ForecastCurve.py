import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import numpy as np


class ForecastCurve(object):
    def __init__(self, timeseries):
        self.ts = timeseries
        self.r_forecastobject = None
        self.fitted = None
        self.forecasted = None
        pandas2ri.activate()
        ro.r('library(forecast)')

    def __del__(self):
        pandas2ri.deactivate()

    def forecast(self, h=5, level=[80,95], fan=False, robust=False, lambdav=None,
                 findfrequency=False):
        # Forecast some h ahead
        ro.globalenv['h'] = 5
        ro.globalenv['level'] = ro.FloatVector(level)
        ro.globalenv['fan'] = ro.rinterface.TRUE if fan else ro.rinterface.FALSE
        ro.globalenv['robust'] = ro.rinterface.TRUE if robust else ro.rinterface.FALSE
        # Not support lamda right now because it is tricky
        #ro.globalenv['lambda'] = ro.rinterface.NA_Real if lambdav is None else lambdav
        ro.globalenv['find.frequency'] = ro.rinterface.TRUE if findfrequency else ro.rinterface.FALSE
        #ro.globalenv['r_forecastobject'] = self.r_forecastobject
        sr_forecast = 'forecast(r_forecastobject, h=h, level=level, fan=fan, robust=robust, ' \
            'find.frequency=find.frequency)'
        # Make the forecast
        fcst = ro.r(sr_forecast)
        self.forecasted = np.asarray(fcst)
        return self.forecasted