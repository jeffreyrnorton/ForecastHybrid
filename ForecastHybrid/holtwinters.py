import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import time


class holtwinters(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def fitR(self, **kwargs):
        ro.r("rm(list=ls())")
        self.setTimeSeries(period=1)
        command = self.setREnv("HoltWinter", **kwargs)
        return self.fitKernel(command)

    def fitKernel(self, command):
        try:
            # Fit the time series
            self.dumpRCommandEnv(command)
            start_time = time.time()
            self.r_forecastobject = ro.r(command)
            logging.info("[R]holtwinters ran in {} sec".format(time.time() - start_time))
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            # Fitted points
            self.fitted = ro.r('fitted(r_forecastobject)').ravel()  # numpy.ndarray (unraveled to 1D)
            logging.info("HoltWinter fit successful")
        except:
            logging.debug(self.rtracebackerror())
            logging.warning("Running HoltWinter without any arguments except for the time series")
            try:
                command = 'HoltWinter(r_timeseries)'
                self.r_forecastobject = ro.r(command)
                ro.globalenv['r_forecastobject'] = self.r_forecastobject
                # Fitted points
                self.fitted = ro.r('fitted(r_forecastobject)').ravel()  # numpy.ndarray (unraveled to 1D)
            except:
                logging.error("Failure to fit data with HoltWinter")

        return self.fitted

    def fit(self, alpha = None, beta = None, gamma = None, seasonal = ["additive", "multiplicative"],
            start_periods = 2, l_start = None, b_start = None, s_start = None,
            optim_start = {'alpha' : 0.3, 'beta' : 0.1, 'gamma' : 0.1}) :

        self.setTimeSeries(period=None)

        nperiods = ro.r("frequency(r_timeseries)")
        nperiods = int(nperiods[0])
        if nperiods < 2:
            logging.error("Cannot fit HoltWinters on time series with less than 2 periods")
            return self.fitted

        aargs = {}
        if alpha is not None: aargs['alpha'] = alpha
        if beta is not None: aargs['beta'] = beta
        if gamma is not None: aargs['gamma'] = gamma
        if isinstance(seasonal, list): aargs['seasonal'] = 'additive'
        else: aargs['seasonal'] = seasonal
        aargs['start.periods'] = start_periods
        if l_start is not None: aargs['l.start'] = l_start
        if b_start is not None: aargs['b.start'] = b_start
        if s_start is not None: aargs['s_start'] = s_start
        aargs['optim.start'] = optim_start
        return self.fitR(**aargs)


    def forecast(self, h=5, level=[80,95], fan=False, robust=False, lambdav=None,
                 findfrequency=False):
        # Make the forecast
        fcst = self.rforecast(h, level, fan, robust, lambdav, findfrequency)
        self.forecasted = self.extractRFcst(fcst, indices={'fidx':1, 'nbands':2, 'lower':4, 'upper':5})
        return self.forecasted