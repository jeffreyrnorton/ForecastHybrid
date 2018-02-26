import rpy2.robjects as ro
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import time
from rpy2.robjects import pandas2ri
import numpy as np
import traceback
import pandas as pd


# Arima is a wrapper on the R function auto.arima
#
class Arima(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)
        self.p = self.d = self.q = None

    def myname(self):
        return "auto.arima"

    def fitR(self, **kwargs):
        ro.r("rm(list=ls())")
        self.setTimeSeries(period=1)
        command = self.setREnv("auto.arima", **kwargs)
        return self.fitKernel(command)

    def fitKernel(self, command):
        try:
            # Fit the time series
            self.dumpRCommandEnv(command)
            start_time = time.time()
            self.r_forecastobject = ro.r(command)
            logging.info("[R]auto.arima ran in {} sec".format(time.time()-start_time))
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            self.extractFit(indices={'fidx':3, 'nbands':2, 'lower':4, 'upper':5})
            # Orders
            [self.p, self.d, self.q] = np.asarray(ro.r('arimaorder(r_forecastobject)')).ravel()  # numpy.ndarray
            logging.info("auto.arima fit successful")
        except:
            traceback.print_exc()
            logging.debug(self.rtracebackerror())
            logging.warning("Running auto.arima without any arguments except for the time series")
            try:
                command = 'auto.arima(r_timeseries)'
                self.r_forecastobject = ro.r(command)
                ro.globalenv['r_forecastobject'] = self.r_forecastobject
                logging.info("auto.arima successful")        # Fitted points
                self.extractFit(indices={'fidx': 3, 'nbands': 2, 'lower': 4, 'upper': 5})
                # Orders
                [self.p, self.d, self.q] = ro.r('arimaorder(r_forecastobject)').ravel() # numpy.ndarray
                logging.info("auto.arima successful with only time series")
            except:
                logging.error("Failure to fit data with auto.arima")

        return self.fitted

    def fit(self, d = None, D = None, maxp = 5, maxq = 5, maxP = 2,
            maxQ = 2, maxorder = 5, maxd = 2, maxD = 1, startp = 2,
            startq = 2, startP = 1, startQ = 1, stationary = False,
            seasonal = True, ic = ['aicc', 'aic', 'bic'], stepwise = True,
            trace = False, approximation = None,
            truncate = None, xreg = None, test = ["kpss", "adf", "pp"],
            seasonaltest = ["ocsb", "ch"], allowdrift = True, allowmean = True,
            lambdav = None, biasadj = False, parallel = False, numcores = 2):

        ro.r("rm(list=ls())")

        self.setTimeSeries(period=1)
        if approximation is None:
            rapprox = np.asarray(ro.r("length(r_timeseries) > 150 | frequency(r_timeseries) > 12")).ravel()
            approximation = bool(rapprox[0] == 1)

        aargs = self.convertArgsToR(d, D, maxp, maxq, maxP, maxQ, maxorder,
                                    maxd, maxD, startp, startq, startP, startQ,
                                    stationary, seasonal, ic, stepwise, trace,
                                    approximation, truncate, xreg, test, seasonaltest,
                                    allowdrift, allowmean, lambdav, biasadj, parallel, numcores)
        command = self.setREnv("auto.arima", **aargs)
        return self.fitKernel(command)

    def convertArgsToR(self, d = None, D = None, maxp = 5, maxq = 5, maxP = 2,
            maxQ = 2, maxorder = 5, maxd = 2, maxD = 1, startp = 2,
            startq = 2, startP = 1, startQ = 1, stationary = False,
            seasonal = True, ic = ['aicc', 'aic', 'bic'], stepwise = True,
            trace = False, approximation = None,
            truncate = None, xreg = None, test = ["kpss", "adf", "pp"],
            seasonaltest = ["ocsb", "ch"], allowdrift = True, allowmean = True,
            lambdav = None, biasadj = False, parallel = False, numcores = 2):

        ro.r("rm(list=ls())")

        self.setTimeSeries(period=1)
        if approximation is None:
            rapprox = ro.r("length(r_timeseries) > 150 | frequency(r_timeseries) > 12").ravel()
            approximation = bool(rapprox[0] == 1)

        aargs = {}
        if d is not None: aargs['d'] = d
        if D is not None: aargs['D'] = D
        aargs['max.p'] = maxp
        aargs['max.q'] = maxq
        aargs['max.P'] = maxP
        aargs['max.Q'] = maxQ
        aargs['max.order'] = maxorder
        aargs['max.d']   = maxd
        aargs['max.D']   = maxD
        aargs['start.p'] = startp
        aargs['start.q'] = startq
        aargs['start.P'] = startP
        aargs['start.Q'] = startQ
        aargs['stationary'] = stationary
        aargs['seasonal'] = seasonal
        aargs['ic'] = "aic" if isinstance(ic,list) else ic
        aargs['stepwise'] = stepwise
        aargs['trace'] = trace
        aargs['approximation'] = approximation
        if truncate is not None: aargs['truncate'] = truncate

        # Currently not supporting xreg, test, or seasonal.test.  These tend to act poorly
        # when set and need a lot of tender care - let the underlying algorithm take care of it.
        #
        #ro.globalenv['xreg'] = ro.rinterface.NA_Real if xreg is None else pandas2ri.ri2py(xreg)
        #ro.globalenv['test'] = ro.rinterface.NA_Character if test is list else test
        #ro.globalenv['seasonal.test'] = ro.rinterface.NA_Character if seasonaltest is list else seasonaltest
        aargs['allowdrift'] = allowdrift
        aargs['allowmean'] = allowmean
        # Also dropping support for lambda at this point and letting the algorithm do its thing
        #ro.globalenv['lambda'] = ro.rinterface.NA_Real if lambdav is None else lambdav
        aargs['biasadj'] = biasadj
        aargs['parallel'] = parallel
        aargs['num.cores'] = numcores
        return aargs


    def forecast(self, h=5, level=[80,95], fan=False, robust=False, lambdav=None,
                 findfrequency=False):
        # Arima shift is one - tends to be one for most types
        # Make the forecast
        fcst = self.rforecast(h, level, fan, robust, lambdav, findfrequency)
        self.forecasted = self.extractRFcst(fcst, indices={'fidx':3, 'nbands':2, 'lower':4, 'upper':5})
        return self.forecasted


    def refit(self, ts):
        # Go ahead and reset the data and the model
        rdf = pandas2ri.py2ri(ts)
        # Create a call string setting variables as necessary
        arr = np.array(self.r_forecastobject)
        ro.r("library(forecast)")
        ro.globalenv['rts'] = rdf
        ro.globalenv['arimamod'] = self.r_forecastobject
        refitR = ro.r("Arima(y=rts, model=arimamod)")
        self.r_forecastobject = refitR
        ro.globalenv['r_forecastobject'] = refitR
        self.fitted = ro.r('fitted(r_forecastobject)')
        return self.fitted