import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import time
import numpy as np
from math import sqrt
import statsmodels


class thetam(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)
        ro.r("library(stats)")

    def fitR(self, **kwargs):
        ro.r("rm(list=ls())")
        self.setTimeSeries(period=1)
        return self.fitKernel(None)

    def fit(self):
        return self.fitR()

    def i_check_seasonal(self):
        n = len(self.ts)
        m = int(ro.r("m = frequency(r_timeseries)")[0])
        if n <= m: return None

        ro.globalenv['n'] = n
        seasonal = False
        if m > 1:
            r = ro.r("r = as.numeric(acf(r_timeseries, lag.max = m, plot = FALSE)$acf)[-1]")
            stat = ro.r("stat <- sqrt((1 + 2 * sum(r[-m] ^ 2)) / n)")
            seasonal = ro.r("seasonal <- (abs(r[m]) / stat > stats::qnorm(0.95))")[0] != 0
        return seasonal

    def i_decompose(self, seasonal):
        origy = ro.r('origy <- r_timeseries')
        if seasonal:
            decomo = ro.r('decomp = decompose(r_timeseries, type="multiplicative")')
            ro.r('r_timeseries <- seasadj(decomp)')

    def i_set_r_object(self, seasonal):
        # Set the R model
        if seasonal:
            ro.r('r_forecastobject$theta.seasadj <- utils::tail(decomp$seasonal, m)')
            ro.r('r_forecastobject$theta.seasadjhist <- decomp$seasona')
        ro.r('r_forecastobject$theta.seasonal <- seasonal')
        ro.r('r_forecastobject$theta.x <- origy')
        ro.r('r_forecastobject$theta.drift <- stats::lsfit(0: (n - 1), r_timeseries)$coef[2] / 2')
        ro.r('r_forecastobject$theta.method <- "Theta"')
        ro.r('class(r_forecastobject) <- c("thetam", "ets")')
        self.r_forecastobject = ro.globalenv['r_forecastobject']

    def fitKernel(self, command):
        try:
            # Fit the time series
            self.dumpRCommandEnv(command)
            start_time = time.time()

            seasonal = self.i_check_seasonal()
            if seasonal is None:
                logging.error('Unable to fit thetam as the time series length is too short.')

            self.i_decompose(seasonal)

            # Fit the model
            fit_ets_model = ro.r('r_forecastobject <- ets(r_timeseries, mode="ANN", opt.crit="mse")')

            # Set the R model
            self.i_set_r_object(seasonal)

            logging.info("[R]thetam ran in {} sec".format(time.time()-start_time))
            # Fitted points
            self.extractFit(indices={'fidx':1, 'nbands':2, 'lower':4, 'upper':5})
            logging.info("thetam fit successful")
        except:
            logging.debug(self.rtracebackerror())
            logging.error("Failure to fit data with thetam")

        return self.fitted


    def forecast(self, h=5, level=[80,95], fan=False, robust=False, lambdav=None,
                 findfrequency=False):
        # Make the forecast
        fcst = self.rforecast(h, level, fan, robust, lambdav, findfrequency)
        self.forecasted = self.extractRFcst(fcst, indices={'fidx':1, 'nbands':2, 'lower':4, 'upper':5})
        return self.forecasted

    def refit(self, ts):
        # Go ahead and reset the data and the model
        self.ts = ts
        rdf = pandas2ri.py2ri(ts)
        # Create a call string setting variables as necessary
        arr = np.array(self.r_forecastobject)
        ro.r("library(forecast)")
        ro.globalenv['r_timeseries'] = rdf
        ro.globalenv['etsmodel'] = self.r_forecastobject
        try:

            seasonal = self.i_check_seasonal()

            if seasonal is None:
                logging.error('Unable to fit thetam as the time series length is too short.')

            self.i_decompose(seasonal)

            start_time = time.time()

            # Set the class on etsmodel to ets
            ro.r('class(etsmodel) <- "ets"')
            # And perform the refitu
            fit_ets_model = ro.r('r_forecastobject <- ets(r_timeseries, model=etsmodel, use.initial.values=TRUE)')
            self.i_set_r_object(seasonal)

            logging.info("[R]thetam ran in {} sec".format(time.time()-start_time))
            # Fitted points
            self.fitted = np.asarray(ro.r('fitted(r_forecastobject)')).ravel()
            logging.info("thetam refit successful")
        except:
            logging.debug(self.rtracebackerror())
            logging.error("Failure to fit data with thetam")

        return self.fitted