import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import time


class thetam(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def fitR(self, **kwargs):
        ro.r("rm(list=ls())")
        self.setTimeSeries(period=1)
        return self.fitKernel(None)

    def fit(self):
        return self.fitR()

    def fitKernel(self, command):
        try:
            # A dirty trick here - we are just going to put a robust spinoff of the original
            # hybrid forecast code.
            command =  """
              library(stats)
              library(forecast)
              n <- length(r_timeseries)
              m <- stats::frequency(r_timeseries)
              if (n <= m)
              {
                 return(NULL)
              }
              if (m > 1)
              {
                 r <- as.numeric(stats::acf(r_timeseries, lag.max = m, plot = FALSE)$acf)[-1]
                 stat <- sqrt((1 + 2 * sum(r[-m] ^ 2)) / n)
                 seasonal <- (abs(r[m]) / stat > stats::qnorm(0.95))
              } else {
                 seasonal <- FALSE
              }
              origy <- r_timeseries
              if (seasonal)
              {
                 decomp <- stats:: decompose(r_timeseries, type="multiplicative")
                 r_timeseries <- forecast::seasadj(decomp)
              }
              object <- forecast::ets(r_timeseries, model="ANN", opt.crit = "mse")
              if (seasonal) {
                 object$seasadj <- utils::tail(decomp$seasonal, m)
                 object$seasadjhist <- decomp$seasona
              }
              object$seasonal <- seasonal
              object$x <- origy
              object$drift <- stats::lsfit(0: (n - 1), r_timeseries)$coef[2] / 2
              object$method <- "Theta"
              class(object) <- c("thetam", "ets")
              return (object)"""

            # Fit the time series
            self.dumpRCommandEnv(command)
            start_time = time.time()
            self.r_forecastobject = ro.r(command)
            logging.info("[R]thetam ran in {} sec".format(time.time()-start_time))
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            # Fitted points
            self.fitted = ro.r('fitted(r_forecastobject)').ravel() # numpy.ndarray (unraveled to 1D)
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


