import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import ForecastHybrid.frequency
import statsmodels.tsa.stattools as stattools
import statsmodels.api as statapi
import logging
import scipy.stats
import math


class thetam(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def fit(self):

        try:
            self.fitted = None

            # Convert the Python time series to an R time series
            rdf = pandas2ri.py2ri(self.ts)
            # Create a call string setting variables as necessary
            ro.globalenv['y'] = rdf
            # A dirty trick here - we are just going to put a robust spinoff of the original
            # hybrid forecast code.
            command =  """
              library(stats)
              library(forecast)
              n <- length(y)
              m <- stats::frequency(y)
              if (n <= m)
              {
                 return(NULL)
              }
              if (m > 1)
              {
                 r <- as.numeric(stats::acf(y, lag.max = m, plot = FALSE)$acf)[-1]
                 stat <- sqrt((1 + 2 * sum(r[-m] ^ 2)) / n)
                 seasonal <- (abs(r[m]) / stat > stats::qnorm(0.95))
              } else {
                 seasonal <- FALSE
              }
              origy <- y
              if (seasonal)
              {
                 decomp <- stats:: decompose(y, type="multiplicative")
                 y <- forecast::seasadj(decomp)
              }
              object <- forecast::ets(y, model="ANN", opt.crit = "mse")
              if (seasonal) {
                 object$seasadj <- utils::tail(decomp$seasonal, m)
                 object$seasadjhist <- decomp$seasona
              }
              object$seasonal <- seasonal
              object$x <- origy
              object$drift <- stats::lsfit(0: (n - 1), y)$coef[2] / 2
              object$method <- "Theta"
              class(object) <- c("thetam", "ets")
              return (object)"""

            # Fit the time series
            self.r_forecastobject = ro.r(command)
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            # Fitted points
            self.fitted = ro.r('fitted(r_forecastobject)').ravel() # numpy.ndarray (unraveled to 1D)
            logging.info("thetam fit successful")
        except:
            logging.error("Failure to fit data with thetam")

        return self.fitted




