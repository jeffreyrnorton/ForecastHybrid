import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import ForecastHybrid.frequency
import logging
import sys


# Arima is a wrapper on the R function auto.arima
#
class Arima(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)
        self.p = self.d = self.q = None

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
        self.fitted = None

        try:

            ro.globalenv['d'] = ro.rinterface.NA_Integer if d is None else d
            ro.globalenv['D'] = ro.rinterface.NA_Integer if D is None else D
            command += ', d=d, D=D'

            ro.globalenv['max.p'] = maxp
            ro.globalenv['max.q'] = maxq
            ro.globalenv['max.P'] = maxP
            ro.globalenv['max.Q'] = maxQ
            ro.globalenv['max.order'] = maxorder
            command += ', max.p=max.p, max.q=max.q, max.P=max.P, max.Q=max.Q, max.order=max.order'

            ro.globalenv['max.d']   = maxd
            ro.globalenv['max.D']   = maxD
            ro.globalenv['start.p'] = startp
            ro.globalenv['start.q'] = startq
            ro.globalenv['start.P'] = startP
            ro.globalenv['start.Q'] = startQ
            command += ", max.d=max.d, max.D=max.D, start.p=start.p, start.q=start.q, start.P=start.P, start.Q=start.Q"

            ro.globalenv['stationary'] = ro.rinterface.TRUE if stationary else ro.rinterface.FALSE
            command += ', stationary=stationary'
            ro.globalenv['seasonal'] = ro.rinterface.TRUE if seasonal else ro.rinterface.FALSE
            command += ', seasonal=seasonal'

            ic_s = "aic" if len(ic) > 1 else ic
            ro.globalenv['ic'] = ic_s
            ro.globalenv['stepwise'] = ro.rinterface.TRUE if stepwise else ro.rinterface.FALSE
            ro.globalenv['trace'] = ro.rinterface.TRUE if trace else ro.rinterface.FALSE
            ro.globalenv['approximation'] = ro.rinterface.TRUE if approximation else ro.rinterface.FALSE
            ro.globalenv['truncate'] = ro.rinterface.NULL if truncate is None else\
                  ro.rinterface.TRUE if truncate else ro.rinterface.FALSE

            command += ', ic=ic, stepwise=stepwise, trace=trace, approximation=approximation, truncate=truncate'

            # Currently not supporting xreg, test, or seasonal.test.  These tend to act poorly
            # when set and need a lot of tender care - let the underlying algorithm take care of it.
            #
            #ro.globalenv['xreg'] = ro.rinterface.NA_Real if xreg is None else pandas2ri.ri2py(xreg)
            #ro.globalenv['test'] = ro.rinterface.NA_Character if len(test) > 1 else test
            #ro.globalenv['seasonal.test'] = ro.rinterface.NA_Character if len(seasonaltest) > 1 else seasonaltest
            ro.globalenv['allowdrift'] = ro.rinterface.TRUE if allowdrift else ro.rinterface.FALSE
            command += ", allowdrift=allowdrift"

            ro.globalenv['allowmean'] = ro.rinterface.TRUE if allowmean else ro.rinterface.FALSE
            # Also dropping support for lambda at this point and letting the algorithm do its thing
            #ro.globalenv['lambda'] = ro.rinterface.NA_Real if lambdav is None else lambdav
            ro.globalenv['biasadj'] = ro.rinterface.TRUE if biasadj else ro.rinterface.FALSE
            ro.globalenv['parallel'] = ro.rinterface.TRUE if parallel else ro.rinterface.FALSE
            ro.globalenv['num.cores'] = numcores
            command += ', allowmean=allowmean, biasadj=biasadj, parallel=parallel, num.cores=num.cores'
            command += ')'

            # Fit the time series
            self.r_forecastobject = ro.r(command)
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            self.fitted = ro.r('fitted(r_forecastobject)').ravel()  # numpy.ndarray (unraveled to 1D)
            # Orders
            [self.p, self.d, self.q] = ro.r('arimaorder(r_forecastobject)').ravel()  # numpy.ndarray
            logging.info("auto.arima fit successful")

        except:
            logging.warning(sys.exc_info()[0])
            logging.warning("Running auto.arima without any arguments except for the time series")
            try:
                command = 'auto.arima(r_timeseries)'
                self.r_forecastobject = ro.r(command)
                ro.globalenv['r_forecastobject'] = self.r_forecastobject
                logging.info("auto.arima successful")        # Fitted points
                self.fitted = ro.r('fitted(r_forecastobject)').ravel() # numpy.ndarray (unraveled to 1D)
                # Orders
                [self.p, self.d, self.q] = ro.r('arimaorder(r_forecastobject)').ravel() # numpy.ndarray
                logging.info("auto.arima successful with only time series")
            except:
                logging.error("Failure to fit data with auto.arima")

        return self.fitted


