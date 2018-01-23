import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import sys


class stlm(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def fit(self, s_window = 7, robust = False, method = ['ets', 'arima'],
            modelfunction = None, model=None, etsmodel = 'ZZN',
            lambday = None, biasadj = None, xreg = None, allow_multiplicative_trend = False):

        # Convert the Python time series to an R time series
        rdf = pandas2ri.py2ri(self.ts)
        # Create a call string setting variables as necessary
        ro.globalenv['r_timeseries'] = rdf
        command = 'stlm(r_timeseries'
        self.fitted = None

        # Run a check to see if the data can be fit with stlm
        icode = ro.r(
            """
            if (frequency(r_timeseries) < 2L) {
               return(1)
            } else if (frequency(y) * 2L >= length(y)) {
               return(2)
            }
            return(0)
            """)
        icode == True if icode[0] == 1 else False
        if icode == 1:
            logging.error("The stlm model requires that the input data be a seasonal ts object. The stlm model will not be used.")
            return self.fitted
        else:
            logging.error("The stlm model requres a series more than twice as long as the seasonal period. The stlm model will not be used.")
            return self.fitted

        try:
            ro.globalenv['s.window'] = s_window
            ro.globalenv['robust'] = ro.rinterface.TRUE if robust else ro.rinterface.FALSE
            if len(method) == 1:
                ro.globalenv['method'] = method
            else:
                ro.globalenv['method'] = 'ets'
            #modelfunction not yet supported
            #model not yet supported
            ro.globalenv['etsmodel'] = etsmodel


            #xreg is tricky - not yet supporting (TODO)
            #ro.globalenv['xreg']

            ro.globalenv['lambda'] = ro.rinterface.NULL if lambday is None else lambday
            command += ", s.window=s.window, robust=robust, method=method, etsmodel=etsmodel, lambda=lambda"

            # Model - we need to input a previous model, this probably should be a true false, see other models as well
            #ro.globalenv['model']
            #ro.globalenv['subset]
            ro.globalenv['biasadj'] = ro.rinterface.TRUE if biasadj else ro.rinterface.FALSE
            #xreg not supported
            ro.globalenv['allow.multiplicative.trend'] = ro.rinterface.TRUE if allow_multiplicative_trend else ro.rinterface.FALSE
            command += ", biasadj=biasadj, allow.multiplicative.trend=allow.multiplicative.trend"
            command += ')'

            # Fit the time series
            self.r_forecastobject = ro.r(command)
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            # Fitted points
            self.fitted = ro.r('fitted(r_forecastobject)').ravel() # numpy.ndarray (unraveled to 1D)
            logging.info("tbats fit successful")
        except:
            print(sys.exc_info()[0])
            logging.warning(sys.exc_info()[0])
            logging.warning("Running nnetar without any arguments except for the time series")
            try:
                command = 'nnetar(r_timeseries)'
                self.r_forecastobject = ro.r(command)
                ro.globalenv['r_forecastobject'] = self.r_forecastobject
                # Fitted points
                self.fitted = ro.r('fitted(r_forecastobject)').ravel()  # numpy.ndarray (unraveled to 1D)
            except:
                logging.error("Failure to fit data with nnetar")

        return self.fitted


