import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import sys


class nnetar(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def fit(self, P = 1, repeats = 20, xreg = None, lambday = None, model = None,
            subset = None, scale_inputs = True):

        #p, size missing args?

        # Convert the Python time series to an R time series
        rdf = pandas2ri.py2ri(self.ts)
        # Create a call string setting variables as necessary
        ro.globalenv['r_timeseries'] = rdf
        command = 'nnetar(r_timeseries'
        self.fitted = None

        try:
            #ro.globalenv['p'] = p
            ro.globalenv['P'] = P
            ro.globalenv['repeats'] = repeats

            #xreg is tricky - not yet supporting (TODO)
            #ro.globalenv['xreg']

            ro.globalenv['lambda'] = ro.rinterface.NULL if lambday is None else lambday
            # Model - we need to input a previous model, this probably should be a true false, see other models as well
            #ro.globalenv['model']
            #ro.globalenv['subset]
            ro.globalenv['scale.inputs'] = ro.rinterface.TRUE if scale_inputs else ro.rinterface.FALSE
            command += ", P=P, repeats=repeats, lambda=lambda, scale.inputs=scale.inputs"
            command += ')'

            # Fit the time series
            self.r_forecastobject = ro.r(command)
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            # Fitted points
            self.fitted = ro.r('fitted(r_forecastobject)').ravel() # numpy.ndarray (unraveled to 1D)
            logging.info("nnetar fit successful")
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


