import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import sys


class tbats(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def fit(self, use_box_cox = None, use_trend = None, use_damped_trend = None,
            seasonal_periods = None, use_arma_errors = True,
            use_parallel = None, num_cores = 2, bc_lower = 0,
            bc_upper = 1, biasadj = False):

        if use_parallel is None:
            if len(self.ts) > 1000:
                use_parallel = True
            else:
                use_parallel = False

        # Convert the Python time series to an R time series
        rdf = pandas2ri.py2ri(self.ts)
        # Create a call string setting variables as necessary
        ro.globalenv['r_timeseries'] = rdf
        command = 'tbats(r_timeseries'
        self.fitted = None

        try:
            if use_box_cox is None:
                ro.globalenv['use.box.cox'] = ro.rinterface.NULL
            else:
                ro.globalenv['use.box.cox'] = ro.rinterface.TRUE if use_box_cox else ro.rinterface.FALSE
            if use_trend is None:
                ro.globalenv['use.trend'] = ro.rinterface.NULL
            else:
                ro.globalenv['use.trend'] = ro.rinterface.TRUE if use_trend else ro.rinterface.FALSE
            if use_damped_trend is None:
                ro.globalenv['use.damped.trend'] = ro.rinterface.NULL
            else:
                ro.globalenv['use.damped.trend'] = ro.rinterface.TRUE if use_damped_trend else ro.rinterface.FALSE

            command += ", use.box.cox=use.box.cox, use.trend=use.trend, use.damped.trend=use.damped.trend"

            # seasonal.periods - (this should be an array - we need to check data types throughout I think!

            ro.globalenv['use.arma.errors'] = ro.rinterface.TRUE if use_arma_errors else ro.rinterface.FALSE
            ro.globalenv['use.parallel'] = ro.rinterface.TRUE if use_parallel else ro.rinterface.FALSE
            ro.globalenv['num.cores'] = num_cores
            command += ", use.arma.errors = use.arma.errors, use.parallel=use.parallel, num.cores=num.cores"
            ro.globalenv['bc.lower'] = bc_lower
            ro.globalenv['bc.upper'] = bc_upper
            ro.globalenv['biasadj'] = ro.rinterface.TRUE if biasadj else ro.rinterface.FALSE
            command += ", bc.lower=bc.lower, bc.upper=bc.upper, biasadj=biasadj"
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
            logging.warning("Running tbats without any arguments except for the time series")
            try:
                command = 'tbats(r_timeseries)'
                self.r_forecastobject = ro.r(command)
                ro.globalenv['r_forecastobject'] = self.r_forecastobject
                # Fitted points
                self.fitted = ro.r('fitted(r_forecastobject)').ravel()  # numpy.ndarray (unraveled to 1D)
            except:
                logging.error("Failure to fit data with tbats")

        return self.fitted


    def forecast(self, h=5, level=[80,95], fan=False, robust=False, lambdav=None,
                 findfrequency=False):
        # Make the forecast
        fcst = self.rforecast(h, level, fan, robust, lambdav, findfrequency)
        self.forecasted = self.extractRFcst(fcst, indices={'fidx':1, 'nbands':2, 'lower':6, 'upper':5})
        return self.forecasted