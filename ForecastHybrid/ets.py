import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import sys


class ets(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def fit(self, model='ZZZ', damped=None, alpha=None, beta=None,
            gamma=None, phi=None, additive_only=False, lambdal=None,
            biasadj=False, lower=[1.0e-4, 1.0e-4, 1.0e-4, 0.8], upper=[0.9999, 0.9999, 0.9999, 0.98],
            opt_crit=['lik', 'amse', 'mse', 'sigma', 'mae'], nmse=3,
            bounds=['both', 'usual', 'admissible'], ic=['aicc', 'aic', 'bic'],
            restrict=True, allow_multiplication_trend=False, use_initial_values=False):
        # Convert the Python time series to an R time series
        rdf = pandas2ri.py2ri(self.ts)
        # Create a call string setting variables as necessary
        ro.globalenv['r_timeseries'] = rdf
        command = 'ets(r_timeseries'
        self.fitted = None

        try:
            ro.globalenv['model'] = model
            if damped is None:
                ro.globalenv['damped'] = ro.rinterface.NULL
            else:
                ro.globalenv['damped'] = ro.rinterface.TRUE if damped else ro.rinterface.FALSE
            ro.globalenv['alpha'] = ro.rinterface.NULL if alpha is None else alpha
            ro.globalenv['beta'] = ro.rinterface.NULL if beta is None else beta
            ro.globalenv['gamma'] = ro.rinterface.NULL if gamma is None else gamma
            ro.globalenv['phi'] = ro.rinterface.NULL if phi is None else phi
            command += ', model=model, damped=damped, alpha=alpha, beta=beta, gamma=gamma, phi=phi'

            ro.globalenv['additive.only'] = ro.rinterface.TRUE if additive_only else ro.rinterface.FALSE
            ro.globalenv['lambda'] = ro.rinterface.TRUE if lambdal else ro.rinterface.FALSE
            ro.globalenv['biasadj'] = ro.rinterface.TRUE if biasadj else ro.rinterface.FALSE
            command += ', additive.only=additive.only, lambda=lambda, biasadj=biasadj'

            ro.globalenv['lower'] = pandas2ri.FloatSexpVector(lower)
            ro.globalenv['upper'] = pandas2ri.FloatSexpVector(upper)
            if len(opt_crit) == 1:
                ro.globalenv['opt.crit'] = opt_crit
            else:
                ro.globalenv['opt.crit'] = 'mse'
            ro.globalenv['nmse'] = nmse
            if len(bounds) == 1:
                ro.globalenv['bounds'] = bounds
            else:
                ro.globalenv['bounds'] = 'both'
            if len(ic) == 1:
                ro.globalenv['ic'] = ic
            else:
                ro.globalenv['ic'] = 'aicc'
            command += ', lower=lower, upper=upper, opt.crit=opt.crit, bounds=bounds, ic=ic'

            ro.globalenv['restrict'] = ro.rinterface.TRUE if restrict else ro.rinterface.FALSE
            ro.globalenv['allow.multiplicative.trend'] = ro.rinterface.TRUE if allow_multiplication_trend else ro.rinterface.FALSE
            ro.globalenv['use.initial.values'] = ro.rinterface.TRUE if use_initial_values else ro.rinterface.FALSE
            command += ', restrict=restrict, allow.multiplicative.trend=allow.multiplicative.trend, use.initial.values=use.initial.values'
            command += ')'

            # Fit the time series
            self.r_forecastobject = ro.r(command)
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            # Fitted points
            self.fitted = ro.r('fitted(r_forecastobject)').ravel() # numpy.ndarray (unraveled to 1D)
            logging.info("ets fit successful")
        except:
            print(sys.exc_info()[0])
            logging.warning(sys.exc_info()[0])
            logging.warning("Running ets without any arguments except for the time series")
            try:
                command = 'ets(r_timeseries)'
                self.r_forecastobject = ro.r(command)
                ro.globalenv['r_forecastobject'] = self.r_forecastobject
                # Fitted points
                self.fitted = ro.r('fitted(r_forecastobject)').ravel()  # numpy.ndarray (unraveled to 1D)
            except:
                logging.error("Failure to fit data with ets")

        return self.fitted

    def forecast(self, h=5, level=[80,95], fan=False, robust=False, lambdav=None,
                 findfrequency=False):
        # Make the forecast
        fcst = self.rforecast(h, level, fan, robust, lambdav, findfrequency)
        self.forecasted = self.extractRFcst(fcst, indices={'fidx':1, 'nbands':2, 'lower':4, 'upper':5})
        return self.forecasted