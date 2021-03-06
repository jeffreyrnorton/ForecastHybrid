import rpy2.robjects as ro
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import time
from rpy2.robjects import pandas2ri
import numpy as np


class ets(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def myname(self):
        return "ets"

    def fitR(self, **kwargs):
        ro.r("rm(list=ls())")
        rtperiod = 1
        if kwargs.get('period', None) is not None:
            try:
                rtperiod = int(kwargs['period'])
            except:
                rtperiod = 1
            kwargs.pop('period')
        self.setTimeSeries(period=rtperiod)
        command = self.setREnv("ets", **kwargs)
        return self.fitKernel(command)

    def fitKernel(self, command):
        try:
            # Fit the time series
            self.dumpRCommandEnv(command)
            start_time = time.time()
            self.r_forecastobject = ro.r(command)
            logging.info("[R]ets ran in {} sec".format(time.time() - start_time))
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            # Fitted points
            self.extractFit(indices={'fidx': 1, 'nbands': 2, 'lower': 4, 'upper': 5})

#            py_rep = dict(zip(self.r_forecastobject.names, list(self.r_forecastobject)))

            # Extract out the model

            logging.info("ets fit successful")
        except:
            logging.debug(self.rtracebackerror())
            logging.warning("Running ets without any arguments except for the time series")
            try:
                command = 'ets(r_timeseries)'
                self.r_forecastobject = ro.r(command)
                ro.globalenv['r_forecastobject'] = self.r_forecastobject
                # Fitted points
                self.extractFit(indices={'fidx': 1, 'nbands': 2, 'lower': 4, 'upper': 5})
            except:
                logging.error("Failure to fit data with ets")

        return self.fitted

    def fit(self, model='ZZZ', damped=None, alpha=None, beta=None,
            gamma=None, phi=None, additive_only=False, lambdal=None,
            biasadj=False, lower=[1.0e-4, 1.0e-4, 1.0e-4, 0.8], upper=[0.9999, 0.9999, 0.9999, 0.98],
            opt_crit=['lik', 'amse', 'mse', 'sigma', 'mae'], nmse=3,
            bounds=['both', 'usual', 'admissible'], ic=['aicc', 'aic', 'bic'],
            restrict=True, allow_multiplication_trend=False, use_initial_values=False):

        ro.r("rm(list=ls())")
        self.setTimeSeries(period=1)
        aargs = self.convertArgsToR(model, damped, alpha, beta, gamma, phi, additive_only,
                                    lambdal, biasadj, lower, upper, opt_crit, nmse, bounds, ic, restrict,
                                    allow_multiplication_trend, use_initial_values)
        command = self.setREnv("ets", **aargs)
        return self.fitKernel(command)

    def convertArgsToR(self, model='ZZZ', damped=None, alpha=None, beta=None,
            gamma=None, phi=None, additive_only=False, lambdal=None,
            biasadj=False, lower=[1.0e-4, 1.0e-4, 1.0e-4, 0.8], upper=[0.9999, 0.9999, 0.9999, 0.98],
            opt_crit=['lik', 'amse', 'mse', 'sigma', 'mae'], nmse=3,
            bounds=['both', 'usual', 'admissible'], ic=['aicc', 'aic', 'bic'],
            restrict=True, allow_multiplication_trend=False, use_initial_values=False):
        aargs = {}
        aargs['model'] = model
        if damped is not None: aargs['damped'] = damped
        if alpha is not None: aargs['alpha'] = alpha
        if beta is not None: aargs['beta'] = beta
        if gamma is not None: aargs['gamma'] = gamma
        if phi is not None: aargs['phi'] = phi
        if additive_only is not None: aargs['additive.only'] = additive_only
        if lambdal is not None: aargs['lambda'] = lambdal
        aargs['biasadj'] = biasadj
        aargs['lower'] = lower
        aargs['upper'] = upper
        if isinstance(opt_crit, list): aargs['opt.crit'] = 'mse'
        else: aargs['opt.crit'] = opt_crit
        aargs['nmse'] = nmse
        if isinstance(bounds, list): aargs['bounds'] = 'both'
        else: aargs['bounds'] = bounds
        if isinstance(ic, list): aargs['ic'] = 'aicc'
        else: aargs['ic'] = ic
        aargs['restrict'] = restrict
        aargs['allow.multiplicative.trend'] = allow_multiplication_trend
        aargs['use.initial.values'] = use_initial_values
        return aargs


    def forecast(self, h=5, level=[80,95], fan=False, robust=False, lambdav=None,
                 findfrequency=False):
        # Make the forecast
        fcst = self.rforecast(h, level, fan, robust, lambdav, findfrequency)
        self.forecasted = self.extractRFcst(fcst, indices={'fidx':1, 'nbands':2, 'lower':4, 'upper':5})
        return self.forecasted

    def refit(self, ts):
        # Go ahead and reset the data and the model
        rdf = pandas2ri.py2ri(ts)
        # Create a call string setting variables as necessary
        arr = np.array(self.r_forecastobject)
        ro.r("library(forecast)")
        ro.globalenv['rts'] = rdf
        ro.globalenv['etsmod'] = self.r_forecastobject
        refitR = ro.r("ets(y=rts, model=etsmod)")
        self.r_forecastobject = refitR
        ro.globalenv['r_forecastobject'] = refitR
        self.fitted = ro.r('fitted(r_forecastobject)')
        return self.fitted