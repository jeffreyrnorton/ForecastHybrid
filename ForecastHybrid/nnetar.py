import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import time
import numpy as np


class nnetar(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def myname(self):
        return "nnetar"

    def fitR(self, **kwargs):
        ro.r("rm(list=ls())")
        self.setTimeSeries(period=1)
        command = self.setREnv("nnetar", **kwargs)
        return self.fitKernel(command)

    def fitKernel(self, command):
        try:
            # Fit the time series
            self.dumpRCommandEnv(command)
            start_time = time.time()
            self.r_forecastobject = ro.r(command)
            logging.info("[R]nnetar ran in {} sec".format(time.time() - start_time))
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            # Fitted points
            self.extractFit(indices={'fidx':15, 'nbands':None, 'lower':None, 'upper':None})
            logging.info("nnetar fit successful")
        except:
            logging.debug(self.rtracebackerror())
            logging.warning("Running nnetar without any arguments except for the time series")
            try:
                command = 'nnetar(r_timeseries)'
                self.r_forecastobject = ro.r(command)
                ro.globalenv['r_forecastobject'] = self.r_forecastobject
                # Fitted points
                self.extractFit(indices={'fidx': 15, 'nbands': None, 'lower': None, 'upper': None})
            except:
                logging.error("Failure to fit data with nnetar")

        return self.fitted


    def fit(self, P = 1, repeats = 20, xreg = None, lambday = None, model = None,
            subset = None, scale_inputs = True):
        aargs = self.convertArgsToR(P, repeats, xreg, lambday, model, subset, scale_inputs)
        return self.fitR(**aargs)

    def convertArgsToR(self, P=1, repeats=20, xreg=None, lambday=None, model=None,
            subset=None, scale_inputs=True):

            aargs = {}
            aargs['P'] = P
            aargs['repeats'] = repeats

            #xreg is tricky - not yet supporting (TODO)
            #aargs['xreg']

            if lambday is not None: aargs['lambda'] = lambday
            # Model - we need to input a previous model, this probably should be a true false, see other models as well
            #aargs['model']
            #aargs['subset]
            aargs['scale.inputs'] = scale_inputs
            return aargs


    def forecast(self, h=5, level=[80,95], fan=False, robust=False, lambdav=None,
                 findfrequency=False):
        # Make the forecast
        fcst = self.rforecast(h, level, fan, robust, lambdav, findfrequency)
        self.forecasted = self.extractRFcst(fcst, indices={'fidx':15, 'nbands':None, 'lower':None, 'upper':None})
        return self.forecasted

    def refit(self, ts):
        # Go ahead and reset the data and the model
        rdf = pandas2ri.py2ri(ts)
        # Create a call string setting variables as necessary
        arr = np.array(self.r_forecastobject)
        ro.r("library(forecast)")
        ro.globalenv['rts'] = rdf
        ro.globalenv['nnetarmod'] = self.r_forecastobject
        refitR = ro.r("nnetar(y=rts, model=nnetarmod)")
        self.r_forecastobject = refitR
        ro.globalenv['r_forecastobject'] = refitR
        self.fitted = ro.r('fitted(r_forecastobject)')
        return self.fitted