import rpy2.robjects as ro
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import time


class stlm(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def fitR(self, **kwargs):
        ro.r("rm(list=ls())")
        self.setTimeSeries(period=None)
        command = self.setREnv("ets", **kwargs)
        return self.fitKernel(command)

    def fitKernel(self, command):
        try:
            # Fit the time series
            self.dumpRCommandEnv(command)
            start_time = time.time()
            self.r_forecastobject = ro.r(command)
            logging.info("[R]stlm ran in {} sec".format(time.time() - start_time))
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            # Fitted points
            self.fitted = ro.r('fitted(r_forecastobject)').ravel()  # numpy.ndarray (unraveled to 1D)
            logging.info("tbats fit successful")
        except:
            logging.debug(self.rtracebackerror())
            logging.warning("Running stlm without any arguments except for the time series")
            try:
                command = 'stlm(r_timeseries)'
                self.r_forecastobject = ro.r(command)
                ro.globalenv['r_forecastobject'] = self.r_forecastobject
                # Fitted points
                self.fitted = ro.r('fitted(r_forecastobject)').ravel()  # numpy.ndarray (unraveled to 1D)
            except:
                logging.error("Failure to fit data with stlm")

        return self.fitted

    def fit(self, s_window = 7, robust = False, method = ['ets', 'arima'],
            modelfunction = None, model=None, etsmodel = 'ZZN',
            lambday = None, biasadj = None, xreg = None, allow_multiplicative_trend = False):

        # for k in range(2,22):
        #     self.setTimeSeries(k)
        #     zzz = ro.r('stlm(r_timeseries)')
        #     print(k)
        #     print(zzz[1][1][0])

        period = ro.r("frequency(r_timeseries)")[0]
        if period < 2:
            logging.error("Cannot use STLM fit on time series with period < 2")
            return self.fitted

        aargs = {}
        aargs['s.window'] = s_window
        aargs['robust'] = robust
        if isinstance(method, list): aargs['method'] = 'ets'
        else: aargs['method]'] = method

        #modelfunction not yet supported
        #model not yet supported
        aargs['etsmodel'] = etsmodel

        #xreg is tricky - not yet supporting (TODO)
        #ro.globalenv['xreg']

        if lambday is not None: aargs['lambda'] = lambday

        # Model - we need to input a previous model, this probably should be a true false, see other models as well
        #ro.globalenv['model']
        #ro.globalenv['subset]
        aargs['biasadj'] = biasadj
        #xreg not supported
        aargs['allow.multiplicative.trend'] = allow_multiplicative_trend

        return self.fitR(**aargs)

    def forecast(self, h=5, level=[80,95], fan=False, robust=False, lambdav=None,
                 findfrequency=False):
        # Make the forecast
        fcst = self.rforecast(h, level, fan, robust, lambdav, findfrequency)
        self.forecasted = self.extractRFcst(fcst, indices={'fidx':1, 'nbands':2, 'lower':4, 'upper':5})
        return self.forecasted


