import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import sys


class holtwinters(ForecastCurve.ForecastCurve):
    def __init__(self, timeseries):
        super().__init__(timeseries)

    def fit(self, alpha = None, beta = None, gamma = None, seasonal = ["additive", "multiplicative"],
            start_periods = 2, l_start = None, b_start = None, s_start = None,
            optim_start = {'alpha' : 0.3, 'beta' : 0.1, 'gamma' : 0.1}) :

        # Convert the Python time series to an R time series
        rdf = pandas2ri.py2ri(self.ts)
        # Create a call string setting variables as necessary
        ro.globalenv['r_timeseries'] = rdf
        command = 'HoltWinters(r_timeseries'
        self.fitted = None

        nperiods = ro.r("frequency(r_timeseries)")
        nperiods = int(nperiods[0])
        if nperiods < 2:
            logging.error("Cannot fit HoltWinters on time series with less than 2 periods")
            return self.fitted

        try:
            ro.globalenv['alpha'] = ro.rinterface.NULL if alpha is None else alpha
            ro.globalenv['beta'] = ro.rinterface.NULL if beta is None else beta
            ro.globalenv['gamma'] = ro.rinterface.NULL if gamma is None else gamma
            if len(seasonal) == 1:
                ro.globalenv['seasonal'] = seasonal
            else:
                ro.globalenv['seasonal'] = 'additive'
            ro.globalenv['start.periods'] = start_periods
            ro.globalenv['l.start'] = ro.rinterface.NULL if l_start is None else l_start
            ro.globalenv['b.start'] = ro.rinterface.NULL if b_start is None else b_start
            ro.globalenv['s.start'] = ro.rinterface.NULL if s_start is None else s_start
            ro.globalenv['optim.start'] = ro.ListVector(optim_start)
            command += ', alpha=alpha, beta=beta, gamma=gamma, seasonal=seasonal, start.periods=start.periods'
            command += ', l.start=l.start, b.start=b.start, s.start=s.start, optim.start=optim.start'
            command += ')'

            # Fit the time series
            self.r_forecastobject = ro.r(command)
            ro.globalenv['r_forecastobject'] = self.r_forecastobject
            # Fitted points
            self.fitted = ro.r('fitted(r_forecastobject)').ravel() # numpy.ndarray (unraveled to 1D)
            logging.info("HoltWinter fit successful")
        except:
            print(sys.exc_info()[0])
            logging.warning(sys.exc_info()[0])
            logging.warning("Running HoltWinter without any arguments except for the time series")
            try:
                command = 'HoltWinter(r_timeseries)'
                self.r_forecastobject = ro.r(command)
                ro.globalenv['r_forecastobject'] = self.r_forecastobject
                # Fitted points
                self.fitted = ro.r('fitted(r_forecastobject)').ravel()  # numpy.ndarray (unraveled to 1D)
            except:
                logging.error("Failure to fit data with HoltWinter")

        return self.fitted
