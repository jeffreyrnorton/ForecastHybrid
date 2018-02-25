import numpy as np
from rpy2.robjects import pandas2ri
import statsmodels.api as sm
from scipy.signal import welch
import logging
import rpy2.robjects as ro
import pandas as pd


class ForecastCurve(object):
    def __init__(self, timeseries):
        self.ts = timeseries     # Original data
        self.r_forecastobject = None
        self.fitted = None       # Data that will be fit (drop the first point): len(self.ts)-1
        self.x = None            # The fit of len(self.ts)-1.
        self.forecasted = None
        ro.r('suppressPackageStartupMessages(library(forecast))')
        pandas2ri.activate()


    #def __del__(self):
        #pandas2ri.deactivate()

    def rtracebackerror(self):
        return ro.r('toString(traceback(max.lines=1)[1])')[0]

    def dumpRCommandEnv(self, command):
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            for ritem in ro.r.ls(ro.globalenv):
                logging.debug('{}:{}'.format(ritem, ro.r[ritem]))
            logging.debug('Running R command:{}'.format(command))

    def setREnv(self, call, **kwargs):
        command = '{}(r_timeseries'.format(call)
        for key, item in kwargs.items():
            if isinstance(item, bool):
                ro.globalenv[key] = ro.rinterface.TRUE if item else ro.rinterface.FALSE
            elif isinstance(item, list) and all(isinstance(x, float) for x in item):
                ro.globalenv[key] = pandas2ri.FloatSexpVector(item)
            elif isinstance(item, list) and all(isinstance(x, int) for x in item):
                ro.globalenv[key] = pandas2ri.IntSexpVector(item)
            elif isinstance(item, dict):
                ro.globalenv[key] = ro.ListVector(item)
            else:
                try:
                    ro.globalenv[key] = item
                except:
                    logging.error('Variable {} - Traceback - {}'.format(key, self.rtracebackerror()))

            command += ", {}={}".format(key,key)
        command += ")"
        return command

    def setTimeSeries(self, period=None):

        # Convert the Python time series to an R time series
        rdf = pandas2ri.py2ri(self.ts)
        # Create a call string setting variables as necessary
        ro.globalenv['r_timeseries'] = rdf

        if period is None:
            try:
                # Decompose into STL
                stl = sm.tsa.seasonal_decompose(self.ts)
                # Use Welch to calculate period on seasonal part of time series
                fs, Pxx = welch(stl.seasonal, fs=len(stl.seasonal), nperseg=len(stl.seasonal))
                period = int(np.round(np.mean(np.ediff1d(np.asarray(fs)))))
                # The above may not be correct - what we are trying to do is minimize AIC on fit
            except Exception as e:
                period = 1 # Safe, but cannot do STLM or other seasonal methods
                ro.r('library(stats)')
                period = ro.r('stl(r_timeseries)')
                logging.warning(str(e))

        # Add frequency to the time series - must be greater than 1 to make stlm work
        command = 'r_timeseries <- ts(r_timeseries, frequency=' + str(period) + ')'
        ro.r(command)

    def extractFit(self, indices={'fidx':1, 'nbands':2, 'lower':4, 'upper':5}, offset=1):
        # It appears that the fit (and the forecast) always lag the true data by one
        # This really appears to be an error in the R libraries because I noticed it in
        # the pure R implementation as well.  So we have to play tricks in both the
        # fit and the forecast to align to the data.
        # First, forecast just one point

        rvec = np.asarray(ro.r('r_forecastobject$fitted')).ravel()
        data = rvec[1:]
        date = self.ts.index[1:]
        self.fitted = pd.Series(data=data, index=date)


    def extractRFcst(self, fcst, indices={'fidx':1, 'nbands':2, 'lower':4, 'upper':5}):

        nprres = np.asarray(fcst)
        # Build a dictionary (map) of the forecast output
        vfor = np.asarray(nprres[indices['fidx']])

        # Build a time index for the future.
        average_period = np.ediff1d(self.ts.index).mean()
        last_date = self.ts.index[len(self.ts) - 1]
        idx = np.arange(start=last_date+average_period, step=average_period,
                        stop =last_date+average_period + (len(vfor)+1)*average_period)
        idx = idx[0:len(vfor)]

        results = {'forecast':pd.Series(index=idx, data=vfor)}

        try:
            if indices['nbands'] is not None:
                nbands = len(nprres[indices['nbands']])
                vbanlower = nprres[indices['lower']]
                vbanupper = nprres[indices['upper']]

                # Flatten the matrices to an array - we will refill them...
                listlower = np.matrix(vbanlower).flatten().tolist()[0]
                listupper = np.matrix(vbanupper).flatten().tolist()[0]

                # the two arrays need to be the same size
                arrlower = np.empty([nbands, int(len(listlower)/nbands)], dtype=np.float64)
                arrupper = np.empty([nbands, int(len(listlower)/nbands)], dtype=np.float64)

                for i in range(0, int(len(listlower)/len(nprres[2]))):
                    for j in range(0, nbands):
                        arrlower[j][i] = listlower[i*nbands+j]
                        arrupper[j][i] = listupper[i*nbands+j]

                for i in range(0, len(nprres[2])):
                    k1 = str(nprres[2][i])+'_lower'
                    k2 = str(nprres[2][i])+'_upper'
                    results[k1] = pd.Series(index=idx, data=arrlower[i][:len(idx)])
                    results[k2] = pd.Series(index=idx, data=arrupper[i][:len(idx)])
        except:
            logging.error("Unable to retrieve confidence levels for type" + str(type(self)))

        return results

    def rforecast(self, h=5, level=[80,95], fan=False, robust=False, lambdav=None,
                  findfrequency=False):
        ro.globalenv['h'] = h
        ro.globalenv['level'] = ro.FloatVector(level)
        ro.globalenv['fan'] = ro.rinterface.TRUE if fan else ro.rinterface.FALSE
        ro.globalenv['robust'] = ro.rinterface.TRUE if robust else ro.rinterface.FALSE
        # Not supporting lamda right now because it is tricky
        #ro.globalenv['lambda'] = ro.rinterface.NA_Real if lambdav is None else lambdav
        ro.globalenv['find.frequency'] = ro.rinterface.TRUE if findfrequency else ro.rinterface.FALSE
        #ro.globalenv['r_forecastobject'] = self.r_forecastobject
        sr_forecast = 'forecast(r_forecastobject, h=h, level=level, fan=fan, robust=robust, ' \
            'find.frequency=find.frequency)'
        return ro.r(sr_forecast)

    def refit(self, ts):
        logging.fatal("Need to create subtype")
