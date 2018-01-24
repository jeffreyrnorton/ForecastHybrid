import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import numpy as np
import logging

class ForecastCurve(object):
    def __init__(self, timeseries):
        self.ts = timeseries
        self.r_forecastobject = None
        self.fitted = None
        self.forecasted = None
        pandas2ri.activate()
        ro.r('library(forecast)')

    def __del__(self):
        pandas2ri.deactivate()

    def extractRFcst(self, fcst, indices={'fidx':1, 'nbands':2, 'lower':4, 'upper':5}):
        nprres = np.asarray(fcst)
        # Build a dictionary (map) of the forecast output
        vfor = np.asarray(nprres[indices['fidx']])
        results = {'forecast':vfor}

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
                    results[k1] = arrlower[i]
                    results[k2] = arrupper[i]
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
