import sys

# A temporary solution - I need to make sure to reset path
pycharmprojects = "/home/osboxes/PycharmProjects"
if pycharmprojects not in sys.path:
    sys.path.append(pycharmprojects)

from ForecastHybrid import Arima
import pandas as pd
import numpy as np
from sklearn import metrics
import math
import logging

from ForecastHybrid import cvts
from ForecastHybrid import ets
from ForecastHybrid import tbats
from ForecastHybrid import thetam
from ForecastHybrid import nnetar


# Read a csv in python...
bikeData = pd.read_csv('/home/osboxes/Documents/day.csv')
bikeData.temp = bikeData.temp.astype('float')
bikeData.dteday = pd.to_datetime(bikeData.dteday)
ts = pd.Series(index=bikeData.dteday, data=bikeData.temp.values)

# Get some periodic data for STLM and HoltWinters
accdata = pd.read_csv('/home/osboxes/Documents/USAccDeaths.csv')
accdata.USAccDeaths = accdata.USAccDeaths.astype('float')
tsl = pd.Series(index=accdata.time, data=accdata.USAccDeaths.values)

logging.basicConfig(filename='logging.log', level=logging.INFO)

import matplotlib
#tsl.plot()

#print("Arima")
#ar = Arima.Arima(tsl)
#if ar.fit() is not None:
#    ar.forecast()
#ar.fitR(**{'parallel':True, 'num.cores':4})

#cvts = cvts.cvts()
#for errmtd in ['MAE', 'MSE', 'RMSE', 'MSLE', 'MEAE']:
#    for mod in [Arima.Arima, ets.ets, tbats.tbats, thetam.thetam, nnetar.nnetar]:
#        ares = cvts.rolling(tsl, mod, None, 'a', error_method=errmtd)
    #    pd.Series(ares[0][:,0]).plot()
    #    pd.Series(ares[0][:,1]).plot()
#        print("{} Error:{}".format(errmtd, ares[1]))


# MSLE looks best for this case
from ForecastHybrid import HybridForecast

print("MODEL: bike")
for errormodel in ["MAE", "MSE", "MSLE", "MEAE", "RSME"]:
    for weightmodel in ['cv.errors', 'equal', 'errors']:
        fh = HybridForecast.HybridForecast(ts)
        fitted = fh.fit(weights=weightmodel, error_method=errormodel, models="aefnt")
        equal_error = fh.error('RMSE')
        print('Error Model {}, Weight Model {}, RMSE total error = {}'.format(errormodel, weightmodel, equal_error['error']))

print("PERIODIC MODEL: accident")
for errormodel in ["MAE", "MSE", "MSLE", "MEAE", "RSME"]:
    for weightmodel in ['equal', 'errors']:#, 'cv.errors']:
        fh = HybridForecast.HybridForecast(tsl)
        fitted = fh.fit(weights=weightmodel, error_method=errormodel, period=12)
        equal_error = fh.error('RMSE')
        print('Error Model {}, Weight Model {}, RMSE total error = {}'.format(errormodel, weightmodel, equal_error['error']))


asffasdafdsa = 4




print("STLM")
from ForecastHybrid import stlm
stlm1 = stlm.stlm(tsl)
if stlm1.fit(period=12) is not None:
    print(stlm1.forecast())

print("HoltWinters")
from ForecastHybrid import holtwinters
hw = holtwinters.holtwinters(tsl)
if hw.fitR(**{'period':12}) is not None:
    print(hw.forecast())

print("ETS")
ets = ets.ets(ts)
if ets.fit() is not None:
    print(ets.forecast())
ets.fitR(**{'model':'ZZZ'})

print("TBATS")
tb = tbats.tbats(ts)
if tb.fit() is not None:
    print(tb.forecast())

print("NNETAR")
nn = nnetar.nnetar(ts)
if nn.fit() is not None:
    print(nn.forecast())

print("THETAM")
theta = thetam.thetam(ts)
if theta.fit() is not None:
    print(theta.forecast())
    theta.refit(ts)









