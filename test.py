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

# Read a csv in python...
bikeData = pd.read_csv('/home/osboxes/Documents/day.csv')
bikeData.temp = bikeData.temp.astype('float')
bikeData.dteday = pd.to_datetime(bikeData.dteday)
ts = pd.Series(index=bikeData.dteday, data=bikeData.temp.values)

# Get some periodic data for STLM and HoltWinters
accdata = pd.read_csv('/home/osboxes/Documents/USAccDeaths.csv')
accdata.USAccDeaths = accdata.USAccDeaths.astype('float')
tsl = pd.Series(index=accdata.time, data=accdata.USAccDeaths.values)

logging.basicConfig(filename='logging.log', level=logging.DEBUG)

# Writing cvts
from ForecastHybrid import cvts

#bestres = cvts.cvts(ts, Arima.Arima)
#bestres['model'].refit(ts)

# MSLE looks best for this case
from ForecastHybrid import HybridForecast
fh = HybridForecast.HybridForecast(ts)
fh.fit(atomic_arguments={'a':{'parallel':False}}, weights='cv.errors', error_method='MSE')
# err1 = math.sqrt(metrics.mean_squared_error(np.asarray(ts), fh.fitted))
# fh.fit(weights='cv.errors', error_method='MSE')
# err2 = math.sqrt(metrics.mean_squared_error(np.asarray(ts), fh.fitted))
# fh.fit(weights='cv.errors', error_method='MAE')
# err3 = math.sqrt(metrics.mean_squared_error(np.asarray(ts), fh.fitted))
# fh.fit(weights='cv.errors', error_method='MSLE')
# err4 = math.sqrt(metrics.mean_squared_error(np.asarray(ts), fh.fitted))
# fh.fit(weights='cv.errors', error_method='MEAE')
# err5 = math.sqrt(metrics.mean_squared_error(np.asarray(ts), fh.fitted))
# fh.fit(weights='cv.errors', error_method='RMSE')
# err6 = math.sqrt(metrics.mean_squared_error(np.asarray(ts), fh.fitted))

asffasdafdsa = 4

print("Arima")
ar = Arima.Arima(ts)
if ar.fit() is not None:
    print(ar.forecast())
ar.fitR(**{'parallel':True, 'num.cores':4})

print("ETS")
from ForecastHybrid import ets
ets = ets.ets(ts)
if ets.fit() is not None:
    print(ets.forecast())
ets.fitR(**{'model':'ZZZ'})

print("THETAM")
from ForecastHybrid import thetam
theta = thetam.thetam(ts)
if theta.fit() is not None:
    print(theta.forecast())

print("STLM")
from ForecastHybrid import stlm
stlm1 = stlm.stlm(ts)
if stlm1.fit() is not None:
    print(stlm1.forecast())
stlm2 = stlm.stlm(tsl)
if stlm2.fit() is not None:
    print(stlm2.forecast())

print("TBATS")
from ForecastHybrid import tbats
tb = tbats.tbats(ts)
if tb.fit() is not None:
    print(tb.forecast())

print("NNETAR")
from ForecastHybrid import nnetar
nn = nnetar.nnetar(ts)
if nn.fit() is not None:
    print(nn.forecast())

print("HoltWinters")
from ForecastHybrid import holtwinters
hw = holtwinters.holtwinters(ts)
if hw.fit() is not None:
    print(hw.forecast())









