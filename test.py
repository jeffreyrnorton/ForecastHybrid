import sys

# A temporary solution - I need to make sure to reset path
pycharmprojects = "/home/osboxes/PycharmProjects"
if pycharmprojects not in sys.path:
    sys.path.append(pycharmprojects)

from ForecastHybrid import Arima
import pandas as pd
import numpy as np

# Read a csv in python...
bikeData = pd.read_csv('/home/osboxes/Documents/day.csv')
bikeData.temp = bikeData.temp.astype('float')
bikeData.dteday = pd.to_datetime(bikeData.dteday)
ts = pd.Series(index=bikeData.dteday, data=bikeData.temp.values)

print("Arima")
ar = Arima.Arima(ts)
if ar.fit() is not None:
    print(ar.forecast())

print("STLM")
from ForecastHybrid import stlm
stlm = stlm.stlm(ts)
if stlm.fit() is not None:
    print(stlm.forecast())

print("ETS")
from ForecastHybrid import ets
ets = ets.ets(ts)
if ets.fit() is not None:
    print(ets.forecast())

print("THETAM")
from ForecastHybrid import thetam
theta = thetam.thetam(ts)
if theta.fit() is not None:
    print(theta.forecast())

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

asdf = 44
