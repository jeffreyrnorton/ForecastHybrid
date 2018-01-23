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

ar = Arima.Arima(ts)
ar.fit()
sot = ar.forecast()

from ForecastHybrid import stlm
stlm = stlm.stlm(ts)
if stlm.fit() is not None:
    stsot = stlm.forecast()

from ForecastHybrid import ets
ets = ets.ets(ts)
ets.fit()
etsot = ets.forecast()

from ForecastHybrid import thetam
theta = thetam.thetam(ts)
theta.fit()
tsot = theta.forecast()

from ForecastHybrid import tbats
tb = tbats.tbats(ts)
tb.fit()
zsot = tb.forecast()

from ForecastHybrid import nnetar
nn = nnetar.nnetar(ts)
if nn.fit() is not None:
    nnsot = nn.forecast()

from ForecastHybrid import holtwinters
hw = holtwinters.holtwinters(ts)
if hw.fit() is not None:
    hwsot = hw.forecast()

asdf = 44
