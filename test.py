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

from ForecastHybrid import ets
ets = ets.ets(ts)
ets.fit()
etsot = ets.forecast()
a=44