import sys

# A temporary solution - I need to make sure to reset path
pycharmprojects = "/home/osboxes/PycharmProjects"
if pycharmprojects not in sys.path:
    sys.path.append(pycharmprojects)

from ForecastHybrid import Arima
import pandas as pd
import numpy as np

rng = pd.date_range('1/1/2011', periods=72, freq='H')
ts = pd.Series(np.random.randn(len(rng)), index=rng)

ar = Arima.Arima(ts)
ar.fit()
sot = ar.forecast()
a = 44