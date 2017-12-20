import ForecastHybrid
import pandas as pd
import numpy as np

rng = pd.date_range('1/1/2011', periods=72, freq='H')
ts = pd.Series(np.random.randn(len(rng)), index=rng)

ar = ForecastHybrid.Arima(ts)
ar.fit()
ar.forecast()