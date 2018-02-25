import numpy as np
from sklearn import metrics
import logging
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import multiprocessing
from math import sqrt


# A bunch of comments not included
# Plus - we are going for simple on this I think and then will add...
class cvts:

    def rolling(self, x, FUN, args, code, window_size=84, num_cores=2, error_method='MSLE', pool=None):

        # No checks on types at this point...

        # This code starts with code from
        # http://francescopochetti.com/pythonic-cross-validation-time-series-pandas-scikit-learn/

        # We are going to do rolling validation - check args
        # ROLLING

        h = 1
        tscv_split = TimeSeriesSplit(max_train_size = None, n_splits=len(x)-h)

        itlist = list()

        for train_index, test_index in tscv_split.split(x):
            train_data  = pd.Series(np.asarray(x)[:test_index[0]])
            if len(train_data) > 4:
                test_data = pd.Series(np.asarray(x)[test_index[0]:])
                itlist.append((test_data, train_data, FUN, h, args))
                # Delete me when the pool goes back in.
#                cvts_worker(test_data, train_data, FUN, h, args)

        own_pool = True if pool is None else False
        if own_pool is True:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        self.model_results = np.asarray(pool.starmap(cvts_worker, itlist))
        if own_pool is True:
            pool.close()

        # Calculate error
        tval = self.model_results[:,0]
        pval = self.model_results[:,1]

        if error_method == 'MAE':
            err = metrics.mean_absolute_error(tval[pd.isnull(pval) == False], pval[pd.isnull(pval) == False])
        elif error_method == 'MSE':
            err = metrics.mean_squared_error(tval[pd.isnull(pval) == False], pval[pd.isnull(pval) == False])
        elif error_method == 'RMSE':
            err = sqrt(metrics.mean_squared_error(tval[pd.isnull(pval) == False], pval[pd.isnull(pval) == False]))
        elif error_method == 'MSLE':
            err = metrics.mean_squared_log_error(tval[pd.isnull(pval) == False], pval[pd.isnull(pval) == False])
        elif error_method == 'MEAE':
            err = metrics.median_absolute_error(tval[pd.isnull(pval) == False], pval[pd.isnull(pval) == False])

        return self.model_results, err


def cvts_worker(test_data, train_data, FUN, h=1, args=None):
    fnc = FUN(train_data)
    fnc.fit() if args is None else fnc.fitR(**args)
    forecast_data = fnc.forecast(h=h)
    forecast_point = np.array(forecast_data['forecast'])[h-1]
#    print("{} : {}".format(test_data.values[h - 1], forecast_point))
    return test_data.values[h - 1], forecast_point, fnc
           #train_data.values[len(train_data)-1], fitted_data[len(fitted_data)-1]
