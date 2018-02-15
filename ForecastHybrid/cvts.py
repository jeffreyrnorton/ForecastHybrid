import numpy as np
from sklearn import metrics
from math import sqrt
import logging
import pandas as pd


# A bunch of comments not included
# Plus - we are going for simple on this I think and then will add...
class cvts:

    def rolling(self, x, FUN, args, code, window_size=84, num_cores=2, error_method='MSLE'):

        # No checks on types at this point...

        # This code starts with code from
        # http://francescopochetti.com/pythonic-cross-validation-time-series-pandas-scikit-learn/

        # We are going to do rolling validation - check args
        # ROLLING

        # Split the series - get number of folds
        nfolds = int(np.floor(float(len(x))/window_size))
        logging.info("{} Cross Validation (Rolling): Using {} folds with window size {}".format(
            code, nfolds, window_size
        ))
        accuracies = np.zeros(nfolds-1)

        # This is where we will set up the multiprocessing

        # Set up the arguments for each of the nfold-1 arguments
#        aargs = list()
        val = float("infinity")
        best_model = None
        for i in range(2, nfolds-1):

            split = window_size*(i-1)

            logging.info("Before measure, i={}".format(i))
            index, acc, f = self.measure(i, FUN,
                                         pd.Series(np.asarray(x)[:split]),
                                         pd.Series(np.asarray(x)[split:]),
                                         error_method)
            logging.info("After measure, i={}, acc={}".format(index, acc))

            if acc < val:
                val = acc
                best_model = f

        return {
            "model_code":code,
            "model":best_model,
            "measure":val
        }


    def measure(self, index, FUN, train, test, error_method='RMSE'):
        f = FUN(train)
        f.fit()
        pred = f.forecast(h=len(test))['forecast']

        if error_method == 'MAE':
            acc = metrics.mean_absolute_error(test, pred)
        elif error_method == 'MSE':
            acc = metrics.mean_squared_error(test, pred)
        elif error_method == 'RMSE':
            acc = sqrt(metrics.mean_squared_error(test, pred))
        elif error_method == 'MSLE':
            acc = metrics.mean_squared_log_error(test, pred)
        elif error_method == 'MEAE':
            acc = metrics.median_absolute_error(test, pred)
        return index, acc, f


