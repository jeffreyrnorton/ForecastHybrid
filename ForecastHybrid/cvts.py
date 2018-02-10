import numpy as np
from sklearn import metrics
from math import sqrt
import logging
import multiprocessing

# A bunch of comments not included
# Plus - we are going for simple on this I think and then will add...
def cvts(x, FUN, args, code, window_size=84, num_cores=2, error_method='MSLE'):

    # No checks on types at this point...

    # This code starts with code from
    # http://francescopochetti.com/pythonic-cross-validation-time-series-pandas-scikit-learn/

    # We are going to do rolling validation - check args
    # ROLLING

    # Split the series - get number of folds
    nfolds = int(np.floor(float(len(x))/window_size))
    logging.info("Cross Validation (Rolling): Using {} folds with window size {}".format(
        nfolds, window_size
    ))
    accuracies = np.zeros(nfolds-1)

    # This is where we will set up the multiprocessing

    # Set up the arguments for each of the nfold-1 arguments
    aargs = list()
    for i in range(2, nfolds-1):

        split = window_size*(i-1)

        aargs.append((i, FUN, x[0:split], x[split:], error_method))

    pool = multiprocessing.Pool(processes=num_cores)
    model_results = pool.starmap(measure, aargs)
    pool.close()

    # Select the best of the models and return it!
    idx = 0
    val = model_results[0][1]
    for i in range(1, len(model_results)):
        if model_results[i][1] < val:
            idx = i

    return {
        "model":model_results[idx][2],
        "measure":model_results[idx][1]
    }


def measure(index, FUN, train, test, error_method='RMSE'):
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


