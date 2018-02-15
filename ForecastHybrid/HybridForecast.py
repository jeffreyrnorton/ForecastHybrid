# ' Hybrid time series modelling
# '
# ' Create a hybrid time series model with two to five component models.
# '
# ' @export
# ' @import forecast
# ' @import stats
# ' @import graphics
# ' @import zoo
# ' @param y A numeric vector or time series.
# ' @param lambda
# ' Box-Cox transformation parameter.
# ' Ignored if NULL. Otherwise, data transformed before model is estimated.
# ' @param models A character string of up to six characters indicating which contributing models to use:
# ' a (\code{\link[forecast]{auto.arima}}), e (\code{\link[forecast]{ets}}),
# ' f (\code{\link{thetam}}), n (\code{\link[forecast]{nnetar}}),
# ' s (\code{\link[forecast]{stlm}}), t (\code{\link[forecast]{tbats}}), and z (\code{\link[forecast]{snaive}}).
# ' @param a.args an optional \code{list} of arguments to pass to \code{\link[forecast]{auto.arima}}. See details.
# ' @param e.args an optional \code{list} of arguments to pass to \code{\link[forecast]{ets}}. See details.
# ' @param n.args an optional \code{list} of arguments to pass to \code{\link[forecast]{nnetar}}. See details.
# ' @param s.args an optional \code{list} of arguments to pass to \code{\link[forecast]{stlm}}. See details.
# ' @param t.args an optional \code{list} of arguments to pass to \code{\link[forecast]{tbats}}. See details.
# ' @param weights method for weighting the forecasts of the various contributing
# ' models.  Defaults to \code{equal}, which has shown to be robust and better
# ' in many cases than giving more weight to models with better in-sample performance. Cross validated errors--implemented with \code{link{cvts}}
# ' should produce the best forecast, but the model estimation is also the slowest. Note that extra arguments
# ' passed in \code{a.args}, \code{e.args}, \code{n.args}, \code{s.args}, and \code{t.args} are not used
# ' during cross validation. See further explanation in \code{\link{cvts}}.
# ' Weights utilizing in-sample errors are also available but not recommended.
# ' @param errorMethod  method of measuring accuracy to use if weights are not
# ' to be equal.
# ' Root mean square error (\code{RMSE}), mean absolute error (\code{MAE})
# ' and mean absolute scaled error (\code{MASE})
# ' are supported.
# ' @param parallel a boolean indicating if parallel processing should be used between models.
# ' This is currently unimplemented.
# ' Parallelization will still occur within individual models that suport it and can be controlled using \code{a.args} and \code{t.args}.
# ' @param num.cores If \code{parallel=TRUE}, how many cores to use.
# ' @param cvHorizon If \code{weights = "cv.errors"}, this controls which forecast to horizon to use
# ' for the error calculations.
# ' @param windowSize length of the window to build each model, only used when \code{weights = "cv.errors"}.
# ' @param horizonAverage If \code{weights = "cv.errors"}, setting this to \code{TRUE} will average
# ' all forecast horizons up to \code{cvHorizon} for calculating the errors instead of using
# ' the single horizon given in \code{cvHorizon}.
# ' @param verbose Should the status of which model is being fit/cross validated be printed to the terminal?
# ' @seealso \code{\link{forecast.hybridModel}}, \code{\link[forecast]{auto.arima}},
# ' \code{\link[forecast]{ets}}, \code{\link{thetam}}, \code{\link[forecast]{nnetar}},
# ' \code{\link[forecast]{stlm}}, \code{\link[forecast]{tbats}}
# ' @return An object of class hybridModel.
# ' The individual component models are stored inside of the object
# ' and can be accessed for all the regular manipulations available in the forecast package.
# ' @details The \code{hybridModel} function fits multiple individual model specifications to allow easy creation
# ' of ensemble forecasts. While default settings for the individual component models work quite well
# ' in most cases, fine control can be exerted by passing detailed arguments to the component models in the
# ' \code{a.args}, \code{e.args}, \code{n.args}, \code{s.args}, and \code{t.args} lists.
# ' Note that if \code{xreg} is passed to the \code{a.args}, \code{n.args}, or \code{s.args} component models
# ' it must be passed as a dataframe instead of the matrix object
# ' that the "forecast" package functions usually accept.
# ' This is due to a limitation in how the component models are called.
# ' \cr
# ' \cr
# ' Characteristics of the input series can cause problems for certain types of models and paramesters.
# ' For example, \code{\link[forecast]{stlm}} models require that the input series be seasonal;
# ' furthemore, the data must include at least two seasons of data (i.e. \code{length(y) >= 2 * frequency(y)})
# ' for the decomposition to succeed.
# ' If this is not the case, \code{hybridModel()}
# ' will remove the \code{stlm} model so an error does not occur.
# ' Similarly, \code{nnetar} models require that
# ' \code{length(y) >= 2 * frequency(y)}, so these models will be removed if the condition is not satisfied
# ' The \code{\link[forecast]{ets}} model does not handle
# ' a series well with a seasonal period longer than 24 and will ignore the seasonality. In this case,
# ' \code{hybridModel()} will also drop the \code{ets} model from the ensemble.
# '
# ' @examples
# ' \dontrun{
# '
# ' # Fit an auto.arima, ets, thetam, nnetar, stlm, and tbats model
# ' # on the time series with equal weights
# ' mod1 <- hybridModel(AirPassengers)
# ' plot(forecast(mod1))
# '
# ' # Use an auto.arima, ets, and tbats model with weights
# ' # set by the MASE in-sample errors
# ' mod2 <- hybridModel(AirPassengers, models = "aet",
# ' weights = "insample.errors", errorMethod = "MASE")
# '
# ' # Pass additional arguments to auto.arima() to control its fit
# ' mod3 <- hybridModel(AirPassengers, models = "aens",
# ' a.args = list(max.p = 7, max.q = 7, approximation = FALSE))
# '
# ' # View the component auto.arima() and stlm() models
# ' mod3$auto.arima
# ' mod3$stlm
# ' }
# '
# ' @author David Shaub
# '

import sys
import rpy2.robjects as ro
import ForecastHybrid.ForecastCurve as ForecastCurve
import logging
import math
import multiprocessing
import pandas as pd
import numpy as np
from sklearn import metrics
from ForecastHybrid import Arima
from ForecastHybrid import ets
from ForecastHybrid import thetam
from ForecastHybrid import nnetar
from ForecastHybrid import tbats
from ForecastHybrid import stlm
from ForecastHybrid import cvts

# For holt winters we are going to use AAA (Additive ETS) and MAM (Multiplicative ETS) models

def arima_worker(ts, aargs):
    logging.info("Running auto.arima")
    res = Arima.Arima(ts)
    if aargs is None or aargs.get('a', None) is None:
        res.fit()
    else:
        res.fitR(**aargs['a'])
    return ['a', res]

def ets_worker(ts, aargs):
    logging.info("Running ets")
    res = ets.ets(ts)
    if aargs is None or aargs.get('e', None) is None:
        res.fit()
    else:
        res.fitR(**aargs['e'])
    return ['e', res]

def hwmult_worker(ts, aargs):
    logging.info("Running holt-winters multiplicative model in ets")
    res = ets.ets(ts)
    aargs['h']['model'] = 'MAM'
    res.fitR(**aargs['h'])
    return ['h', res]

def hwadd_worker(ts, aargs):
    logging.info("Running holt-winters additive model in ets")
    res = ets.ets(ts)
    aargs['d']['model'] = 'AAA'
    res.fitR(**aargs['d'])
    return ['d', res]

def thetam_worker(ts, aargs):
    logging.info("Running thetam")
    res = thetam.thetam(ts)
    if aargs is None or aargs.get('f', None) is None:
        res.fit()
    else:
        res.fitR(**aargs['f'])
    return ['f', res]

def nnetar_worker(ts, aargs):
    logging.info("Running nnetar")
    res = nnetar.nnetar(ts)
    if aargs is None or aargs.get('n', None) is None:
        res.fit()
    else:
        res.fitR(**aargs['n'])
    return ['n', res]

def tbats_worker(ts, aargs):
    logging.info("Running tbats")
    res = tbats.tbats(ts)
    if aargs is None or aargs.get('t', None) is None:
        res.fit()
    else:
        res.fitR(**aargs['t'])
    return ['t', res]

def stlm_worker(ts, aargs):
    logging.info("Running stlm")
    res = stlm.stlm(ts)
    if aargs is None or aargs.get('s', None) is None:
        res.fit()
    else:
        res.fitR(**aargs['s'])
    return ['s', res]


def all_workers(ts, atomic_arguments, model):
    if model == 'a':
        return arima_worker(ts, atomic_arguments)
    elif model == 'e':
        return ets_worker(ts, atomic_arguments)
    elif model == 'f':
        return thetam_worker(ts, atomic_arguments)
    elif model == 'n':
        return nnetar_worker(ts, atomic_arguments)
    elif model == 's':
        return stlm_worker(ts, atomic_arguments)
    elif model == 't':
        return tbats_worker(ts, atomic_arguments)
    elif model == 'h': # holt winter multiplicative
        return hwmult_worker(ts, atomic_arguments)
    elif model == 'd': # holt winter additive
        return hwadd_worker(ts, atomic_arguments)


def cvts_worker(x, FUN, args, code, window_size=84, num_cores=2, error_method='MSLE'):
    rolling_cvts = cvts.cvts()
    return rolling_cvts.rolling(x, FUN, args, code, window_size, num_cores, error_method)


class HybridForecast(ForecastCurve.ForecastCurve):
    MODELS = {'a':'arima', 'e':'ets', 'f':'thetam', 'n':'nnetar', 's':'stlm', 't':'tbats', 'h':'holt-winters_multiplicative',
              'd':'holt-winters_additive'}
    model_results = {'a':None, 'e':None, 'f':None, 'n':None, 's':None, 't':None, 'h':None, 'd':None}

    def __init__(self, timeseries):
        super().__init__(timeseries)

    # Extreme difference on measure handling from R method where these methods are from sklearn.metrics
    # MAE - Mean Absolute Error
    # MSE - Mean Squared Error
    # MSLE - Mean Squared Log Error
    # MEAE - Median Absolute Error
    def fit(self, models="aefnsthd", lambday = None, atomic_arguments = None,
                       weights = ['equal', 'errors', 'cv.errors'], error_method = ['MAE', 'MSE', 'MSLE', 'MEAE', 'RMSE'],
                       cv_horizon = None, window_size = 84, horizon_average = False,
                       parallel = False, period=None):

        logging.info('Fitting with HybridForecast : models = {}'.format(models))
        if len(self.ts) < 4:
            logging.fatal("The input time series must have 4 or more observations.")
            return self.fitted

        self.setTimeSeries(period)
        rfreq = int(ro.r('frequency(r_timeseries)')[0])

        # Note that if rfreq > 1 at this point, we want to also change atomic_arguments for 's' and 'h' to
        # have the argument 'period'.
        if atomic_arguments is None: atomic_arguments = dict()
        if rfreq > 1:
            if atomic_arguments.get('s', None) == None: atomic_arguments['s'] = {'period':rfreq}
            else: atomic_arguments['s'].update({'period':rfreq})
            if atomic_arguments.get('h', None) == None: atomic_arguments['h'] = {'period':rfreq}
            else: atomic_arguments['h'].update({'period':rfreq})
            if atomic_arguments.get('d', None) == None: atomic_arguments['d'] = {'period':rfreq}
            else: atomic_arguments['d'].update({'period':rfreq})

        # Run checks on all the variables as appropriate
        if weights is list:
            logging.info('Selecting "equal" from ["equal", "errors", "cv.errors"]')
            weights = 'equal'
        elif weights not in ['equal', 'errors', 'cv.errors']:
            logging.warning('Invalid weight count or type - using "equal"')
            weights = 'equal' # Cross-validation errors are better accuracy wise, but slower
        if error_method is list:
            logging.info('Selecting "MSE" from ["MAE", "MSE", "MSLE", "MEAE", "RSME"]')
            error_method = 'MSE'
        elif error_method not in ['MAE', 'MSE', 'MSLE', 'MEAE']:
            logging.warning('Invalid error method - using MSE')
            error_method = 'MSE'
        if cv_horizon is None:
            cv_horizon = rfreq

        wexpanded_models = set(list(models.lower()))
        expanded_models = []
        for m in wexpanded_models:
            if m in list(self.MODELS.keys()):
                expanded_models.append(m)
                logging.info('Using model ' + self.MODELS[m])
        if len(expanded_models) < 1:
            logging.error("At least one component model type must be specified.")
            return self.fitted

        # Validate cores and parallel arguments
        if type(parallel) is not bool:
            logging.warning("Invalid type for parallel - assigning to run in parallel")
            parallel = True

        # Check for problems for specific models (e.g. long seasonality for ets and non-seasonal for stlm or nnetar)
        if rfreq >= 24:
            if 'e' in expanded_models:
                logging.warning('frequency >= 24, the ets model will not be used')
                expanded_models.remove('e')
            if 'f' in expanded_models:
                logging.warning('frequency >= 24, the theta model will not be used')
                expanded_models.remove('f')
        if 'f' in expanded_models and len(self.ts) < rfreq:
            logging.warning('The theta model requires more than a year of data.  The theta model will not be used.')
            expanded_models.remove('f')
        if 's' in expanded_models:
            if rfreq < 2:
                logging.warning("The stlm model requires that the input data be a seasonal ts object.  The stlm model will not be used.")
                expanded_models.remove('s')
            if rfreq * 2 >= len(self.ts):
                logging.warning("The stlm model requres a series more than twice as long as the seasonal period. The stlm model will not be used.")
                expanded_models.remove('s')

        if 'h' in expanded_models or 'd' in expanded_models:
            if rfreq < 2:
                logging.warning("The holt-winters model requires that the input data be a seasonal ts object.  The holt-winters model will not be used.")
                expanded_models.remove('h')

        if 'n' in expanded_models:
            if rfreq * 2 >= len(self.ts):
                logging.warning("The nnetar model requres a series more than twice as long as the seasonal period. The nnetar model will not be used.")
                expanded_models.remove('n')
        if len(expanded_models) < 1:
            logging.error("A hybrid model must contain one component model.")
            return self.fitted

        if weights == 'cv.errors' and window_size > int(len(self.ts)/5):
            window_size = int(len(self.ts)/5)
            if window_size < 5:
                logging.warning('Not enough data for rolling validation, using "error" weighting.')
                weights = 'errors'
            else:
                logging.warning('Window size is too large - reducing size to {}'.format(window_size))

        logging.info('Fitting with models : {}'.format(expanded_models))
        logging.info('Number of cores = {}'.format(multiprocessing.cpu_count()))
        # If we have extra cpus - get several working on tbats because it is slow in the fit
        # if extra_cpus >= 2:
        #     ncpus = min(3, extra_cpus)
        #     logging.info('Extra Core count = {}'.format(ncpus))
        #     if atomic_arguments is None or atomic_arguments.get('t', None) is None:
        #         atomic_arguments = {'t':{'use.parallel':True, 'num.cores':ncpus}}
        #     else:
        #         atomic_arguments['t'].update({'use.parallel':True, 'num.cores':ncpus})
        #     extra_cpus -= ncpus

        if weights == 'cv.errors':  # cv.errors
            try: # There are reasons for exceptions like a list too short for stlm.  We will handle it robustly
                itlist = list()
                if 't' in expanded_models:
                    itlist.append((self.ts, tbats.tbats, atomic_arguments, 't', window_size, 2, error_method))
                if 'n' in expanded_models:
                    itlist.append((self.ts, nnetar.nnetar, atomic_arguments, 'n', window_size, 2, error_method))
                if 'a' in expanded_models:
                    itlist.append((self.ts, Arima.Arima, atomic_arguments, 'a', window_size, 2, error_method))
                if 'e' in expanded_models:
                    itlist.append((self.ts, ets.ets, atomic_arguments, 'e', window_size, 2, error_method))
                if 'f' in expanded_models:
                    itlist.append((self.ts, thetam.thetam, atomic_arguments, 'f', window_size, 2, error_method))
                if 's' in expanded_models:
                    itlist.append((self.ts, stlm.stlm, atomic_arguments, 's', window_size, 2, error_method))
                if 'h' in expanded_models:
                    atomic_arguments['h']['model'] = 'MAM'
                    itlist.append((self.ts, ets.ets, atomic_arguments, 'h', window_size, 2, error_method))
                if 'd' in expanded_models:
                    atomic_arguments['d']['model'] = 'AAA'
                    itlist.append((self.ts, ets.ets, atomic_arguments, 'd', window_size, 2, error_method))

                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                self.model_results = pool.starmap(cvts_worker, itlist)
                pool.close()

                # Turn the array in model results into a dictionary (map) and extract even weighting...
                # Also, sometimes we are going to get NaNs so we are going to have to work with a dataframe
                # and handle each observation separately.
                self.fitted = np.ndarray(shape=[len(self.ts), 1])
                temp = dict()
                column_errors = list()
                df = pd.DataFrame()
                for i in range(0, len(self.model_results)):
                    column_errors.append(self.model_results[i]['measure'])
                    tdict = {'model': self.model_results[i]['model'].refit(self.ts),
                             'error': self.model_results[i]['measure']}
                    temp.update({self.model_results[i]['model_code']: tdict})
                    df = pd.DataFrame(np.asmatrix(self.model_results[i]['model'].fitted)) if df is None else \
                        df.append(pd.DataFrame(np.asmatrix(self.model_results[i]['model'].fitted)), ignore_index=True)

                df = df.transpose()
                self.model_results = temp
                tweights = 1.0 / np.asarray(column_errors)
                weightsnorm = np.linalg.norm(tweights, ord=1)
                weights = tweights / weightsnorm

                # Hybrid weighting which handles NaNs in rows
                for i in range(0, df.shape[0]):
                    good_row_data = df.ix[i].dropna()
                    # Use properly weighted if we have all the data.
                    if len(good_row_data) == len(weights):
                        self.fitted[i] = np.dot(good_row_data, weights)
                    # Weight on the good values
                    elif len(good_row_data) > 0:
                        stweights = tweights[df.ix[i].isna() == False]
                        stweightsn = np.linalg.norm(stweights, ord=1)
                        self.fitted[i] = np.dot(good_row_data, stweights / stweightsn)
                    else:
                        self.fitted[i] = np.NaN
            except:
                logging.warning('Python Traceback: {}'.format(str(sys.exc_info())))
                logging.warning('R traceback: {}'.format(self.rtracebackerror()))
                logging.warning('Cannot use weighting "cv.errors", using "errors"')
                weights = 'errors'

        if weights == 'equal' or weights == 'errors':

            itlist = list()
            if 't' in expanded_models:
                itlist.append((self.ts, atomic_arguments, 't'))
            if 'n' in expanded_models:
                itlist.append((self.ts, atomic_arguments, 'n'))
            if 'a' in expanded_models:
                itlist.append((self.ts, atomic_arguments, 'a'))
            if 'e' in expanded_models:
                itlist.append((self.ts, atomic_arguments, 'e'))
            if 'f' in expanded_models:
                itlist.append((self.ts, atomic_arguments, 'f'))
            if 's' in expanded_models:
                itlist.append((self.ts, atomic_arguments, 's'))
            if 'h' in expanded_models:
                atomic_arguments['h']['model'] = 'MAM'
                itlist.append((self.ts, atomic_arguments, 'h'))
            if 'd' in expanded_models:
                atomic_arguments['h']['model'] = 'AAA'
                itlist.append((self.ts, atomic_arguments, 'd'))

            # Initial fit of all models in parallel!
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            self.model_results = pool.starmap(all_workers, itlist)
            pool.close()

            # Turn the array in model results into a dictionary (map) and extract even weighting...
            # Also, sometimes we are going to get NaNs so we are going to have to work with a dataframe
            # and handle each observation separately.
            temp = dict()
            df = pd.DataFrame()
            for i in range(0, len(self.model_results)):
                temp.update({self.model_results[i][0]:self.model_results[i][1]})
                df = pd.concat([df, pd.DataFrame(self.model_results[i][1].fitted)], axis=1, ignore_index=True)
            self.fitted = np.ndarray(shape=[len(self.ts),1])
            self.model_results = temp

            if weights == 'equal':

                for i in range(0, df.shape[0]):
                    good_row_data = df.ix[i].dropna()
                    if len(good_row_data) > 0:
                        weights = np.ndarray(shape=(len(good_row_data),1))
                        weights.fill(1.0/len(good_row_data))
                        self.fitted[i] = np.dot(good_row_data, weights)
                    else:
                        self.fitted[i] = np.NaN

            else:
                # Measure the error - using some error measure between self.ts and df[,i]
                # ['MAE', 'MSE', 'MSLE', 'MEAE']
                column_errors = list()
                for i in range(0, df.shape[1]):
                    fitted_value = df[df.columns[i]]
                    if error_method == 'MAE':
                        column_errors.append(metrics.mean_absolute_error(self.ts.values[fitted_value.isna()==False],
                                                                         fitted_value[fitted_value.isna()==False]))
                    elif error_method == 'MSE':
                        column_errors.append(metrics.mean_squared_error(self.ts.values[fitted_value.isna()==False],
                                                                        fitted_value[fitted_value.isna()==False]))
                    elif error_method == 'RMSE':
                        column_errors.append(math.sqrt(metrics.mean_squared_error(self.ts.values[fitted_value.isna()==False],
                                                                        fitted_value[fitted_value.isna()==False])))
                    elif error_method == 'MSLE':
                        column_errors.append(metrics.mean_squared_log_error(self.ts.values[fitted_value.isna()==False],
                                                                            fitted_value[fitted_value.isna()==False]))
                    elif error_method == 'MEAE':
                        column_errors.append(metrics.median_absolute_error(self.ts.values[fitted_value.isna() == False],
                                                                            fitted_value[fitted_value.isna() == False]))
                tweights = 1.0/np.asarray(column_errors)
                weightsnorm = np.linalg.norm(tweights, ord=1)
                weights = tweights/weightsnorm
                # Hybrid weighting which handles NaNs in rows
                for i in range(0, df.shape[0]):
                    good_row_data = df.ix[i].dropna()
                    # Use properly weighted if we have all the data.
                    if len(good_row_data) == len(weights):
                        self.fitted[i] = np.dot(good_row_data, weights)
                    # Weight on the good values
                    elif len(good_row_data) > 0:
                        stweights = tweights[df.ix[i].isna() == False]
                        stweightsn = np.linalg.norm(stweights, ord=1)
                        self.fitted[i] = np.dot(good_row_data, stweights/stweightsn)
                    else:
                        self.fitted[i] = np.NaN


        return self.fitted

    def error(self, error_method = ['MAE', 'MSE', 'MSLE', 'MEAE', 'RMSE']):
        if self.fitted is None:
            return None
        if error_method is list:
            logging.info('Selecting "MSE" from ["MAE", "MSE", "MSLE", "MEAE", "RSME"]')
            error_method = 'MSE'
        elif error_method not in ['MAE', 'MSE', 'MSLE', 'MEAE']:
            logging.warning('Invalid error method - using MSE')
            error_method = 'MSE'

        def RMSE(ts, fit): math.sqrt(metrics.mean_squared_error(ts,fit))

        res = dict()
        if error_method == 'MAE': FUN = metrics.mean_absolute_error
        elif error_method == 'MSE': FUN = metrics.mean_squared_error
        elif error_method == 'RMSE': FUN = RMSE
        elif error_method == 'MSLE': FUN = metrics.mean_squared_log_error
        elif error_method == 'MEAE': FUN = metrics.median_absolute_error

        res['error'] = FUN(self.ts, self.fitted)

        modelkeys = self.model_results.keys()
        for key in modelkeys:
            try:
                res[self.MODELS[key]] = FUN(self.ts, self.model_results[key].fitted)
            except:
                res[self.MODELS[key]] = None
        return res