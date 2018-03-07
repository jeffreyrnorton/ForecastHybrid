import numpy as np
from sklearn import metrics
import logging
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import multiprocessing
from math import sqrt
import rpy2.robjects as ro

# Note that seq(start,end) for indices will be
# np.linspace(start=start, end=end, num=end-start+1, dtype=int)
def p_seq(start, end):
    return(np.linspace(start=start, stop=end, num=end-start+1, dtype=int))


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


    def RcvtsWrapper(self, x, curve, windowSize, ncores=2, **kwargs):

        windowSize = int(windowSize)

        # Make sure we don't have too many cores
        if int(len(x)/windowSize) < ncores:
            ncores = int(len(x)/windowSize)
            logging.info("Reducing number of cores to {}".format(ncores))

        curve.setTimeSeries(rtsname="x", tts=x)

        # Define the function
        command = curve.setREnv(curve.myname(), tsname='x', inline=True, **kwargs)
        funcname = "hfTSFIT"
        curve.generateRForCVTS(command, namets="x", funcname=funcname)

        # Define cvts in R in the R environment
        ro.r(self.cvtsRSource())
        # Call R cvts with input arguments
        cvtsr = 'cvts(x, FUN={}, FCFUN=forecast, windowSize={}, rolling=FALSE, num.cores={})'.format(funcname, windowSize, ncores)

        logging.info("FUN={}".format(cvtsr))

        try:
            res = ro.r(cvtsr)
            return res
        except:
            logging.error(curve.rtracebackerror())
        return None


    ## cvts.R from HybridForecast R package as of Feb 2018
    def cvtsRSource(self):
        return """
    library(foreach)
    library(doParallel)
       
    cvts <- function(x, FUN = NULL, FCFUN = NULL,
                     rolling = FALSE, windowSize = 84,
                     maxHorizon = 5,
                     horizonAverage = FALSE,
                     xreg = NULL,
                     saveModels = ifelse(length(x) > 500, FALSE, TRUE),
                     saveForecasts = ifelse(length(x) > 500, FALSE, TRUE),
                     verbose = TRUE, num.cores = 2L, extraPackages = NULL,
                     ...){
      # Default model function
      # This can be useful for methods that estimate the model and forecasts in one step
      # e.g. GMDH() from the "GMDH" package or thetaf()/meanf()/rwf() from "forecast". In this case,
      # no model function is used but the forecast function is applied in FCFUN
      if(is.null(FUN)){
        FUN <- function(x){
          return(x)
        }
      }
      # Determine which packages will need to be sent to the parallel workers
      excludePackages <- c("", "R_GlobalEnv")
      includePackages <- "forecast"
      funPackage <- environmentName(environment(FUN))
      if(!is.element(funPackage, excludePackages)){
        includePackages <- c(includePackages, funPackage)
      }

      # Default forecast function
      if(is.null(FCFUN)){
        FCFUN <- forecast
      }
      # Determine which packages will need to be sent to the parallel workers
      fcfunPackage <- environmentName(environment(FCFUN))
      if(!is.element(fcfunPackage, excludePackages)){
        includePackages <- c(includePackages, fcfunPackage)
      }
      if(!is.null(extraPackages)){
        includePackages <- c(includePackages, extraPackages)
      }
      includePackages <- unique(includePackages)

      f = frequency(x)
      tspx <- tsp(x)
      if(is.null(tspx)){
        x <- ts(x, frequency = f)
      }

      if(any(sapply(c(x, windowSize, maxHorizon), FUN = function(x) !is.numeric(x)))){
        stop("The arguments x, windowSize, and maxHorizon must all be numeric.")
      }

      if(any(c(windowSize, maxHorizon) < 1L)){
        stop("The arguments windowSize, and maxHorizon must be positive integers.")
      }

      if(any(c(windowSize, maxHorizon) %% 1L != 0)){
        stop("The arguments windowSize, and maxHorizon must be positive integers.")
      }

      # Ensure at least two periods are tested
      if(windowSize + 2 * maxHorizon > length(x)){
        stop("The time series must be longer than windowSize + 2 * maxHorizon.")
      }

      # Check if fitting function accepts xreg when xreg is not NULL
      xregUse <- FALSE
      if (!is.null(xreg)) {
        fitArgs <- formals(FUN)
        if (any(grepl("xreg", names(fitArgs)))) {
          xregUse <- TRUE
          xreg <- as.matrix(xreg)
        } else
          warning("Ignoring xreg parameter since fitting function does not accept xreg")
      }

      # Combined code for rolling/nonrolling CV
      nrow = ifelse(rolling, length(x) - windowSize - maxHorizon + 1,
                    as.integer((length(x) - windowSize) / maxHorizon))
      resultsMat <- matrix(NA, nrow = nrow, ncol = maxHorizon)

      forecasts <- fits <- vector("list", nrow(resultsMat))
      slices <- tsPartition(x, rolling, windowSize, maxHorizon)

      # Perform the cv fits
      # adapted from code from Rob Hyndman at http://robjhyndman.com/hyndsight/tscvexample/
      # licensend under >= GPL2 from the author

      cl <- parallel::makeCluster(num.cores)
      doParallel::registerDoParallel(cl)
      on.exit(parallel::stopCluster(cl))
      # Appease R CMD CHECK with sliceNum declaration
      sliceNum <- NULL
      results <- foreach::foreach(sliceNum = seq_along(slices),
                                  .packages = includePackages) %dopar% {
        if(verbose){
          cat("Fitting fold", sliceNum, "of", nrow(resultsMat), "\n")
        }
        results <- list()

        trainIndices <- slices[[sliceNum]]$trainIndices
        testIndices <- slices[[sliceNum]]$testIndices
        
        tsSubsetWithIndices <- function(x, indices) {
            xtime <- time(x)
            minIndex <- min(indices)
            maxIndex <- max(indices)
    
            if (maxIndex > length(xtime)){
                stop("Max subset index cannot exceed time series length")
            }
            if (all(seq(minIndex, maxIndex, 1) != indices)){
                stop("Time series can only be subset with continuous indices")
            }
            return(window(x, start = xtime[minIndex], end = xtime[maxIndex]))
        }

        tsTrain <- tsSubsetWithIndices(x, trainIndices)
        tsTest <- tsSubsetWithIndices(x, testIndices)

        if(xregUse){
          xregTrain <- xreg[trainIndices, ,drop = FALSE]
          xregTest <- xreg[testIndices, ,drop = FALSE]
          mod <- do.call(FUN, list(tsTrain, xreg = xregTrain, ...))
          fc <- do.call(FCFUN, list(mod, xreg = xregTest, h = maxHorizon))
        }else{
          mod <- do.call(FUN, list(tsTrain, ...))
          fc <- do.call(FCFUN, list(mod, h = maxHorizon))
        }

        if(saveModels){
          results$fits <- mod
        }

        if(saveForecasts){
          results$forecasts <- fc
        }

        results$resids <- tsTest - fc$mean
        results
      }

      # Gather the parallel chunks
      residlist <- lapply(results, function(x) unlist(x$resids))
      resids <- matrix(unlist(residlist, use.names = FALSE),
                       ncol = maxHorizon, byrow = TRUE)
      forecasts <- lapply(results, function(x) x$forecasts)
      fits <- lapply(results, function(x) x$fits)

      # Average the results from all forecast horizons up to maxHorizon
      if(horizonAverage){
        resids <- as.matrix(rowMeans(resids), ncol = 1)
      }

      if(!saveModels){
        fits <- NULL
      }
      if(!saveForecasts){
        forecasts <- NULL
      }

      params <- list(FUN = FUN,
                     FCFUN = FCFUN,
                     rolling = rolling,
                     windowSize = windowSize,
                     maxHorizon = maxHorizon,
                     horizonAverage = horizonAverage,
                     saveModels = saveModels,
                     saveForecasts = saveForecasts,
                     verbose = verbose,
                     num.cores = num.cores,
                     extra = list(...))

      result <- list(x = x,
                   xreg = xreg,
                   params = params,
                   forecasts = forecasts, 
                   models = fits, 
                   residuals = resids)

      class(result) <- "cvts"
      return(result)
    }

    tsPartition <- function(x, rolling, windowSize, maxHorizon) {
      numPartitions <- ifelse(rolling, length(x) - windowSize - maxHorizon + 1, as.integer((length(x) - windowSize) / maxHorizon))

      slices <- rep(list(NA), numPartitions)
      start <- 1

        for (i in 1:numPartitions) {
            if(rolling){
                trainIndices <- seq(start, start + windowSize - 1, 1)
                testIndices <-  seq(start + windowSize, start + windowSize + maxHorizon - 1)
                start <- start + 1
            }
            ## Sample the correct slice for nonrolling
            else{
                trainIndices <- seq(start, start + windowSize - 1 + maxHorizon * (i - 1), 1)
                testIndices <- seq(start + windowSize + maxHorizon * (i - 1), start + windowSize - 1 + maxHorizon * i)
            }

            slices[[i]] <- list(trainIndices = trainIndices, testIndices = testIndices)
        }

        return(slices)
    }

    extractForecasts <- function(cv, horizon = 1) {
          if (horizon > cv$params$maxHorizon) 
             stop("Cannot extract forecasts with a horizon greater than the model maxHorizon")
          pointfList <- Map(function(fcast) {
             pointf <- fcast$mean
             window(pointf, start = time(pointf)[horizon], 
                    end = time(pointf)[horizon])
             },
             cv$forecasts) 

          pointf <- Reduce(tsCombine, pointfList)

          #Ensure all points in the original series are represented (makes it easy for comparisons)
          template <- replace(cv$x, c(1:length(cv$x)), NA)
          return(tsCombine(pointf, template))
    }

        """

    ### Python translation of cvts.R
    def cvtsp(self, x, FUN, thread_pool=None,
              rolling=False, windowSize=84, maxHorizon=5,
              horizonAverage=False, saveModels=False,
              saveForecasts=False, num_cores=2, **kwargs):

        windowSize = int(windowSize)
        # A bunch of checking ...

        # Check if fitting function accepts xreg when xreg is not None

        slices = tsPartition(x, rolling, windowSize, maxHorizon)

        if thread_pool is None:
            thread_pool = multiprocessing.Pool(processes=
                                               min(len(slices), multiprocessing.cpu_count()))

        itlist = list()
        function_args = None if kwargs is None else dict(**kwargs)

        for slice in slices:
            itlist.append((x, FUN, slice, maxHorizon, saveModels, saveForecasts, function_args))

        # I think we want to set chunk size to limit the number of threads used, but not sure yet.
        cvts_results = np.asarray(thread_pool.starmap(cvtsp_worker, itlist))

        return None

def cvts_worker(test_data, train_data, FUN, h=1, args=None):
    fnc = FUN(train_data)
    fnc.fit() if args is None else fnc.fitR(**args)
    forecast_data = fnc.forecast(h=h)
    forecast_point = np.array(forecast_data['forecast'])[h-1]
#    print("{} : {}".format(test_data.values[h - 1], forecast_point))
    return test_data.values[h - 1], forecast_point, fnc
           #train_data.values[len(train_data)-1], fitted_data[len(fitted_data)-1]


def tsPartition(x, rolling, windowSize, maxHorizon):
    numPartitions = len(x) - windowSize - maxHorizon + 1 if rolling \
        else int((len(x) - windowSize)/maxHorizon)
    slices = []
    start = 0
    last = None
    for i in range(0, numPartitions):
        if rolling:
            trainIndices = [start, start+windowSize-1]
            testIndices  = [start+windowSize, min(start+windowSize+maxHorizon-1, len(x)-1)]
            start += 1
        else:
            if last is None: last = start + windowSize - 1
            trainIndices = [start, last]
            last += maxHorizon
            testIndices  = [trainIndices[1]+1, last]
        slices.append({'trainIndices':trainIndices, 'testIndices':testIndices})
    return slices


# Not supporting xreg at this time...
def cvtsp_worker(x, FUN, slice, maxHorizon, saveModels=False, saveForecasts=False, function_args=None):
    trainIndices = slice['trainIndices']
    testIndices  = slice['testIndices']

    def tsSubsetWithIndices(x, indices):
        xtime = x.copy()
        minIndex = min(indices)
        maxIndex = max(indices)
        if maxIndex > len(x):
            logging.warning("Max subset index cannot exceed time series length")
            return None
        return xtime.iloc[minIndex:maxIndex + 1]

    tsTrain = tsSubsetWithIndices(x, trainIndices)
    tsTest  = tsSubsetWithIndices(x, testIndices)

    fnc = FUN(tsTrain)
    fc = None
    if fnc.fitR(**function_args) is not None:
        logging.warning("Could not calculate fit with model")
        fc = fnc.forecast(h=maxHorizon)

    results = {}
    if saveModels:
        results['model'] = fnc
    if fc is not None:
        if saveForecasts:
            results['forecast'] = fc
        results['resids'] = tsTest.values - fc['forecast']
    return results
