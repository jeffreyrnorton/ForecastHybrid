import numpy as np

#' Generate training and test indices for time series cross validation
#'
#' Training and test indices are generated for time series cross validation.
#' Generated indices are based on the training windowSize, forecast horizons
#' and whether a rolling or non-rolling cross validation procedure is desired.
#'
#' @export
#' @param x A time series
#' @param rolling Should indices be generated for a rolling or non-rolling procedure?
#' @param windowSize Size of window for training
#' @param maxHorizon Maximum forecast horizon
#'
#' @return List containing train and test indices for each fold
#'
#' @author Ganesh Krishnan
#' @examples
#' tsPartition(AirPassengers, rolling = TRUE, windowSize = 10, maxHorizon = 2)
#'

def tsPartition(x, rolling, windowSize, maxHorizon):
    numPartitions = (len(x) - windowSize - maxHorizon + 1) if rolling else (int((len(x) - windowSize))/maxHorizon)
    slices = ()
    start = 0
    for i in range(0, numPartitions):
        if rolling:
            trainIndices = np.linspace(start, start+windowSize-1, 1)
            testIndices  = np.linspace(start+windowSize, start + windowSize + maxHorizon - 1)
            start += 1
        else:
            trainIndices = np.linspace(start, start + windowSize - 1 + maxHorizon * i, 1)
            testIndices =  np.linspace(start + windowSize + maxHorizon * i, start + windowSize - 1 + maxHorizon * (i+1))

        slices.add({'trainIndices':trainIndices, 'testIndices':testIndices})
    return slices
