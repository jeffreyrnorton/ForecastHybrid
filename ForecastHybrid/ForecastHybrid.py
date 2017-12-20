import pandas
import rpy2.robjects.packages as importr
import logging


class ForecastHybrid:
    def __init__(self, y):
        if type(y) is not list:
            logging.fatal("Input y must be a a list.")
            exit()
        if len(y) < 4:
            logging.fatal("Input y must contain 4 or more data points.")
            exit()
        self.y = y

        