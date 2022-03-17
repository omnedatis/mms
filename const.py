# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

"""
from enum import Enum
import datetime

PREDICT_PERIODS = [5, 10, 20, 40, 60, 120]
DEFAULT_TRAIN_BEGIN_DATE = datetime.date(2007, 7, 1)
DEFAULT_TGAP = 3
MIN_BACKTEST_LEN = 180
LOCAL_DB = '_local_db'
MIN_Y_SAMPLES = 30
Y_OUTLIER = 0.05
BATCH_EXE_CODE = 'nlKJ12avTYHDlw956evclk2b'

class PredictResultField(Enum):
    """Fields of predict result table on DB.
    
    Membors
    -------
    MODEL_ID: ID of model.
    MARKET_ID: ID of market.
    DATE: Predicting base date.
    PERIOD: Predicting period.
    UPPER_BOUND: Upper-bound of predicting range.
    LOWER_BOUND: Lower-bound of predicting range.
    PREDICT_VALUE: value of predicting result.
    
    """
    MODEL_ID = 'MODEL_ID'
    MARKET_ID = 'MARKET_CODE'
    DATE = 'DATA_DATE'
    PERIOD = 'DATE_PERIOD'
    UPPER_BOUND = 'UPPER_BOUND'
    LOWER_BOUND = 'LOWER_BOUND'
    PREDICT_VALUE = 'DATA_VALUE'
    
class PatternResultField(Enum):
    """Fields of pattern result table on DB.
    
    Membors
    -------
    PATTERN_ID: ID of pattern.
    MARKET_ID: ID of market.
    DATE: Trading date.
    VALUE: Value of pattern.
    
    """
    PATTERN_ID = 'PATTERN_ID'
    MARKET_ID = 'MARKET_CODE'
    DATE = 'DATA_DATE'
    VALUE = 'OCCUR_YN'

class ModelExecution(str, Enum):
    """Execution types of Model."""
    ADD_PREDICT = '0'
    ADD_BACKTEST = '1'
    BATCH_PREDICT = '2'

class ModelStatus(int, Enum):
    """Status of Model on DB."""
    FAILED = -1
    ADDED = 1
    CREATED = 2
    COMPLETE = 3

class ModelExecution(str, Enum):
    """Execution types of Model."""
    ADD_PREDICT = '0'
    ADD_BACKTEST = '1'
    BATCH_PREDICT = '2'