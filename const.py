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
DATA_LOC = '_data'
MIN_Y_SAMPLES = 30
Y_OUTLIER = 0.05
BATCH_EXE_CODE = 'nlKJ12avTYHDlw956evclk2b'
QUEUE_LIMIT = 2

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
    """Execution types of Model.
    
    新增模型時預測 ADD_PREDICT = 'AP'
    新增模型時回測 ADD_BACKTEST = 'AB'
    批次執行預測 BATCH_PREDICT = 'BP'
    新增模型時預測完成 ADD_PREDICT_FINISHED = 'APF'
    新增模型時回測完成 ADD_BACKTEST_FINISHED = 'ABF'
    批次執行預測完成 BATCH_PREDICT_FINISHED = 'BPF'
    """
    ADD_PREDICT = 'AP'
    ADD_BACKTEST = 'AB'
    BATCH_PREDICT = 'BP'
    ADD_PREDICT_FINISHED = 'APF'
    ADD_BACKTEST_FINISHED = 'ABF'
    BATCH_PREDICT_FINISHED = 'BPF'

class ModelStatus(int, Enum):
    """Status of Model on DB."""
    FAILED = -1
    ADDED = 1
    CREATED = 2
    COMPLETE = 3

class MarketOccurField(Enum):
    """Fields of pattern market occur stat info table on DB.
    
    Membors
    -------
    PATTERN_ID: ID of pattern.
    MARKET_ID: ID of market.
    DATE_PERIOD: date Period.
    OCCUR_CNT: pattern occur count in history.
    NON_OCCUR_CNT: pattern non-occur count in history.
    MARKET_RISE_CNT: rise count after pattern occurs.
    MARKET_FLAT_CNT: flat count after pattern occurs.
    MARKET_FALL_CNT: fall count after pattern occurs.
    """
    PATTERN_ID = 'PATTERN_ID'
    MARKET_ID = 'MARKET_CODE'
    DATE_PERIOD = 'DATE_PERIOD'
    OCCUR_CNT = 'OCCUR_CNT'
    NON_OCCUR_CNT = 'NON_OCCUR_CNT'
    MARKET_RISE_CNT = 'MARKET_RISE_CNT'
    MARKET_FLAT_CNT = 'MARKET_FLAT_CNT'
    MARKET_FALL_CNT = 'MARKET_FALL_CNT'

class MarketDistField(Enum):
    """Fields of pattern market occur stat info table on DB.
    
    Membors
    -------
    PATTERN_ID: ID of pattern.
    MARKET_ID: ID of market.
    DATE_PERIOD: date Period.
    RETURN_MEAN: return mean.
    RETURN_STD: return standard deviation.
    """
    PATTERN_ID = 'PATTERN_ID'
    MARKET_ID = 'MARKET_CODE'
    DATE_PERIOD = 'DATE_PERIOD'
    RETURN_MEAN = 'RETURN_MEAN'
    RETURN_STD = 'RETURN_STD'
