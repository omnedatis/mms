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
LOCAL_DB = '../_local_db'
DATA_LOC = '_data'
LOG_LOC = './log'
EXCEPT_DATA_LOC = '_except'
MIN_Y_SAMPLES = 30
Y_OUTLIER = 0.05
BATCH_EXE_CODE = 'nlKJ12avTYHDlw956evclk2b'
QUEUE_LIMIT = 2
PORT = 8080
FUNC_CACHE_SIZE = 2000
MULTI_PATTERN_CACHE_SIZE = 2000 * 100

class BatchType(str, Enum):
    """Batch 的執行型態

    Membors
    -------
    INIT_BATCH: 初始化時的批次執行.
    SERVICE_BATCH: 對外部提供的批次執行.

    """
    INIT_BATCH = 'init'
    SERVICE_BATCH = 'service'

class ExecMode(Enum):
    """Fields of excution modes. """

    DEV = "dev"
    PROD = "prod"
    UAT = 'uat'


    @classmethod
    def get(cls, value):
        for each in cls:
            if value == each.value:
                return each.value
        else:
            return None

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

class MarketScoreField(Enum):
    """市場分數標籤表的欄位資訊

    Membors
    -------
    MARKET_ID: ID of market.
    DATE_PERIOD: date Period.
    MARKET_SCORE: 分數標籤
    UPPER_BOUND: 分數標籤對應的報酬上界
    LOWER_BOUND: 分數標籤對應的報酬下界
    """
    MARKET_ID = 'MARKET_CODE'
    DATE_PERIOD = 'DATE_PERIOD'
    MARKET_SCORE = 'MARKET_SCORE'
    UPPER_BOUND = 'UPPER_BOUND'
    LOWER_BOUND = 'LOWER_BOUND'

class MarketPeriodField(Enum):
    """市場各天期歷史報酬的欄位資訊

    Membors
    -------
    MARKET_ID: ID of market.
    DATE_PERIOD: date Period.
    PRICE_DATE: 價格日期
    DATA_DATE: 計算報酬的計算日期
    NET_CHANGE: 漲跌幅
    NET_CHANGE_RATE: 報酬率
    """
    MARKET_ID = 'MARKET_CODE'
    DATE_PERIOD = 'DATE_PERIOD'
    PRICE_DATE = 'PRICE_DATE'
    DATA_DATE = 'DATA_DATE'
    NET_CHANGE = 'NET_CHANGE'
    NET_CHANGE_RATE = 'NET_CHANGE_RATE'

class MarketStatField(Enum):
    """市場統計資訊的欄位資訊

    Membors
    -------
    MARKET_ID: ID of market.
    DATE_PERIOD: date Period.
    RETURN_MEAN: return mean.
    RETURN_STD: return standard deviation.
    RETURN_CNT: 所有報酬的樣本個數
    """
    MARKET_ID = 'MARKET_CODE'
    DATE_PERIOD = 'DATE_PERIOD'
    RETURN_MEAN = 'RETURN_MEAN'
    RETURN_STD = 'RETURN_STD'
    RETURN_CNT = 'RETURN_CNT'

class ScoreMetaField(Enum):
    """分數標籤統計資訊

    Membors
    -------
    SCORE_CODE: 分數代碼
    SCORE_VALUE: 分數值
    UPPER_BOUND: 分數上界的標準差倍率
    LOWER_BOUND: 分數下界的標準差倍率
    """
    SCORE_CODE = "SCORE_CODE"
    SCORE_VALUE = "SCORE_VALUE"
    UPPER_BOUND = "UPPER_BOUND"
    LOWER_BOUND = "LOWER_BOUND"

class MarketInfoField(Enum):
    """市場資訊

    Membors
    -------
    MARKET_CODE: 市場代碼
    MARKET_SOURCE_TYPE: 市場來源型態(BLB, TEJ)
    MARKET_SOURCE_CODE: 市場於來源端資料的編碼
    MARKET_NAME: 市場名稱
    """
    MARKET_CODE = "MARKET_CODE"
    MARKET_SOURCE_TYPE = "MARKET_SOURCE_TYPE"
    MARKET_SOURCE_CODE = "MARKET_SOURCE_CODE"
    MARKET_NAME = "MARKET_NAME"

class DSStockInfoField(Enum):
    """TEJ 市場資訊

    Membors
    -------
    STOCK_CODE: 股票代碼
    ISIN_CODE: ISIN 代碼
    COMPANY_CODE: 公司代碼
    EXCHANGE_TYPE: 上市別
    INDUSTRY_CODE: TEJ 產業別代碼
    TSE_INDUSTRY_CODE: TSE 產業別代碼
    CUR_CODE: 幣別
    TSE_IPO_DATE: 首次 TSE 上市日
    OTC_IPO_DATE: 首次 OTC 上市日
    REG_IPO_DATE: 首次 REG 上市日
    DELISTING_DATE: 下市日期
    """
    STOCK_CODE = "STOCK_CODE"
    ISIN_CODE = "ISIN_CODE"
    COMPANY_CODE = "COMPANY_CODE"
    EXCHANGE_TYPE = "EXCHANGE_TYPE"
    INDUSTRY_CODE = "INDUSTRY_CODE"
    TSE_INDUSTRY_CODE = "TSE_INDUSTRY_CODE"
    CUR_CODE = "CUR_CODE"
    TSE_IPO_DATE = "TSE_IPO_DATE"
    OTC_IPO_DATE = "OTC_IPO_DATE"
    REG_IPO_DATE = "REG_IPO_DATE"
    DELISTING_DATE = "DELISTING_DATE"
