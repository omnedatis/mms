# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

"""
from enum import Enum, IntEnum
import datetime
from typing import Optional
import re

PREDICT_PERIODS = [5, 10, 20, 40, 60, 120]
DEFAULT_TRAIN_BEGIN_DATE = datetime.date(2007, 7, 1)
DEFAULT_TRAIN_GAP = 3
MIN_BACKTEST_LEN = 180
LOCAL_DB = '../_local_db'
DATA_LOC = '_data'
LOG_LOC = './log'
EXCEPT_DATA_LOC = '_except'
Y_LABELS = 5
MIN_Y_SAMPLES = 30 * Y_LABELS
Y_OUTLIER = 0.025  # 150 * 0.025 = 3.75 vs 30 -> 10%
DEFAULT_ALGORITHM = 'DECISION_TREE_CLASSIFIER'
BATCH_EXE_CODE = 'nlKJ12avTYHDlw956evclk2b'
MODEL_QUEUE_LIMIT = 2
PATTERN_QUEUE_LIMIT = 1
PATTERN_UPDATE_CPUS = 7
PORT = 8080
FUNC_CACHE_SIZE = 2000
MULTI_PATTERN_CACHE_SIZE = 2000 * 100
MIN_DB_WRITING_LENGTH = 0
MA_GRAPH_SAMPLE_NUM = 10

class TypeCheckMap(str,Enum):
    STRING = "string", "字串", str, "[A-z]+"
    INT = "int", "整數", int, "\d+\.?"
    FLOAT = "float", "浮點數", float, "\d+\.\d*"

    def __new__(cls, value, cstring, type, pattern):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.cstring = cstring
        obj.type = type
        obj._pattern = re.compile(pattern)
        return obj


    def check(self, value):
        return not not (self._pattern.fullmatch(value))


class HttpResponseCode(IntEnum):

    OK = 200, "OK"
    ACCEPTED = 202, "Accepted"
    BAD_REQUEST = 400, "Bad request"
    NOT_FOUND = 404, "Not found"
    METHOD_NOT_ALLOWED = 405, "Method not allowed"
    INTERNAL_SERVER_ERROR = 500, "Internal server error"

    def __new__(cls, value, phrase):
        obj = int.__new__(cls, value)
        obj._value_ = value

        obj.phrase = phrase
        return obj


    def format(self, data:Optional[dict]=None) -> dict:
        return {"status":self.value, "message":self.phrase, "data":data}

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

class TableName(Enum):
    """資料表名稱

    Membors
    -------
    DS_S_STOCK: TEJ 台股資料表
    MKT_INFO: Mimosa 市場資料表
    MKT_HISTORY_PRICE: 市場歷史價格表
    MKT_PERIOD: 市場最新各天期報酬表
    MKT_PERIOD_HISTORY: 市場歷史各天期報酬表
    MKT_DIST: 市場統計資訊表
    MKT_SCORE: 市場分數範圍資訊表
    PAT_INFO: 現象資訊表
    PAT_PARAM: 現象參數資訊表
    PATTERN_RESULT: 現象當日發生狀況資訊表
    PAT_MKT_DIST: 現象最新統計結果資訊表
    PAT_MKT_OCCUR: 現象最新發生次數統計表
    MACRO_INFO: Macro資訊表
    MACRO_PARAM: Macro參數資訊表
    MODEL_INFO: 觀點資訊表
    MODEL_EXECUTION: 觀點執行狀態資訊表
    MODEL_MKT_MAP: 觀點使用市場資訊表
    MODEL_PAT_MAP: 觀點使用現象資訊表
    PREDICT_RESULT: 觀點最新預測結果資訊表
    PREDICT_RESULT_HISTORY: 觀點歷史預測結果資訊表
    SCORE_META: 分數標籤統計資訊表
    MODEL_MKT_HIT_SUM: 觀點命中次數記錄表
    """
    DS_S_STOCK = "DS_S_STOCK"
    MKT_INFO = "FCST_MKT"
    MKT_HISTORY_PRICE = "FCST_MKT_PRICE_HISTORY"
    MKT_PERIOD = "FCST_MKT_PERIOD"
    MKT_PERIOD_HISTORY = "FCST_MKT_PERIOD_HISTORY"
    MKT_DIST = "FCST_MKT_DIST"
    MKT_SCORE = "FCST_MKT_SCORE"
    PAT_INFO = "FCST_PAT"
    PAT_PARAM = "FCST_PAT_PARAM"
    PATTERN_RESULT = "FCST_PAT_MKT_EVENT"
    PAT_MKT_DIST = "FCST_PAT_MKT_DIST"
    PAT_MKT_OCCUR = "FCST_PAT_MKT_OCCUR"
    MACRO_INFO = "FCST_MACRO"
    MACRO_PARAM = "FCST_MACRO_PARAM"
    MODEL_INFO = "FCST_MODEL"
    MODEL_EXECUTION = "FCST_MODEL_EXECUTION"
    MODEL_MKT_MAP = "FCST_MODEL_MKT_MAP"
    MODEL_PAT_MAP = "FCST_MODEL_PAT_MAP"
    PREDICT_RESULT = "FCST_MODEL_MKT_VALUE"
    PREDICT_RESULT_HISTORY = "FCST_MODEL_MKT_VALUE_HISTORY"
    SCORE_META = "FCST_SCORE"
    MODEL_MKT_HIT_SUM = "FCST_MODEL_MKT_HIT_SUM"
    PATTERN_EXECUTION = 'FCST_PAT_EXECUTION'

class StoredProcedule(Enum):
    """SP名稱

    Membors
    -------
    TRUNCATE_AI_SWAP: 清空計算時所需要為空的資料表的 SP
    GET_SERIAL_NO: 取得觀點執行代碼的 SP
    UPDATE_FCST_MODEL_MKT_HIT_SUM_SWAP: 統計觀點準確率的 SP
    SWAP_FCST: 將 SWAP 切換至正式資料表的SP
    """
    TRUNCATE_AI_SWAP = "SP_TRUNCATE_AI_SWAP"
    GET_SERIAL_NO = "SP_GET_SERIAL_NO"
    UPDATE_FCST_MODEL_MKT_HIT_SUM_SWAP = "SP_UPDATE_FCST_MODEL_MKT_HIT_SUM_SWAP"
    SWAP_FCST = "SP_SWAP_FCST"

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

class PatternExecution(str, Enum):
    """Execution types of Model.

    跑批時計算 BATCH_SERVICE = 'AP'
    新增現象時計算 ADD_PATTERN = 'AB'
    跑批時計算完成 BATCH_SERVICE_FINISHED = 'APF'
    新增現象時計算完成 ADD_PATTERN_FINISHED = 'BPF'
    """
    BATCH_SERVICE = 'BS'
    ADD_PATTERN = 'AP'
    BATCH_SERVICE_FINISHED = 'BSF'
    ADD_PATTERN_FINISHED = 'APF'

class PatternExecutionField(Enum):
    """觀點執行狀態資訊

    Members
    -------
    EXEC_ID: 執行ID
    PATTERN_ID: 觀點ID
    STATUS_CODE: 執行狀態代碼
    START_DT: 執行起始時間
    END_DT: 執行結束時間
    """
    EXEC_ID = "EXEC_ID"
    PATTERN_ID = "PATTERN_ID"
    STATUS_CODE = "STATUS_CODE"
    START_DT = "START_DT"
    END_DT = "END_DT"

class ModelStatus(int, Enum):
    """Status of Model on DB."""
    FAILED = -1
    ADDED = 1
    CREATED = 2
    COMPLETE = 3

class DBModelStatus(str, Enum):
    """ 觀點在資料庫中的狀態

    Members
    -------
    PRIVATE_AND_VALID: 未發布且有效
    PUBLIC_AND_VALID: 發布且有效
    DRAFT: 草稿
    PRIVATE_AND_INVALID: 未發布且無效
    PUBLIC_AND_INVALID: 發布且無效

    """
    PRIVATE_AND_VALID = '0'
    PUBLIC_AND_VALID = '1'
    DRAFT = '2'
    PRIVATE_AND_INVALID = '3'
    PUBLIC_AND_INVALID = '4'

class DBPatternStatus(str, Enum):
    """ 現象在資料庫中的狀態

    Members
    -------
    PRIVATE_AND_VALID: 未發布且有效
    PUBLIC_AND_VALID: 發布且有效
    DRAFT: 草稿
    PRIVATE_AND_INVALID: 未發布且無效
    PUBLIC_AND_INVALID: 發布且無效

    """
    PRIVATE_AND_VALID = '0'
    PUBLIC_AND_VALID = '1'
    DRAFT = '2'
    PRIVATE_AND_INVALID = '3'
    PUBLIC_AND_INVALID = '4'

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

class MarketHistoryPriceField(Enum):
    """歷史市場價格資訊

    Members
    -------
    MARKET_CODE: 市場代碼
    PRICE_DATE: 價格日期
    OPEN_PRICE: 開盤價
    HIGH_PRICE: 最高價
    LOW_PRICE: 最低價
    CLOSE_PRICE: 收盤價
    VOLUME: 成交量
    NET_CHANGE: 單日漲跌幅
    NET_CHANGE_RATE: 單日報酬率
    """
    MARKET_CODE = "MARKET_CODE"
    PRICE_DATE = "PRICE_DATE"
    OPEN_PRICE = "OPEN_PRICE"
    HIGH_PRICE = "HIGH_PRICE"
    LOW_PRICE = "LOW_PRICE"
    CLOSE_PRICE = "CLOSE_PRICE"
    VOLUME = "VOLUME"
    NET_CHANGE = "NET_CHANGE"
    NET_CHANGE_RATE = "NET_CHANGE_RATE"

class PatternInfoField(Enum):
    """現象基本資訊

    Members
    -------
    PATTERN_ID: 現象ID
    PATTERN_NAME: 現象名稱
    PATTERN_DESC: 現象描述
    MACRO_ID: 現象使用的 Macro ID
    PATTERN_STATUS: 現象狀態
    """
    PATTERN_ID = "PATTERN_ID"
    PATTERN_NAME = "PATTERN_NAME"
    PATTERN_DESC = "PATTERN_DESC"
    MACRO_ID = "MACRO_ID"
    PATTERN_STATUS = "PATTERN_STATUS"

class PatternParamField(Enum):
    """現象參數基本資訊

    Members
    -------
    PATTERN_ID: 現象ID
    MACRO_ID: 現象所使用的 Macro ID
    PARAM_CODE: 現象下的參數代碼
    PARAM_VALUE: 現象下的參數設定值
    """
    PATTERN_ID = "PATTERN_ID"
    MACRO_ID = "MACRO_ID"
    PARAM_CODE = "PARAM_CODE"
    PARAM_VALUE = "PARAM_VALUE"

class MacroInfoField(Enum):
    """Macro 基本資訊

    Members
    -------
    MACRO_ID: Macro ID
    MACRO_NAME: Macro 名稱
    MACRO_DESC: Macro 描述
    FUNC_CODE: Macro 所使用的 Function 代碼
    """
    MACRO_ID = "MACRO_ID"
    MACRO_NAME = "MACRO_NAME"
    MACRO_DESC = "MACRO_DESC"
    FUNC_CODE = "FUNC_CODE"

class MacroParamField(Enum):
    """Macro 參數基本資訊

    Members
    -------
    MACRO_ID: Macro ID
    PARAM_CODE: Macro 參數代碼
    PARAM_NAME: Macro 參數名稱
    PARAM_DESC: Macro 參數描述
    PARAM_DEFAULT: Macro 參數預設值
    PARAM_TYPE: Macro 參數型態(str, int, float)
    """
    MACRO_ID = "MACRO_ID"
    PARAM_CODE = "PARAM_CODE"
    PARAM_NAME = "PARAM_NAME"
    PARAM_DESC = "PARAM_DESC"
    PARAM_DEFAULT = "PARAM_DEFAULT"
    PARAM_TYPE = "PARAM_TYPE"

class ModelInfoField(Enum):
    """觀點基本資訊

    Members
    -------
    MODEL_ID: 觀點ID
    MODEL_NAME: 觀點名稱
    MODEL_DESC: 觀點描述
    MODEL_STATUS: 觀點狀態
    TRAIN_START_DT: 觀點訓練起始日
    TRAIN_END_DT: 觀點訓練終止日
    RETRAIN_CYCLE: 觀點重訓練區間(日)
    """
    MODEL_ID = "MODEL_ID"
    MODEL_NAME = "MODEL_NAME"
    MODEL_DESC = "MODEL_DESC"
    MODEL_STATUS = "MODEL_STATUS"
    TRAIN_START_DT = "TRAIN_START_DT"
    TRAIN_END_DT = "TRAIN_END_DT"
    RETRAIN_CYCLE = "RETRAIN_CYCLE"

class ModelMarketMapField(Enum):
    """觀點使用的市場資訊

    Members
    -------
    MODEL_ID: 觀點ID
    MARKET_CODE: 市場代碼
    """
    MODEL_ID = "MODEL_ID"
    MARKET_CODE = "MARKET_CODE"

class ModelPatternMapField(Enum):
    """觀點使用的現象資訊

    Members
    -------
    MODEL_ID: 觀點ID
    PATTERN_ID: 現象代碼
    """
    MODEL_ID = "MODEL_ID"
    PATTERN_ID = "PATTERN_ID"

class ModelExecutionField(Enum):
    """觀點執行狀態資訊

    Members
    -------
    EXEC_ID: 執行ID
    MODEL_ID: 觀點ID
    STATUS_CODE: 執行狀態代碼
    START_DT: 執行起始時間
    END_DT: 執行結束時間
    """
    EXEC_ID = "EXEC_ID"
    MODEL_ID = "MODEL_ID"
    STATUS_CODE = "STATUS_CODE"
    START_DT = "START_DT"
    END_DT = "END_DT"

class ModelMarketHitSumField(Enum):
    """觀點準確率資訊

    Members
    -------
    MODEL_ID: 觀點ID
    MARKET_CODE: 市場代碼
    DATE_PERIOD: 預測天期
    HIT: 命中次數
    FCST_CNT: 預測總數
    """
    MODEL_ID = "MODEL_ID"
    MARKET_CODE = "MARKET_CODE"
    DATE_PERIOD = "DATE_PERIOD"
    HIT = "HIT"
    FCST_CNT = "FCST_CNT"


class SerialNoType(Enum):
    """呼叫取得序列號時需要使用的參數

    EXECUTION: 執行狀態的序列號
    """
    EXECUTION = 'EXEC_ID'


class DataType(Enum):
    """資料庫中的資料型態

    STRING: 字串
    INT: 整數
    FLOAT: 浮點數
    DATETIME: 日期時間
    DATE: 日期
    BOOLEAN: 布林值
    """
    STRING = 'str'
    INT = 'int32'
    FLOAT = 'float32'
    DATETIME = 'datetime64[s]'
    DATE = 'datetime64[D]'
    BOOLEAN = 'bool'


class CacheName(Enum):
    MKT_HISTORY_PRICE = TableName.MKT_HISTORY_PRICE._value_
    MKT_INFO = TableName.MKT_INFO.value
    DS_S_STOCK = TableName.DS_S_STOCK.value
    PAT_INFO = TableName.PAT_INFO.value
    PAT_PARAM = TableName.PAT_PARAM.value
    MACRO_INFO = TableName.MACRO_INFO.value
    MACRO_PARAM = TableName.MACRO_PARAM.value
    MODEL_INFO = TableName.MODEL_INFO.value
    MODEL_EXECUTION = TableName.MODEL_EXECUTION.value
    MODEL_MKT_MAP = TableName.MODEL_MKT_MAP.value
    MODEL_PAT_MAP = TableName.MODEL_PAT_MAP.value
    SCORE_META = TableName.SCORE_META.value
    PATTERNS = 'patterns'
    MODELS = 'models'


