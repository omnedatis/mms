# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

"""

import datetime
import logging
import os
import shutil
from threading import Lock
import time
from typing import Any, List, Optional, NamedTuple, Dict
# from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm
from sklearn.tree import DecisionTreeClassifier as Dtc

from const import *
from utils import *
from func._tp import *

# 設定 logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(lineno)s - %(levelname)s - %(message)s')

db = None


def get_db():
    return db


def set_db(_db):
    global db
    db = _db


class PatternInfo(NamedTuple):
    """Pattern Info.

    fileds
    ------
    pid: pattern ID
    func: macro function
    kwargs: a dict mapping the name of parameters of macro function to 
            corresponding values given by the pattern.

    """
    pid: str
    func: str
    kwargs: Dict[str, Any]

    def run(self, market_id: str) -> pd.Series:
        """Evaulate the pattern for given market.

        Parameters
        ----------
        market_id: str

        Returns
        -------
        pd.Series

        """
        return eval(f"{self.func}('{market_id}', **{self.kwargs})").to_pandas()


def get_markets():
    """get market IDs from DB.

    Returns
    -------
    list of str

    """
    db = get_db()
    result = db.get_markets()
    return result


def get_patterns():
    """get patterns from DB.

    Returns
    -------
    list of PatternInfo

    """
    db = get_db()
    result = db.get_patterns()
    return result


def save_pattern_results(market_id, data):
    """Save pattern results to DB.

    Parameters
    ----------
    market_id: str
        ID of market.
    data: Pandas's Series
        A time-series of boolean which name is the pattern ID.

    """
    db = get_db()
    result = db.save_pattern_results(market_id, data)
    return result


def save_latest_pattern_results(data):
    """Save latest pattern results to DB.

    Parameters
    ----------
    data: Pandas's DataFrame
        A table of pattern results with columns for market_id, pattern_id, 
        price_date and value.

    See Also
    --------
    PatternResultField

    """
    db = get_db()
    result = db.save_latest_pattern_results(data)
    return result

def clean_db_cache():
    """Clean local cache."""
    db = get_db()
    db.clean_db_cache()

def clone_db_cache():
    """Build local cache from db."""
    db = get_db()
    db.clone_db_cache()

def get_pattern_results(market_id, patterns, begin_date):
    """Get pattern results from begining date to the latest of given market from DB.

    Parameters
    ----------
    market_id: str
        ID of market.
    patterns: list of str
        ID of patterns.
    begin_date: datetime.date, optional
        The begin date of the designated data range. If it is not set, get all
        history results.

    Returns
    -------
    Pandas's DataFrame
        A panel of boolean with all id of patterns as columns.

    """
    db = get_db()
    result = db.get_pattern_results(market_id, patterns, begin_date)
    return result


def get_filed_name_of_future_return(period: int) -> str:
    return f'FR{period}'


def gen_future_return(market: str, period: int):
    cps = get_market_data(market)['CP']
    ret = cps.shift(-period) / cps - 1
    ret[cps.values <= 0] = np.nan
    name = get_filed_name_of_future_return(period)
    ret = ret.rename(name)
    return ret


def get_future_return(market_id, period, begin_date):
    """Get future return from begining date to the latest of given market from DB.

    Parameters
    ----------
    market_id: str
        ID of market.
    period: int
        Predicting period.
    begin_date: datetime.date, optional
        The begin date of the designated data range. If it is not set, get all
        history results.

    Returns
    -------
    Pandas's Series
        A timeseries of float.

    """
    db = get_db()
    result = db.get_future_return(market_id, period, begin_date)
    return result


def save_future_return(market_id, data):
    """Save futrue return results to DB.

    Parameters
    ----------
    market_id: str
        ID of market.
    data: Pandas's DataFrame
        A panel of float which columns TF(d) for each predicting period as d.

    """
    db = get_db()
    result = db.save_future_return(market_id, data)
    return result


def save_latest_pattern_occur(data: pd.DataFrame):
    """Save pattern occured info

    Parameters
    ----------
    data: pd.DataFram
    """
    db = get_db()
    db.save_latest_pattern_occur(data)


def save_latest_pattern_distribution(data: pd.DataFrame):
    """Save pattern occured info

    Parameters
    ----------
    data: pd.DataFram
    """
    db = get_db()
    db.save_latest_pattern_distribution(data)


def get_latest_dates(model_id):
    """get dates of the latest predict results of markets for given model.

    Paratmeters
    -----------
    model_id: str
        ID of model.

    Returns
    -------
    dict from str to datetime.date
        a dict mapping from marketID to the corresponding latest predicting 
        date.

    """
    db = get_db()
    result = db.get_latest_dates(model_id)
    return result


def get_earliest_dates(model_id):
    """get dates of the earliest predict results of markets for given model.

    Paratmeters
    -----------
    model_id: str
        ID of model.

    Returns
    -------
    dict from str to datetime.date
        a dict mapping from marketID to the corresponding earliest predicting 
        date.

    """
    db = get_db()
    result = db.get_earliest_dates(model_id)
    return result


class ModelInfo():
    """Model Info.

    fileds
    ------
    model_id: Model ID
    patterns: ids of pattern set of this model.
    markets: ids of market set of this model.
    train_begin: the date, this model to be trained from.
    train_gap: the number of month, this model must be re-trained.

    """

    def __init__(self, model_id: str, patterns: List[str],
                 markets: List[str],
                 train_begin: datetime.date,
                 train_gap: int):
        self._model_id = model_id
        self._patterns = patterns
        self._markets = markets
        self._train_begin = train_begin or DEFAULT_TRAIN_BEGIN_DATE
        self._train_gap = train_gap if train_gap and train_gap > 0 else DEFAULT_TGAP

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def patterns(self) -> List[str]:
        return self._patterns

    @property
    def markets(self) -> Optional[List[str]]:
        return self._markets

    @property
    def train_begin(self) -> datetime.date:
        return self._train_begin

    @property
    def train_gap(self) -> int:
        return self._train_gap

    def get_cur_tdate(self, today: datetime.date) -> datetime.date:
        month = (today.month-1) // self.train_gap * self.train_gap + 1
        ret = datetime.date(today.year, month, 1)
        return ret

    def get_next_tdate(self, today: datetime.date) -> datetime.date:
        recv = self.get_cur_tdate(today)
        year = recv.year
        month = recv.month + self.train_gap
        if month > 12:
            month -= 12
            year += 1
        ret = datetime.date(year, month, 1)
        return ret

    def get_latest_dates(self):
        return get_latest_dates(self.model_id)

    def get_earliest_dates(self):
        return get_earliest_dates(self.model_id)


def get_models():
    """Get models from DB which complete ADD_PREDICT.

    Returns
    -------
    list of str
        IDs of models in DB

    """
    db = get_db()
    result = db.get_models()
    return result


def get_model_info(model_id):
    """Get model info from DB.

    Parameters
    ----------
    model_id: str

    Returns
    -------
    ModelInfo

    """
    db = get_db()
    result = db.get_model_info(model_id)
    return result


def del_model_data(model_id):
    """Delete model information and predicting results from DB.

    Parameters
    ----------
    model_id: str
        ID of model to delete.

    """
    db = get_db()
    result = db.del_model_data(model_id)
    return result


def save_model_results(model_id, data, also_save_latest=False):
    """Save modle predicting results to DB.

    Parameters
    ----------
    model_id: str
        ID of model.
    data: Pandas's DataFrame
        A panel of floating with columns, 'lb_p', 'ub_p', 'pr_p' for each 
        predicting period as p.

    """
    db = get_db()
    result = db.save_model_results(model_id, data, also_save_latest)
    return result


def get_market_data(market_id, begin_date=None):
    """Get market data from the designated date to the latest from DB.

    Parameters
    ----------
    market_id: str
        ID of market.
    begin_date: datetime.date, optional
        The begin date of the designated data range. If it is not set, get all
        historical data.

    Returns
    -------
    Pandas's DataFrame
        A panel of floating with columns, 'OP', 'LP', 'HP', and 'CP'.

    """
    db = get_db()
    result = db.get_market_data(market_id, begin_date)
    return result


class MarketDataFromDb:
    def get_ohlc(self, market_id: str) -> pd.DataFrame:
        return get_market_data(market_id)


def set_model_execution_start(model_id, exection):
    """Set model execution to start.

    Parameters
    ----------
    model_id: str
        Id of model.
    exection: str
        Code of model execution type.

    Returns
    -------
    str
        ID of this model execution.

    """
    db = get_db()
    result = db.set_model_execution_start(model_id, exection)
    return result


def set_model_execution_complete(exection_id):
    """Set model execution to complete.

    Parameters
    ----------
    exection_id: str
        ID of this model execution.        

    """
    db = get_db()
    db.set_model_execution_complete(exection_id)


def get_recover_model_execution():
    """ 取得模型最新執行狀態

    Parameters
    ----------
    None.

    Returns
    -------
    exec_info: list of tuple
        [(model_id, ModelExecution), ...]

    """
    db = get_db()
    result = db.get_recover_model_execution()
    return result


def set_model_status(model_id, status):
    """Set model status on DB.

    Parameters
    ----------
    model_id: str
        ID of model.
    stauts: int
        stauts code set to this model.

    """
    pass


class ThreadController:
    """多執行緒執行控制器

    在多執行緒環境下，支援外部程序控制(提前中止)其他執行緒的執行

    Properties
    ----------
    isactive: bool
        True, if this controller is switch-on; otherwise, Flase.

    Methods
    -------
    switch_off:
        Turn-off the switch of this controller.

    """

    def __init__(self):
        self._state = True

    @property
    def isactive(self) -> bool:
        return self._state

    def switch_off(self):
        self._state = False


class ModelThreadManager:
    """模型多執行緒管理器

    提供多執行緒環境下，模型操作的共享控制器管理

    Methods
    -------
    exists: 詢問指定模型是否還有未完成的操作正在執行中
    acquire: 請求指定模型的控制器
    release: 釋出指定模型的控制器    

    Notes
    -----
    執行一個模型操作前，應先請求其控制器，並於操作完成(或因故中止)後，將其釋出

    """

    def __init__(self):
        self._controllers = {}
        self._lock = Lock()

    def acquire(self, model_id: str) -> ThreadController:
        """請求指定模型的Controller

        如果該模型的controller不存在，則建立後，回傳，計數器設定為1；
        否則回傳該模型的controller，並將計數器累加1。

        Parameters
        ----------
        model_id: str
            ID of the desigated model.

        Returns
        -------
        ThreadController
            Controller of the desigated model.

        """
        self._lock.acquire()
        if model_id not in self._controllers:
            ret = ThreadController()
            self._controllers[model_id] = {'controller': ret, 'requests': 1}
        else:
            ret = self._controllers[model_id]['controller']
            self._controllers[model_id]['requests'] += 1
        self._lock.release()
        return ret

    def release(self, model_id: str):
        """釋放指定模型的Controller

        指定模型的計數器減1，若計數器歸零，則刪除該模型的controller。

        Parameters
        ----------
        model_id: str
            ID of the desigated model.

        """
        self._lock.acquire()
        if model_id not in self._controllers:
            raise RuntimeError('release an not existed controller')
        self._controllers[model_id]['requests'] -= 1
        if self._controllers[model_id]['requests'] <= 0:
            del self._controllers[model_id]
        self._lock.release()

    def exists(self, model_id: str) -> bool:
        """指定模型的Controller是否存在

        主要用途是讓外部使用者可以知道是否還有其他程序正在對該指定模型進行操作。

        Parameters
        ----------
        model_id: str
            ID of the desigated model.

        Returns
        -------
        bool
            False, if no thread operating the designated model; otherwise, True.

        """
        return model_id in self._controllers


MT_MANAGER = ModelThreadManager()


class Labelization:
    def __init__(self, n: int = 2):
        self._n = n
        self._creeds = None
        self._cuts = None
        self._cents = None
        self._lowers = None
        self._uppers = None

    @property
    def isfitted(self):
        return self._cuts is not None

    def fit(self, data, outlier=0):
        data = data.flatten()
        data.sort()
        if outlier > 0:
            outlier = int(len(data) * outlier)
            data = data[outlier: -outlier]
        pps = np.arange(1, 2 * self._n + 1) / (2 * self._n + 1)
        idxs = (pps * len(data)).astype(int)
        self._cuts = data[idxs]
        self._cents = np.array([data[:idxs[0]].mean()] +
                               [data[idxs[i]: idxs[i+1]].mean()
                                for i in range(len(idxs) - 1)] +
                               [data[idxs[-1]:].mean()])
        self._lowers = np.concatenate([[2 * data[0] - data[1]],
                                       self._cuts])
        self._uppers = np.concatenate([self._cuts, [2 * data[-1] - data[-2]]])
        zs = norm(data.mean(), data.std()).isf(1 - pps)
        self._creeds = (np.array([self._cents] * 2 * self._n).T >
                        zs).sum(axis=1) - self._n
        self._creeds[self._creeds * self._cents < 0] = 0
        for idx in range(self._n-1):
            self._creeds = np.concatenate([np.max([self._creeds[:self._n],
                                                   self._creeds[1: self._n+1]-1], axis=0),
                                           [0],
                                           np.min([self._creeds[self._n:-1] + 1,
                                                   self._creeds[-self._n:]], axis=0)])

    def fit_transform(self, data, outlier=0):
        self.fit(data, outlier)
        return self.transform(data)

    def transform(self, data):
        if not self.isfitted:
            raise RuntimeError("transform without fitting")
        data = np.broadcast_to(data.T, (2 * self._n, len(data))).T
        data = (data > self._cuts).sum(axis=1)
        return data

    def get_upperbound(self, labels):
        if not self.isfitted:
            raise RuntimeError("transform without fitting")
        return self._uppers[labels]

    def get_lowerbound(self, labels):
        if not self.isfitted:
            raise RuntimeError("transform without fitting")
        return self._lowers[labels]

    def get_center(self, labels):
        if not self.isfitted:
            raise RuntimeError("transform without fitting")
        return self._cents[labels]

    def get_creed(self, labels):
        return self._creeds[labels]

    def save(self, file):
        data = {'n': self._n,
                'cuts': self._cuts,
                'cents': self._cents,
                'lowers': self._lowers,
                'uppers': self._uppers}
        pickle_dump(data, file)

    @classmethod
    def load(cls, file):
        data = pickle_load(file)
        ret = cls(data['n'])
        ret._cuts = data['cuts']
        ret._cents = data['cents']
        ret._lowers = data['lowers']
        ret._uppers = data['uppers']
        return ret


def _get_model_dir(model_id: str) -> str:
    return f'{LOCAL_DB}/views/{model_id}'


def _get_model_file(model_id: str, tdate: datetime.date, period: int) -> str:
    return f'{_get_model_dir(model_id)}/{tdate}/{period}/model.pkl'


def _get_ycoder_file(model_id: str, market: str, tdate: datetime.date,
                     period: int) -> str:
    return f'{_get_model_dir(model_id)}/{tdate}/{period}/{market}.pkl'


def get_model(model: ModelInfo, tdate: datetime.date, period: int) -> Dtc:
    """取得指定的預測模型

        如果該模型已經存在，則直接載入並回傳之；否則，先建立該模型
        ，並儲存後，再回傳之

    Parameters
    ----------
    model: ModelInfo
        Information of the designated model.
    tdate: datetime.date
        Target-date of the designated model.
    period: int
        Predicting period of the designated model.

    Returns
    -------
    sklearn.tree.DecisionTreeClassifier

    See Also
    --------
    model_train

    """
    file = _get_model_file(model.model_id, tdate, period)
    if not os.path.exists(file):
        model_train(model, tdate)
    return pickle_load(file)


def gen_ycoder(model: ModelInfo, market: str, tdate: datetime.date, period: int):
    """建立指定的預測模型、指定市場的Y-coder

    Parameters
    ----------
    model: ModelInfo
        Information of the designated model.
    market: str
        ID of the designated market.
    tdate: datetime.date
        Target-date of the designated model.
    period: int
        Predicting period of the designated model.


    See Also
    --------
    Labelization

    """
    train_begin = model.train_begin
    train_end = tdate
    y_data = get_ydata(market, period, train_begin, train_end).dropna().values
    ycoder = Labelization()
    if len(y_data) >= MIN_Y_SAMPLES:
        ycoder.fit(y_data, outlier=Y_OUTLIER)
    ycoder.save(_get_ycoder_file(model.model_id, market, tdate, period))


def get_ycoder(model: ModelInfo, market: str, tdate: datetime.date, period: int
               ) -> Labelization:
    """取得指定的預測模型、指定市場的Y-coder

        如果該Y-coder已經存在，則直接載入並回傳之；否則，先建立該Y-coder
        ，並儲存後，再回傳之

    Parameters
    ----------
    model: ModelInfo
        Information of the designated model.
    market: str
        ID of the designated market.
    tdate: datetime.date
        Target-date of the designated model.
    period: int
        Predicting period of the designated model.

    Returns
    -------
    Labelization

    See Also
    --------
    gen_ycoder

    """
    file = _get_ycoder_file(model.model_id, market, tdate, period)
    if not os.path.exists(file):
        gen_ycoder(model, market, tdate, period)
    return Labelization.load(file)


def get_xdata(market: str, patterns: List[str],
              begin_date: Optional[datetime.date] = None,
              end_date: Optional[datetime.date] = None) -> pd.DataFrame:
    """取得指定市場在指定期間內的指定現象集數據

    Parameters
    ----------
    market: str
        ID of the designated market.
    patterns: list of str
        List of IDs of the designated pattern set.
    begin_date: datetime.date, optional
        The including begining of the designated interval. 
    end_date: datetime.date, optional
        The excluding end of the designated interval.  

    Returns
    -------
    Pandas' DataFrame
        A panel of boolean with all ID of patterns as columns.

    See Also
    --------
    get_pattern_results

    """
    ret = get_pattern_results(market, patterns, begin_date)
    if end_date:
        ret = ret[ret.index.values.astype('datetime64[D]') < end_date]
    return ret


def get_ydata(market: str, period: int,
              begin_date: datetime.date,
              end_date: datetime.date) -> pd.Series:
    """取得指定市場在指定期間內的指定天期未來報酬率

    Parameters
    ----------
    market: str
        ID of the designated market.
    period: int
        The designated period of future return.
    begin_date: datetime.date, optional
        The including begining of the designated interval. 
    end_date: datetime.date, optional
        The excluding end of the designated interval.  

    Returns
    -------
    Pandas' Series
        A time-series of floating.

    """
    ret = get_future_return(market, period, begin_date)
    # cps = get_market_data(market, begin_date)['CP']
    # ret = cps.shift(-period) / cps - 1
    ret = ret[ret.index.values.astype('datetime64[D]') < end_date]
    return ret


def model_train(model: ModelInfo, tdate: datetime.date):
    """建立(訓練)各預測天期的指定預測模型

    Parameters
    ----------
    model: ModelInfo
        Information of the designated model.
    tdate: datetime.date
        Target-date of the designated model.       

    """
    markets = model.markets if model.markets else get_markets()
    # Step1: 取得各市場的x_data
    x_data = {mid: get_xdata(mid, model.patterns,
                             begin_date=model.train_begin,
                             end_date=tdate) for mid in markets}
    for period in PREDICT_PERIODS:
        cur_x = []
        cur_y = []
        # Step2: 取得個別市場、目標天期的x_data與y_data
        # 1. 取得目標天期的未來報酬
        # 2. 與x_data對齊後去除NA
        # 3. 建立Y-coder並對未來報酬進行編碼產生y_data
        for mid in markets:
            temp = pd.concat([x_data[mid],
                              get_ydata(mid, period, model.train_begin, tdate)],
                             axis=1, sort=True).dropna()
            ycoder = Labelization()
            if len(temp) >= MIN_Y_SAMPLES:
                # 資料少於Y-coder的最小有效訓練長度的市場，不參與模型訓練
                cur_x.append(temp.values[:, :-1])
                cur_y.append(ycoder.fit_transform(
                    temp.values[:, -1], outlier=Y_OUTLIER))
            ycoder.save(_get_ycoder_file(model.model_id, mid, tdate, period))
        # Step3: 使用所有市場的x_data與y_data建立(訓練)決策樹模型
        tree = Dtc(max_depth=20, criterion='entropy', random_state=0,
                   max_leaf_nodes=200, min_samples_leaf=30)
        tree.fit(np.concatenate(cur_x, axis=0),
                 np.concatenate(cur_y, axis=0))
        pickle_dump(tree, _get_model_file(model.model_id, tdate, period))


def _model_predict(model: ModelInfo, market: str, tdate: datetime.date,
                   period: int, x_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """使用指定預測模型對目標市場進行預測

    Parameters
    ----------
    model: ModelInfo
        Information of the designated model.
    market: str
        ID of the designated market.
    tdate: datetime.date
        Target-date of the designated model.       
    period: int
        Predicting period of the designated model.
    x_data: Pandas's DataFrame
        A panel used to be the x_data of predicting model.

    Returns
    -------
    Pandas' DataFrame
        A panel consisted of time-series as follows:
        - predicting upper-bound: floating
        - predicting lower-bound: floating
        - predicting value: floating
        - market ID: str
        - predict period: int
        - model ID: str

    Notes
    -----
    基於程式碼的相依性，此呼叫函數應該提供預測區間(begin_date與end_date)，由函數
    自行取得對應的x-data，而非由呼叫者提供；但為了減少資料庫的存取，在此模組中
    採取統一由外部函數一次性取得所有欲預測的x-data，再分段使用此函數進行預測。

    See Also
    --------
    PredictResultField

    """
    # 1. 取得指定預測模型
    # 2. 取得指定預測模型、指定市場的Y-coder
    # 3. 若Y-coder is fitted，則用x-data進行預測後，再用Y-coder取得預測結果對應
    #    之預測報酬下界、預測報酬下界、預測報酬率與預測趨勢觀點
    tree = get_model(model, tdate, period)
    ycoder = get_ycoder(model, market, tdate, period)
    if ycoder.isfitted:
        y_result = tree.predict(x_data.values)
        ret = pd.DataFrame(
            {PredictResultField.DATE.value: x_data.index.values.astype('datetime64[D]'),
             PredictResultField.UPPER_BOUND.value: ycoder.get_upperbound(y_result),
             PredictResultField.LOWER_BOUND.value: ycoder.get_lowerbound(y_result),
             PredictResultField.PREDICT_VALUE.value: ycoder.get_center(y_result)})
        ret[PredictResultField.MARKET_ID.value] = market
        ret[PredictResultField.PERIOD.value] = period
        ret[PredictResultField.MODEL_ID.value] = model.model_id
        return ret


def model_predict(model: ModelInfo, market: str,
                  latest_date: Optional[datetime.date] = None,
                  max_len: Optional[int] = MIN_BACKTEST_LEN,
                  controller: Optional[ThreadController] = None
                  ) -> Optional[pd.DataFrame]:
    """使用指定預測模型對目標市場進行預測

    此函數目的為系統批次的model_update提供指定模型/指定市場的預測結果服務，主要
    工作包含：
    1. 根據latest_date與max_len參數判斷需要更新的預測結果範圍
    2. 根據1.中決定的範圍取得x_data
    3. 根據回測區間，對2.所取得的x_data分段進行預測(呼叫_model_predict)
    4. 串接3.所有預測結果，並回傳之

    Parameters
    ----------
    model: ModelInfo
        Information of the designated model.
    market: str
        ID of the designated market.
    latest_date: datetime.date, optional
        Date of the latest predicting result of the designated model. If it is
        not set, there is no predicting result of the model for given market.
    controller: ThreadController, optional
        Controllor used to control execution outside.

    Returns
    -------
    Pandas' DataFrame
        A panel consisted of time-series as follows:
        - predicting upper-bound: floating
        - predicting lower-bound: floating
        - predicting value: floating
        - market ID: str
        - predict period: int
        - model ID: str

    See Also
    --------
    PredictResultField

    """
    if latest_date:
        x_data = get_xdata(market, model.patterns,
                           latest_date + datetime.timedelta(1)).dropna()
    else:
        x_data = get_xdata(market, model.patterns).dropna()[-max_len:]
    if len(x_data) <= 0:
        return  # donothing
    ret_buffer = []
    for period in PREDICT_PERIODS:
        if controller and not controller.isactive:
            return
        dates = x_data.index.values.astype('datetime64[D]')
        next_tdate = model.get_cur_tdate(dates[0].tolist())
        eidx = 0
        result_buffer = []
        while dates[-1] >= next_tdate:
            if controller and not controller.isactive:
                return
            cur_tdate, next_tdate = next_tdate, model.get_next_tdate(
                next_tdate)
            bidx, eidx = eidx, (dates < next_tdate).sum()
            if bidx < eidx:
                result = _model_predict(model, market, cur_tdate, period,
                                        x_data=x_data[bidx: eidx])
                if result is not None:
                    result_buffer.append(result)
        if result_buffer:
            ret_buffer.append(pd.concat(result_buffer, axis=0))
    if ret_buffer:
        ret = pd.concat(ret_buffer, axis=0)
        ret.index = np.arange(len(ret))
        return ret


def _is_all_none(recv: List[Any]) -> bool:
    for each in recv:
        if each is not None:
            return False
    return True


def model_update(model_id: str, batch_controller: ThreadController):
    """更新模型預測結果

    此函數目的為更新模型的預測結果：
    1. 從DB取得指定模型資訊
    2. 從DB取得指定模型下所有市場的最新預測結果日期
    3. 呼叫model_predict計算個別市場的更新預測結果
    4. 更新DB的指定模型預測結果

    Parameters
    ----------
    model_id: str
        ID of the designated model.

    """
    controller = MT_MANAGER.acquire(model_id)
    if not controller.isactive:
        MT_MANAGER.release(model_id)
        return
    if not batch_controller.isactive:
        MT_MANAGER.release(model_id)
        return
    exection_id = set_model_execution_start(
        model_id, ModelExecution.BATCH_PREDICT)
    model = get_model_info(model_id)
    latest_dates = model.get_latest_dates()
    markets = model.markets if model.markets else get_markets()
    ret_buffer = []
    for mid in markets:
        if not controller.isactive:
            MT_MANAGER.release(model_id)
            return
        if not batch_controller.isactive:
            MT_MANAGER.release(model_id)
            return
        if mid in latest_dates:
            ret_buffer.append(model_predict(model, mid, latest_dates[mid],
                                            controller=controller))
        else:
            ret_buffer.append(model_predict(model, mid, controller=controller))
    if ret_buffer and not _is_all_none(ret_buffer):
        ret = pd.concat(ret_buffer, axis=0)
        ret.index = np.arange(len(ret))
        if controller.isactive:
            save_model_results(model.model_id, ret, True)
    if controller.isactive:
        set_model_execution_complete(exection_id)
    MT_MANAGER.release(model_id)


def add_model(model_id: str):
    """新增模型

    此函數目的為新增模型：
    1. 呼叫model_create建立模型，並計算各市場、各天期的最近一日的預測結果
    2. 呼叫model_backtest計算各市場、各天期的歷史回測結果

    Parameters
    ----------
    model_id: str
        ID of the designated model.

    See Also
    --------
    model_create, model_backtest

    """

    controller = MT_MANAGER.acquire(model_id)
    if not controller.isactive:
        MT_MANAGER.release(model_id)
        return
    model = get_model_info(model_id)
    model_create(model, controller)
    model_backtest(model, controller)
    MT_MANAGER.release(model_id)


def model_recover(model_id: str, status: ModelStatus):
    """重啟模型

    此函數目的為重啟因故中斷的模型：
    如果模型狀態為 ADDED 則
    1. 呼叫model_create建立模型，並計算各市場、各天期的最近一日的預測結果
    2. 呼叫model_backtest計算各市場、各天期的歷史回測結果
    如果模型狀態為 CREATED 則跳過1. 直接執行 2.

    Parameters
    ----------
    model_id: str
        ID of the designated model.
    status: ModelStatus
        Status of Model.

    See Also
    --------
    model_create, model_backtest, ModelStatus

    """
    controller = MT_MANAGER.acquire(model_id)
    if not controller.isactive:
        MT_MANAGER.release(model_id)
        return
    model = get_model_info(model_id)
    if status < ModelStatus.CREATED:
        model_create(model, controller)
    model_backtest(model, controller)
    MT_MANAGER.release(model_id)


def remove_model(model_id):
    """移除模型

    此函數目的為移除指定模型：
    1. 取得指定模型控制器，以中斷所有該模型正在執行中的工作
    2. 等待所有該模型正在執行中的工作中斷
    3. 呼叫del_model_data，於DB中移除該模型相關資料
    4. 移除本地端與該模型相關的資料

    Parameters
    ----------
    model_id: str
        ID of the designated model.

    See Also
    --------
    model_create, model_backtest

    """
    MT_MANAGER.acquire(model_id).switch_off()
    MT_MANAGER.release(model_id)
    while MT_MANAGER.exists(model_id):
        time.sleep(1)
    # del_model_data(model_id)
    model_dir = _get_model_dir(model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)


def model_create(model: ModelInfo, controller: ThreadController):
    """建立模型

        建立模型的工作流程如下：
        1. 使用指定模型產生各目標市場、各預測天期最新的預測結果
        2. 上傳1. 所產生的預測結果至DB
        3. 更新DB上指定模型的狀態為"CREATED"

    Paramaters
    ----------
    model: ModelInfo
        Information of the designated model.
    controller: ThreadController
        Controllor used to control execution outside.

    See Also
    --------
    ModelExecution, model_predict

    """
    if not controller.isactive:
        return
    exection_id = set_model_execution_start(
        model.model_id, ModelExecution.ADD_PREDICT)
    markets = model.markets if model.markets else get_markets()
    ret_buffer = []
    for mid in markets:
        if not controller.isactive:
            return
        ret_buffer.append(model_predict(
            model, mid, max_len=1, controller=controller))
    if ret_buffer:
        ret = pd.concat(ret_buffer, axis=0)
        ret.index = np.arange(len(ret))
        if controller.isactive:
            save_model_results(model.model_id, ret, True)
    if controller.isactive:
        set_model_execution_complete(exection_id)


def model_backtest(model: ModelInfo, controller: ThreadController):
    """執行模型回測

        執行模型回測的工作流程如下：
        1. 取得指定模型的各市場在DB中的第一筆預測結果日期
        2. 取得各市場的的回測資料集
        -> 從第一筆預測結果向前取到目標筆數(若市場資料長度允許)
        3. (根據重訓練區間)分段對各市場各天期向前進行回測，於每次分段回測完成時，
           上傳該分段全市場、全天期的所有回測結果至DB
        4. 更新DB上指定模型的狀態為"COMPLETE"

    Paramaters
    ----------
    model: ModelInfo
        Information of the designated model.
    controller: ThreadController
        Controllor used to control execution outside.

    See Also
    --------
    ModelExecution, model_predict

    """
    if not controller.isactive:
        return
    exection_id = set_model_execution_start(
        model.model_id, ModelExecution.ADD_BACKTEST)
    earliest_dates = model.get_earliest_dates()
    # 取得所有市場的目標回測資料
    x_data = {mid: get_xdata(mid, model.patterns, end_date=edate
                             ).dropna()[-MIN_BACKTEST_LEN:]
              for mid, edate in earliest_dates.items()}
    tdate = max(earliest_dates.values())
    while x_data:
        ret_buffer = []
        # 在每輪迴圈開始時，tdate是上輪迴圈模型可預測的最小日期
        # 故 -1 後，即為本輪迴圈模型可預測的最大日期，
        # 送入 get_cur_tdate 後可以取得此輪迴圈模型的Target-date(可預測的最小日期)
        tdate = model.get_cur_tdate(tdate - datetime.timedelta(1))
        for mid in list(x_data.keys()):
            if not controller.isactive:
                return
            if len(x_data[mid]) <= 0:
                # 該市場回測工作已完成
                del x_data[mid]
                continue
            # 取得該市場此輪的回測長度
            blen = (x_data[mid].index.values.astype(
                'datetime64[D]') >= tdate).sum()
            if blen <= 0:
                continue
            # 擷取該市場此輪的回測資料
            cur_x = x_data[mid][-blen:]
            result_buffer = []
            for period in PREDICT_PERIODS:
                if not controller.isactive:
                    return
                result = _model_predict(model, mid, tdate, period, cur_x)
                if result is not None:
                    result_buffer.append(result)
            # 將該市場此輪的回測資料自回測資料中移除
            x_data[mid] = x_data[mid][:-blen]
            if result_buffer:
                ret_buffer.append(pd.concat(result_buffer, axis=0))
        if ret_buffer:
            ret = pd.concat(ret_buffer, axis=0)
            ret.index = np.arange(len(ret))
            if not controller.isactive:
                return
            # 將此輪所有市場的回測結果上傳至DB
            save_model_results(model.model_id, ret)
    if controller.isactive:
        # 回測完成，更新指定模型在DB上的狀態為'COMPLETE'
        set_model_execution_complete(exection_id)


def pattern_update(controller: ThreadController):
    patterns = get_patterns()
    markets = get_markets()
    ret_buffer = []
    ret_market_dist = []
    ret_market_occur = []
    for market in markets:
        result_buffer = []
        for pattern in patterns:
            if not controller.isactive:
                return
            result_buffer.append(pattern.run(market).rename(pattern.pid))
        pattern_result = fast_concat(result_buffer)
        if not controller.isactive:
            return
        save_pattern_results(market, pattern_result)
        cur_ret = pd.DataFrame()
        cur_ret[PatternResultField.VALUE.value] = pattern_result.values[-1]
        cur_ret[PatternResultField.DATE.value] = pattern_result.index.values.astype(
            'datetime64[D]').tolist()[-1]
        cur_ret[PatternResultField.MARKET_ID.value] = market
        cur_ret[PatternResultField.PATTERN_ID.value] = list(
            pattern_result.columns)
        ret_buffer.append(cur_ret)
        result_buffer = []
        for period in PREDICT_PERIODS:
            if not controller.isactive:
                return
            result_buffer.append(gen_future_return(market, period))
        return_result = fast_concat(result_buffer)
        save_future_return(market, return_result)

        market_occur, market_dist = get_pattern_stats_info(
            pattern_result, return_result, market)
        ret_market_dist.append(market_dist)
        ret_market_occur.append(market_occur)
    ret = pd.concat(ret_buffer, axis=0)
    ret_market_dist = pd.concat(ret_market_dist, axis=0)
    ret_market_occur = pd.concat(ret_market_occur, axis=0)
    if not controller.isactive:
        return
    pickle_dump(ret, './latest_pattern_results.pkl')
    save_latest_pattern_results(ret)
    if not controller.isactive:
        return
    pickle_dump(ret_market_occur, './latest_pattern_occur.pkl')
    save_latest_pattern_occur(ret_market_occur)
    if not controller.isactive:
        return
    pickle_dump(ret_market_dist, './latest_pattern_distribution.pkl')
    save_latest_pattern_distribution(ret_market_dist)
    return ret


def get_pattern_stats_info(pattern, freturn, market):

    market_occur = {MarketOccurField.OCCUR_CNT.value: [],
                    MarketOccurField.NON_OCCUR_CNT.value: [],
                    MarketOccurField.MARKET_RISE_CNT.value: [],
                    MarketOccurField.MARKET_FLAT_CNT.value: [],
                    MarketOccurField.MARKET_FALL_CNT.value: [],
                    MarketOccurField.MARKET_ID.value: [],
                    MarketOccurField.PATTERN_ID.value: [],
                    MarketOccurField.DATE_PERIOD.value: []}
    market_dist = {MarketDistField.RETURN_MEAN.value: [],
                   MarketDistField.RETURN_STD.value: [],
                   MarketDistField.MARKET_ID.value: [],
                   MarketDistField.PATTERN_ID.value: [],
                   MarketDistField.DATE_PERIOD.value: []}

    for pname in pattern:

        for each_r in freturn:
            fret = freturn[each_r][(
                (~pattern[pname].isnull()) & (~freturn[each_r].isnull()))]
            ptn = pattern[pname][((~pattern[pname].isnull())
                                  & (~freturn[each_r].isnull()))]
            market_occur[MarketOccurField.OCCUR_CNT.value].append(
                np.sum(ptn.values == 1))
            market_occur[MarketOccurField.NON_OCCUR_CNT.value].append(
                np.sum(ptn.values == 0))
            market_occur[MarketOccurField.MARKET_RISE_CNT.value].append(np.sum(
                (ptn.values == 1) & (fret.values > 0)))
            market_occur[MarketOccurField.MARKET_FLAT_CNT.value].append(np.sum(
                (ptn.values == 1) & (fret.values == 0)))
            market_occur[MarketOccurField.MARKET_FALL_CNT.value].append(np.sum(
                (ptn.values == 1) & (fret.values < 0)))
            market_occur[MarketOccurField.MARKET_ID.value].append(market)
            market_occur[MarketOccurField.PATTERN_ID.value].append(pname)
            market_occur[MarketOccurField.DATE_PERIOD.value].append(
                each_r.replace('FR', ''))
            ret_occur = fret[ptn.values == 1].values
            if len(ret_occur) != 0:
                market_dist[MarketDistField.RETURN_STD.value].append(
                    np.std(ret_occur))
                market_dist[MarketDistField.RETURN_MEAN.value].append(
                np.mean(ret_occur))
            else:
                market_dist[MarketDistField.RETURN_STD.value].append(np.nan)
                market_dist[MarketDistField.RETURN_MEAN.value].append(np.nan)
            market_dist[MarketDistField.MARKET_ID.value].append(market)
            market_dist[MarketDistField.PATTERN_ID.value].append(pname)
            market_dist[MarketDistField.DATE_PERIOD.value].append(
                each_r.replace('FR', ''))

    return pd.DataFrame(market_occur), pd.DataFrame(market_dist)


def model_execution_recover():
    for model, etype in get_recover_model_execution():
        if etype == ModelExecution.ADD_PREDICT:
            model_recover(model, ModelStatus.ADDED)
        if etype == ModelExecution.ADD_BACKTEST:
            model_recover(model, ModelStatus.CREATED)


def init_db():
    controller = MT_MANAGER.acquire("init")
    pattern_update(controller)
    model_execution_recover()


def batch(excute_id, logger):
    logging.info("batch start")
    controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
    try:
        if controller.isactive:
            logger.info(f"api_batch {excute_id} start")
            clean_db_cache()
            clone_db_cache()
            pattern_update(controller)
            for model in get_models():
                model_update(model, controller)
            logger.info(f"api_batch {excute_id} complete")
            MT_MANAGER.release(BATCH_EXE_CODE)

    except Exception as esp:
        logging.info("batch failed")
        MT_MANAGER.release(BATCH_EXE_CODE)
        raise esp
    logging.info("batch finished")
