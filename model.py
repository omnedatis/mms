# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

"""

from abc import ABCMeta, abstractmethod, abstractproperty
import datetime
import os
import shutil
from threading import Lock
import time
import threading as mt
import multiprocessing as mp
from typing import Any, List, Optional, NamedTuple, Dict, Callable
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm
from sklearn.tree import DecisionTreeClassifier as Dtc
import traceback
from const import *
from utils import *
from func._tp import *
import logging
from logging import handlers
from func._td import MD_CACHE

PATTERN_UPDATE_CPUS = 7

smd_lock = Lock()
SMD_IDX_FILE = f'{LOCAL_DB}/smd_idx.pkl'
smd_idx = pickle_load(SMD_IDX_FILE) if os.path.exists(SMD_IDX_FILE) else 1

exec_mode = ExecMode.DEV.value

def get_exec_mode():
    return exec_mode

def set_exec_mode(mode):
    global exec_mode
    exec_mode = mode

def swap_smd_index():
    global smd_idx
    smd_lock.acquire()
    smd_idx = (smd_idx + 1) % 2
    pickle_dump(smd_idx, SMD_IDX_FILE)
    smd_lock.release()

class SMD:
    def __init__(self, mids: List[str]=[], pids: List[str]=[], mdates: List[np.ndarray]=[],
                       pvalues: List[np.ndarray]=[], freturns: List[np.ndarray]=[]):
        self._mids = {mid: idx for idx, mid in enumerate(mids)}
        self._pids = {pid: idx for idx, pid in enumerate(pids)}
        self._mdates = mdates
        self._pvalues = pvalues
        self._freturns = freturns

    @staticmethod
    def pencode(values: np.ndarray) -> np.ndarray:
        ret = values.astype('int8')
        ret[np.isnan(values)] = -1
        return ret

    @staticmethod
    def rencode(values: np.ndarray) -> np.ndarray:
        return values.astype('float32')

    @staticmethod
    def dencode(values: np.ndarray) -> np.ndarray:
        return values.astype('datetime64[D]').astype('int32')

    @staticmethod
    def pdecode(values: np.ndarray) -> np.ndarray:
        ret = values.astype(float)
        ret[ret < 0] = np.nan
        return ret

    @staticmethod
    def rdecode(values: np.ndarray) -> np.ndarray:
        return values.astype(float)

    @staticmethod
    def ddecode(values: np.ndarray) -> np.ndarray:
        return values.astype('datetime64[D]')

    def copy_from(self, other: 'SMD'):
        self._mids = other._mids
        self._pids = other._pids
        self._mdates = other._mdates
        self._pvalues = other._pvalues
        self._freturns = other._freturns

    def clear(self):
        self._mids = {}
        self._pids = {}
        self._mdates = []
        self._pvalues = []
        self._freturns = []

    def update(self, mids: Optional[List[str]]=None,
               pids: Optional[List[str]]=None,
               mdates: Optional[List[np.ndarray]]=None,
               pvalues: Optional[List[np.ndarray]]=None,
               freturns: Optional[List[np.ndarray]]=None):
        if mids:
            self._mids = {mid: idx for idx, mid in enumerate(mids)}
        if pids:
            self._pids = {pid: idx for idx, pid in enumerate(pids)}
        if mdates:
            self._mdates = mdates
        if pvalues:
            self._pvalues = pvalues
        if freturns:
            self._freturns = freturns

    def dump(self):
        smd_lock.acquire()
        idx = (smd_idx + 1) % 2
        pickle_dump({'mids': self._mids, 'pids': self._pids,
                     'mdates': self._mdates, 'pvalues': self._pvalues,
                     'freturns': self._freturns}, f'{LOCAL_DB}/smd_{idx}.pkl')
        smd_lock.release()

    @classmethod
    def load(cls):
        ret = cls()
        smd_lock.acquire()
        file = f'{LOCAL_DB}/smd_{smd_idx}.pkl'
        if os.path.exists(file):
            recv = pickle_load(file)
            ret._mids = recv['mids']
            ret._pids = recv['pids']
            ret._mdates = recv['mdates']
            ret._pvalues = recv['pvalues']
            ret._freturns = recv['freturns']
        smd_lock.release()
        return ret

    @property
    def raw_data(self):
        # Without copy, be careful!!
        return self._mids, self._pids, self._mdates, self._pvalues, self._freturns

    def isempty(self):
        return len(self._mids) == 0

db = None

batch_lock = Lock()
smd = SMD()
smd_swap = SMD()

def load_smd():
    global smd
    smd = SMD.load()

def get_db():
    return db


def set_db(_db):
    global db
    db = _db

def swap_smd():
    smd_lock.acquire()
    smd.copy_from(smd_swap)
    smd_swap.clear()
    smd_lock.release()

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

def save_mkt_score(data: pd.DataFrame):
    """save mkt score to DB."""
    db = get_db()
    db.save_mkt_score(data)

def save_latest_mkt_period(data: pd.DataFrame):
    """save mkt period to DB."""
    db = get_db()
    db.save_latest_mkt_period(data)

def save_model_hit_sum(data: pd.DataFrame):
    """save mkt score to DB."""
    db = get_db()
    db.save_model_hit_sum(data)

def update_model_hit_sum(data: pd.DataFrame):
    """save mkt score to DB."""
    db = get_db()
    db.update_model_hit_sum(data)

def get_model_results(model_id: str):
    """save mkt period to DB."""
    db = get_db()
    return db.get_model_results(model_id)

def save_mkt_period(data: pd.DataFrame):
    """save mkt period to DB."""
    db = get_db()
    db.save_mkt_period(data)

def save_mkt_dist(data: pd.DataFrame):
    """save mkt dist to DB."""
    db = get_db()
    db.save_mkt_dist(data)

def get_score_meta_info():
    """get score meta info from DB."""
    db = get_db()
    result = db.get_score_meta_info()
    return result

def get_markets(market_type=None, category_code=None):
    """get market IDs from DB.

    Returns
    -------
    list of str

    """
    db = get_db()
    result = db.get_markets(market_type, category_code)
    return result

def truncate_swap_tables():
    db = get_db()
    db.truncate_swap_tables()

def get_pattern_info(pid) -> PatternInfo:
    """get patterns from DB.

    Returns
    -------
    list of PatternInfo

    """
    db = get_db()
    result = db.get_pattern_info(pid)
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
    db.save_latest_pattern_results(data)

def update_latest_pattern_results(pid: str, data: pd.DataFrame):
    """Save latest pattern results to DB.

    Parameters
    ----------
    pattern_id: `str`
        要更新的現象 ID
    data: Pandas's DataFrame
        A table of pattern results with columns for market_id, pattern_id,
        price_date and value.

    See Also
    --------
    PatternResultField

    """
    db = get_db()
    db.update_latest_pattern_results(pid, data)

def clean_db_cache():
    """Clean local cache."""
    db = get_db()
    db.clean_db_cache()

def clone_db_cache(batch_type):
    """Build local cache from db."""
    db = get_db()
    db.clone_db_cache(batch_type)

def clone_model_results(controller):
    db = get_db()
    return db.clone_model_results(controller)

def get_pattern_results(market_id, patterns, begin_date=None, batch_type:BatchType=None):
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
    smd_lock.acquire()
    if batch_type == BatchType.INIT_BATCH:
        mids, pids, mdates, pvalues, _ = smd.raw_data
    elif batch_type == BatchType.SERVICE_BATCH:
        mids, pids, mdates, pvalues, _ = smd_swap.raw_data
    smd_lock.release()

    midx = mids[market_id]
    values = pvalues[midx]
    dates = mdates[midx]
    if patterns:
        columns = patterns
        pidxs = [pids[pid] for pid in patterns]
        values = values[:, pidxs]
    else:
        columns = list(pids.keys())
    values = SMD.pdecode(values)
    dates = SMD.ddecode(dates)
    ret = pd.DataFrame(values, index=dates, columns=columns)
    if begin_date:
        ret = ret[ret.index.values.astype('datetime64[D]') >= begin_date]
    return ret

def get_filed_name_of_future_return(period: int) -> str:
    return f'FR{period}'


def gen_future_return(market: str, period: int):
    cps = get_market_data(market)['CP']
    ret = cps.shift(-period) / cps - 1
    ret[cps.values <= 0] = np.nan
    name = get_filed_name_of_future_return(period)
    ret = ret.rename(name)
    return ret


def get_future_return(market_id, period, begin_date, batch_type:BatchType = None):
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
    smd_lock.acquire()
    if batch_type == BatchType.INIT_BATCH:
        mids, _, mdates, _, freturns = smd.raw_data
    elif batch_type == BatchType.SERVICE_BATCH:
        mids, _, mdates, _, freturns = smd_swap.raw_data
    smd_lock.release()

    midx = mids[market_id]
    pidx = PREDICT_PERIODS.index(period)
    values = SMD.rdecode(freturns[midx][:, pidx])
    dates = SMD.ddecode(mdates[midx])
    ret = pd.Series(values, index=dates, name=market_id)
    if begin_date:
        ret = ret[ret.index.values.astype('datetime64[D]') >= begin_date]
    return ret

def get_future_returns(market_id, begin_date=None):
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
    smd_lock.acquire()
    mids, _, mdates, _, freturns = smd.raw_data
    smd_lock.release()

    midx = mids[market_id]
    columns = [f'FR{period}' for period in PREDICT_PERIODS]
    values = SMD.rdecode(freturns[midx])
    dates = SMD.ddecode(mdates[midx])
    ret = pd.DataFrame(values, index=dates, columns=columns)
    if begin_date:
        ret = ret[ret.index.values.astype('datetime64[D]') >= begin_date]
    return ret

def save_latest_pattern_occur(data: pd.DataFrame):
    """Save pattern occured info

    Parameters
    ----------
    data: pd.DataFram
    """
    db = get_db()
    db.save_latest_pattern_occur(data)

def update_latest_pattern_occur(pid: str, data: pd.DataFrame):
    """Save pattern occured info

    Parameters
    ----------
    pattern_id: `str`
        要更新的現象 ID
    data: pd.DataFram
    """
    db = get_db()
    db.update_latest_pattern_occur(pid, data)

def save_latest_pattern_distribution(data: pd.DataFrame):
    """Save pattern occured info

    Parameters
    ----------
    data: pd.DataFram
    """
    db = get_db()
    db.save_latest_pattern_distribution(data)

def update_latest_pattern_distribution(pid: str, data: pd.DataFrame):
    """Save pattern occured info

    Parameters
    ----------
    pattern_id: `str`
        要更新的現象 ID
    data: pd.DataFram
    """
    db = get_db()
    db.update_latest_pattern_distribution(pid, data)

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


def checkout_fcst_data():
    """Save pattern occured info

    Parameters
    ----------
    None.
    """
    db = get_db()
    db.checkout_fcst_data()

def update_model_accuracy():
    """Save pattern occured info

    Parameters
    ----------
    None.
    """
    db = get_db()
    db.update_model_accuracy()

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


def save_model_results(model_id, data, exec_type: ModelExecution):
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
    result = db.save_model_results(model_id, data, exec_type)
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


def get_model(model: ModelInfo, tdate: datetime.date, period: int, batch_type:BatchType = None) -> Dtc:
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
        model_train(model, tdate, batch_type=batch_type)
    return pickle_load(file)


def gen_ycoder(model: ModelInfo, market: str, tdate: datetime.date, period: int,
               batch_type:BatchType = None):
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
    y_data = get_ydata(market, period, train_begin, train_end, batch_type=batch_type).dropna().values
    ycoder = Labelization()
    if len(y_data) >= MIN_Y_SAMPLES:
        ycoder.fit(y_data, outlier=Y_OUTLIER)
    ycoder.save(_get_ycoder_file(model.model_id, market, tdate, period))


def get_ycoder(model: ModelInfo, market: str, tdate: datetime.date, period: int,
               batch_type:BatchType = None) -> Labelization:
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
        gen_ycoder(model, market, tdate, period, batch_type=batch_type)
    return Labelization.load(file)


def get_xdata(market: str, patterns: List[str],
              begin_date: Optional[datetime.date] = None,
              end_date: Optional[datetime.date] = None,
              batch_type:BatchType = None) -> pd.DataFrame:
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
    ret = get_pattern_results(market, patterns, begin_date, batch_type=batch_type)
    if end_date:
        ret = ret[ret.index.values.astype('datetime64[D]') < end_date]
    return ret


def get_ydata(market: str, period: int,
              begin_date: datetime.date,
              end_date: datetime.date,
              batch_type:BatchType = None) -> pd.Series:
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
    ret = get_future_return(market, period, begin_date, batch_type=batch_type)
    # cps = get_market_data(market, begin_date)['CP']
    # ret = cps.shift(-period) / cps - 1
    ret = ret[ret.index.values.astype('datetime64[D]') < end_date]
    return ret


def model_train(model: ModelInfo, tdate: datetime.date, batch_type:BatchType = None):
    """建立(訓練)各預測天期的指定預測模型

    Parameters
    ----------
    model: ModelInfo
        Information of the designated model.
    tdate: datetime.date
        Target-date of the designated model.

    """
    markets = model.markets if model.markets else get_markets()
    if len(markets) <= 0:
        pickle_dump(None, _get_model_file(model.model_id, tdate, period))
        return
    # Step1: 取得各市場的x_data
    x_data = {mid: get_xdata(mid, model.patterns,
                             begin_date=model.train_begin,
                             end_date=tdate, batch_type=batch_type) for mid in markets}
    for period in PREDICT_PERIODS:
        cur_x = []
        cur_y = []
        # Step2: 取得個別市場、目標天期的x_data與y_data
        # 1. 取得目標天期的未來報酬
        # 2. 與x_data對齊後去除NA
        # 3. 建立Y-coder並對未來報酬進行編碼產生y_data
        for mid in markets:
            temp = pd.concat([x_data[mid],
                              get_ydata(mid, period, model.train_begin, tdate, batch_type=batch_type)],
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
                   period: int, x_data: pd.DataFrame, batch_type:BatchType = None) -> Optional[pd.DataFrame]:
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
    tree = get_model(model, tdate, period, batch_type=batch_type)
    if tree is None:
        return None
    ycoder = get_ycoder(model, market, tdate, period, batch_type=batch_type)
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
                  controller: Optional[ThreadController] = None,
                  batch_type:BatchType = None) -> Optional[pd.DataFrame]:
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
                           latest_date + datetime.timedelta(1), batch_type=batch_type).dropna()
    else:
        x_data = get_xdata(market, model.patterns, batch_type=batch_type).dropna()[-max_len:]
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
                                        x_data=x_data[bidx: eidx], batch_type=batch_type)
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


def get_hit_sum(data, market_id, batch_type):
    if data is not None and len(data) > 0:
        hits = []
        cnts = []
        for period in PREDICT_PERIODS:
            cur = data[[PredictResultField.DATE.value,
                       PredictResultField.UPPER_BOUND.value,
                       PredictResultField.LOWER_BOUND.value]][data[PredictResultField.PERIOD.value].values == period
                       ].set_index(PredictResultField.DATE.value).sort_index()
            if len(cur) <= 0:
                cnts.append(0)
                hits.append(0)
                continue
            frs = get_future_return(market_id, period, cur.index.values[0].astype('datetime64[D]').tolist(), batch_type) * 100
            values = pd.concat([cur, frs], axis=1, sort=True).dropna().values
            if len(values) > 0:
                cnts.append(len(values))
                hits.append(((values[:,0] >= values[:,-1]) & (values[:,1] <= values[:,-1])).sum())
            else:
                cnts.append(0)
                hits.append(0)
    else:
        hits = [0 for p in PREDICT_PERIODS]
        cnts = [0 for p in PREDICT_PERIODS]
    ret = pd.DataFrame()
    ret[ModelMarketHitSumField.HIT.value] = hits
    ret[ModelMarketHitSumField.FCST_CNT.value] = cnts
    ret[ModelMarketHitSumField.DATE_PERIOD.value] = PREDICT_PERIODS
    ret[ModelMarketHitSumField.MARKET_CODE.value] = market_id
    return ret

def model_update(model_id: str, batch_controller: ThreadController, batch_type:BatchType):
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
        logging.info('model update terminated')
        return
    if not batch_controller.isactive:
        MT_MANAGER.release(model_id)
        logging.info('batch terminated')
        return
    exection_id = set_model_execution_start(
        model_id, ModelExecution.BATCH_PREDICT)
    model = get_model_info(model_id)
    logging.info(f'start model update on {model_id}')
    latest_dates = model.get_latest_dates()
    old_results = get_model_results(model_id)
    markets = model.markets if model.markets else get_markets()
    ret_buffer = []
    for mid in markets:
        if not controller.isactive:
            MT_MANAGER.release(model_id)
            logging.info('model update terminated')
            return
        if not batch_controller.isactive:
            MT_MANAGER.release(model_id)
            logging.info('batch terminated')
            return
        if mid in latest_dates:
            cur = model_predict(model, mid, latest_dates[mid], controller=controller, batch_type=batch_type)
        else:
            cur = model_predict(model, mid, controller=controller, batch_type=batch_type)
        ret_buffer.append(cur)
    def save_result(data, model_id, exection_id, controller):
        markets = model.markets if model.markets else get_markets()
        if data and not _is_all_none(data):
            ret = pd.concat(data, axis=0)
            ret.index = np.arange(len(ret))
            logging.info(f'finish model update on {model_id}')
            if controller.isactive:
                save_model_results(model_id, ret, ModelExecution.BATCH_PREDICT)
        if controller.isactive:
            set_model_execution_complete(exection_id)
        if controller.isactive:
            model_results = get_model_results(model_id)
            ret = [get_hit_sum(model_results[mid], mid, BatchType.SERVICE_BATCH) for mid in markets]
            if ret:
                ret = pd.concat(ret, axis=0)
                ret.index = np.arange(len(ret))
                ret['MODEL_ID'] = model_id
                save_model_hit_sum(ret)
        MT_MANAGER.release(model_id)
    t = CatchableTread(target=save_result, args=(ret_buffer, model_id, exection_id, controller))
    t.start()
    return t

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
    try:
        model = get_model_info(model_id)
        model_create(model, controller)
        model_backtest(model, controller)
        markets = model.markets if model.markets else get_markets()
        model_results = get_model_results(model_id)
        ret = [get_hit_sum(model_results[mid], mid, BatchType.INIT_BATCH) for mid in markets]
        if ret:
            ret = pd.concat(ret, axis=0)
            ret.index = np.arange(len(ret))
            ret['MODEL_ID'] = model_id
            update_model_hit_sum(ret)
        MT_MANAGER.release(model_id)

    except Exception as esp:
        MT_MANAGER.release(model_id)

        logging.error(f"add model failed ")
        logging.error(traceback.format_exc())
        return


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
    try:
        model = get_model_info(model_id)
        if status < ModelStatus.CREATED:
            model_create(model, controller)
        model_backtest(model, controller)
        markets = model.markets if model.markets else get_markets()
        model_results = get_model_results(model_id)
        ret = [get_hit_sum(model_results[mid], mid, BatchType.INIT_BATCH) for mid in markets]
        if ret:
            ret = pd.concat(ret, axis=0)
            ret.index = np.arange(len(ret))
            ret['MODEL_ID'] = model_id
            update_model_hit_sum(ret)
        MT_MANAGER.release(model_id)

    except Exception as esp:
        logging.error(f"recover model failed")
        logging.error(traceback.format_exc())
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
    logging.info('start remove model')
    MT_MANAGER.acquire(model_id).switch_off()
    MT_MANAGER.release(model_id)
    while MT_MANAGER.exists(model_id):
        time.sleep(1)
    # del_model_data(model_id)
    model_dir = _get_model_dir(model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    logging.info('finish remove model')


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
    batch_type = BatchType.INIT_BATCH
    logging.info('start model create')
    if not controller.isactive:
        logging.info('model create terminated')
        return
    exection_id = set_model_execution_start(
        model.model_id, ModelExecution.ADD_PREDICT)
    markets = model.markets if model.markets else get_markets()
    ret_buffer = []
    for mid in markets:
        if not controller.isactive:
            logging.info('model create terminated')
            return
        ret_buffer.append(model_predict(
            model, mid, max_len=1, controller=controller, batch_type=batch_type))
    if ret_buffer:
        ret = pd.concat(ret_buffer, axis=0)
        ret.index = np.arange(len(ret))
        logging.info('finish model create')
        if controller.isactive:
            save_model_results(model.model_id, ret, ModelExecution.ADD_PREDICT)
    if controller.isactive:
        set_model_execution_complete(exection_id)
    if ret_buffer:
        return ret


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
    batch_type = BatchType.INIT_BATCH
    logging.info('start model backtest')
    if not controller.isactive:
        logging.info('model backtest terminated')
        return
    exection_id = set_model_execution_start(
        model.model_id, ModelExecution.ADD_BACKTEST)
    markets = model.markets if model.markets else get_markets()
    if len(markets) > 0:
        earliest_dates = model.get_earliest_dates()
        # 取得所有市場的目標回測資料
        x_data = {mid: get_xdata(mid, model.patterns,
                                 end_date=earliest_dates[mid] if mid in earliest_dates else None
                                 ,batch_type=batch_type).dropna()[-MIN_BACKTEST_LEN:]
                  for mid in markets}
        tdate = max(earliest_dates.values()) if earliest_dates else datetime.date.today()
        all_results = []
        while x_data:
            ret_buffer = []
            # 在每輪迴圈開始時，tdate是上輪迴圈模型可預測的最小日期
            # 故 -1 後，即為本輪迴圈模型可預測的最大日期，
            # 送入 get_cur_tdate 後可以取得此輪迴圈模型的Target-date(可預測的最小日期)
            tdate = model.get_cur_tdate(tdate - datetime.timedelta(1))
            for mid in list(x_data.keys()):
                if not controller.isactive:
                    logging.info('model backtest terminated')
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
                        logging.info('model backtest terminated')
                        return
                    result = _model_predict(model, mid, tdate, period, cur_x, batch_type=batch_type)
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
                    logging.info('model backtest terminated')
                    return
                # 將此輪所有市場的回測結果上傳至DB
                logging.info('finish model backtest')
                save_model_results(model.model_id, ret, ModelExecution.ADD_BACKTEST)
                all_results.append(ret)
    if controller.isactive:
        # 回測完成，更新指定模型在DB上的狀態為'COMPLETE'
        set_model_execution_complete(exection_id)
    if all_results:
        return pd.concat(all_results, axis=0)

class CatchableTread:

    def __init__(self, target, args=None, name=None):
        self._args = args
        self._target = target
        self.esp = None
        self._thread = mt.Thread(name=name, target=self._run)

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            if self._args is not None:
                self._target(*self._args)
            else:
                self._target()
        except Exception as esp:
            logging.error(traceback.format_exc())
            self.esp = traceback.format_exc()

    def join(self):
        self._thread.join()
class ExecQueue:

    def __init__(self, limit, name):
        self._limit = limit
        self._occupants = 0
        self._queue = []
        self.isactive = True
        self.threads = []
        self._lock = Lock()
        self._thread = CatchableTread(self._run, name=name)

    def _run(self):
        while self.isactive:
            # logging.debug(self._occupants)
            if self._queue:
                func, args, size = self._pop(0)
                self._occupants += size
                if self._occupants <= self._limit:
                    def callback():
                        cur_size = size
                        logging.info(f'Do {func.__name__}')
                        if args is not None:
                            ret = func(*args)
                        else:
                            ret = func()
                        self._occupants -= cur_size
                        return ret
                    t = CatchableTread(target=callback)
                    t.start()
                    self.threads.append(t)
                else:
                    self.cut_line(func, args=args, size=size)
                    self._occupants -= size
            time.sleep(1)

    def start(self):
        self._thread.start()

    def _pop(self, index):
        self._lock.acquire()
        item = self._queue.pop(index)
        self._lock.release()
        return item

    def push(self, func:Callable, *, size:int, args:tuple=None):
        self._lock.acquire()
        assert size <= self._limit, 'exceed queue size'
        self._queue.append((func, args, size))
        self._lock.release()


    def cut_line(self, func:Callable, *, size:int, args:tuple=None):
        self._lock.acquire()
        assert size <= self._limit, 'exceed queue size'
        self._queue.insert(0, (func, args, size))
        self._lock.release()

    def stop(self):
        self.isactive = False

    def collect_threads(self):
        self._thread.join()
        self.threads.append(self._thread)
        return self.threads

model_queue = ExecQueue(MODEL_QUEUE_LIMIT, 'm_queue')
pattern_queue = ExecQueue(PATTERN_QUEUE_LIMIT, 'p_queue')

class DbWriterBase(metaclass=ABCMeta):
    def __init__(self, controller, stime=1):
        self._controller = controller
        self._active = False
        self._pool = mp.Manager().list()
        self._thread = CatchableTread(target=self._run, args=(stime,))
        self._thread.start()

    def add(self, data: pd.DataFrame):
        self._pool.append(data)

    def get_thread(self):
        return self._thread

    @abstractproperty
    def _TASK_NAME(self):
        pass

    @abstractmethod
    def _save(self, data: pd.DataFrame):
        raise NotImplementedError()

    def _run(self, stime):
        self._active = True
        logging.info(f'start writing {self._TASK_NAME}')
        tlen = 0
        while self._controller.isactive:
            if self._pool:
                recv = list(self._pool)
                self._pool[:] = self._pool[len(recv):]
                data = pd.concat(recv, axis=0)
                self._save(data)
                logging.info(f'writing {self._TASK_NAME}: #{tlen} ~ #{tlen + len(data)} records')
                tlen = tlen + len(data)
            else:
                if self._active:
                    time.sleep(stime)
                else:
                    break
        logging.info(f'Writing {self._TASK_NAME} finished')

    def stop(self):
        self._active = False

    @property
    def pool(self):
        return self._pool

class HistoryReturnWriter(DbWriterBase):

    def _save(self, data):
        save_mkt_period(data)

    @property
    def _TASK_NAME(self):
        return 'market period values'

class LatestReturnWriter(DbWriterBase):

    def _save(self, data):
        save_latest_mkt_period(data)

    @property
    def _TASK_NAME(self):
        return 'market latest period values'

class PatternOccurWriter(DbWriterBase):

    def _save(self, data):
        save_latest_pattern_occur(data)

    @property
    def _TASK_NAME(self):
        return 'pattern occurs'

class PatternDistWriter(DbWriterBase):

    def _save(self, data):
        save_latest_pattern_distribution()

    @property
    def _TASK_NAME(self):
        return 'pattern distributions'

class ReturnScoreWriter(DbWriterBase):

    def _save(self, data):
        save_mkt_score(data)

    @property
    def _TASK_NAME(self):
        return 'market return scores'

class ReturnDistWriter(DbWriterBase):

    @classmethod
    def _save(self, data):
        save_mkt_dist(data)

    @property
    def _TASK_NAME(self):
        return 'market return distributions'

class LatestPatternWriter(DbWriterBase):

    @classmethod
    def _save(self, data):
        save_latest_pattern_results(data)

    @property
    def _TASK_NAME(self):
        return 'latest pattern results'


def gen_return_value(mid: str):
    cps = get_market_data(mid)['CP'].dropna()
    ds = cps.index.values
    vs = cps.values
    mis = []
    dps = []
    pds = []
    dds = []
    ncs = []
    ncrs = []
    rps = []
    rms = []
    rss = []
    rcs = []
    for p in PREDICT_PERIODS:
        if len(vs) <= p:
            continue
        mis.append(np.full(len(vs) - p, p))
        dps.append(np.full(len(vs) - p, p))
        pds.append(ds[p:])
        dds.append(ds[:-p])
        nc = vs[p:] - vs[:-p]
        ncs.append(nc)
        ncr = nc / vs[:-p]
        ncrs.append(ncr)
        rps.append(p)
        rms.append(ncr.mean())
        rss.append(ncr.std())
        rcs.append(len(ncr))
    ret = pd.DataFrame()
    if dps:
        ret[MarketPeriodField.DATE_PERIOD.value] = np.concatenate(dps)
        ret[MarketPeriodField.PRICE_DATE.value] = np.concatenate(pds)
        ret[MarketPeriodField.DATA_DATE.value] = np.concatenate(dds)
        ret[MarketPeriodField.NET_CHANGE.value] = np.concatenate(ncs)
        ret[MarketPeriodField.NET_CHANGE_RATE.value] = np.concatenate(ncrs)
        ret[MarketPeriodField.MARKET_ID.value] = mid

    ret_1 = pd.DataFrame()
    ret_2 = pd.DataFrame()
    if rms:
        ret_1[MarketStatField.DATE_PERIOD.value] = rps
        ret_1[MarketStatField.RETURN_MEAN.value] = rms
        ret_1[MarketStatField.RETURN_STD.value] = rss
        ret_1[MarketStatField.RETURN_CNT.value] = rcs
        ret_1[MarketStatField.MARKET_ID.value] = mid

        dps = []
        mss = []
        ubs = []
        lbs = []

        s_, l_, u_ = list(zip(*get_score_meta_info()))
        s_ = np.array(s_)
        l_ = np.array(l_)
        u_ = np.array(u_)
        for p, m, s in zip(rps, rms, rss):
            dps.append(np.full(len(s_), p))
            mss.append(s_)
            ubs.append(m + s * u_)
            lbs.append(m + s * l_)

        ret_2[MarketScoreField.DATE_PERIOD.value] = np.concatenate(dps)
        ret_2[MarketScoreField.MARKET_SCORE.value] = np.concatenate(mss)
        ret_2[MarketScoreField.UPPER_BOUND.value] = np.concatenate(ubs)
        ret_2[MarketScoreField.LOWER_BOUND.value] = np.concatenate(lbs)
        ret_2[MarketScoreField.MARKET_ID.value] = mid
    return ret, ret_1, ret_2

def get_latest_patterns(mids, pids, dates, pvalues):
    ret = pd.DataFrame()
    values = np.full(len(pvalues), 'N')
    values[pvalues == 1] = 'Y'
    ret[PatternResultField.VALUE.value] = values
    ret[PatternResultField.DATE.value] = dates.astype('datetime64[D]').tolist()
    ret[PatternResultField.MARKET_ID.value] = mids
    ret[PatternResultField.PATTERN_ID.value] = pids
    return ret

def gen_return_value_v2(mid: str):
    cps = get_market_data(mid)['CP'].dropna()
    ds = cps.index.values
    vs = cps.values
    mis = []
    dps = []
    pds = []
    dds = []
    ncs = []
    ncrs = []
    rps = []
    rms = []
    rss = []
    rcs = []
    frs = []
    for p in PREDICT_PERIODS:
        if len(vs) <= p:
            frs.append(np.full(len(vs), np.nan))
            continue
        dps.append(p)
        pds.append(ds[-1])
        dds.append(ds[-p-1])
        nc = vs[p:] - vs[:-p]
        ncs.append(nc[-1])
        ncr = nc / vs[:-p]
        ncrs.append(ncr[-1])
        frs.append(np.concatenate([ncr, np.full(p, np.nan)]))
        rps.append(p)
        rms.append(ncr.mean())
        rss.append(ncr.std())
    ret = pd.DataFrame()
    if dps:
        ret[MarketPeriodField.DATE_PERIOD.value] = dps
        ret[MarketPeriodField.PRICE_DATE.value] = np.array(pds)
        ret[MarketPeriodField.DATA_DATE.value] = np.array(dds)
        ret[MarketPeriodField.NET_CHANGE.value] = np.array(ncs)
        ret[MarketPeriodField.NET_CHANGE_RATE.value] = np.array(ncrs)
        ret[MarketPeriodField.MARKET_ID.value] = mid

    ret_1 = np.array(frs).T
    ret_2 = pd.DataFrame()
    if rms:
        dps = []
        mss = []
        ubs = []
        lbs = []

        s_, l_, u_ = list(zip(*get_score_meta_info()))
        s_ = np.array(s_)
        l_ = np.array(l_)
        u_ = np.array(u_)
        for p, m, s in zip(rps, rms, rss):
            dps.append(np.full(len(s_), p))
            mss.append(s_)
            ubs.append(m + s * u_)
            lbs.append(m + s * l_)

        ret_2[MarketScoreField.DATE_PERIOD.value] = np.concatenate(dps)
        ret_2[MarketScoreField.MARKET_SCORE.value] = np.concatenate(mss)
        ret_2[MarketScoreField.UPPER_BOUND.value] = np.concatenate(ubs)
        ret_2[MarketScoreField.LOWER_BOUND.value] = np.concatenate(lbs)
        ret_2[MarketScoreField.MARKET_ID.value] = mid
    return ret, ret_1, ret_2

def do_pattern_task(market, patterns, mode,
                    ret_mdates, ret_pvalues, ret_freturns,
                    ret_values, ret_mscores, ret_lptterns):
    try:
        from func._td import set_market_data_provider
        from dao import MimosaDB

        class MDP:
            def __init__(self, mode):
                self._db = MimosaDB(mode=mode)
            def get_ohlc(self, market_id: str) -> pd.DataFrame:
                return self._db.get_market_data(market_id)

        set_exec_mode(mode)
        set_db(MimosaDB(mode=mode))
        set_market_data_provider(MDP(mode))

        rvalues, freturns, mscores = gen_return_value_v2(market)
        if ret_values is not None:
            ret_values.append(rvalues)
        if ret_mscores is not None:
            ret_mscores.append(mscores)
        result_buffer = []
        for pattern in patterns:
            result_buffer.append(pattern.run(market).rename(pattern.pid))
        pattern_result = fast_concat(result_buffer)
        mdates = SMD.dencode(pattern_result.index.values)
        pvalues = SMD.pencode(pattern_result.values)
        freturns = SMD.rencode(freturns)
        ret_mdates[market] = mdates
        ret_pvalues[market] = pvalues
        ret_freturns[market] = freturns
        if ret_lptterns is not None and len(pattern_result) > 0:
            ret_lptterns.append(get_latest_patterns(market,
                                                    list(pattern_result.columns),
                                                    mdates[-1], pvalues[-1]))
        return f'update patterns: {market}'
    except Exception as esp:
        return f'update patterns error: {market} - {esp}'

def pattern_update(controller, batch_type=BatchType.SERVICE_BATCH):
    MD_CACHE.clear()
    smd_buffer = {}
    patterns = get_patterns()
    markets = get_markets()
    if len(patterns) <= 0 or len(markets) <= 0:
        logging.debug(f'get markets: \n{markets}')
        logging.debug(f'get patterns: \n{patterns}')
        return
    if batch_type == BatchType.SERVICE_BATCH:
        latest_pattern_writer = LatestPatternWriter(controller)
        ret_values = mp.Manager().list()
        ret_mscores = mp.Manager().list()
        ret_lpatterns = latest_pattern_writer.pool
    else:
        ret_values = None
        ret_mscores = None
        ret_lpatterns = None
    smd_buffer['mids'] = {mid: idx for idx, mid in enumerate(markets)}
    smd_buffer['pids'] = {each.pid: idx for idx, each in enumerate(patterns)}
    smd_buffer['mdates'] = mp.Manager().dict()
    smd_buffer['pvalues'] = mp.Manager().dict()
    smd_buffer['freturns'] = mp.Manager().dict()
    patterns = mp.Manager().list(patterns)
    pool = mp.Pool(PATTERN_UPDATE_CPUS)

    for market in markets:
        pool.apply_async(do_pattern_task,
                         (market, patterns, exec_mode,
                          # return buffers
                          smd_buffer['mdates'], smd_buffer['pvalues'], smd_buffer['freturns'],
                          # optional return buffers
                          ret_values, ret_mscores, ret_lpatterns),
                          callback=logging.debug)


    pool.close()
    pool.join()

    smd_buffer['mdates'] = [smd_buffer['mdates'][mid] for mid in markets]
    smd_buffer['pvalues'] = [smd_buffer['pvalues'][mid] for mid in markets]
    smd_buffer['freturns'] = [smd_buffer['freturns'][mid] for mid in markets]
    smd_swap.update(**smd_buffer)
    del smd_buffer
    tsmd = CatchableTread(target=smd_swap.dump)
    tsmd.start()
    ret = [tsmd]
    if batch_type == BatchType.SERVICE_BATCH:
        return_score_writer = ReturnScoreWriter(controller)
        return_score_writer.add(pd.concat(ret_mscores, axis=0))
        market_return_writer = LatestReturnWriter(controller)
        market_return_writer.add(pd.concat(ret_values, axis=0))
        latest_pattern_writer.stop()
        return_score_writer.stop()
        market_return_writer.stop()
        if not controller.isactive:
            logging.info('batch terminated')
            return
        ret += [market_return_writer.get_thread(), latest_pattern_writer.get_thread(),
                return_score_writer.get_thread()]
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


def get_pattern_stats_info_v2(pattern, freturn, market):
    ret_buffer_1 = []
    ret_buffer_2 = []
    for p in PREDICT_PERIODS:
        cur_1 = pd.DataFrame()
        cur_2 = pd.DataFrame()
        cur_r = freturn[get_filed_name_of_future_return(p)].values
        cur_p = pattern.values.copy()

        # 正常市場資料下的寫法，比較快
        cur_r = cur_r[:-p]
        cur_p = cur_p[:-p]
        # 不正常市場資料下的寫法，比較慢
        #is_valid_r = ~np.isnan(cur_r)
        #cur_r = cur_r[is_valid_r]
        #cur_p = cur_p[is_valid_r]

        is_occur = cur_p == 1
        is_not_occur = cur_p == 0
        cur_p[is_not_occur] = np.nan
        effective_r = (cur_r * cur_p.T).T

        cur_1[MarketDistField.RETURN_MEAN.value] = np.nanmean(effective_r, axis=0)
        cur_1[MarketDistField.RETURN_STD.value] = np.nanstd(effective_r, axis=0, ddof=0)
        cur_1[MarketDistField.PATTERN_ID.value] = pattern.columns.values
        cur_1[MarketDistField.MARKET_ID.value] = market
        cur_1[MarketDistField.DATE_PERIOD.value] = p
        cur_2[MarketOccurField.OCCUR_CNT.value] = is_occur.sum(axis=0)
        cur_2[MarketOccurField.NON_OCCUR_CNT.value] = is_not_occur.sum(axis=0)
        cur_2[MarketOccurField.MARKET_RISE_CNT.value] = (effective_r>0).sum(axis=0)
        cur_2[MarketOccurField.MARKET_FLAT_CNT.value] = (effective_r==0).sum(axis=0)
        cur_2[MarketOccurField.MARKET_FALL_CNT.value] = (effective_r<0).sum(axis=0)
        cur_2[MarketOccurField.PATTERN_ID.value] = pattern.columns.values
        cur_2[MarketOccurField.MARKET_ID.value] = market
        cur_2[MarketOccurField.DATE_PERIOD.value] = p
        ret_buffer_1.append(cur_1)
        ret_buffer_2.append(cur_2)
    return pd.concat(ret_buffer_1, axis=0), pd.concat(ret_buffer_2, axis=0)

def model_execution_recover(batch_type:BatchType):
    logging.info('Start model execution recover')
    for model, etype in get_recover_model_execution():
        if etype == ModelExecution.ADD_PREDICT:
            model_recover(model, ModelStatus.ADDED)
        if etype == ModelExecution.ADD_BACKTEST:
            model_recover(model, ModelStatus.CREATED)
    logging.info('End model execution recover')

def init_db():
    try:
        logging.info('start initiate db')
        model_queue.cut_line(batch, size=MODEL_QUEUE_LIMIT, args=(BatchType.INIT_BATCH,))
        pattern_queue.cut_line(batch, size=PATTERN_QUEUE_LIMIT, args=(BatchType.INIT_BATCH,))
        logging.info('init finished')
    except Exception as esp:
        logging.error(f"init db failed")
        logging.error(traceback.format_exc())

def batch(batch_type=BatchType.SERVICE_BATCH):
    batch_lock.acquire()
    if MT_MANAGER.exists('batch executing'):
        batch_lock.release()
        while MT_MANAGER.exists('batch executing'):
            time.sleep(1)
        return
    else:
        MT_MANAGER.acquire('batch executing')
        batch_lock.release()
        _batch(batch_type)
        MT_MANAGER.release('batch executing')
        return

def _batch(batch_type):
    MT_MANAGER.acquire(BATCH_EXE_CODE).switch_off()
    MT_MANAGER.release(BATCH_EXE_CODE)
    while MT_MANAGER.exists(BATCH_EXE_CODE):
        time.sleep(1)
    controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
    try:
        if controller.isactive:
            logging.info(f"Start batch ")
            clean_db_cache()
            clone_db_cache(batch_type)
            mrt = clone_model_results(controller)
            logging.info("Start pattern update")
            if batch_type == BatchType.SERVICE_BATCH:
                truncate_swap_tables()
            if batch_type == BatchType.INIT_BATCH:
                if smd.isempty():
                    ts = pattern_update(controller, batch_type) or []
                    if controller.isactive:
                        swap_smd_index()
                        swap_smd()
                else:
                    ts = []
            else:
                ts = pattern_update(controller, batch_type) or []
            logging.info("End pattern update")
            mrt.join()
            if controller.isactive and (batch_type == BatchType.INIT_BATCH):
                logging.info('Start model execution recover')
                model_execution_recover(batch_type)
                logging.info('End model execution recover')
            if batch_type == BatchType.SERVICE_BATCH:
                logging.info("Start model update")
                for model in get_models():
                    t = model_update(model, controller, batch_type)
                    if t is not None:
                        ts.append(t)
                for t in ts:
                    t.join()
                    if t.esp is not None:
                        logging.error(t.esp)
                logging.info("End model update")
                if controller.isactive:
                    checkout_fcst_data()
                    swap_smd_index()
                    swap_smd()

            MT_MANAGER.release(BATCH_EXE_CODE)
            logging.info(f"End batch")

    except Exception as esp:
        logging.error(f"batch failed")
        logging.error(traceback.format_exc())
        MT_MANAGER.release(BATCH_EXE_CODE)


def get_mix_pattern_occur(market_id: str, patterns: List, start_date:str=None, end_date:str=None):
    if smd.isempty():
        return []
    smd_lock.acquire()
    mids, pids, mdates, pvalues, _ = smd.raw_data
    smd_lock.release()

    midx = mids.get(market_id)
    if midx is None:
        return []
    pidxs = [pids[pid] for pid in patterns]
    valeus = pvalues[midx][:,pidxs]
    dates = mdates[midx]
    dates = dates[(valeus==1).all(axis=1)]

    ret = SMD.ddecode(dates)
    if start_date is not None:
        start_date = np.datetime64(start_date)
        ret = ret[(ret>=start_date).sum():]
    if end_date is not None:
        end_date = np.datetime64(end_date)
        ret = ret[:(ret<=end_date).sum()]
    return ret.tolist()

def get_pattern_occur(market_id: str, pattern_id):
    return get_mix_pattern_occur(market_id, [pattern_id])

def get_mix_pattern_occur_cnt(patterns, market_type=None, category_code=None):
    def func(vs):
        recv = np.array([((v>=0).all(axis=1).sum(), (v==1).all(axis=1).sum()) for v in vs])
        if len(vs) <= 0:
            return 0, 0
        cnts, occurs = recv.sum(axis=0).tolist()
        return occurs, cnts - occurs
    if smd.isempty():
        return 0, 0

    smd_lock.acquire()
    mids, pids, mdates, pvalues, _ = smd.raw_data
    smd_lock.release()

    markets = get_markets(market_type, category_code)
    if not markets:
        return 0, 0
    midxs = [mids[mid] for mid in markets if mid in mids]
    if len(midxs) == 0:
        return 0, 0
    pidxs = [pids[pid] for pid in patterns]
    values = [pvalues[idx][:,pidxs] for idx in midxs]
    return func(values)

def get_pattern_occur_cnt(pattern_id, market_type=None, category_code=None):
    return get_mix_pattern_occur_cnt([pattern_id], market_type, category_code)

def get_mix_pattern_rise_prob(patterns, period, market_type=None, category_code=None):
    def func(v, r):
        ret = r[(v==1).all(axis=1) & (r==r)]
        return len(ret), (ret>0).sum().tolist()

    if smd.isempty():
        return 0

    smd_lock.acquire()
    mids, pids, _, pvalues, freturns = smd.raw_data
    smd_lock.release()

    markets = get_markets(market_type, category_code)
    if not markets:
        return 0
    pidx = PREDICT_PERIODS.index(period)
    midxs = [mids[mid] for mid in markets if mid in mids]
    if len(midxs) == 0:
        return 0
    pidxs = [pids[pid] for pid in patterns]
    values = [pvalues[idx][:,pidxs] for idx in midxs]
    returns = [freturns[idx][:,pidx] for idx in midxs]
    stats = np.array([func(v, r) for v, r in zip(values, returns)])
    cnts, ups = stats.sum(axis=0)
    return ups / cnts * 100

def get_pattern_rise_prob(pattern_id, period, market_type=None, category_code=None):
    return get_mix_pattern_rise_prob([pattern_id], period, market_type, category_code)

def get_mix_pattern_mkt_dist_info(patterns, period, market_type=None, category_code=None):
    def func(v, r):
        ret = r[(v==1).all(axis=1) & (r==r)]
        if len(ret) == 0:
            return 0, 0, 0
        return ret.mean() * 100, ret.std() * 100, len(ret)

    if smd.isempty():
        return {}

    smd_lock.acquire()
    mids, pids, _, pvalues, freturns = smd.raw_data
    smd_lock.release()

    markets = get_markets(market_type, category_code)
    if not markets:
        return {}
    pidx = PREDICT_PERIODS.index(period)
    midxs = [mids[mid] for mid in markets if mid in mids]
    if len(midxs) == 0:
        return {}
    pidxs = [pids[pid] for pid in patterns]
    values = [pvalues[idx][:,pidxs] for idx in midxs]
    returns = [freturns[idx][:,pidx] for idx in midxs]
    stats = np.array([func(v, r) for v, r in zip(values, returns)])
    drops = ~np.isnan(stats).any(axis=1)
    markets = np.array(markets)[drops].tolist()
    stats = stats[drops]
    return {m: (v, r, int(c)) for m, (v, r, c) in zip(markets, stats)}

def get_pattern_mkt_dist_info(pattern_id, period, market_type=None, category_code=None):
    return get_mix_pattern_mkt_dist_info([pattern_id], period, market_type, category_code)

def add_pattern(pid):
    smd_lock.acquire()
    mids, pids, _, old_pvalues, freturns = smd.raw_data
    smd_lock.release()

    pvalues = []
    pattern = get_pattern_info(pid)
    latest_dates = []
    latest_values = []
    pattern_occurs = []
    pattern_dists = []
    if pid in pids:
        pidx = pids[pid]
    else:
        pids = pids.copy()
        pids[pid] = len(pids)
        pidx = -1

    for mid, idx in mids.items():
        pattern_result = pattern.run(mid).rename(pattern.pid)
        if pidx > 0:
            values = old_pvalues[idx].copy()
            values[:, pidx] = SMD.pencode(pattern_result.values)
        else:
            values = np.concatenate([old_pvalues[idx], SMD.pencode(pattern_result.values).reshape((-1,1))], axis=1)
        pvalues.append(values)
        return_result = get_future_returns(mid)
        latest_dates.append(pattern_result.index.values[-1])
        latest_values.append(pattern_result.values[-1])
        market_dist, market_occur = get_pattern_stats_info_v2(pd.DataFrame(pattern_result), return_result, mid)
        pattern_occurs.append(market_occur)
        pattern_dists.append(market_dist)
    update_latest_pattern_results(get_latest_patterns(list(mids.keys()), pid, np.array(latest_dates), np.array(latest_values)))
    update_latest_pattern_occur(pd.concat(pattern_occurs, axis=0))
    update_latest_pattern_distribution(pd.concat(pattern_dists, axis=0))
    smd_lock.acquire()
    if pidx > 0:
        smd.update(pvalues=pvalues)
    else:
        smd.update(pids=pids, pvalues=pvalues)
    smd_lock.release()

def get_market_rise_prob(period, market_type=None, category_code=None):
    def func(r):
        ret = r[(r==r)]
        return len(ret), (ret>0).sum().tolist()

    if smd.isempty():
        return 0

    smd_lock.acquire()
    mids, pids, _, pvalues, freturns = smd.raw_data
    smd_lock.release()

    markets = get_markets(market_type, category_code)
    if not markets:
        return 0
    pidx = PREDICT_PERIODS.index(period)
    midxs = [mids[mid] for mid in markets if mid in mids]
    if len(midxs) == 0:
        return 0
    returns = [freturns[idx][:,pidx] for idx in midxs]
    stats = np.array([func(r) for r in returns])
    cnts, ups = stats.sum(axis=0)
    return (ups / cnts) * 100

def get_mkt_dist_info(period, market_type=None, category_code=None):
    def func(r):
        ret = r[(r==r)]
        if len(ret) == 0:
            return 0, 0, 0
        return ret.mean() * 100, ret.std() * 100, len(ret)

    if smd.isempty():
        return {}

    smd_lock.acquire()
    mids, pids, _, pvalues, freturns = smd.raw_data
    smd_lock.release()

    markets = get_markets(market_type, category_code)
    if not markets:
        return {}
    pidx = PREDICT_PERIODS.index(period)
    midxs = [mids[mid] for mid in markets if mid in mids]
    if len(midxs) == 0:
        return {}
    returns = [freturns[idx][:,pidx] for idx in midxs]
    stats = np.array([func(r) for r in returns])
    drops = ~np.isnan(stats).any(axis=1)
    markets = np.array(markets)[drops].tolist()
    stats = stats[drops]
    return {m: (v, r, int(c)) for m, (v, r, c) in zip(markets, stats)}

def get_market_price_dates(market_id: str, begin_date: Optional[datetime.date] = None):
    def _extend_dates(bdate, length):
        ext_dates = np.arange(1, (int(length / 5) + 1) * 7 + 1) + bdate
        ext_dates = ext_dates[(ext_dates - 4) % 7 < 5][:length]
        return ext_dates

    if smd.isempty():
        return []

    smd_lock.acquire()
    mids, _, mdates, _, _ = smd.raw_data
    smd_lock.release()

    if market_id not in mids:
        return []
    midx = mids[market_id]
    mdates = mdates[midx]
    ext_dates = SMD.ddecode(_extend_dates(mdates[-1], PREDICT_PERIODS[-1]))
    mdates = SMD.ddecode(mdates)
    if begin_date:
        didx = (mdates < begin_date).sum()
    else:
        didx = 0
    ret = []
    mdates = np.concatenate([mdates[didx:], ext_dates]).tolist()
    eidx = len(mdates) - PREDICT_PERIODS[-1]
    for period in PREDICT_PERIODS:
        ret += [{MarketPeriodField.DATA_DATE.value: a,
                 MarketPeriodField.PRICE_DATE.value: b,
                 MarketPeriodField.DATE_PERIOD.value: period
                 } for a, b in zip(mdates[:eidx],
                                   mdates[period:eidx+period])]
    return ret

def verify_pattern(func, kwargs):
    try:
        return eval(f"{func}.check(kwargs)")
    except AttributeError:
        return {}
