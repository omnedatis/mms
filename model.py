# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

"""

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
import datetime
import logging
import os
import shutil
from threading import Lock
import time
import multiprocessing as mp
from typing import Any, List, Optional, Dict, Callable
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm
from sklearn.tree import DecisionTreeClassifier as Dtc
import traceback

from _core import Pattern as PatternInfo
from _core import View as ModelInfo
from func._tp import *
from _view import view_backtest, view_create, view_update
from const import *
from utils import *
from const import (BatchType, MarketPeriodField, MarketScoreField,
                   ModelExecution, ModelMarketHitSumField, ModelStatus,
                   PatternExecution, PatternResultField, PredictResultField,
                   MIN_BACKTEST_LEN, PREDICT_PERIODS)
from utils import extend_working_dates, CatchableTread, ThreadController

#from func._tp import

from _core import Pattern
from _db import get_dao, set_dao, MimosaDBManager

batch_lock = Lock()

def get_db():
    """Get Mimosa DB accessing object"""
    return get_dao()

def set_db(_db):
    """Set Mimosa DB accessing object"""
    set_dao(_db)

def clean_db_cache():
    """Clean Minosa DB cache"""
    db = get_db()
    db.clean_db_cache()

def clone_db_cache(batch_type):
    """Clone Mimosa DB to cache"""
    db = get_db()
    db.clone_db_cache(batch_type)

def save_mkt_score(recv: Dict[str, pd.DataFrame]):
    """save mkt score to DB."""
    def gen_records(recv):
        ret = []
        scores, lfactor, ufactor = list(zip(*get_db().get_score_meta_info()))
        lfactor = np.array(lfactor)
        ufactor = np.array(ufactor)
        for mid, freturns in recv.items():
            # Note: without making `dropna` for performance
            for period, freturn in zip(PREDICT_PERIODS, freturns.values.T):
                if len(freturn) <= period:
                    continue
                cur = pd.DataFrame()
                freturn = freturn[:-period]
                mean = freturn.mean()
                std = freturn.std(ddof=1)

                cur[MarketScoreField.MARKET_SCORE.value] = scores
                cur[MarketScoreField.UPPER_BOUND.value] = ufactor * std + mean
                cur[MarketScoreField.LOWER_BOUND.value] = lfactor * std + mean
                cur[MarketScoreField.DATE_PERIOD.value] = period
                cur[MarketScoreField.MARKET_ID.value] = mid
                ret.append(cur)
        return ret

    if len(recv) > 0:
        records = gen_records(recv)
        if len(records) > 0:
            db = get_db()
            db.save_mkt_score(pd.concat(records, axis=0))

def save_latest_mkt_period(recv: Dict[str, pd.DataFrame]):
    """save latest mkt period to DB."""
    def gen_records(recv):
        ret = []
        for mid, cps in recv.items():
            # Note: without making `dropna` for performance
            periods = list(filter(lambda p: p < len(cps), PREDICT_PERIODS))
            if len(periods) <= 0:
                continue
            cur = pd.DataFrame()
            offsets = list(map(lambda p: -(p+1), periods))
            dates = cps.index.values[offsets + [-1]].astype('datetime64[D]').tolist()
            bps = cps.values[offsets]
            changes = cps.values[-1] - bps
            cur[MarketPeriodField.DATE_PERIOD.value] = periods
            cur[MarketPeriodField.PRICE_DATE.value] = dates[-1]
            cur[MarketPeriodField.DATA_DATE.value] = dates[:-1]
            cur[MarketPeriodField.NET_CHANGE.value] = changes
            cur[MarketPeriodField.NET_CHANGE_RATE.value] = changes / bps
            cur[MarketPeriodField.MARKET_ID.value] = mid
            ret.append(cur)
        return ret

    if len(recv) > 0:
        records = gen_records(recv)
        if len(records) > 0:
            db = get_db()
            db.save_latest_mkt_period(pd.concat(records, axis=0))

def save_latest_pattern_results(recv: Dict[str, pd.DataFrame], update:bool=False):
    """Save latest pattern results to DB."""
    def trans2dbformat(recv):
        ret_buffer = []
        for mid, pdata in recv.items():
            cur = pd.DataFrame()
            mdate = pdata.index.values[-1].astype('datetime64[D]').tolist()
            values = np.full(len(pdata.columns), 'N')
            values[pdata.values[-1] == 1] = 'Y'
            cur[PatternResultField.VALUE.value] = values
            cur[PatternResultField.DATE.value] = mdate
            cur[PatternResultField.MARKET_ID.value] = mid
            cur[PatternResultField.PATTERN_ID.value] = pdata.columns
            ret_buffer.append(cur)
        return pd.concat(ret_buffer, axis=0)

    if len(recv) > 0:
        db = get_db()
        if update:
            for pid, data in trans2dbformat(recv).groupby(by=PatternResultField.PATTERN_ID.value):
                db.update_latest_pattern_results(pid, data)
        else:
            db.save_latest_pattern_results(trans2dbformat(recv))

def _evaluate_hit_sum(market, results, freturns):
    ret = pd.DataFrame()
    ret[ModelMarketHitSumField.DATE_PERIOD.value] = PREDICT_PERIODS
    ret[ModelMarketHitSumField.MARKET_CODE.value] = market
    if results is None:
        ret[ModelMarketHitSumField.HIT.value] = 0
        ret[ModelMarketHitSumField.FCST_CNT.value] = 0
        return ret

    results = results.set_index(PredictResultField.DATE.value)
    result_group = {p: d for p, d in results.groupby(PredictResultField.PERIOD.value)}
    cnts = []
    hits = []
    for period in PREDICT_PERIODS:
        cur_result = result_group.get(period)
        if cur_result is None:
            cnts.append(0)
            hits.append(0)
            continue
        cur_result = cur_result[[PredictResultField.UPPER_BOUND.value,
                                 PredictResultField.LOWER_BOUND.value]]
        cur_freturn = freturns[period].rename('FR')
        ubs, lbs, rs = pd.concat([cur_result, cur_freturn], axis=1, sort=True
                                 ).dropna().values.T
        rs *= 100  # trans to percents
        cnts.append(len(rs))
        hits.append(((ubs >= rs) & (lbs <= rs)).sum())
    ret[ModelMarketHitSumField.HIT.value] = hits
    ret[ModelMarketHitSumField.FCST_CNT.value] = cnts
    return ret

def get_model_hit_sum(model_id: str, batch_type: BatchType):
    """Get model Hit-sum."""
    if batch_type == BatchType.INIT_BATCH:
        _db = MimosaDBManager().current_db
    else:
        _db = MimosaDBManager().next_db
    results = get_db().get_model_results(model_id)
    recv = [_evaluate_hit_sum(market, results.get(market),
                              _db.get_future_returns(market, PREDICT_PERIODS)
                              ) for market in _db.get_markets()]
    ret = pd.concat(recv, axis=0)
    ret['MODEL_ID'] = model_id
    return ret

def save_model_hit_sum(model_id: str, batch_type: BatchType):
    """Save model hit sum to DB."""
    recv = get_model_hit_sum(model_id, batch_type)
    db = get_db()
    if batch_type == BatchType.INIT_BATCH:
        db.update_model_hit_sum(recv)
    else:
        db.save_model_hit_sum(recv)

def _get_backtest_length(market: str, earlist_date: datetime.date):
    db = get_db()
    dates = db.get_market_data(market).index.values.astype('datetime64[D]')
    ret = min([MIN_BACKTEST_LEN, len(dates)]) - (dates >= earlist_date).sum()
    return ret


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


def _update_model(model_id: str, controller):
    latest_dates = get_db().get_latest_dates(model_id)
    recv = view_update(get_db().get_model_info(model_id), latest_dates, controller)
    if recv is None:
        return recv
    return recv.dropna()

def add_model(model_id: str):
    """新增模型

    此函數目的為新增模型：
    1. 呼叫_create_model建立模型，並計算各市場、各天期的最近一日的預測結果
    2. 呼叫_backtest_model計算各市場、各天期的歷史回測結果

    Parameters
    ----------
    model_id: str
        ID of the designated model.

    See Also
    --------
    _create_model, _backtest_model

    """
    if MT_MANAGER.exists(model_id):
        return
    controller = MT_MANAGER.acquire(model_id)
    if not controller.isactive:
        MT_MANAGER.release(model_id)
        return
    try:
        model = get_db().get_model_info(model_id)
        _create_model(model, controller)
        _backtest_model(model, controller)
        save_model_hit_sum(model_id, BatchType.INIT_BATCH)
        logging.info('add model finished')
    except Exception:
        logging.error("add model failed ")
        logging.error(traceback.format_exc())
    finally:
        MT_MANAGER.release(model_id)

def model_recover(model_id: str, status: ModelStatus):
    """重啟模型

    此函數目的為重啟因故中斷的模型：
    如果模型狀態為 ADDED 則
    1. 呼叫_create_model建立模型，並計算各市場、各天期的最近一日的預測結果
    2. 呼叫_backtest_model計算各市場、各天期的歷史回測結果
    如果模型狀態為 CREATED 則跳過1. 直接執行 2.

    Parameters
    ----------
    model_id: str
        ID of the designated model.
    status: ModelStatus
        Status of Model.

    See Also
    --------
    _create_model, _backtest_model, ModelStatus

    """
    controller = MT_MANAGER.acquire(model_id)
    if not controller.isactive:
        MT_MANAGER.release(model_id)
        return
    try:
        model = get_db().get_model_info(model_id)
        if status < ModelStatus.CREATED:
            _create_model(model, controller)
        _backtest_model(model, controller)
        save_model_hit_sum(model_id, BatchType.INIT_BATCH)
    except Exception:
        logging.error("recover model failed")
        logging.error(traceback.format_exc())
    finally:
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

    """
    logging.info('start remove model')
    MT_MANAGER.acquire(model_id).switch_off()
    MT_MANAGER.release(model_id)
    while MT_MANAGER.exists(model_id):
        time.sleep(1)
    get_db().del_model_data(model_id)
    model_dir = ModelInfo.get_dir(model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    logging.info('finish remove model')

def edit_model(model_id: str):
    remove_model(model_id)
    add_model(model_id)
    return

def _create_model(model: ModelInfo, controller: ThreadController):
    logging.info('start model create')
    if not controller.isactive:
        logging.info('model create terminated')
        return
    exection_id = get_db().set_model_execution_start(
        model.view_id, ModelExecution.ADD_PREDICT)
    recv = view_create(model, controller).dropna()
    logging.info('finish model create')
    if recv is not None and controller.isactive:
        get_db().save_model_results(model.view_id, recv, ModelExecution.ADD_PREDICT)
    if controller.isactive:
        get_db().set_model_execution_complete(exection_id)
        get_db().stamp_model_execution([exection_id])

def _backtest_model(model: ModelInfo, controller: ThreadController):
    logging.info('start model backtest')
    exection_id = get_db().set_model_execution_start(
        model.view_id, ModelExecution.ADD_BACKTEST)
    earlist_dates = get_db().get_earliest_dates(model.view_id)
    cur_thread = None
    while controller.isactive:
        earlist_dates = {market: date for market, date in earlist_dates.items()
                         if _get_backtest_length(market, date) > 0}

        if len(earlist_dates) == 0:
            cur_thread and cur_thread.join()
            break
        recv = view_backtest(model, earlist_dates, controller)
        if recv is None:
            cur_thread and cur_thread.join()
            break
        cur_thread and cur_thread.join()
        cur_thread = CatchableTread(target=get_db().save_model_results,
                                    args=(model.view_id, recv.dropna(),
                                          ModelExecution.ADD_BACKTEST))
        cur_thread.start()
        for mid, data in recv.groupby(PredictResultField.MARKET_ID.value):
            earlist_dates[mid] = data[PredictResultField.DATE.value
                                      ].values.min().astype('datetime64[D]').tolist()

    if not controller.isactive:
        logging.info('model backtest terminated')
        return

    # 回測完成，更新指定模型在DB上的狀態為'COMPLETE'
    logging.info('finish model backtest')
    get_db().set_model_execution_complete(exection_id)
    get_db().stamp_model_execution([exection_id])

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
                        try:
                            if args is not None:
                                ret = func(*args)
                            else:
                                ret = func()
                        except Exception as esp:
                            logging.error(traceback.format_exc())
                            ret = None
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

def _db_update(batch_type=BatchType.SERVICE_BATCH):
    markets = get_db().get_markets()
    patterns = get_db().get_patterns()
    if batch_type == BatchType.SERVICE_BATCH:
        _db = MimosaDBManager().next_db
    else:
        _db = MimosaDBManager().current_db
    if batch_type == BatchType.SERVICE_BATCH:
        exec_type = PatternExecution.BATCH_SERVICE
        exec_ids = [get_db().set_pattern_execution_start(each.pid, exec_type)
                    for each in patterns]
    _db.update(markets, patterns)
    ret = [CatchableTread(target=_db.dump)]
    if batch_type == BatchType.SERVICE_BATCH:
        ret += [CatchableTread(target=save_latest_pattern_results,
                               args=(_db.get_latest_pattern_values(), )),
                CatchableTread(target=save_latest_mkt_period,
                               args=({mid: _db.get_market_prices(mid)
                                      for mid in markets}, )),
                CatchableTread(target=save_mkt_score,
                               args=({mid: _db.get_future_returns(mid)
                                      for mid in markets}, ))]
        for exec_id in exec_ids:
            get_db().set_pattern_execution_complete(exec_id)
    for t in ret:
        t.start()
    return ret


def model_execution_recover(batch_type:BatchType):
    logging.info('Start model execution recover')
    for model, etype in get_db().get_recover_model_execution():
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
    except Exception:
        logging.error("init db failed")
        logging.error(traceback.format_exc())

def batch(batch_type=BatchType.SERVICE_BATCH):
    batch_lock.acquire()
    if MT_MANAGER.exists('batch executing'):
        batch_lock.release()
        while MT_MANAGER.exists('batch executing'):
            time.sleep(1)
    else:
        MT_MANAGER.acquire('batch executing')
        batch_lock.release()
        _batch(batch_type)
        MT_MANAGER.release('batch executing')

def _model_update():
    ModelUpdateMoniter = namedtuple('_ModelUpdateMoniter',
                                    ['controller', 'exec_id', 'thread'])
    def save_result(data, model_id, exec_id, controller):
        if controller.isactive:
            if data is not None:
                get_db().save_model_results(model_id, data.dropna(), ModelExecution.BATCH_PREDICT)
        if controller.isactive:
            get_db().set_model_execution_complete(exec_id)
        if controller.isactive:
            save_model_hit_sum(model_id, BatchType.SERVICE_BATCH)

    moniters = {}
    for model_id in get_db().get_models():
        controller = MT_MANAGER.acquire(model_id)

        exec_id = get_db().set_model_execution_start(model_id,
                                                     ModelExecution.BATCH_PREDICT)
        logging.info(f'start model update on {model_id}')
        recv = _update_model(model_id, controller)
        if not controller.isactive:
            MT_MANAGER.release(model_id)
            logging.info('model update terminated')
        else:
            logging.info(f'finish model update on {model_id}')
            thread = CatchableTread(target=save_result,
                                    args=(recv, model_id, exec_id, controller))
            thread.start()
            moniters[model_id] = ModelUpdateMoniter(controller, exec_id, thread)

    exec_ids = []
    for model_id, (controller, exec_id, thread) in moniters.items():
        if not controller.isactive:
            logging.info('model update terminated')
        else:
            thread.join()
            if thread.esp is not None:
                logging.error(thread.esp)
            else:
                exec_ids.append(exec_id)
        MT_MANAGER.release(model_id)
    logging.info("End model update")
    return exec_ids

def _batch(batch_type):
    try:
        logging.info("Start batch ")
        clean_db_cache()
        clone_db_cache(batch_type)
        model_prepare_thread = get_db().clone_model_results(ThreadController())
        logging.info("Start pattern update")
        threads = []
        if batch_type == BatchType.SERVICE_BATCH:
            get_db().truncate_swap_tables()
            threads = _db_update(batch_type) or []
        elif not MimosaDBManager().is_ready():
            threads = _db_update(batch_type) or []
        logging.info("End pattern update")

        model_prepare_thread.join()
        if batch_type == BatchType.INIT_BATCH:
            logging.info('Start model execution recover')
            model_execution_recover(batch_type)
            logging.info('End model execution recover')
        if batch_type == BatchType.SERVICE_BATCH:
            model_exec_ids = _model_update()
            for t in threads:
                t.join()
                t.esp and logging.error(t.esp)
            get_db().checkout_fcst_data()
            get_db().stamp_model_execution(model_exec_ids)
            MimosaDBManager().swap_db()
        logging.info("End batch")
    except Exception:
        logging.error("batch failed")
        logging.error(traceback.format_exc())

def get_mix_pattern_occur(market_id: str, patterns: List, start_date:str=None,
                          end_date:str=None):
    _db = MimosaDBManager().current_db
    if not _db.is_initialized():
        return []

    pdata = _db.get_pattern_values(market_id, patterns)

    values = pdata.values
    dates = pdata.index.values.astype('datetime64[D]')
    ret = dates[(values==1).all(axis=1)]

    if start_date is not None:
        ret = ret[-(ret>=start_date).sum():]
        if (ret>=start_date).sum() == 0:
            ret = ret[:0]
    if end_date is not None:
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

    _db = MimosaDBManager().current_db
    markets = get_db().get_markets(market_type, category_code)
    if not markets or not _db.is_initialized():
        return 0, 0

    pvalues = [_db.get_pattern_values(mid, patterns).values for mid in markets]
    return func(pvalues)

def get_pattern_occur_cnt(pattern_id, market_type=None, category_code=None):
    return get_mix_pattern_occur_cnt([pattern_id], market_type, category_code)

def get_mix_pattern_rise_prob(patterns, period, market_type=None, category_code=None):
    def func(v, r):
        ret = r[(v==1).all(axis=1) & (r==r)]
        return len(ret), (ret>0).sum().tolist()

    _db = MimosaDBManager().current_db
    markets = get_db().get_markets(market_type, category_code)
    if not markets or not _db.is_initialized():
        return 0

    returns = [_db.get_future_returns(mid, [period]).values[:,0] for mid in markets]
    pvalues = [_db.get_pattern_values(mid, patterns).values for mid in markets]
    stats = np.array([func(v, r) for v, r in zip(pvalues, returns)])
    cnts, ups = stats.sum(axis=0).tolist()
    return (ups / cnts) * 100 if cnts > 0 else 0

def get_pattern_rise_prob(pattern_id, period, market_type=None, category_code=None):
    return get_mix_pattern_rise_prob([pattern_id], period, market_type, category_code)

def get_mix_pattern_mkt_dist_info(patterns, period, market_type=None, category_code=None):
    def func(v, r):
        ret = r[(v==1).all(axis=1) & (r==r)]
        if len(ret) == 0:
            return 0, 0, 0
        return ret.mean() * 100, ret.std() * 100, len(ret)

    _db = MimosaDBManager().current_db
    markets = get_db().get_markets(market_type, category_code)
    if not markets or not _db.is_initialized():
        return {}

    returns = [_db.get_future_returns(mid, [period]).values[:,0] for mid in markets]
    pvalues = [_db.get_pattern_values(mid, patterns).values for mid in markets]
    stats = np.array([func(v, r) for v, r in zip(pvalues, returns)])
    drops = ~np.isnan(stats).any(axis=1)
    markets = np.array(markets)[drops].tolist()
    stats = stats[drops]
    return {m: (v, r, int(c)) for m, (v, r, c) in zip(markets, stats)}

def get_pattern_mkt_dist_info(pattern_id, period, market_type=None, category_code=None):
    return get_mix_pattern_mkt_dist_info([pattern_id], period, market_type, category_code)

def add_pattern(pid):
    pattern = get_db().get_pattern_info(pid)

    _db = MimosaDBManager().current_db
    sid = get_db().set_pattern_execution_start(pid, PatternExecution.ADD_PATTERN)
    _db.add_pattern(pattern)
    latest_pattern_values = _db.get_latest_pattern_values([pid])

    th1 = CatchableTread(target=_db.dump)
    th1.start()
    th2 = CatchableTread(target=save_latest_pattern_results,
                         args=(latest_pattern_values, True))
    th2.start()
    th1.join()
    th2.join()
    get_db().set_pattern_execution_complete(sid)

def del_pattern_data(pattern_id:str):
    db = get_db()
    db.del_pattern_data(pattern_id)

def edit_pattern(pattern_id:str):
    del_pattern_data(pattern_id)
    add_pattern(pattern_id)
    return

def get_market_rise_prob(period, market_type=None, category_code=None):
    def func(r):
        ret = r[(r==r)]
        return len(ret), (ret>0).sum().tolist()

    _db = MimosaDBManager().current_db
    markets = get_db().get_markets(market_type, category_code)
    if not markets or not _db.is_initialized():
        return 0

    returns = [_db.get_future_returns(mid, [period]).values[:,0] for mid in markets]
    stats = np.array([func(r) for r in returns])
    cnts, ups = stats.sum(axis=0).tolist()
    return (ups / cnts) * 100 if cnts > 0 else 0

def get_mkt_dist_info(period, market_type=None, category_code=None):
    def func(r):
        ret = r[(r==r)]
        if len(ret) == 0:
            return 0, 0, 0
        return ret.mean() * 100, ret.std() * 100, len(ret)

    _db = MimosaDBManager().current_db
    markets = get_db().get_markets(market_type, category_code)
    if not markets or not _db.is_initialized():
        return {}

    returns = [_db.get_future_returns(mid, [period]).values[:,0] for mid in markets]
    stats = np.array([func(r) for r in returns])
    drops = ~np.isnan(stats).any(axis=1)
    markets = np.array(markets)[drops].tolist()
    stats = stats[drops]
    return {m: (v, r, int(c)) for m, (v, r, c) in zip(markets, stats)}

def get_market_price_dates(market_id: str, begin_date: Optional[datetime.date] = None):
    ret = []
    _db = MimosaDBManager().current_db
    if not _db.is_initialized():
        return []
    mdates = _db.get_market_dates(market_id)
    if len(mdates) == 0 or (begin_date is not None and mdates[-1] < begin_date):
        return []
    if begin_date:
        mdates = mdates[(mdates < begin_date).sum():]
    mdates = extend_working_dates(mdates, PREDICT_PERIODS[-1]).tolist()
    eidx = len(mdates) - PREDICT_PERIODS[-1]
    for period in PREDICT_PERIODS:
        ret += [{MarketPeriodField.DATA_DATE.value: a,
                 MarketPeriodField.PRICE_DATE.value: b,
                 MarketPeriodField.DATE_PERIOD.value: period
                 } for a, b in zip(mdates[:eidx],
                                   mdates[period:eidx+period])]
    return ret

def get_macro_params(func_code):
    db = get_db()
    return db.get_macro_param_type(func_code)

def check_macro_info(func):
    macro_info = get_macro_params(func)
    if 'market_id' in macro_info:
        del macro_info['market_id']

    params = eval(f'{func}.params')
    invalids = 0
    for each in params:
        if each.code not in macro_info:
            invalids += 1
        elif each.dtype.value.code != macro_info[each.code]:
            invalids +=1

    return invalids

def cast_macro_kwargs(func, macro_kwargs):
    macro_info = get_macro_params(func)
    if 'market_id' in macro_info:
        del macro_info['market_id']
    try:
        ret = {}
        msg = {}
        for key, value in macro_kwargs.items():
            if key not in macro_info:
                raise KeyError(f'keyword argument {key} not in macro info')
            macro_type = macro_info[key]
            is_valid = TypeCheckMap(macro_type).check(value)
            if is_valid:
                ret[key] = TypeCheckMap(macro_type).type(value)
            else:
                msg[key] = '參數型態錯誤'+f'，應為{TypeCheckMap(macro_type).cstring}'
        if len(macro_kwargs) != len(macro_info):
            raise KeyError('missing keyword arguments')
        return ret, msg

    except Exception as esp:
        raise esp

def verify_pattern(func, kwargs):
    pattern = PatternInfo.make(pid="", code=func, params=kwargs)
    return pattern.check()

def get_plot(func, kwargs):
    pattern = PatternInfo.make(pid="", code=func, params=kwargs)
    return pattern.plot()

def get_frame(func, kwargs):
    pattern = PatternInfo.make(pid="", code=func, params=kwargs)
    return pattern.frame()
