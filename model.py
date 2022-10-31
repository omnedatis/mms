# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

definition of MIMOSA python service functions

develop guilds
--------------

* function specifics
    dao-actions:

    api-referenced:

    batch-exclusive:

    queue-pushed:

    queue-prioritized-task:

* logging specification


* batch controller
"""
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple, defaultdict
import datetime
import logging
import os
import shutil
from threading import Lock
import time
import multiprocessing as mp
from typing import Any, List, NamedTuple, Optional, Dict, Callable, Tuple
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm
from sklearn.tree import DecisionTreeClassifier as Dtc
import traceback

from _core import Pattern as PatternInfo
from _core import View as ModelInfo
from _core import MarketInfo
from _core._macro import MacroManager, MacroParaEnumManager
from func.common import ParamType
from func._tp import *
from _view import backtest_view, create_view, update_view
from const import *
from utils import *
from const import (BatchType, MarketPeriodField, MarketScoreField,
                   ModelExecution, ModelMarketHitSumField, ModelStatus,
                   PatternExecution, PatternResultField, PredictResultField,
                   MIN_BACKTEST_LEN, PREDICT_PERIODS, BATCH_EXE_CODE, TaskCode)
from utils import extend_working_dates, CatchableTread, ThreadController
from _core import Pattern
from _db import get_dao, set_dao, MimosaDBManager


batch_lock = Lock()


task_queue = QueueManager({
    TaskCode.PATTERN: ExecQueue('pattern_queue'),
    TaskCode.MODEL: ExecQueue('model_queue')
})


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


def get_markets():
    db = get_db()
    minfo = db.get_market_info()
    cinfo = db.get_category_info()
    cinfo = {key: value for key, value in zip(cinfo.index.values.tolist(),
                                              cinfo.values.tolist())}
    ret = []
    for mid, (mtype, code) in zip(minfo.index.values, minfo.values):
        mcate = cinfo.get(code)
        ret.append(MarketInfo.make(mid, mtype, mcate))
    return ret


def del_pattern_data(pattern_id: str):
    logging.info('Deleting pattern data')
    get_db().del_pattern_data(pattern_id)


def del_view_execution(model_id: str):
    # ???
    # MT_MANAGER.acquire(model_id).switch_off()
    # MT_MANAGER.release(model_id)
    # while MT_MANAGER.exists(model_id):
    #     time.sleep(1)
    logging.info("Deleting view execution")
    get_db().del_model_execution(model_id)


def del_view_data(model_id: str):
    logging.info(f"Deleting view data predict history on {model_id}")
    get_db().del_model_data(model_id)
    logging.info(f"Deleting view data local on {model_id}")
    model_dir = ModelInfo.get_dir(model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

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
                if not controller.isactive:
                    break
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
    try:
        controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
        logging.info("Saving market score to db started")
        if len(recv) > 0:
            records = gen_records(recv)
            if len(records) > 0:
                if controller.isactive:
                    db = get_db()
                    db.save_mkt_score(pd.concat(records, axis=0))
        if not controller.isactive:
            logging.info("Saving market score to db terminated")
        else:
            logging.info("Saving market score to db finished")
    except Exception as esp:
        logging.info("Saving market score to db failed")
        MT_MANAGER.release(BATCH_EXE_CODE)
        raise esp


def save_latest_mkt_period(recv: Dict[str, pd.DataFrame]):
    """save latest mkt period to DB."""
    def gen_records(recv):
        ret = []
        for mid, cps in recv.items():
            if controller.isactive:
                # Note: without making `dropna` for performance
                periods = list(filter(lambda p: p < len(cps), PREDICT_PERIODS))
                if len(periods) <= 0:
                    continue
                cur = pd.DataFrame()
                offsets = list(map(lambda p: -(p+1), periods))
                dates = cps.index.values[offsets + [-1]
                                        ].astype('datetime64[D]').tolist()
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

    try:
        controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
        logging.info("Saving latest market period to db started")
        if len(recv) > 0:
            records = gen_records(recv)
            if len(records) > 0:
                if controller.isactive:
                    db = get_db()
                    db.save_latest_mkt_period(pd.concat(records, axis=0))
        if not controller.isactive:
            logging.info("Saving latest market period to db terminated")
        else:
            logging.info("Saving latest market period to db finished")
        MT_MANAGER.release(BATCH_EXE_CODE)
    except Exception as esp:
        logging.info("Saving latest market period to db failed")
        MT_MANAGER.release(BATCH_EXE_CODE)
        raise esp


def save_latest_pattern_results(recv: Dict[str, pd.DataFrame], update: bool = False):
    """Save latest pattern results to DB."""
    def trans2dbformat(recv):
        ret_buffer = []
        for mid, pdata in recv.items():
            if controller.isactive:
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
    try:
        controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
        logging.info('Saving lastest pattern result to db started')
        if len(recv) > 0:
            db = get_db()
            if update:
                for pid, data in trans2dbformat(recv).groupby(
                    by=PatternResultField.PATTERN_ID.value):
                    if not controller.isactive:
                        break
                    db.update_latest_pattern_results(pid, data)
            else:
                if controller.isactive:
                    db.save_latest_pattern_results(trans2dbformat(recv))
        if not controller.isactive:
            logging.info('Saving lastest pattern result to db terminated')
        else:
            logging.info('Saving lastest pattern result to db fnished')
        MT_MANAGER.release(BATCH_EXE_CODE)
    except Exception as esp:
        logging.info('Saving lastest pattern result to db failed')
        MT_MANAGER.release(BATCH_EXE_CODE)
        raise esp


def _evaluate_hit_sum(market: str, results: Dict[str, pd.DataFrame], freturns):
    ret = pd.DataFrame()
    ret[ModelMarketHitSumField.DATE_PERIOD.value] = PREDICT_PERIODS
    ret[ModelMarketHitSumField.MARKET_CODE.value] = market
    if results is None:
        ret[ModelMarketHitSumField.HIT.value] = 0
        ret[ModelMarketHitSumField.FCST_CNT.value] = 0
        return ret

    results = results.set_index(PredictResultField.DATE.value)
    result_group = {p: d for p, d in results.groupby(
        PredictResultField.PERIOD.value)}
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
    results: Dict[str, pd.DataFrame] = get_db().get_model_results(model_id)
    recv = [_evaluate_hit_sum(market, results.get(market),
                              _db.get_future_returns(market, PREDICT_PERIODS)
                              ) for market in _db.get_markets()]
    if len(recv) > 0:
        ret = pd.concat(recv, axis=0)
        ret['MODEL_ID'] = model_id
        return ret


def save_model_hit_sum(model_id: str, batch_type: BatchType):
    """Save model hit sum to DB."""
    recv = get_model_hit_sum(model_id, batch_type)
    if recv is not None:
        db = get_db()
        if batch_type == BatchType.INIT_BATCH:
            db.update_model_hit_sum(recv)
        else:
            db.save_model_hit_sum(recv)


def _get_backtest_length(market: str, earlist_date: datetime.date):
    db = get_db()
    dates = db.get_market_data(market).index.values.astype('datetime64[D]')
    ret = min([MIN_BACKTEST_LEN, len(dates)]) - (dates >= earlist_date).sum() #???
    return ret


def _db_update(batch_type: BatchType = BatchType.SERVICE_BATCH):
    markets = get_markets()
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
                               args=({each.mid: _db.get_market_prices(each.mid)
                                      for each in markets}, )),
                CatchableTread(target=save_mkt_score,
                               args=({each.mid: _db.get_future_returns(each.mid)
                                      for each in markets}, ))]
        for exec_id in exec_ids:
            get_db().set_pattern_execution_complete(exec_id)
    for t in ret:
        t.start()
    return ret


def _batch_recover_executions():
    logging.info('Batch view execution recover started')
    try:
        exec_info = get_db().get_recover_model_execution()
    except:
        logging.info('Batch view execution recover failed')
    for model, etype in exec_info:
        if etype == ModelExecution.ADD_PREDICT:
            status = ModelStatus.ADDED
        elif etype == ModelExecution.ADD_BACKTEST:
            status = ModelStatus.CREATED
        else:
            raise RuntimeError(f'Invalid model execution type {etype} ecountered')
        _batch_recover_views(model, status)
    else:
        logging.info('Batch view execution recover finished')
    logging.info('Batch view execution recover terminated')


def _batch_recover_views(model_id: str, status: ModelStatus):
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
    try:
        logging.info(f"Recovering on view {model_id} started")
        controller = MT_MANAGER.acquire(model_id)
        model = get_db().get_model_info(model_id)
        # the following thrown error will be catched
        if status < ModelStatus.CREATED:
            del_view_execution(model_id)  
            del_view_data(model_id) 
            _batch_create_view(model, controller)
        _batch_backtest_view(model, controller) 
        save_model_hit_sum(model_id, BatchType.INIT_BATCH)
        MT_MANAGER.release(model_id)
    except Exception as esp:
        logging.error(f"Recovering on view {model_id} failed")
        MT_MANAGER.release(model_id)
        raise esp


def _batch_create_view(model: ModelInfo, controller: ThreadController):
    batch_controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
    logging.info(f'Creating on view {model.view_id} started')
    if controller.isactive and batch_controller.isactive:
        exection_id = get_db().set_model_execution_start(
            model.view_id, ModelExecution.ADD_PREDICT)
    if controller.isactive and batch_controller.isactive:
        recv = create_view(model, controller)
    if controller.isactive and batch_controller.isactive:
        get_db().save_model_results(model.view_id, recv.dropna(), ModelExecution.ADD_PREDICT)
        get_db().set_model_execution_complete(exection_id)
        get_db().stamp_model_execution([exection_id])
    if not batch_controller.isactive:
        logging.info('Batch terminated')
        MT_MANAGER.release(BATCH_EXE_CODE)
        return
    if not controller.isactive:
        logging.info(f'Creating on view {model.view_id} terminated')
    else:
        logging.info(f'Creating on view {model.view_id} finished')


def _batch_backtest_view(model: ModelInfo, controller: ThreadController):
    batch_controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
    logging.info(f'Backtesting on view {model.view_id} started')
    exection_id = get_db().set_model_execution_start(
        model.view_id, ModelExecution.ADD_BACKTEST)
    earlist_dates = get_db().get_earliest_dates(model.view_id)
    cur_thread = None
    while controller.isactive:
        if not batch_controller.isactive:
            break
        earlist_dates = {market: date for market, date in earlist_dates.items()
                        if _get_backtest_length(market, date) > 0}

        if len(earlist_dates) == 0:
            # no more market has non-zero backtest length
            cur_thread and cur_thread.join()
            break
        if not batch_controller.isactive:
            break
        recv = backtest_view(model, earlist_dates, controller)
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
        logging.info('Backtesting on view {model.view_id} terminated')
        return

    # 回測完成，更新指定模型在DB上的狀態為'COMPLETE'
    smd = get_db().set_model_train_complete(model.view_id)
    get_db().set_model_execution_complete(exection_id)
    get_db().stamp_model_execution([exection_id, smd])
    logging.info('Backtesting on view {model.view_id} finished')


def _batch_update_views():
    class ModelUpdateMoniter(NamedTuple):
        controller:ThreadController
        exec_id:str
        smd:bool
        complete_exce_id:str
        thread:CatchableTread
    def save_result(data: pd.DataFrame, model_id: str, exec_id: str,
                    controller: ThreadController):
        if controller.isactive:
            if data is not None:
                get_db().save_model_results(
                    model_id, data.dropna(), ModelExecution.BATCH_PREDICT)
        if controller.isactive:
            get_db().set_model_execution_complete(exec_id)
        if controller.isactive:
            save_model_hit_sum(model_id, BatchType.SERVICE_BATCH)
    try:
        batch_controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
        logging.info('Updating view on started')
        moniters: Dict[str, ModelUpdateMoniter] = {}
        for model_id in get_db().get_models():
            controller = MT_MANAGER.acquire(model_id)
            logging.info(f'Updating view on {model_id} started')
            if batch_controller.isactive and controller.isactive:
                exec_id = get_db().set_model_execution_start(
                    model_id, ModelExecution.BATCH_PREDICT)
            if batch_controller.isactive and controller.isactive:
                recv, smd = _batch_update_view(model_id, controller)
            if batch_controller.isactive and controller.isactive:
                if smd:
                    complete_exce_id:str = get_db().set_model_train_complete(model_id)
                thread = CatchableTread(target=save_result, 
                                        args=(recv, model_id, exec_id, controller))
                thread.start()
                moniters[model_id] = ModelUpdateMoniter(
                    controller, exec_id, smd, complete_exce_id, thread)
                logging.info(f'Updaing view on {model_id} finished')
            if not controller.isactive:
                logging.info('Updating view on {model_id} terminated')
            MT_MANAGER.release(model_id)
        if not batch_controller.isactive:
            logging.info('Batch terminated')
            MT_MANAGER.release(model_id)
            return

        exec_ids = []
        for model_id, moniter in moniters.items():
            batch_controller = MT_MANAGER.acquire(model_id)
            logging.info(f'Joining view thread on {model_id}')
            moniter.thread.join()
            if moniter.controller.isactive and batch_controller.isactive:
                if moniter.thread.esp is None:
                    exec_ids.append(moniter.exec_id)
                    if moniter.smd:
                        exec_ids.append(moniter.complete_exce_id) 
            if not moniter.controller.isactive:
                logging.info(f'Joining view thread on {model_id} terminated')
            MT_MANAGER.release(model_id)
        if not batch_controller.isactive:
            logging.info('Batch terminated')
            MT_MANAGER.release(BATCH_EXE_CODE)
            return []
        else:
            logging.info(f'Updating view finished')
            return exec_ids
    except Exception as esp:
        logging.info(f'Updating view failed')
        MT_MANAGER.release(BATCH_EXE_CODE)
        raise esp


def _batch_update_view(model_id: str, controller: ThreadController,
        ) -> Tuple[Union[pd.DataFrame, None], bool]:
    latest_dates = get_db().get_latest_dates(model_id)
    recv, is_retrained = update_view(get_db().get_model_info(
        model_id), latest_dates, controller)  
    if recv is not None:
        recv = recv.dropna()
    return recv, is_retrained


def _batch_del_view_data():
    try:
        controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
        logging.info("Batch deleting view data started")
        for model_id in get_db().get_removed_model():
            if controller.isactive:
                del_view_data(model_id)
            else:
                break
        else:
            MT_MANAGER.release(BATCH_EXE_CODE)
            logging.info("Batch deleting view data finished")
            return
        MT_MANAGER.release(BATCH_EXE_CODE)
        logging.info("Batch deleting view data terminated")
    except Exception as esp:
        logging.info("Batch deleting view data falied")
        MT_MANAGER.release(BATCH_EXE_CODE)
        raise esp
    

def _init_db():
    try:
        batch_type = BatchType.INIT_BATCH
        controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
        logging.info('Batch init started')
        clean_db_cache()
        clone_db_cache(batch_type)
        model_prepare_thread = get_db().clone_model_results(controller)
        #
        logging.info("Batch pattern update started")
        if not MimosaDBManager().is_ready():
            _db_update(batch_type)
        logging.info("Batch pattern update finished")
        #
        model_prepare_thread.join()
        _batch_del_view_data()
        _batch_recover_executions()
        logging.info('Batch init finished')
    except Exception as esp:
        logging.error("Batch init failed")
        logging.error(traceback.format_exc())
        MT_MANAGER.release(BATCH_EXE_CODE)


def init_db():
    task_queue.do_prioritized_task(
        _init_db, name='init_batch')

def _batch():
    try:
        batch_type = BatchType.SERVICE_BATCH
        controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
        logging.info("Batch service started")
        clean_db_cache()
        clone_db_cache(batch_type)
        model_prepare_thread = get_db().clone_model_results(controller)
        #
        logging.info("Batch pattern update started")
        threads = []
        get_db().truncate_swap_tables()
        threads = _db_update(batch_type) or []
        logging.info("Bacth pattern update finished") # should not be here
        #
        model_prepare_thread.join()
        _batch_del_view_data()
        model_exec_ids = _batch_update_views()
        for t in threads:
            if controller.isactive:
                t.join()
        if controller.isactive:
            get_db().checkout_fcst_data()
            get_db().stamp_model_execution(model_exec_ids)
            MimosaDBManager().swap_db()
            logging.info("Batch finished")
            MT_MANAGER.release(BATCH_EXE_CODE)
            return
        logging.info("Batch Terminated")
        MT_MANAGER.release(BATCH_EXE_CODE)

    except Exception:
        logging.error("Batch failed")
        logging.error(traceback.format_exc())
        MT_MANAGER.release(BATCH_EXE_CODE)


def batch():
    now = datetime.datetime.now()
    date = now.date().day
    hour = now.hour
    minute = now.minute
    sec =  now.second
    task_queue.do_prioritized_task(
        _batch, name=f'service_batch_{date}{hour}{minute}{sec}')
    


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
        logging.info('Adding view started')
        del_view_data(model_id)
        model = get_db().get_model_info(model_id)
        _batch_create_view(model, controller)
        _batch_backtest_view(model, controller)
        save_model_hit_sum(model_id, BatchType.INIT_BATCH)
        MT_MANAGER.release(model_id)
    except Exception as esp:
        logging.error("Adding view failed ")
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

    """
    logging.info('removing view started')
    MT_MANAGER.acquire(model_id).switch_off()
    MT_MANAGER.release(model_id)
    while MT_MANAGER.exists(model_id):
        time.sleep(1)
    del_view_data(model_id)
    logging.info('removing mviewodel started')


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


def get_mix_pattern_occur(market_id: str, patterns: List, start_date: str = None,
                          end_date: str = None):
    _db = MimosaDBManager().current_db
    if not _db.is_initialized():
        return []

    pdata = _db.get_pattern_values(market_id, patterns)

    values = pdata.values
    dates = pdata.index.values.astype('datetime64[D]')
    ret = dates[(values == 1).all(axis=1)]

    if start_date is not None:
        ret = ret[-(ret >= start_date).sum():]
        if (ret >= start_date).sum() == 0:
            ret = ret[:0]
    if end_date is not None:
        ret = ret[:(ret <= end_date).sum()]
    return ret.tolist()


def get_patterns_occur_dates(market_id: str, patterns: List[str], start_date: str = None,
                             end_date: str = None) -> List[Dict[str, str]]:
    """取得指定市場, 指定時間區段複數現象的發生時間

    Parameters
    ----------
    market_id: str
        指定市場的市場ID
    patterns: List[str]
        要查看發生時間點的現象 ID 清單
    start_date: str
        查看時間起始日
    end_date: str
        查看時間終止日

    Returns
    -------
    results: List[Dict[str, str]]
        各現象發生時間點
    """
    _db = MimosaDBManager().current_db
    if not _db.is_initialized():
        return []

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    results = []
    for pattern in patterns:
        pdata = _db.get_pattern_values(market_id, [pattern])

        values = pdata.values
        dates = pdata.index.values.astype('datetime64[D]')
        ret = dates[(values == 1).all(axis=1)]

        if start_date is not None:
            ret = ret[-(ret >= start_date).sum():]
            if (ret >= start_date).sum() == 0:
                ret = ret[:0]
        if end_date is not None:
            ret = ret[:(ret <= end_date).sum()]
        dates: List[datetime.date] = ret.tolist()
        result = [{'patternId': pattern, "occurDate": date.strftime(
            '%Y-%m-%d')} for date in dates]
        results += result
    return results


def get_pattern_occur(market_id: str, pattern_id):
    return get_mix_pattern_occur(market_id, [pattern_id])


def get_mix_pattern_occur_cnt(patterns, markets, begin_date=None, end_date=None):
    def func(vs):
        recv = np.array(
            [((v >= 0).all(axis=1).sum(), (v == 1).all(axis=1).sum()) for v in vs])
        if len(vs) <= 0:
            return 0, 0
        cnts, occurs = recv.sum(axis=0).tolist()
        return occurs, cnts - occurs

    _db = MimosaDBManager().current_db
    if not markets or not _db.is_initialized():
        return 0, 0
    pvalues = [_db.get_pattern_values(mid, patterns) for mid in markets]
    pdata = []
    for each_p in pvalues:
        dates = each_p.index.values.astype('datetime64[D]')
        ret = each_p.values
        if begin_date is not None:
            ret = ret[-(dates >= begin_date).sum():]
            if (dates >= begin_date).sum() == 0:
                ret = ret[:0]
        if end_date is not None:
            ret = ret[:(dates <= end_date).sum()]
        pdata.append(ret)
    return func(pdata)


def _get_mix_pattern_rise_prob(markets, patterns, period):
    def func(v, r):
        ret = r[(v == 1).all(axis=1) & (r == r)]
        return (len(r), (r > 0).sum().tolist(), (r < 0).sum().tolist(),
                len(ret), (ret > 0).sum().tolist(), (ret < 0).sum().tolist())

    _db = MimosaDBManager().current_db
    if not _db.is_initialized():
        return 0, 0, 0, 0, 0, 0
    returns = [_db.get_future_returns(
        mid, [period]).values[:, 0] for mid in markets]
    pvalues = [_db.get_pattern_values(mid, patterns).values for mid in markets]
    stats = np.array([func(v, r) for v, r in zip(pvalues, returns)])
    mcnt, mup, mdown, pcnt, pup, pdown = stats.sum(axis=0).tolist()
    return mcnt, mup, mdown, pcnt, pup, pdown


def get_mix_pattern_rise_prob(markets, patterns):
    market_rise = {'dataType': 'marketAverage',
                   'upDownType': 'rise',
                   'values': []}
    market_fall = {'dataType': 'marketAverage',
                   'upDownType': 'fall',
                   'values': []}
    pattern_rise = {'dataType': 'patternOccur',
                    'upDownType': 'rise',
                    'values': []}
    pattern_fall = {'dataType': 'patternOccur',
                    'upDownType': 'fall',
                    'values': []}
    for period in PREDICT_PERIODS:
        (mcnt, mup, mdown, pcnt, pup, pdown
         ) = _get_mix_pattern_rise_prob(markets, patterns, period)
        market_rise['values'].append({'datePeriod': period,
                                      'prob': mup / mcnt * 100 if mcnt > 0 else 0})
        market_fall['values'].append({'datePeriod': period,
                                      'prob': mdown / mcnt * 100 if mcnt > 0 else 0})
        pattern_rise['values'].append({'datePeriod': period,
                                       'prob': pup / pcnt * 100 if pcnt > 0 else 0})
        pattern_fall['values'].append({'datePeriod': period,
                                       'prob': pdown / pcnt * 100 if pcnt > 0 else 0})
    return [market_rise, market_fall, pattern_rise, pattern_fall]


def get_occurred_patterns(date, patterns):
    _db = MimosaDBManager().current_db
    markets = _db.get_markets()
    if not markets or not _db.is_initialized():
        return []
    date = np.datetime64(date)
    pvalues = []
    for mid in markets:
        try:
            pvalues.append(_db.get_pattern_values(mid, patterns).loc[date])
        except KeyError:
            continue
    if len(pvalues) == 0:
        return []
    pvalues = pd.concat(pvalues, axis=1).fillna(False).astype(bool).T
    ret = pvalues.columns.values[pvalues.values.any(axis=0)].tolist()
    return ret


def get_mix_pattern_mkt_dist_info(patterns, period, markets: List[str]) -> List[Dict[str, Any]]:
    def func(v, r):
        ret = r[(v == 1).all(axis=1) & (r == r)]
        return ret

    _db = MimosaDBManager().current_db
    if not markets or not _db.is_initialized():
        return {}

    returns = [_db.get_future_returns(
        mid, [period]).values[:, 0] for mid in markets]
    pvalues = [_db.get_pattern_values(mid, patterns).values for mid in markets]
    # 發生後
    market_occured_future_rets = np.concatenate(
        [func(v, r) for v, r in zip(pvalues, returns)], axis=0)
    drops = ~np.isnan(market_occured_future_rets)
    market_occured_future_rets: np.ndarray = market_occured_future_rets[drops]
    market_occured_future_rets.sort()
    # 全歷史報酬
    future_rets = np.concatenate(returns, axis=0)
    if len(future_rets) == 0:
        return []
    drops = ~np.isnan(future_rets)
    future_rets: np.ndarray = future_rets[drops]
    future_rets.sort()

    size = 100
    diff = (future_rets[-1] - future_rets[0])/size
    segments = []
    for i in range(1, size+1):
        if len(segments) == 0:
            segments.append(future_rets[0])
        else:
            segments.append(segments[-1]+diff)

    segs = []
    for i in range(1, len(segments)):
        min = segments[i-1]
        max = segments[i]
        p_seg = market_occured_future_rets[
            (market_occured_future_rets >= min) &
            (market_occured_future_rets < max)]
        seg = future_rets[
            (future_rets >= min) &
            (future_rets < max)
        ]
        if len(market_occured_future_rets) > 0:
            segs.append({
                'type': "pattern",
                'rangeUp': max,
                'rangeDown': min,
                'name': np.round(min, 1),
                'value': np.round(len(p_seg)/len(market_occured_future_rets) * 100, 2)
            })
        segs.append({
            'type': "market",
            'rangeUp': max,
            'rangeDown': min,
            'name': np.round(min, 1),
            'value': np.round(len(seg)/len(future_rets) * 100, 2)
        })
    return segs


def get_pattern_mkt_dist_info(pattern_id, period, market_type=None, category_code=None):
    return get_mix_pattern_mkt_dist_info([pattern_id], period, market_type, category_code)


def get_market_rise_prob(period, market_type=None, category_code=None):
    def func(r):
        ret = r[(r == r)]
        return len(ret), (ret > 0).sum().tolist()

    _db = MimosaDBManager().current_db
    markets = _db.get_markets(market_type, category_code)
    if not markets or not _db.is_initialized():
        return 0

    returns = [_db.get_future_returns(
        mid, [period]).values[:, 0] for mid in markets]
    stats = np.array([func(r) for r in returns])
    cnts, ups = stats.sum(axis=0).tolist()
    return (ups / cnts) * 100 if cnts > 0 else 0


def get_mkt_dist_info(period, markets: List[str]) -> List[Dict[str, Any]]:

    _db = MimosaDBManager().current_db
    if not markets or not _db.is_initialized():
        return {}

    returns = [_db.get_future_returns(
        mid, [period]).values[:, 0] for mid in markets]
    future_rets = np.concatenate(returns, axis=0)
    drops = ~np.isnan(future_rets)
    future_rets: np.ndarray = future_rets[drops]
    future_rets.sort()

    size = 100
    diff = (future_rets[-1] - future_rets[0])/size
    segments = []
    for i in range(1, size+1):
        if len(segments) == 0:
            segments.append(future_rets[0])
        else:
            segments.append(segments[-1]+diff)

    segs = []
    for i in range(1, len(segments)):
        min = segments[i-1]
        max = segments[i]
        seg = future_rets[(future_rets >= min) & (future_rets < max)]
        segs.append({
            'type': "market",
            'rangeUp': max,
            'rangeDown': min,
            'name': np.round(min, 1),
            'value': np.round(len(seg)/len(future_rets) * 100, 2)
        })

    return segs


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
    """Check mismatch of definitions between code and db."""
    macro_info = get_macro_params(func)
    if 'market_id' in macro_info:
        del macro_info['market_id']

    params = MacroManager.get(func).parameters
    invalids = 0
    for each in params:
        # senario 1: definition cannot be found in db
        if each.code not in macro_info:
            invalids += 1
        # senario 2: definition type_code does not match what found in db
        if each.dtype.code != macro_info[each.code]:
            invalids += 1

    return invalids


def cast_macro_kwargs(func, macro_kwargs):
    """Check if exist invalid or missing arguments,
       and cast marcro params type accordingly."""
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
            if MacroParaEnumManager.get(macro_type, value) is not None:
                valid_param = MacroParaEnumManager.get(macro_type, value)
            elif ParamType.get(macro_type) is not None:
                try:
                    valid_param = ParamType.get(macro_type).value.type(value)
                except ValueError:
                    valid_param = None
            if valid_param is not None:
                ret[key] = valid_param
            else:
                msg[key] = '參數型態錯誤'
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


def get_draft_date(func_code: str, kwargs: Dict[str, Any], market_id: str,
                   start_date: Optional[datetime.date] = None,
                   end_date: Optional[datetime.date] = None):
    _db = MimosaDBManager().current_db
    if not _db.is_initialized():
        return []

    pattern = PatternInfo.make('draft_ptn', func_code, kwargs)
    pdata = _db.gen_pattern_data(market_id, pattern)
    index = pdata.index.values.astype('datetime64[D]')
    ret = index[pdata.values == True]
    if start_date is not None:
        ret = ret[ret >= start_date]
    if end_date is not None:
        ret = ret[ret <= end_date]
    return ret.tolist()


def get_mkt_trend_score(market_id: str,
                        start_date: Optional[datetime.date] = None,
                        end_date: Optional[datetime.date] = None,
                        dates: Optional[List[datetime.date]] = None
                        ) -> pd.DataFrame:
    """取得一組或一段指定時間, 指定市場 ID 的各指定天期未來趨勢

    Parameters
    ----------
    market_id: str
        要取得的市場 ID
    start_date: datetime.date
        要取得資料的起始時間(包含)
    end_date: datetime.date
        要取得資料的結束時間(包含)
    dates: list of datetime.date
        要取得資料的時間
    periods: List[int]
        要取得趨勢的天期

    Returns
    -------
    results: pd.DataFrame
        指定一組或一段時間指定市場的各指定天期未來趨勢, Index 為日期, 欄位有 CP, 各天期未來趨勢
    """
    _db = MimosaDBManager().current_db
    future_rets = _db.get_future_returns(market_id)
    if dates is not None:
        start_date = min(dates)
        end_date = max(dates)
    sidx = (0 if start_date is None else
            (future_rets.index.values.astype('datetime64[D]') < start_date).sum())
    eidx = (len(future_rets) if end_date is None else
            (future_rets.index.values.astype('datetime64[D]') <= end_date).sum())
    cps = _db.get_market_prices(market_id)[sidx:eidx]
    scores, lfactor, ufactor = list(zip(*get_db().get_score_meta_info()))
    lfactor = np.array(lfactor)
    ufactor = np.array(ufactor)
    result = {p: {} for p in PREDICT_PERIODS}
    idxs = []
    if dates is not None:
        dates = np.array(dates).astype('datetime64[D]')
    for i in range(sidx, eidx):
        date = future_rets.index.values[i]
        if dates is not None and date not in dates:
            idxs.append(False)
            continue
        idxs.append(True)
        record = future_rets.iloc[i]
        _rets = future_rets[:i+1]
        for period, freturn in zip(PREDICT_PERIODS, _rets.values.T):
            if len(freturn) <= period:
                result[period][date] = np.nan
                continue
            freturn = freturn[:-period]
            mean = freturn.mean()
            std = freturn.std(ddof=1)

            ret = record[period]
            uppers = ufactor * std + mean
            lowers = lfactor * std + mean
            for score, upper, lower in zip(scores, uppers, lowers):
                if np.isnan(lower) and ret <= upper:
                    result[period][date] = score
                    break
                elif np.isnan(upper) and ret >= lower:
                    result[period][date] = score
                    break
                elif ret >= lower and ret <= upper:
                    result[period][date] = score
                    break
    if dates is not None:
        cps = cps[idxs]
    result = pd.DataFrame(result)
    result = pd.concat([cps, result], axis=1)
    return result
