# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

definition of MIMOSA python service functions

develop guilds
--------------

* function specifics
    db-actions: db action that as no side effects on assigned db
        get_db, set_db, clean_db_cache, clone_db_cache, get_markets

    db-wb-actions: create/update/delete action of assigned db
        add_model, remove_model, add_pattern, del_pattern_data,
        del_view_execution, del_view_data

    module-privates:
        prefixed by ${_}

    batch-related:
        prefixed by ${_batch}

    api-referenced:
        public function for server api's

    queue-pushed:
        add_model, remove_model, add_pattern
        NOTE: Handle by `CatchableThread`

    queue-prioritized-task:
        batch, init_db
        NOTE: NOT handle by `CatchableThread`, need its own try-catch

* logging specification
    task: Union[noun, present-particle sentence]
    description: Optional[Any]
    state:literal[started, finished, failed, terminated]
    logging.(info/error/debug/...)(f'{task}{description}{state})
    NOTE:For the flow of the task to be fully logged, the function needs to
         have its own try-catch to differentiate finished, failed, terminated,
         i.e.

* thread controller
    NOTE:There are batch controller in not batch exclusive function, but it
         woundn't affect the expected flow (not necessarily so in the future).
         (add_model and add_pattern would not be trigger in batch or init, so
         it wouldn't be terminated by batch controller. The two controller
         should be exculsive, or the function called be separated.)
"""
import datetime
import glob
import logging
import os
import shutil
import time
from typing import Any, List, NamedTuple, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as Dtc
import traceback
from werkzeug.exceptions import BadRequest

from _core import Pattern as PatternInfo
from _core import View as ModelInfo
from _core import MarketInfo
from _core._macro import MacroManager, MacroParaEnumManager
from _core._view.model import Labelization
from func.common import ParamType
from func._tp import *
from const import *
from utils import *
from const import (BatchType, MarketPeriodField, MarketScoreField, 
                   ModelExecution, ModelMarketHitSumField, ModelStatus,
                   PatternExecution, PatternResultField, PredictResultField,
                   MIN_BACKTEST_LEN, PREDICT_PERIODS, BATCH_EXE_CODE, TaskCode,
                   Y_LABELS, Y_OUTLIER)
from utils import (extend_working_dates, CatchableTread, ThreadController,
                   pickle_dump, pickle_load)
from _core import Pattern
from _db import get_dao, set_dao, MimosaDBManager, MimosaDB
from collections import defaultdict
import multiprocessing as mp
from threading import Thread, Lock
from _core._view.model import COMMON_DECISION_TREE_PARAMS, Labelization, MIN_Y_SAMPLES


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


def get_markets() -> List[MarketInfo]: # why not dao method ?
    "Get all the markets and its information"
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
    ViewManagerFactory.get()._add(model_id)


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
    ViewManagerFactory.get()._remove(model_id)


def edit_model(model_id):
    """JKJKJKJKJK

    """
    ViewManagerFactory.get()._remove(model_id)
    time.sleep(1)
    ViewManagerFactory.get()._add(model_id)

def add_pattern(pid):
    pattern = get_db().get_pattern_info(pid)

    _db = MimosaDBManager().current_db
    sid = get_db().set_pattern_execution_start(pid, PatternExecution.ADD_PATTERN)
    _db.add_pattern(pattern)
    latest_pattern_values = _db.get_latest_pattern_values([pid])

    th1 = CatchableTread(target=_db.dump)
    th1.start()
    th2 = CatchableTread(target=_save_latest_pattern_results,
                         args=(latest_pattern_values, True))
    th2.start()
    th1.join()
    th2.join()
    get_db().set_pattern_execution_complete(sid)


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


def _save_latest_pattern_results(recv: Dict[str, pd.DataFrame], update: bool = False):
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
        controller = mt_manager.acquire(BATCH_EXE_CODE)
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
            logging.info('Batch terminated')
        else:
            logging.info('Saving lastest pattern result to db fnished')
        mt_manager.release(BATCH_EXE_CODE)
    except Exception as esp:
        logging.error(traceback.format_exc())
        logging.error('Saving lastest pattern result to db failed')
        mt_manager.release(BATCH_EXE_CODE)
        raise esp



def _batch_db_update(batch_type: BatchType) -> List[ThreadController]:
    try:
        controller = mt_manager.acquire(BATCH_EXE_CODE)
        logging.info('DB update started')
        markets = get_markets() # henceforth, ...
        patterns = get_db().get_patterns()
        if batch_type == BatchType.SERVICE_BATCH:
            _db = MimosaDBManager().next_db
        else:
            _db = MimosaDBManager().current_db
         # ... can be writen in the function to form a standalone step
        _db.update(markets, patterns, batch_type)
        db_dump = [CatchableTread(target=_db.dump)]
        db_dump[0].start()
        ret:List[ThreadController] = []
        if batch_type == BatchType.SERVICE_BATCH:
            exec_type = PatternExecution.BATCH_SERVICE # redundant definition in const
            exec_ids = []
            for each in patterns:
                if not controller.isactive:
                    break
                exec_ids.append(get_db().set_pattern_execution_start(each.pid, exec_type))
            ret += [CatchableTread(target=_save_latest_pattern_results, args=(
                _db.get_latest_pattern_values(),))]
            for exec_id in exec_ids:
                if not controller.isactive:
                    break
                get_db().set_pattern_execution_complete(exec_id)
        for t in ret:
            if not controller.isactive:
                break
            t.start()
        if controller.isactive:
            logging.info('DB update finished')  
        else:
            logging.info('DB update terminated')
        mt_manager.release(BATCH_EXE_CODE)
        if batch_type == BatchType.SERVICE_BATCH:
            return ret + db_dump
        if batch_type == BatchType.INIT_BATCH:
             return db_dump
    except Exception as esp:
        logging.error(traceback.format_exc())
        logging.error('DB update failed')
        mt_manager.release(BATCH_EXE_CODE)
        raise esp


def _batch_recover_executions():
    logging.info('Recovering view execution started')
    for model in get_db().get_recover_models():
        logging.info(f"Calling recover view: {model}")
        ViewManagerFactory.get()._add(model)
    else:
        logging.info('Recovering view execution finished')


def _batch_del_view_data():
    try:
        logging.info("Removing view data started")
        for model_id in get_db().get_removed_model():
            logging.info(f"Calling remove view: {model_id}")
            ViewManagerFactory.get()._remove(model_id)
        else:
            logging.info("Removing view data finished")
            return
    except Exception as esp:
        logging.info("Removing view data falied")
        raise esp

def _get_x_data(db_: MimosaDB, markets: List[str], patterns: List[str],
                controller: ThreadController) -> Dict[str, pd.DataFrame]:
    # Note: Bad performance!!
    ret = {}
    for market in markets:
        if not controller.isactive:
            return {}
        ret[market] = db_.get_pattern_values(market, patterns)
    return ret

def _get_y_data(db_: MimosaDB, markets: List[str], period: int,
                controller: ThreadController) -> Dict[str, pd.Series]:
    # Note: Bad performance!!
    ret = {}
    for market in markets:
        if not controller.isactive:
            return {}
        ret[market] = db_.get_future_returns(market, [period])[period]
    return ret

def _get_train_dataset(db_: MimosaDB, markets: List[str], patterns: List[str],
                       tdate: datetime.date, edate: datetime.date, period: int,
                       controller: ThreadController, min_samples: int=MIN_Y_SAMPLES
                       ) -> Tuple[Dict[str, pd.DataFrame],
                                  Dict[str, pd.DataFrame]]:
    if markets is None or len(markets) <= 0:
        markets = db_.get_markets()
    x_train = _get_x_data(db_, markets, patterns, controller)
    y_train = _get_y_data(db_, markets, period, controller)
    ret_x = []
    ret_y = []
    for market in markets:
        if not controller.isactive:
            return np.array([]), np.array([])
        dates = x_train[market].index.values.astype('datetime64[D]')
        sidx = (dates < tdate).sum()
        eidx = (dates < edate).sum() - period
        eidx = max([sidx, eidx])
        if eidx > sidx:
            values = pd.concat([x_train[market][sidx: eidx],
                                y_train[market][sidx: eidx]], axis=1).dropna().values
            if len(values) >= MIN_Y_SAMPLES:
                ret_x.append(values[:,:-1])
                ret_y.append(Labelization(Y_LABELS).fit_transform(values[:, -1],
                                                                  Y_OUTLIER))
    if not controller.isactive or len(ret_x) == 0 or len(ret_y) == 0:
        return np.array([]), np.array([])
    return np.concatenate(ret_x, axis=0), np.concatenate(ret_y, axis=0)


class ViewManagerFactory:
    _INSTANCE = None
    @classmethod
    def get(cls, cpus: int=6):
        if cls._INSTANCE is None:
            cls._INSTANCE = _ViewManager(cpus)
        else:
            cls._INSTANCE.set_cpus(cpus)
        return cls._INSTANCE

class _ViewManager:
    def __init__(self, cpus:int=6):
        self._controllers = {}
        self._requires = defaultdict(int)
        self._lock = Lock()
        self._tasks = defaultdict(list)
        self._queue = []
        self._max_cpus = cpus
        self._cpus = 0

    def set_cpus(self, cpus: int):
        self._max_cpus = cpus

    def _acquire(self, view_id: str, locked=False):
        if locked:
            if view_id not in self._controllers:
                self._controllers[view_id] = ThreadController()
            ret = self._controllers[view_id]
            self._requires[view_id] += 1
        else:
            self._lock.acquire()
            if view_id not in self._controllers:
                self._controllers[view_id] = ThreadController()
            ret = self._controllers[view_id]
            self._requires[view_id] += 1
            self._lock.release()
        return ret

    def _release(self, view_id: str, locked=False):
        if locked:
            if view_id not in self._controllers:
                raise RuntimeError('release an not existed controller')
            self._requires[view_id] -= 1
        else:
            self._lock.acquire()
            if view_id not in self._controllers:
                raise RuntimeError('release an not existed controller')
            self._requires[view_id] -= 1
            self._lock.release()

    def _isready(self, view_id, locked=False):
        if locked:
            ret = self._requires[view_id] == 0
        else:
            self._lock.acquire()
            ret = self._requires[view_id] == 0
            assert self._requires[view_id] >= 0
            self._lock.release()
        return ret

    def _remove(self, view_id):
        self._lock.acquire()
        self._tasks[view_id] = ['r']
        if view_id in self._queue:
            self._queue.remove(view_id)
        self._lock.release()
        self._acquire(view_id).switch_off()
        self._release(view_id)
        while not self._isready(view_id):
            time.sleep(1)
        self._lock.acquire()
        self._queue.append(view_id)
        self._lock.release()

    def _add(self, view_id):
        self._lock.acquire()
        if view_id in self._queue:
            self._queue.remove(view_id)
        if len(self._tasks[view_id]) > 0 and self._tasks[view_id][0] == ['r']:
            self._tasks[view_id] = ['r', 'a']
        else:
            self._tasks[view_id] = ['a']
        self._lock.release()
        self._acquire(view_id).switch_off()
        self._release(view_id)
        while not self._isready(view_id):
            time.sleep(1)
        self._lock.acquire()
        self._queue.append(view_id)
        self._lock.release()

    def _update(self, view_id):
        self._lock.acquire()
        if len(self._tasks[view_id]) <= 0 or self._tasks[view_id][-1] != ['u']:
            self._tasks[view_id].append('u')
        if view_id not in self._queue:
            self._queue.append(view_id)
        self._lock.release()

    def _acquire_cpu(self, locked=False):
        if locked:
            self._cpus += 1
        else:
            self._lock.acquire()
            self._cpus += 1
            self._lock.release()

    def _release_cpu(self, locked=False):
        if locked:
            self._cpus -= 1
        else:
            self._lock.acquire()
            self._cpus -= 1
            self._lock.release()

    def _do_remove(self, view_id):
        try:
            get_db().del_model_execution(view_id)
            get_db().del_model_kernel(view_id)
            path = f'..\_local_db\models\{view_id}'
            if os.path.exists(path):
                shutil.rmtree(path)
                logging.info(f"Removing local models of view {view_id}")
            logging.info(f"Removing view finished: {view_id}")
        except:
            logging.error(traceback.format_exc())
        self._release(view_id)
        self._release_cpu()

    def _get_bdates(self, view_id):
        path = f'..\_local_db\models\{view_id}'
        ret = []
        if os.path.exists(path):
            ret = [datetime.datetime.strptime(each[:-4], "%Y-%m-%d").date()
                   for each in os.listdir(path) if each[-4:] == '.pkl']
            ret.sort()
        return ret

    def _get_tdates(self, tg:int):
        today = datetime.date.today()
        ret = [datetime.date(today.year, today.month, 1)]
        for i in range(12):
            ret.append(self._get_prev_bdate(ret[-1], tg))
            if (today - ret[-1]).days >= 360:
                break
        return ret[::-1]

    def _get_prev_bdate(self, next_: datetime.date, tg: int) -> datetime.date:
        year = next_.year
        month = next_.month - tg
        t = (month - 1) // 12
        ret = datetime.date(year+t, month-12*t, 1)
        return ret

    def _do_add(self, view_id, controller):
        try:
            db_ = MimosaDBManager().current_db
            view = get_db().get_model_info(view_id)
            bdates = self._get_bdates(view_id)
            prev = None
            for bdate in self._get_tdates(view.train_gap):
                if bdate in bdates:
                    prev = bdate
                    continue
                if not controller.isactive:
                    break
                models = self._train(db_, view, bdate, controller)
                if controller.isactive:
                    pickle_dump(models, f'..\_local_db\models\{view_id}\{bdate}.pkl')
                    if prev is not None:
                        get_db().update_view_kernel(view_id, prev, bdate-datetime.timedelta(1))
                    get_db().save_view_kernel(view_id, bdate)
                prev = bdate
            if controller.isactive:
                logging.info(f"Adding view finished: {view_id}")
                get_db().set_model_train_complete(view_id)
        except:
            logging.error(traceback.format_exc())
        self._release(view_id)
        self._release_cpu()

    def _get_next_bdate(self, prev: datetime.date, tg: int) -> datetime.date:
        year = prev.year
        month = prev.month + tg
        t = (month - 1) // 12
        ret = datetime.date(year+t, month-12*t, 1)
        return ret

    def _train(self, db_, view, bdate, controller):
        models = {}
        for p in PREDICT_PERIODS:
            x, y = _get_train_dataset(db_, view.markets, view.patterns,
                                      view.train_begin, bdate, p, controller)
            if not controller.isactive:
                break
            if len(x) <= 0 or len(y) <= 0:
                continue
            models[p] = Dtc(**COMMON_DECISION_TREE_PARAMS)
            models[p].fit(x,y)
        if controller.isactive:
            return models
        return None

    def _do_update(self, view_id, controller):
        try:
            db_ = MimosaDBManager().current_db
            view = get_db().get_model_info(view_id)
            bdates = self._get_bdates(view_id)
            if len(bdates) <= 0:
                raise RuntimeError("update model before adding")
            bdate = self._get_next_bdate(bdates[-1], view.train_gap)
            if datetime.date.today() >= bdate:
                models = self._train(db_, view, bdate, controller)
                if controller.isactive:
                    pickle_dump(models, f'..\_local_db\models\{view_id}\{bdate}.pkl')
                    get_db().update_view_kernel(view_id, bdates[-1], bdate-datetime.timedelta(1))
                    get_db().save_view_kernel(view_id, bdate)
                    get_db().set_model_train_complete(view_id)
                    logging.info(f"Updating view complete: {view_id}")
        except:
            logging.error(traceback.format_exc())

        self._release(view_id)
        self._release_cpu()

    def _run(self):
        cnt = 0
        while True:
            cnt += 1
            self._lock.acquire()
            if len(self._queue) <= 0 or self._cpus >= self._max_cpus:
                self._lock.release()
                time.sleep(1)
                continue
            for view_id in self._queue:
                if self._isready(view_id, locked=True):
                    if len(self._tasks[view_id]) > 0:
                        t = self._tasks[view_id][0]
                        self._tasks[view_id] = self._tasks[view_id][1:]
                        self._acquire_cpu(locked=True)
                        controller = self._acquire(view_id, locked=True)
                        self._lock.release()
                        controller.switch_on()
                        if t == 'r':
                            Thread(target=self._do_remove, args=(view_id, )).start()
                        elif t == 'a':
                            Thread(target=self._do_add, args=(view_id, controller)).start()
                            #self._do_add(view_id, controller)
                        elif t == 'u':
                            Thread(target=self._do_update, args=(view_id, controller)).start()
                        else:
                            raise RuntimeError(f"invalid execution type {t}")
                        time.sleep(1)
                        break
                    else:
                        self._queue = self._queue[1:]
                        self._lock.release()
                        break
            else:
                self._lock.release()
                time.sleep(1)


def init_db():
    try:
        controller = mt_manager.acquire(BATCH_EXE_CODE)
        batch_type = BatchType.INIT_BATCH
        logging.info('Initial batch started')
        clean_db_cache()
        clone_db_cache(batch_type)
        t = None
        if not MimosaDBManager().is_ready():
            t = _batch_db_update(batch_type)
        _batch_del_view_data()
        _batch_recover_executions()
        if t is not None:
            t[0].join()
        if controller.isactive:
            logging.info('Initial batch finished')
        else:
            logging.info('Initial batch terminated')
        mt_manager.release(BATCH_EXE_CODE)
    except Exception as esp:
        logging.error(traceback.format_exc())
        logging.error("Initial batch failed")
        mt_manager.release(BATCH_EXE_CODE)


def batch():
    try:
        batch_type = BatchType.SERVICE_BATCH
        controller = mt_manager.acquire(BATCH_EXE_CODE)
        logging.info("Batch service started")
        clean_db_cache()
        if controller.isactive:
            clone_db_cache(batch_type) # ineffective without clone
        if controller.isactive:
            get_db().truncate_swap_tables()
        threads = _batch_db_update(batch_type) or [] # standalone step
        # update threads
        for t in threads:
            if not controller.isactive:
                break
            t.join()
        if controller.isactive:
            get_db().checkout_fcst_data()
            MimosaDBManager().swap_db()
            _batch_del_view_data() # fast
            for model_id in get_db().get_models(): # a function ?
                ViewManagerFactory.get()._update(model_id)
            logging.info("Service batch finished")
            mt_manager.release(BATCH_EXE_CODE)
            return
        logging.info("Service batch terminated")
        mt_manager.release(BATCH_EXE_CODE)

    except Exception:
        logging.error(traceback.format_exc())
        logging.error("Service batch failed")
        mt_manager.release(BATCH_EXE_CODE)

task_queue = QueueManager({TaskCode.PATTERN:ExecQueue('pattern_queue')})



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


def get_patterns_occur_dates(market_id: str, patterns: List[str], 
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None) -> List[Dict[str, str]]:
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


def get_mix_pattern_occur_cnt(patterns:List[str], markets:List[str],
         begin_date:Optional[datetime.date]=None, 
         end_date:Optional[datetime.date]=None) -> Tuple[int, int]:
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


def _get_mix_pattern_rise_prob(markets, patterns, period
        ) -> Tuple[int, int, int, int, int, int]:
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


def get_mix_pattern_rise_prob(markets:List[str], patterns:List[str]):
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


def get_occurred_patterns(date:datetime.date, patterns:List[str]) -> List[str]:
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
    ret:List[str] = pvalues.columns.values[pvalues.values.any(axis=0)].tolist()
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


def get_market_rise_prob(period:str, market_type:Optional[str]=None, 
        category_code:Optional[str]=None) -> float:
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
    if not macro_info:
        raise BadRequest('macro info not found')
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


def get_targetdate_pred(view_id: str, market_id: str, target_date: datetime.date
        ) -> List[Dict[str, Any]]:
    """使用指定觀點, 取得目標預測日期往前各天期的預測結果, 例如: 目標日期為 12/31,
    那麼就會取得 12/26 的 5 天期預測結果(往前 5 日), 12/21 的 10 天期預測結果(往前
    10 日), 12/11 的 20 天期預測結果(往前 20 日)... 依此類推至 120 天期.

    Parameters
    ----------
    view_id: str
        要用來預測的觀點 ID
    market_id: str
        要進行預測的市場 ID
    target_date: datetime.date
        要預測的目標日期

    Returns
    -------
    result: List[Dict[str, Any]]
        各天期預測結果, 若無該天期結果則不會填入陣列. 內容物件格式為
         - dataDate: datetime.date
            基底日期
         - price: float
            基底價格
         - datePeriod: int
            預測天期
         - upperBound: float
            預測上界報酬
         - lowerBound: float
            預測下界報酬
         - score: int
            預測標籤
         - priceDate: datetime.date
            目標預測日期
    """
    def _get_model_kernels(model_id: str) -> Dict[datetime.date, Dict[int, Optional[Dtc]]]:
        """取得觀點訓練完成模型, 會根據起始日期由大到小將字典進行排序

        Parameters
        ----------
        model_id: str
            觀點 ID

        Returns
        -------
        result: Dict[datetime.date, Dict[int, DecisionTreeClassifier | None]
            觀點歷史訓練完成模型, 結構為: [可用起始日]-[目標天期]-DecisionTreeClassifier | None
        """
        result = {}
        model_path = f'{LOCAL_DB}/models'
        if not os.path.exists(model_path) or not os.path.exists(f'{model_path}/{model_id}'):
            return result
        model_names = os.listdir(f'{model_path}/{model_id}')
        for model_name in model_names:
            start_date = datetime.datetime.strptime(model_name[:-4], '%Y-%m-%d').date()
            model_info: Optional[Dict[int, Dtc]] = pickle_load(f'{model_path}/{view_id}/{model_name}')
            result[start_date] = model_info
        result = dict(sorted(result.items(), reverse=True))
        return result
    results = []
    _db = MimosaDBManager().current_db
    if not _db.is_initialized():
        return results
    _dao = get_dao()
    market_dates = _db.get_market_dates(market_id)
    market_cps = _db.get_market_prices(market_id)
    models = _get_model_kernels(view_id)

    for period in PREDICT_PERIODS:
        base_dates = market_dates[market_dates < target_date]
        if len(base_dates) < period:
            continue
        base_date = base_dates[-period].tolist()
        cpidx = market_cps.index.values.astype('datetime64[D]') == base_date
        base_cp = market_cps[cpidx].values[0]
        for start_date in models:
            if start_date < base_date:
                model = models[start_date].get(period)
                if model is not None:
                    view = _dao.get_model_info(view_id).get_model(period, start_date)
                    x_data = _db.get_pattern_values(market_id, view.patterns)
                    y_data = _db.get_future_returns(market_id, [period])[period]
                    y_data = y_data[~np.isnan(y_data.values)]
                    dates = y_data.index.values.astype('datetime64[D]')
                    sidx = (dates < view.train_begin_date).sum()
                    eidx = (dates < view.effective_date).sum() - view.predict_period
                    eidx = max([sidx, eidx])
                    y_coder = Labelization(Y_LABELS)
                    y_coder.fit(y_data.values, Y_OUTLIER)

                    xidx = x_data.index.values.astype('datetime64[D]') == base_date
                    x = x_data[xidx].values
                    y = model.predict(x)
                    lower_bound = y_coder.label2lowerbound(y)[0]
                    upper_bound = y_coder.label2upperbound(y)[0]
                    ret = y_coder.label2mean(y)[0]

                    results.append({
                        'dataDate': str(base_date),
                        'price': base_cp,
                        'datePeriod': period,
                        'upperBound': upper_bound,
                        'lowerBound': lower_bound,
                        'score': int(y[0]),
                        'priceDate': str(target_date)
                    })
                break
    return results


class DateBaseDateResp(NamedTuple):
    dataDate:str
    price:float
    datePeriod:int
    upperBound:float
    lowerBound:float
    score:int
    priceDate:str


def get_basedate_pred(view_id:str, market_id:str, 
        base_date:Optional[datetime.date]=None) -> List[DateBaseDateResp]:
    class Scorer(NamedTuple):
        value:int
        upper:float
        lower:float

        def get(self, mean, std):
            lower = self.lower*std if not pd.isna(self.lower) else -np.inf
            upper = self.upper*std if not pd.isna(self.upper) else np.inf
            if mean > lower and mean <= upper:
                return self.value

    V_PATH = LOCAL_DB+f'/models/{view_id}'
    if not os.path.isdir(V_PATH):
        return []
    _db = MimosaDBManager().current_db
    if not _db.is_initialized():
        return []
    ret = []
    
    view_info = get_db().get_model_info(view_id)
    view_begin = view_info.train_begin
    models:Dict[str, Dict[int, Dtc]] = {i.split('\\')[-1][:-4]:pickle.load(
        open(i, 'rb')) for i in glob.glob(fr'{V_PATH}\*.pkl')}
    
    cps = _db.get_market_prices(market_id)
    if base_date is not None:
        if cps.index.values[-1] < np.array(base_date).astype('datetime64[D]'):
            return []
    _base_date = np.array(base_date or cps.index.values[-1]).astype('datetime64[D]')
    cp = cps.loc[_base_date]

    effective_date_time_diff_map:Dict[str, datetime.timedelta] = {
        i:_base_date.tolist()-datetime.datetime.strptime(i, '%Y-%m-%d').date(
        ) for i in models}
    edm = effective_date_time_diff_map
    edm ={k:v for k, v  in edm.items() if v.days>0}
    if len(edm) == 0:
        return []

    model_effective_key = next(iter(sorted(edm, key=lambda k: edm[k].days)))
    model = models[model_effective_key]
    model_effective_date = datetime.datetime.strptime(model_effective_key, '%Y-%m-%d').date()

    ptns = _db.get_pattern_values(market_id, view_info.patterns)
    ptns = ptns.loc[_base_date,:]
    scores = [Scorer(k, upper=u, lower=l) for k, u, l in get_db().get_score_meta_info()]
    
   
    for period in PREDICT_PERIODS:
        pred = model[period].predict(ptns.values.reshape((1,-1))) # reshape to 2D
        y_coder = Labelization(Y_LABELS)
        freturns = _db.get_future_returns(market_id, [period])
        begin = max((freturns.index.values.astype('datetime64[D]') <= view_begin).sum()-1, 0)
        end = max((freturns.index.values.astype('datetime64[D]') <= model_effective_date).sum()-1, 0)
        _freturns = _db.get_future_returns(market_id, [period]).values[begin:end-1-period,0]
        _freturns = _freturns[_freturns == _freturns]
        y_coder.fit(_freturns, Y_OUTLIER)
        lower_bound = y_coder.label2lowerbound(pred)[0]
        upper_bound = y_coder.label2upperbound(pred)[0]
        mean = y_coder.label2mean(pred)[0]
        std = np.std(freturns)
        score = None
        price_dates = extend_working_dates(np.array([_base_date]), period)
        for scorer in scores:
            score = scorer.get(mean, std)
            if score is not None:
                break
        ret.append(DateBaseDateResp(
            _base_date.tolist().strftime('%Y-%m-%d'), cp.tolist(), period, upper_bound.tolist(),
            lower_bound.tolist(), score.tolist(), price_dates[-1].tolist().strftime('%Y-%m-%d'))._asdict())
    return ret

class DateRangePredResp(NamedTuple):
    dataDate:str
    price:float
    upperBound:float
    lowerBound:float
    priceDate:str


def get_daterange_pred(view_id:str, market_id:str, *, period:int,
        start_date:datetime.date, end_date:datetime.date)-> List[DateRangePredResp]:

    V_PATH = LOCAL_DB+f'/models/{view_id}'
    if not os.path.isdir(V_PATH):
        return []
    _db = MimosaDBManager().current_db
    if not _db.is_initialized():
        return []
    if period not in PREDICT_PERIODS:
        return []

    ret = []
    view_info = get_db().get_model_info(view_id)
    view_begin = np.array(view_info.train_begin).astype('datetime64[D]')
    models:Dict[str, Dict[int, Dtc]] = {i.split('\\')[-1][:-4]:pickle.load(
        open(i, 'rb'))for i in glob.glob(fr'{V_PATH}\*.pkl')}
 
    cps = _db.get_market_prices(market_id).rename('cp')

    def get_effective_model_dates(dates:pd.Index) -> List[datetime.date]:
        order_models = dict(sorted(models.items(), 
            key= lambda i : datetime.datetime.strptime(i[0], '%Y-%m-%d'), reverse=True))
        _dates = dates.values.astype('datetime64[D]')
        count = 0
        for each in order_models:
            e_date = np.array(datetime.datetime.strptime(
                each, '%Y-%m-%d').date()).astype('datetime64[D]')
            if (_dates >= e_date).sum() > 0:
                if count == 0:
                    _dates[-(_dates >= e_date).sum():] = e_date
                    count = (_dates >= e_date).sum()
                else:
                    _dates[-(_dates >= e_date).sum():-count] = e_date
                    count = (_dates >= e_date).sum()
            else:
                continue
        mask = np.full(_dates[:-count].shape, np.nan)
        return mask.tolist() + _dates[-count:].tolist()

    ptns = _db.get_pattern_values(market_id, view_info.patterns)
    freturns = _db.get_future_returns(market_id, [period])
    begin =  max((freturns.index.values.astype('datetime64[D]') <= view_begin).sum()-1, 0)
    
    ret = []
    pred_dates = get_effective_model_dates(cps.index)
    for e_date, cp, date in zip(pred_dates, cps.values.tolist(), cps.index.tolist()):
        if date < start_date or date > end_date or not (e_date==e_date):
            continue
        _ptns = ptns.loc[date]
        score = models[e_date.strftime('%Y-%m-%d')][period].predict(_ptns.values.reshape((1,-1)))
        end = max((freturns.index.values.astype('datetime64[D]') <= e_date).sum()-1, 0)
        y_coder = Labelization(Y_LABELS)
        _freturn = freturns.values[begin:end-1-period,0]
        _freturn = _freturn[_freturn==_freturn]
        y_coder.fit(_freturn, Y_OUTLIER)
        lower_bound = y_coder.label2lowerbound(score)[0]
        upper_bound = y_coder.label2upperbound(score)[0]
        price_dates = extend_working_dates(np.array([date]).astype('datetime64[D]'), period)
        ret.append(DateRangePredResp(
            date.strftime('%Y-%m-%d'), cp, upper_bound.tolist(),
            lower_bound.tolist(), price_dates[-1].tolist().strftime('%Y-%m-%d'))._asdict())
    return ret

def _get_freturn_score(freturns:np.ndarray, period:int, market_id:str):
    class Scorer(NamedTuple):
        score:int
        upper:float
        lower:float

        def get(self, mean, std):
            lower = self.lower*std if not pd.isna(self.lower) else -np.inf
            upper = self.upper*std if not pd.isna(self.upper) else np.inf
            ret = ((mean > lower) & (mean <= upper)).astype('int')
            ret[ret == 1] = self.score
            return ret
    _db = MimosaDBManager().current_db
    std = _db.get_future_std(market_id, period)
    if std == np.nan:
        return np.nan
    scorers = [Scorer(k, upper=u, lower=l) for k, u, l in get_db().get_score_meta_info()]
    ret = []
    for scorer in scorers:
        ret.append(scorer.get(freturns, std))
    return np.array(ret).sum(axis=0)
