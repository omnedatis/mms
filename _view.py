# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:45:34 2022

@author: WaNiNi
"""

import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from _core import View, ViewModel, ModelResultField
from _db import MimosaDBManager, MimosaDB
from const import PredictResultField, MIN_BACKTEST_LEN, PREDICT_PERIODS, BATCH_EXE_CODE
from utils import ThreadController, MT_MANAGER



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

def _get_train_dataset(db_: MimosaDB, model: ViewModel, markets: List[str],
                       controller: ThreadController
                       ) -> Tuple[Dict[str, pd.DataFrame],
                                  Dict[str, pd.DataFrame]]:
    x_train = _get_x_data(db_, markets, model.patterns, controller)
    y_train = _get_y_data(db_, markets, model.predict_period, controller)
    for market in markets:
        if not controller.isactive:
            return {}, {}
        dates = x_train[market].index.values.astype('datetime64[D]')
        sidx = (dates < model.train_begin_date).sum()
        eidx = (dates < model.effective_date).sum() - model.predict_period
        eidx = max([sidx, eidx])
        x_train[market] = x_train[market][sidx: eidx]
        y_train[market] = y_train[market][sidx: eidx]
    return x_train, y_train

def _train_model(db_: MimosaDB, model: ViewModel,
                controller: ThreadController) -> bool:
    markets = model.markets or db_.get_markets()
    x_train, y_train = _get_train_dataset(db_, model, markets, controller)
    if controller.isactive:
        model.train(x_train, y_train)
    return controller.isactive

def _update_model_markets(db_: MimosaDB, model: ViewModel, markets: List[str],
                          controller: ThreadController) -> bool:
    x_train, y_train = _get_train_dataset(db_, model, markets, controller)
    if controller.isactive:
        model.update_markets(x_train, y_train)
    return controller.isactive

def _combine_predict_results(view_id: str, period: int,
                             recv: Dict[str, pd.DataFrame]
                             ) -> Optional[pd.DataFrame]:
    if recv is None or len(recv) == 0:
        return None
    ret = []
    for market, data in recv.items():
        if data is None or len(data) == 0:
            continue
        cur = pd.DataFrame()
        if len(data) > 0:
            cur[PredictResultField.DATE.value] = data.index.values
            cur[PredictResultField.UPPER_BOUND.value
                ] = data[ModelResultField.UPPER_BOUND.value].values
            cur[PredictResultField.LOWER_BOUND.value
                ] = data[ModelResultField.LOWER_BOUND.value].values
            cur[PredictResultField.PREDICT_VALUE.value
                ] = data[ModelResultField.MEAN.value].values
            cur[PredictResultField.MARKET_ID.value] = market
            cur[PredictResultField.PERIOD.value] = period
            cur[PredictResultField.MODEL_ID.value] = view_id
        ret.append(cur)
    if len(ret) == 0:
        return None
    return pd.concat(ret, axis=0)

def update_view(view: View, latest_dates: Dict[str, datetime.date],
                controller: ThreadController) -> Tuple[Union[pd.DataFrame, None], bool]:
    batch_controller = MT_MANAGER.acquire(BATCH_EXE_CODE)
    _db = MimosaDBManager().next_db
    # pattern data for markets
    x_data = _get_x_data(_db, _db.get_markets(), view.patterns, controller)
    first_bdates:Dict[str, datetime.date] = {}
    # get backtest dates for each market
    for market, p_data in x_data.items():
        if batch_controller.isactive:
            ldate = latest_dates.get(market)
            if ldate is None: # if none, get minimum backtest length
                idx = -MIN_BACKTEST_LEN 
            else:
                idx = (p_data.index.values.astype('datetime64[D]') <= ldate).sum()
            p_data = p_data[idx:]
            if len(p_data) > 0:
                # get the earliest backtest date for each market
                first_bdates[market] = p_data.index.values[0].astype('datetime64[D]').tolist()
            x_data.update({market:p_data})

    ret = []
    smd = False
    while controller.isactive:
        if len(first_bdates) == 0:
            # no more update necessary
            break
        start_date = min(first_bdates.values())
        expired_date = view.get_expiration_date(start_date)
        cur_model_x = {}
        for market, bdate in list(first_bdates.items()):
            if not batch_controller.isactive:
                break
            if bdate < expired_date:
                p_data = x_data[market]
                idx = (p_data.index.values.astype('datetime64[D]') < expired_date).sum()
                cur_model_x[market] = p_data[:idx]
                x_data[market] = p_data[idx:]
                if len(x_data[market]) > 0:
                    # update earliest backtest date for each market
                    first_bdates[market] = x_data[market].index.values[0].astype('datetime64[D]').tolist()
                else:
                    del first_bdates[market] # no more available data
        # train model for every period and get prediction
        for period in PREDICT_PERIODS:
            if not batch_controller.isactive:
                break
            model = view.get_model(period, start_date)
            if not model.is_trained():
                if not _train_model(_db, model, controller): # model is mutable
                    break
                smd = True
            new_markets = list(set(cur_model_x.keys()) - set(model.trained_markets))
            if (new_markets and
                not _update_model_markets(_db, model, new_markets, controller)):
                break
            recv = model.predict(cur_model_x)
            if recv is None or len(recv) == 0:
                continue
            # flattern data structure to db schema (multiple model under one view)
            recv = _combine_predict_results(view.view_id, period, recv)
            ret.append(recv)
    else:
        # ????
        # view have no update for every model and every period
        return None, smd
    if len(ret) > 0:
        if not batch_controller.isactive:
            logging.info('')
        else:
            MT_MANAGER.release(BATCH_EXE_CODE)
        return pd.concat(ret, axis=0), smd
    # ??? return any ?

def backtest_view(view: View, earlist_dates: Dict[str, datetime.date],
                  controller: ThreadController) -> Optional[pd.DataFrame]:
    _db = MimosaDBManager().current_db
    markets = list(earlist_dates.keys())
    # edate = max(earlist_dates.values())
    x_data:Dict[str, pd.DataFrame] = {} # backtest dates for markets
    latest_dates = []
    for mid, data in _get_x_data(_db, markets, view.patterns, controller).items():
        dates = data.index.values.astype('datetime64[D]')
        # index of to which date backtest is done
        eidx = (dates < earlist_dates[mid]).sum() 
        if eidx > 0:
            x_data[mid] = data[:eidx]
            latest_dates.append(dates[eidx-1])
    sdate = view.get_effective_date(np.array(latest_dates).max().tolist())
    # edate = view.get_expiration_date(sdate)
    for mid, data in list(x_data.items()):
        dates = data.index.values.astype('datetime64[D]')
        # market need not backtest
        if sdate > dates[-1]:
            del x_data[mid]
        else:
            sidx = (dates < sdate).sum()
            x_data[mid] = data[sidx:]
    ret = []
    for period in PREDICT_PERIODS:
        if not controller.isactive:
            return None
        model = view.get_model(period, sdate)
        if (not model.is_trained() and
            not _train_model(_db, model, controller)):
            break
        new_markets = list(set(x_data.keys()) - set(model.trained_markets))
        if (new_markets and
            not _update_model_markets(_db, model, new_markets, controller)):
            break
        recv = model.predict(x_data)
        if recv is None or len(recv) == 0:
            continue
        recv = _combine_predict_results(view.view_id, period, recv)
        ret.append(recv)
    if len(ret) > 0:
        return pd.concat(ret, axis=0)


def create_view(view: View, controller: ThreadController) -> Optional[pd.DataFrame]:
    _db = MimosaDBManager().current_db
    x_data = _get_x_data(_db, _db.get_markets(), view.patterns, controller)
    bdates = {}
    for market, cur_d in x_data.items():
        if len(cur_d) > 0:
            bdates[market] = cur_d.index.values[-1].astype('datetime64[D]').tolist()
            x_data[market] = cur_d[-1:]

    ret = []
    while controller.isactive:
        if len(bdates) == 0:
            break
        sdate = min(bdates.values())
        edate = view.get_expiration_date(sdate)
        cur_x = {}
        for market, bdate in list(bdates.items()):
            if bdate < edate:
                cur_d = x_data[market]
                idx = (cur_d.index.values.astype('datetime64[D]') < edate).sum()
                cur_x[market] = cur_d[:idx]
                x_data[market] = cur_d[idx:]
                if len(x_data[market]) > 0:
                    bdates[market] = x_data[market].index.values.astype('datetime64[D]').tolist()
                else:
                    del bdates[market]
        for period in PREDICT_PERIODS:
            model = view.get_model(period, sdate)
            if (not model.is_trained() and
                not _train_model(_db, model, controller)):
                break
            new_markets = list(set(cur_x.keys()) - set(model.trained_markets))
            if (new_markets and
                not _update_model_markets(_db, model, new_markets, controller)):
                break
            recv = model.predict(cur_x)
            if recv is None or len(recv) == 0:
                continue
            recv = _combine_predict_results(view.view_id, period, recv)
            ret.append(recv)
    else:
        return None
    if len(ret) > 0:
        return pd.concat(ret, axis=0)
