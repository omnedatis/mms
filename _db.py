# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:03:51 2022

@author: WaNiNi
"""

import multiprocessing as mp
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from func._td import set_market_data_provider, MarketDataProvider, MD_CACHE
from _core import Pattern, MarketInfo
from dao import MimosaDB as MimosaDao
from utils import (int2datetime64d, datetime64d2int, print_error_info,
                   pickle_dump, pickle_load, singleton)
from const import PATTERN_UPDATE_CPUS, PREDICT_PERIODS, LOCAL_DB


_dao: MimosaDao = MimosaDao()

def get_dao():
    return _dao

def set_dao(dao: MimosaDao):
    global _dao
    _dao = dao


class ReturnValueCoder:
    @staticmethod
    def encode(values):
        # float32的精確度為 7 位數字
        return values.astype(np.float32)

    @staticmethod
    def decode(values):
        return values.astype(float)

class PatternValueCoder:
    @staticmethod
    def encode(values):
        ret = values.astype('int8')
        ret[np.isnan(values)] = -1
        return ret

    @staticmethod
    def decode(values):
        ret = values.astype(float)
        ret[ret < 0] = np.nan
        return ret

class MarketDateCoder:
    @staticmethod
    def encode(values):
        return datetime64d2int(values)

    @staticmethod
    def decode(values):
        return int2datetime64d(values)

class LocalMarketDataProvider(MarketDataProvider):
    def __init__(self, idx: int, path: str = f'{LOCAL_DB}/mdata'):
        self._path = f'{path}/{idx}'

    def _get_file(self, market_id: str) -> str:
        return f'{self._path}/{market_id}.pkl'

    def get_ohlc(self, market_id: str) -> pd.DataFrame:
        file = self._get_file(market_id)
        if not os.path.exists(file):
            raise FileNotFoundError(f"market is not found: {market_id}")
        return pickle_load(file)

    def update(self):
        dao = get_dao()
        for mid in dao.get_markets():
            file = self._get_file(mid)
            pickle_dump(dao.get_market_data(mid), file)


class MimosaDB:
    def __init__(self, idx: int):
        self._dbid = idx
        self._market_ids: Optional[Dict[str, int]] = None
        self._market_info: Optional[Dict[str, np.ndarray]] = None
        self._pattern_ids: Optional[Dict[str, int]] = None
        self._patterns: Optional[List[Pattern]] = None
        self._period_ids: Dict[int, int] = {
            period: idx for idx, period in enumerate(PREDICT_PERIODS)}
        self._market_dates: Optional[List[np.ndarray]] = None
        self._market_pattern_values: Optional[List[np.ndarray]] = None
        self._market_future_returns: Optional[List[np.ndarray]] = None

    @property
    def _local_path(self):
        return f'{LOCAL_DB}/db_{self._dbid}'

    @property
    def _market_data_provider(self):
        return LocalMarketDataProvider(self._dbid)

    def is_initialized(self):
        if self._market_ids is None:
            return False
        if self._pattern_ids is None:
            return False
        if self._market_dates is None:
            return False
        if self._market_pattern_values is None:
            return False
        if self._market_future_returns is None:
            return False
        return True

    def clear(self):
        self._market_ids = None
        self._pattern_ids = None
        self._patterns = None
        self._market_dates = None
        self._market_pattern_values = None
        self._market_future_returns = None

    def dump(self):
        path = self._local_path
        if not os.path.exists(path):
            os.mkdir(path)
        pickle_dump(self._market_ids, f'{path}/mids.pkl')
        pickle_dump(self._market_info, f'{path}/minfo.pkl')
        pickle_dump(self._pattern_ids, f'{path}/pids.pkl')
        pickle_dump(self._patterns, f'{path}/patterns.pkl')
        pickle_dump(self._market_dates, f'{path}/mdates.pkl')
        pickle_dump(self._market_pattern_values, f'{path}/pvalues.pkl')
        pickle_dump(self._market_future_returns, f'{path}/freturns.pkl')

    def load(self):
        path = self._local_path
        if os.path.exists(path):
            self._market_ids = pickle_load(f'{path}/mids.pkl')
            self._market_info = pickle_load(f'{path}/minfo.pkl')
            self._pattern_ids = pickle_load(f'{path}/pids.pkl')
            self._patterns = pickle_load(f'{path}/patterns.pkl')
            self._market_dates = pickle_load(f'{path}/mdates.pkl')
            self._market_pattern_values = pickle_load(f'{path}/pvalues.pkl')
            self._market_future_returns = pickle_load(f'{path}/freturns.pkl')

    def get_markets(self, mtype=None, category=None) -> List[str]:
        if self._market_ids is None:
            raise RuntimeError("market data is not initialized")
        ret = list(self._market_ids.keys())
        if mtype is not None:
            idxs = self._market_info['mtype'] == mtype
            if category is not None:
                idxs &= self._market_info['category'] == category
            return np.array(ret)[idxs].tolist()
        return list(self._market_ids.keys())

    def _set_markets(self, market_info: List[MarketInfo]):
        self._market_ids = {}
        mtypes = []
        categories = []
        for idx, minfo in enumerate(market_info):
            self._market_ids[minfo.mid] = idx
            mtypes.append(minfo.mtype)
            categories.append(minfo.category)
        self._market_info = {'mtype': np.array(mtypes),
                             'category': np.array(categories)}

    def get_pattern_ids(self) -> List[str]:
        if self._pattern_ids is None:
            raise RuntimeError("pattern information is not initialized")
        return list(self._pattern_ids.keys())

    def _set_patterns(self, patterns: List[str]):
        self._patterns = patterns
        self._pattern_ids = {each.pid: idx for idx, each in enumerate(patterns)}

    def get_market_dates(self, market_id: str) -> np.ndarray:
        if self._market_dates is None:
            raise RuntimeError("market data is not initialized")
        midx = self._market_ids.get(market_id)
        if midx is None:
            raise KeyError(f"market is not found: {market_id}")
        ret = MarketDateCoder.decode(self._market_dates[midx])
        return ret

    def get_market_prices(self, market_id: str) -> pd.Series:
        if self._market_ids is None:
            raise RuntimeError("market data is not initialized")
        if market_id not in self._market_ids:
            raise KeyError(f"market is not found: {market_id}")
        return self._market_data_provider.get_ohlc(market_id)['CP']

    def get_future_returns(self, market_id: str,
                           periods: Optional[List[int]]=None) -> pd.DataFrame:
        if self._market_future_returns is None:
            raise RuntimeError("market return values are not initialized")
        midx = self._market_ids.get(market_id)
        if midx is None:
            raise KeyError(f"market is not found: {market_id}")
        dates = self.get_market_dates(market_id)
        values = self._market_future_returns[midx]
        if periods is not None:
            pidxs = []
            for period in periods:
                pidx = self._period_ids.get(period)
                if pidx is None:
                    raise KeyError(f"invalid period: {period}")
                pidxs.append(pidx)
            values = values[:, pidxs]
        else:
            periods = PREDICT_PERIODS

        ret = pd.DataFrame(ReturnValueCoder.decode(values),
                           index=dates, columns=periods)
        return ret

    def get_pattern_values(self, market_id: str,
                           patterns: Optional[List[str]]=None) -> pd.DataFrame:
        if self._market_pattern_values is None:
            raise RuntimeError("market pattern values are not initialized")
        midx = self._market_ids.get(market_id)
        if midx is None:
            raise KeyError(f"market is not found: {market_id}")

        dates = self.get_market_dates(market_id)
        values = self._market_pattern_values[midx]
        if patterns is not None:
            pidxs = []
            for pid in patterns:
                pidx = self._pattern_ids.get(pid)
                if pidx is None:
                    raise KeyError(f"pattern is not found: {pid}")
                pidxs.append(pidx)
            values = values[:, pidxs]
        else:
            patterns = list(self._pattern_ids.keys())
            # Note: without sorting by its values for performance

        ret = pd.DataFrame(PatternValueCoder.decode(values),
                           index=dates, columns=patterns)
        return ret

    def get_latest_pattern_values(self, patterns: Optional[List[str]]=None
                                  ) -> Dict[str, pd.DataFrame]:
        # Note: similar code with `get_pattern_values` for performance
        if self._market_pattern_values is None:
            raise RuntimeError("market pattern values are not initialized")

        if patterns is not None:
            pidxs = []
            for pid in patterns:
                pidx = self._pattern_ids.get(pid)
                if pidx is None:
                    raise KeyError(f"pattern is not found: {pid}")
                pidxs.append(pidx)
        else:
            pidxs = None
            patterns = list(self._pattern_ids.keys())
            # Note: without sorting by its values for performance

        ret = {}
        for mid, dates, values in zip(self._market_ids.keys(),
                                      self._market_dates,
                                      self._market_pattern_values):
            if pidxs is not None:
                values = values[:, pidxs]
            ret[mid] = pd.DataFrame(PatternValueCoder.decode(values[-1:]),
                                    index=MarketDateCoder.decode(dates[-1:]),
                                    columns=patterns)
        return ret

    @staticmethod
    def _gen_market_data(market_id, dbid: int):
        recv = LocalMarketDataProvider(dbid).get_ohlc(market_id)['CP']
        mdates = MarketDateCoder.encode(recv.index.values.astype('datetime64[D]'))
        cps = recv.values
        freturns = []
        for period in PREDICT_PERIODS:
            if len(cps) > period:
                cur = np.concatenate([cps[period:] / cps[:-period] - 1,
                                      np.full(period, np.nan)], axis=0)
            else:
                cur = np.full(len(cps), np.nan)
            freturns.append(cur)
        return market_id, mdates, ReturnValueCoder.encode(np.array(freturns).T)

    @staticmethod
    def _gen_pattern_data(market_id, patterns, dbid: int):
        set_market_data_provider(LocalMarketDataProvider(dbid))
        recv = [each.run(market_id).rename(each.pid).values for each in patterns]
        return market_id, PatternValueCoder.encode(np.array(recv).T)

    def gen_pattern_data(self, market_id, pattern):
        set_market_data_provider(LocalMarketDataProvider(self._dbid))
        return pattern.run(market_id).rename(pattern.pid)

    def add_pattern(self, pattern):
        set_market_data_provider(self._market_data_provider)
        pidx = self._pattern_ids.get(pattern.pid)
        for mid, midx in self._market_ids.items():
            pvalue = PatternValueCoder.encode(pattern.run(mid).values)
            if pidx is None:
                self._market_pattern_values[midx] = np.concatenate([
                    self._market_pattern_values[midx],
                    pvalue.reshape(-1,1)], axis=1)
            else:
                self._market_pattern_values[midx][:, pidx] = pvalue
        if pidx is None:
            self._pattern_ids[pattern.pid] = len(self._pattern_ids)
            self._patterns.append(pattern)
        else:
            self._patterns[pidx] = pattern

    def update(self, market_info, patterns, processes: int=PATTERN_UPDATE_CPUS):
        MD_CACHE.clear()  # clear cache <- old version
        self._market_data_provider.update()
        self.clear()
        markets = [each.mid for each in market_info]
        if len(markets) > 0:
            mdates = {}
            freturns = {}
            pvalues = {}
            shared_patterns = mp.Manager().list(patterns)
            pool = mp.Pool(processes)

            def _add_mdates(mid, values):
                mdates[mid] = values
            def _add_freturns(mid, values):
                freturns[mid] = values
            def _add_pvalues(mid, values):
                pvalues[mid] = values

            for market in markets:
                pool.apply_async(self._gen_market_data,
                                 (market, self._dbid),
                                 callback=lambda recv: (_add_mdates(recv[0], recv[1]),
                                                        _add_freturns(recv[0], recv[2])),
                                 error_callback=print_error_info)
                if len(patterns) > 0:
                    pool.apply_async(self._gen_pattern_data,
                                     (market, shared_patterns, self._dbid),
                                     callback=lambda recv: _add_pvalues(recv[0], recv[1]),
                                     error_callback=print_error_info)
                else:
                    pvalues[market] = np.array([])

            pool.close()
            pool.join()

        self._set_markets(market_info)
        self._set_patterns(patterns)
        self._market_dates = [mdates[mid] for mid in markets]
        self._market_future_returns = [freturns[mid] for mid in markets]
        self._market_pattern_values = [pvalues[mid] for mid in markets]


@singleton
class MimosaDBManager:
    _DBID_FILE = f'{LOCAL_DB}/dbid.pkl'
    def __init__(self):
        self._dbs = [MimosaDB(idx) for idx in range(2)]
        self._dbid = self._load_dbid()
        self.current_db.load()

    @classmethod
    def _load_dbid(cls):
        file = cls._DBID_FILE
        if not os.path.exists(file):
            pickle_dump(0, file)
        return pickle_load(file)

    def _dump_dbid(self):
        file = self._DBID_FILE
        pickle_dump(self._dbid, file)

    @property
    def current_db(self):
        return self._dbs[self._dbid]

    @property
    def next_db(self):
        return self._dbs[(self._dbid+1) % 2]

    def swap_db(self):
        self._dbid = (self._dbid+1) % 2
        self._dump_dbid()
        self.next_db.clear()

    def is_ready(self):
        return self.current_db.is_initialized()

