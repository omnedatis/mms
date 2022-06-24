# -*- coding: utf-8 -*-
"""Financial Market DataBase."""

from abc import ABCMeta, abstractmethod
from typing import NamedTuple, Optional

import pandas as pd

from ._series import NumericTimeSeries, TimeIndex
from utils import Cache
from const import FUNC_CACHE_SIZE

_DB = None

MD_CACHE = Cache(FUNC_CACHE_SIZE)

class MarketDataProvider(metaclass=ABCMeta):
    @abstractmethod
    def get_ohlc(self, market_id: str) -> pd.DataFrame:
        """Get OHLC data of the designated market.

        Parameters
        ----------
        market_id: str
            ID of the designated market.

        Returns
        -------
        Pandas' DataFrame
            A Pandas' DataFrame consisted of four floating time-series with
            names, `OP`, `HP`, `LP`, and `CP`.

        """
        pass

def set_market_data_provider(db_: MarketDataProvider):
    """

    Parameters
    ----------
    db_: MarketDataProvider

    """
    global _DB
    _DB = db_

def get_market_data_provider() -> MarketDataProvider:
    return _DB

class MarketData(NamedTuple):
    """Named-tuple of market data.

    Properties
    ----------
    cp : NumericTimeSeries
        Closing price of market.
    lp : Optional[NumericTimeSeries]
        Lowest price of market. ``None`` if no data of lowest price.
    hp : Optional[NumericTimeSeries]
        highest price of market. ``None`` if no data of highest price.
    op : Optional[NumericTimeSeries]
        opening price of market. ``None`` if no data of opening price.

    """
    cp : NumericTimeSeries
    lp : Optional[NumericTimeSeries] = None
    hp : Optional[NumericTimeSeries] = None
    op : Optional[NumericTimeSeries] = None

def _get_market_data(market_id: str):
    recv = get_market_data_provider().get_ohlc(market_id)
    index = TimeIndex(recv.index.values.astype('datetime64[D]'))
    cp = NumericTimeSeries(data=recv.CP.values.astype(float), index=index,
                           name=f'{market_id}.CP')
    lp = NumericTimeSeries(data=recv.LP.values.astype(float), index=index,
                           name=f'{market_id}.LP') if 'LP' in recv else None
    hp = NumericTimeSeries(data=recv.HP.values.astype(float), index=index,
                           name=f'{market_id}.HP') if 'HP' in recv else None
    op = NumericTimeSeries(data=recv.OP.values.astype(float), index=index,
                           name=f'{market_id}.OP') if 'OP' in recv else None
    ret = MarketData(cp=cp, lp=lp, hp=hp, op=op)
    return ret

def get_market_data(market_id: str):
    """Get data of the designated market.

    Parameters
    ----------
    market_id : str
        ID of the designated market.

    Returns
    -------
    MarketData

    """
    if market_id not in MD_CACHE:
        MD_CACHE[market_id] = _get_market_data(market_id)
    return MD_CACHE[market_id]

def get_op(market_id: str):
    """Get opening price of the designated market.

    Parameters
    ----------
    market_id : str
        ID of the designated market.

    Returns
    -------
    NumericTimeSeries or None

    """
    recv = get_market_data(market_id)
    return recv.op

def get_hp(market_id: str):
    """Get highest price of the designated market.

    Parameters
    ----------
    market_id : str
        ID of the designated market.

    Returns
    -------
    NumericTimeSeries or None

    """
    recv = get_market_data(market_id)
    return recv.hp

def get_lp(market_id: str):
    """Get lowest price of the designated market.

    Parameters
    ----------
    market_id : str
        ID of the designated market.

    Returns
    -------
    NumericTimeSeries or None

    """
    recv = get_market_data(market_id)
    return recv.lp

def get_cp(market_id: str):
    """Get closing price of the designated market.

    Parameters
    ----------
    market_id : str
        ID of the designated market.

    Returns
    -------
    NumericTimeSeries

    """
    recv = get_market_data(market_id)
    return recv.cp
