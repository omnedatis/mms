# -*- coding: utf-8 -*-
"""Timing Data.

This package include classes and methods for timing data and operations.

Enumerators
-----------
TimeUnit : Definitions of unit of time.

Classes
-------
BooleanTimeSeries : A time-series with boolean values.
NumericTimeSeries : A time-series with numeric values.
TimeIndex : Immutable ndarray of datetimes, used as index of timing data.
TimeSeries: A sequence of data points indexed by time.

"""

# pylint: disable=unused-import
from ._db import (get_cp, get_hp, get_lp, get_op, get_market_data, MarketData,
                  set_market_data_provider, MD_CACHE, MarketDataProvider)
from ._index import TimeIndex, TimeUnit  # noqa: F401
from ._series import BooleanTimeSeries, NumericTimeSeries, TimeSeries  # noqa: F401

# pylint: enable=unused-import

ts_all = BooleanTimeSeries.all
ts_any = BooleanTimeSeries.any
ts_average = NumericTimeSeries.average
ts_max = NumericTimeSeries.max
ts_min = NumericTimeSeries.min

__ALL__ = ['TimeIndex', 'TimeUnit', 'MD_CACHE', 'MarketDataProvider']
__ALL__ += ['BooleanTimeSeries', 'NumericTimeSeries', 'TimeSeries']
__ALL__ += ['get_cp', 'get_hp', 'get_lp', 'get_op', 'get_market_data',
            'MarketData', 'set_db_id']
__ALL__ += ['ts_all', 'ts_any', 'ts_max', 'ts_min']
