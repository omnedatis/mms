# -*- coding: utf-8 -*-
"""
Created on Thur Jun  2 10:30:29 2022

@author: Jeff
"""
from collections import defaultdict
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from ...common import Macro, MacroParam, ParamType, PlotInfo, Ptype, PeriodType
from .common import MAX_PRICE_LEN
from .._context import TimeUnit
from .._context import TechnicalIndicator as TI
import const

MA_GRAPH_SAMPLE_NUM = const.MA_GRAPH_SAMPLE_NUM


def _get_ma(data, period) -> np.ndarray:
    new_shape = (period, data.shape[0]-period+1)
    new_strides = (*data.strides, *data.strides)
    return np.mean(as_strided(data, shape=new_shape, strides=new_strides).copy(), axis=0)


def _is_positive_int(value) -> str:
    if isinstance(value, int) and value > 0:
        return ''
    return '輸入值必須為正整數'

def _is_bounded(value) -> str:
    if value <= MAX_PRICE_LEN:
        return ''
    return f'輸入值超過合法上限 {MAX_PRICE_LEN}'


params = [
    MacroParam(code='ma_long_period', name='長期均線天數', desc='長期均線天數',
               dtype=ParamType.get('int'), default=10),
    MacroParam(code='ma_mid_period', name='中期均線天數', desc='中期均線天數',
               dtype=ParamType.get('int'), default=5),
    MacroParam(code='ma_short_period', name='短期均線天數', desc='短期均線天數',
               dtype=ParamType.get('int'), default=1),
    MacroParam(code='period_type', name='K線週期', desc='K線週期',
               dtype=PeriodType, default=PeriodType.type.DAY)
]

__doc__ = """
短期MA小於中期MA小於長期MA (3條).

規則：
    MA(短期) < MA(中期) < MA(長期).

Arguments
---------
market_id : string
    目標市場ID
period_type : string
    MA 取樣週期，有效值為
    - 'D' : 日
    - 'W' : 週
    - 'M' : 月
ma_short_period : int
    短期均線天數.
ma_mid_period : int
    中期均線天數.
ma_long_period : int
    長期均線天數

"""


def _checker(**kwargs) -> dict:
    if 'period_type' in kwargs:
        del kwargs['period_type']
    ret = defaultdict(list)
    for key, value in kwargs.items():
        if _is_positive_int(value):
            ret[key].append(_is_positive_int(value))
        if _is_bounded(value):
            ret[key].append(_is_bounded(value))
    if kwargs['ma_short_period'] >= kwargs['ma_mid_period']:
        ret['ma_mid_period'].append('輸入值應大於短天期 MA')
    if kwargs['ma_mid_period'] >= kwargs['ma_long_period']:
        ret['ma_long_period'].append('輸入值應大於中短天期 MA')
    return {key: ', '.join(value) for key, value in ret.items() if value}


def _plotter(**kwargs) -> dict:
    if 'period_type' in kwargs:
        del kwargs['period_type']
    periods = [each for each in kwargs.values()]
    base = np.ones(shape=(MA_GRAPH_SAMPLE_NUM+2*max(periods)-min(periods),))
    curve_map = {
        'ma_long_period': {'start': 1, 'stop': 1},
        'ma_mid_period': {'start': 1, 'stop': 0.5},
        'ma_short_period': {'start': 1, 'stop': 0.5}
    }
    for key, value in kwargs.items():
        start = curve_map[key]['start']
        stop = curve_map[key]['stop']
        slope = np.linspace(start=start, stop=stop,
                            num=value+MA_GRAPH_SAMPLE_NUM)
        base[-(value+MA_GRAPH_SAMPLE_NUM)
               :] = base[-(value+MA_GRAPH_SAMPLE_NUM):] * slope

    fluc = (np.random.normal(scale=0.01, size=base.shape)+1)
    line = base*fluc*100
    ret = []
    for prd in periods:
        ret.append(PlotInfo(Ptype.MA, f'MA {prd}',_get_ma(
            line, prd)[-(MA_GRAPH_SAMPLE_NUM+max(periods)-min(periods)):]))
    return ret


def _framer(**kwargs) -> int:
    return 1


def _jack_ma_order_down(market_id: str, **kwargs) -> pd.Series:
    """短期MA小於中期MA小於長期MA (3條).

    規則：
        MA(短期) < MA(中期) < MA(長期).

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    ma_short_period : int
        短期均線天數.
    ma_mid_period : int
        中期均線天數.
    ma_long_period : int
        長期均線天數
    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        ma_short_period = kwargs['ma_short_period']
        ma_mid_period = kwargs['ma_mid_period']
        ma_long_period = kwargs['ma_long_period']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'jack_ma_order_down'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'jack_ma_order_down': {esp}")
    ma_short = TI.MA(market_id, ma_short_period, period_type)
    ma_mid = TI.MA(market_id, ma_mid_period, period_type)
    ma_long = TI.MA(market_id, ma_long_period, period_type)
    ret = (ma_short < ma_mid) & (ma_mid < ma_long)
    ret.rename(f'{market_id}.jack_ma_order_down({kwargs})')
    return ret.to_pandas()


jack_ma_order_down = Macro(code='jack_ma_order_down', name='MA由小至大短中長排列', desc=__doc__,
                           params=params, run=_jack_ma_order_down, check=_checker, plot=_plotter,
                           frame=_framer)
