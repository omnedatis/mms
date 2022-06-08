# -*- coding: utf-8 -*-
""" 
Created on Thur Jun  2 10:30:29 2022

@author: Jeff
"""
from collections import defaultdict
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from ...common import Macro, MacroParam, ParamType, PlotInfo, Ptype
from .._context import TimeUnit, get_cp, ts_any
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


params = [
    MacroParam(code='period_1', name='MA均線天數(最小)', desc='MA均線天數(最小)',
               dtype=ParamType.get('int'), default=3),
    MacroParam(code='period_2', name='MA均線天數(第二小)', desc='MA均線天數(第二小)',
               dtype=ParamType.get('int'), default=10),
    MacroParam(code='period_3', name='MA均線天數(第三小)', desc='MA均線天數(第三小)',
               dtype=ParamType.get('int'), default=15),
    MacroParam(code='period_4', name='MA均線天數(第四小)', desc='MA均線天數(第四小)',
               dtype=ParamType.get('int'), default=20),
    MacroParam(code='period_5', name='MA均線天數(第五小)', desc='MA均線天數(第五小)',
               dtype=ParamType.get('int'), default=40),
    MacroParam(code='period_6', name='MA均線天數(第六小)', desc='MA均線天數(第六小)',
               dtype=ParamType.get('int'), default=60),
    MacroParam(code='period_7', name='MA均線天數(第七小)', desc='MA均線天數(第七小)',
               dtype=ParamType.get('int'), default=80),
    MacroParam(code='period_8', name='MA均線天數(第八小)', desc='MA均線天數(第八小)',
               dtype=ParamType.get('int'), default=100),
    MacroParam(code='period_9', name='MA均線天數(第九小)', desc='MA均線天數(第九小)',
               dtype=ParamType.get('int'), default=120),
    MacroParam(code='period_type', name='K線週期', desc='K線週期',
               dtype=ParamType.get('string'), default='day'),
]

__doc__ = """
任一短天期MA向下穿越長天期MA (9條).

規則：
    任一短天期 MA 向下穿越任一長天期 MA.

Arguments
---------
market_id : string
    目標市場ID
period_type : string
    MA 取樣週期，有效值為
    - 'D' : 日
    - 'W' : 週
    - 'M' : 月
period_1 : int
    MA均線天數(最小).
period_2 : int
    MA均線天數(第二小).
period_3 : int
    MA均線天數(第三小).
period_4 : int
    MA均線天數(第四小).
period_5 : int
    MA均線天數(第五小).
period_6 : int
    MA均線天數(第六小).
period_7 : int
    MA均線天數(第七小).
period_8 : int
    MA均線天數(第八小).
period_9 : int
    MA均線天數(第九小).
    
"""


def _checker(**kwargs) -> dict:
    if 'period_type' in kwargs:
        del kwargs['period_type']
    ret = defaultdict(list)
    for key, value in kwargs.items():
        if _is_positive_int(value):
            ret[key].append(_is_positive_int(value))
    length = ret['period_1']
    period = '第一MA均線天數'
    if ret['period_2'] < length:
        ret['period_2'].append(f'輸入值應大於{period}')
    else:
        length = ret['period_2']
        period = '第二MA均線天數'
    if ret['period_3'] < length:
        ret['period_3'].append(f'輸入值應大於{period}')
    else:
        length = ret['period_3']
        period = '第三MA均線天數'
    if ret['period_4'] < length:
        ret['period_4'].append(f'輸入值應大於{period}')
    else:
        length = ret['period_4']
        period = '第四MA均線天數'
    if ret['period_5'] < length:
        ret['period_5'].append(f'輸入值應大於{period}')
    else:
        length = ret['period_5']
        period = '第五MA均線天數'
    if ret['period_6'] < length:
        ret['period_6'].append(f'輸入值應大於{period}')
    else:
        length = ret['period_6']
        period = '第六MA均線天數'
    if ret['period_7'] < length:
        ret['period_7'].append(f'輸入值應大於{period}')
    else:
        length = ret['period_7']
        period = '第七MA均線天數'
    if ret['period_8'] < length:
        ret['period_8'].append(f'輸入值應大於{period}')
    else:
        length = ret['period_8']
        period = '第八MA均線天數'
    if ret['period_9'] < length:
        ret['period_9'].append(f'輸入值應大於{period}')
    return {key: ', '.join(value) for key, value in ret.items() if value}

def _plotter(**kwargs) -> dict:
    if 'period_type' in kwargs:
        del kwargs['period_type']
    periods = [each for each in kwargs.values()]
    base = np.ones(shape=(MA_GRAPH_SAMPLE_NUM+2*max(periods)-min(periods),))
    curve_map = {
        'period_1': {'start': 1, 'stop': 0.3},
        'period_2': {'start': 1, 'stop': 0.7},
        'period_3': {'start': 1, 'stop': 0.8},
        'period_4': {'start': 1, 'stop': 0.8},
        'period_5': {'start': 1, 'stop': 1},
        'period_6': {'start': 1, 'stop': 1.2},
        'period_7': {'start': 1, 'stop': 1.2},
        'period_8': {'start': 1, 'stop': 1.2},
        'period_9': {'start': 1, 'stop': 1.4},
    }
    for key, value in kwargs.items():
        start = curve_map[key]['start']
        stop = curve_map[key]['stop']
        slope = np.linspace(start=start, stop=stop,
                            num=value+MA_GRAPH_SAMPLE_NUM)
        base[-(value+MA_GRAPH_SAMPLE_NUM):] = base[-(value+MA_GRAPH_SAMPLE_NUM):] * slope
    fluc = (np.random.normal(scale=0.1, size=base.shape)+1)
    line = base*fluc*100
    ret = []
    for prd in periods:
        ret.append(PlotInfo(Ptype.MA, f'MA {prd}', _get_ma(
            line, prd)[-(MA_GRAPH_SAMPLE_NUM+max(periods)-min(periods)):]))
    return ret


def _framer(**kwargs) -> int:
    return MA_GRAPH_SAMPLE_NUM

def _jack_ma_through_ma_down_trend(market_id: str, **kwargs) -> pd.Series:
    """任一短天期MA向下穿越長天期MA (9條).

    規則：
        任一短天期 MA 向下穿越任一長天期 MA.

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    period_1 : int
        MA均線天數(最小).
    period_2 : int
        MA均線天數(第二小).
    period_3 : int
        MA均線天數(第三小).
    period_4 : int
        MA均線天數(第四小).
    period_5 : int
        MA均線天數(第五小).
    period_6 : int
        MA均線天數(第六小).
    period_7 : int
        MA均線天數(第七小).
    period_8 : int
        MA均線天數(第八小).
    period_9 : int
        MA均線天數(第九小).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        periods = [kwargs[f'period_{idx}'] for idx in range(1, 10)]
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'jack_ma_through_ma_down_trend'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'jack_ma_through_ma_down_trend': {esp}")
    mas = [TI.MA(market_id, each, period_type) for each in periods]
    conds = [mas[i] < each for i in range(len(mas)) for each in mas[i+1:]]
    conds = [each & ~each.shift(1, period_type) for each in conds]
    ret = ts_any(*conds)
    ret.rename(f'{market_id}.jack_ma_through_ma_down_trend({kwargs})')
    return ret.to_pandas()


jack_ma_through_ma_down_trend = Macro(code='jack_ma_through_ma_down_trend', name='短期MA向下穿越長期MA', desc=__doc__,
                                       params=params, run=_jack_ma_through_ma_down_trend, check=_checker, plot=_plotter,
                                       frame=_framer)
