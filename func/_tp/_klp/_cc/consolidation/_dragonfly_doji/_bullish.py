# -*- coding: utf-8 -*-
"""Macro for Drafonfly Doji follows a bullish trend.

Created on Mon Aug  8 10:15:12 2022

@author: WaNiNi
"""

import numpy as np
from ._common import _CHT_NAME, _ENG_NAME
from ...common import (COMMON_PARAS, arg_checker, get_candlestick,
                       Candlestick, Macro, NumericTimeSeries,
                       TimeUnit, PlotInfo, Ptype, _TREND_PERIODS)

array = np.array
code = 'klp_cc_bullish_dragonfly_doji'
name = f'商智K線指標(KLP版)-上升{_CHT_NAME}({_ENG_NAME})'
description = (f"[上升趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於上升趨勢(近期收盤價呈向上趨勢)；\n"
               "2. 實體線長度為零或接近於零。\n"
               "3. 不存在上影線或上影線長度接近於零。\n"
               "4. 存在長下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，一開始賣盤壓力大；"
               "驅使價格持續下跌，但後來買盤力量增強，將價格推回開盤價的位置。"
               f"當{_CHT_NAME}出現在上升段時，表示趨勢可能會發生反轉。")
_DEFAULT_SAMPLES = {
    TimeUnit.DAY:
        array([[ 94.54390102,  96.02096828,  92.9417355 ,  96.02096828],
               [ 93.70881981,  96.07910096,  93.64204914,  93.74754742],
               [ 94.60149678,  97.22526779,  94.15712933,  95.87054664],
               [ 95.73593881,  98.77836462,  95.26762343,  98.38980864],
               [ 98.50901711, 100.40683449,  98.00304139,  99.37542269],
               [ 99.58041874,  99.96262541,  96.5912302 ,  97.64841277],
               [ 99.36698001, 103.12899768,  99.36698001, 103.12617434],
               [100.92773414, 102.34703019, 100.34668098, 101.16476277],
               [102.66158933, 106.22865483, 102.61104569, 106.22865483],
               [104.89176393, 107.91845447, 104.68561305, 107.91845447],
               [108.0765034 , 108.07766169, 106.18173324, 107.9362617 ]]).T,
    TimeUnit.WEEK:
        array([[ 89.25051465,  95.5862804 ,  89.25051465,  92.94350453],
               [ 92.47175407,  96.85622687,  92.07562644,  95.31897361],
               [ 94.74787329, 100.60721469,  93.79404979,  97.88499235],
               [100.17712746, 105.4021844 ,  98.69394985, 101.74320013],
               [105.25535972, 109.21728516,  99.88364442, 109.21728516],
               [111.31715728, 111.58198657, 105.45592808, 111.26736643]]).T,
    TimeUnit.MONTH:
        array([[ 81.54873906,  87.9803708 ,  77.63387536,  87.86160826],
               [ 87.0517301 , 105.79609112,  81.63932874, 104.54775214],
               [103.7844365 , 116.71648888, 101.7969603 , 115.16139639],
               [114.56581381, 114.56581381, 104.97999208, 114.36960267]]).T}

def _func(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_bullish
    cond_1 = (cct.is_doji_body &
              cct.is_without_uppershadow &
              cct.is_long_lowershadow)
    ret = cond_0 & cond_1
    return ret

def _macro(market: str, period_type: str):
    tunit = TimeUnit.get(period_type)
    cct = get_candlestick(market, tunit)
    ret = _func(cct).to_pandas().rename(f'{cct.name}.IsBullish{_ENG_NAME}')
    return ret

def _sample(period_type: str):
    tunit = TimeUnit.get(period_type)
    ret = [PlotInfo(Ptype.CANDLE, 'K', _DEFAULT_SAMPLES[tunit])]
    return ret

def _interval(period_type, **kwargs):
    tunit = TimeUnit.get(period_type)
    return 1 + _TREND_PERIODS[tunit][-1]

(klp_cc_bullish_dragonfly_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval,
          "2022-08-08-v1", "2022-08-08-v1", )
