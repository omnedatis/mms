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
code = 'klp_cc_strictly_bullish_dragonfly_doji'
name = f'商智K線指標(KLP版)-嚴格上升{_CHT_NAME}({_ENG_NAME})'
description = (f"[嚴格上升趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於嚴格上升趨勢(近期收盤價與最低價皆呈向上趨勢)；\n"
               "2. 實體線長度為零或接近於零。\n"
               "3. 不存在上影線或上影線長度接近於零。\n"
               "4. 存在長下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，一開始賣盤壓力大；"
               "驅使價格持續下跌，但後來買盤力量增強，將價格推回開盤價的位置。"
               f"當{_CHT_NAME}出現在上升段時，表示趨勢可能會發生反轉。")
_DEFAULT_SAMPLES = {
    TimeUnit.DAY:
        array([[ 91.57859812,  93.6784101 ,  90.96005516,  92.97231488],
               [ 93.37790129,  96.21231288,  92.70183691,  95.62314374],
               [ 96.76280002,  97.87804992,  95.14076773,  95.33600098],
               [ 93.41153239,  98.33974939,  93.12763724,  98.33974939],
               [ 96.62726931, 100.93150214,  96.38182718,  99.99636612],
               [101.18904618, 103.28195933, 100.02531651, 102.40972425],
               [100.12213103, 102.56133085,  98.96329427, 101.69337955],
               [102.34666403, 104.27015375, 101.0077461 , 103.23447119],
               [103.7065092 , 104.32766875, 100.79157477, 101.53103459],
               [102.7077178 , 107.27770481, 102.7077178 , 107.27770481],
               [110.27679296, 110.40467403, 108.25763485, 110.25022371]]).T,
    TimeUnit.WEEK:
        array([[ 84.65889238,  89.31727127,  84.03564681,  89.31727127],
               [ 88.32863625,  96.18015055,  87.65470013,  93.59149162],
               [ 95.60752394,  95.87562245,  90.55912936,  95.6394888 ],
               [ 96.77234034, 104.92879769,  96.77234034, 104.24417712],
               [102.38051201, 117.34473657, 102.25091086, 115.67766905],
               [119.16283509, 119.22593189, 111.40106254, 119.07286168]]).T,
    TimeUnit.MONTH:
        array([[ 89.3661929 ,  98.13890211,  85.50604391,  91.27342523],
               [ 92.84461616, 100.0843463 ,  91.77475179,  98.98149115],
               [100.47164131, 110.56439244,  97.50375414, 110.51230659],
               [110.64102282, 110.92690003, 101.15304575, 110.25716737]]).T}

def _func(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_strictly_bullish
    cond_1 = (cct.is_doji_body &
              cct.is_without_uppershadow &
              cct.is_long_lowershadow)
    ret = cond_0 & cond_1
    return ret

def _macro(market: str, period_type: str):
    tunit = TimeUnit.get(period_type)
    cct = get_candlestick(market, tunit)
    ret = _func(cct).to_pandas().rename(f'{cct.name}.IsStrictlyBullish{_ENG_NAME}')
    return ret

def _sample(period_type: str):
    tunit = TimeUnit.get(period_type)
    ret = [PlotInfo(Ptype.CANDLE, 'K', _DEFAULT_SAMPLES[tunit])]
    return ret

def _interval(period_type, **kwargs):
    tunit = TimeUnit.get(period_type)
    return 1 + _TREND_PERIODS[tunit][-1]

(klp_cc_strictly_bullish_dragonfly_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval,
          "2022-08-08-v1", "2022-08-08-v1", )
