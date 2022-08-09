# -*- coding: utf-8 -*-
"""Macro for Drafonfly Doji follows a strictly bearish trend.

Created on Mon Aug  8 10:15:05 2022

@author: WaNiNi
"""

import numpy as np
from ._common import _CHT_NAME, _ENG_NAME
from ...common import (COMMON_PARAS, arg_checker, get_candlestick,
                      Candlestick, Macro, NumericTimeSeries,
                      TimeUnit, PlotInfo, Ptype, _TREND_PERIODS)
array = np.array
code = 'klp_cc_strictly_bearish_dragonfly_doji'
name = f'商智K線指標(KLP版)-嚴格下降{_CHT_NAME}({_ENG_NAME})'
description = (f"[嚴格下降趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於嚴格下降趨勢(近期收盤價與最高價皆呈向下趨勢)；\n"
               "2. 實體線長度為零或接近於零。\n"
               "3. 不存在上影線或上影線長度接近於零。\n"
               "4. 存在長下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，一開始賣盤壓力大；"
               "驅使價格持續下跌，但後來買盤力量增強，將價格推回開盤價的位置。"
               f"當{_CHT_NAME}出現在下降段時，表示趨勢可能會發生反轉。")
_DEFAULT_SAMPLES = {
    TimeUnit.DAY:
        array([[106.3927353 , 112.21169062, 105.87207035, 110.45970571],
               [108.99758466, 111.10044365, 107.62129051, 110.49772733],
               [110.73353192, 111.28130065, 103.40258036, 103.40258036],
               [104.40811791, 106.24385134, 101.92974839, 102.73587932],
               [101.61941284, 103.51713156, 100.98491207, 103.45044963],
               [100.0137571 , 100.48922284,  98.19727407,  99.74910386],
               [ 98.6310997 ,  99.01482162,  94.75596993,  96.22970816],
               [ 94.43804473,  95.46633925,  94.17414714,  94.17414714],
               [ 97.2141026 ,  97.3293856 ,  92.58895177,  93.82713973],
               [ 92.34439497,  92.34439497,  90.23507282,  90.63615704],
               [ 90.96063428,  90.96521673,  88.49484821,  90.86332121]]).T,
    TimeUnit.WEEK:
        array([[109.81850683, 116.40775092, 108.10015726, 108.10015726],
               [104.07085225, 110.96930232, 104.07085225, 106.74100092],
               [106.1796343 , 111.12329   , 103.73943604, 105.89719377],
               [105.77282425, 105.77282425,  94.31956555,  95.29734917],
               [ 93.45222287,  94.21093056,  85.31506878,  86.57633088],
               [ 87.08562931,  87.08562931,  82.82049633,  87.07299461]]).T,
    TimeUnit.MONTH:
        array([[131.17285735, 131.7442727 , 106.08884004, 113.12783959],
               [111.57295451, 113.69980838,  97.01427141,  98.75277807],
               [ 97.75703771, 100.43858718,  83.76632418,  84.99338006],
               [ 84.66086672,  85.35176695,  76.04190464,  83.8165105 ]]).T}

def _func(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_strictly_bearish
    cond_1 = (cct.is_doji_body &
              cct.is_without_uppershadow &
              cct.is_long_lowershadow)
    ret = cond_0 & cond_1
    return ret

def _macro(market: str, period_type: str):
    tunit = TimeUnit.get(period_type)
    cct = get_candlestick(market, tunit)
    ret = _func(cct).to_pandas().rename(f'{cct.name}.IsStrictlyBearish{_ENG_NAME}')
    return ret

def _sample(period_type: str):
    tunit = TimeUnit.get(period_type)
    ret = [PlotInfo(Ptype.CANDLE, 'K', _DEFAULT_SAMPLES[tunit])]
    return ret

def _interval(period_type, **kwargs):
    tunit = TimeUnit.get(period_type)
    return 1 + _TREND_PERIODS[tunit][-1]

(klp_cc_strictly_bearish_dragonfly_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval,
          "2022-08-08-v1", "2022-08-08-v1", )
