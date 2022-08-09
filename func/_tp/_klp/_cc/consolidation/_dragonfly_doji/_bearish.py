# -*- coding: utf-8 -*-
"""Macro for Drafonfly Doji follows a bearish trend.

Created on Mon Aug  8 10:15:05 2022

@author: WaNiNi
"""

import numpy as np
from ._common import _CHT_NAME, _ENG_NAME
from ...common import (COMMON_PARAS, arg_checker, get_candlestick,
                      Candlestick, Macro, NumericTimeSeries,
                      TimeUnit, PlotInfo, Ptype, _TREND_PERIODS)
array = np.array
code = 'klp_cc_bearish_dragonfly_doji'
name = f'商智K線指標(KLP版)-下降{_CHT_NAME}({_ENG_NAME})'
description = (f"[下降趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於下降趨勢(近期收盤價呈向下趨勢)；\n"
               "2. 實體線長度為零或接近於零。\n"
               "3. 不存在上影線或上影線長度接近於零。\n"
               "4. 存在長下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，一開始賣盤壓力大；"
               "驅使價格持續下跌，但後來買盤力量增強，將價格推回開盤價的位置。"
               f"當{_CHT_NAME}出現在下降段時，表示趨勢可能會發生反轉。")
_DEFAULT_SAMPLES = {
    TimeUnit.DAY:
        array([[105.90482969, 107.70574957, 105.03787571, 105.59437442],
               [105.56081618, 105.76042337, 103.29605731, 104.66848779],
               [103.21134349, 104.57708693, 100.88659467, 104.39578003],
               [104.63641152, 104.63641152, 102.45427259, 104.02850945],
               [103.84358289, 104.35863995, 102.50660382, 104.35863995],
               [101.98534515, 102.1515099 ,  99.60737827, 100.04804947],
               [100.47203007, 102.01311477,  99.84096862, 101.19265552],
               [ 99.96548787, 102.42228315,  98.39147845, 101.94483994],
               [ 97.47984426,  97.47984426,  94.05931943,  94.54382294],
               [ 93.78147085,  93.78147085,  89.82436752,  91.24326268],
               [ 88.17545343,  88.17545343,  85.93401152,  88.06404682]]).T,
    TimeUnit.WEEK:
        array([[105.65086597, 107.93954329, 102.30159911, 106.67897035],
               [103.22077327, 107.95392018, 100.36005304, 106.99441931],
               [106.06673506, 108.83887379, 102.12398971, 102.96523702],
               [104.14751169, 105.22303118,  94.40421122,  99.15798072],
               [ 97.46327826,  97.78092136,  87.83408991,  90.39168659],
               [ 91.5264344 ,  91.57292545,  87.83002366,  91.57292545]]).T,
    TimeUnit.MONTH:
        array([[118.7558332 , 118.7558332 , 104.43156853, 110.09053423],
               [109.93823657, 113.1233852 ,  95.39333255,  95.39333255],
               [ 98.51604355, 100.95216865,  92.27217836,  92.52877023],
               [ 90.7323865 ,  90.7323865 ,  78.08345547,  90.30055471]]).T}
    
def _func(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_bearish
    cond_1 = (cct.is_doji_body &
              cct.is_without_uppershadow &
              cct.is_long_lowershadow)
    ret = cond_0 & cond_1
    return ret

def _macro(market: str, period_type: str):
    tunit = TimeUnit.get(period_type)
    cct = get_candlestick(market, tunit)
    ret = _func(cct).to_pandas().rename(f'{cct.name}.IsBearish{_ENG_NAME}')
    return ret

def _sample(period_type: str):
    tunit = TimeUnit.get(period_type)
    ret = [PlotInfo(Ptype.CANDLE, 'K', _DEFAULT_SAMPLES[tunit])]
    return ret

def _interval(period_type, **kwargs):
    tunit = TimeUnit.get(period_type)
    return 1 + _TREND_PERIODS[tunit][-1]

(klp_cc_bearish_dragonfly_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval,
          "2022-08-08-v1", "2022-08-08-v1", )
