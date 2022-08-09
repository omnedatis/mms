# -*- coding: utf-8 -*-
"""Macro for Doji follows a bullish trend.

Created on Mon Aug  8 10:15:12 2022

@author: WaNiNi
"""

import numpy as np
from ._common import _CHT_NAME, _ENG_NAME
from ...common import (COMMON_PARAS, arg_checker, get_candlestick,
                       Candlestick, Macro, NumericTimeSeries,
                       TimeUnit, PlotInfo, Ptype, _TREND_PERIODS)

array = np.array
code = 'klp_cc_bullish_doji'
name = f'商智K線指標(KLP版)-上升{_CHT_NAME}({_ENG_NAME})'
description = (f"[上升趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於上升趨勢(近期收盤價呈向上趨勢)；\n"
               "2. 實體線長度為零或接近於零。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，市場呈現多空拉鋸的僵局；"
               f"在上升段出現{_CHT_NAME}，表示買盤壓力可能正在減弱，有可能出現反轉。"
               f"但，單獨的{_CHT_NAME}往往不能確定走勢是否會發生反轉，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")
_DEFAULT_SAMPLES = {
    TimeUnit.DAY:
        array([[ 96.59680249,  97.36712993,  95.07854336,  95.07854336],
               [ 95.56891239,  96.08284647,  92.91364088,  93.11928958],
               [ 93.62250584,  93.91649478,  91.65313139,  93.52449884],
               [ 91.62622019,  93.23944117,  91.62224286,  93.03781198],
               [ 93.75759959,  97.22707114,  93.57712441,  97.18303775],
               [100.08137853, 101.3219979 ,  99.10405417, 100.69469416],
               [101.0949989 , 103.73672002, 100.9467206 , 103.1721188 ],
               [102.78614315, 105.6567318 , 102.59957955, 103.4496747 ],
               [104.48977223, 107.73937831, 103.8695856 , 107.38536895],
               [106.86197284, 108.47467189, 105.85490218, 107.8467497 ],
               [109.40464819, 110.51513893, 107.68251383, 109.43759667]]).T,
    TimeUnit.WEEK:
        array([[ 88.89149823,  94.14042939,  88.89149823,  93.65375524],
               [ 94.34470325,  97.10332494,  92.53896716,  95.82581861],
               [ 97.86244248, 104.62054074,  97.53175777, 101.19092806],
               [100.23191707, 103.28760667,  95.79220207, 102.2109229 ],
               [103.72717074, 107.83860948, 102.98839205, 107.83860948],
               [107.08739405, 109.95719088, 105.36848844, 107.07583209]]).T,
    TimeUnit.MONTH:
        array([[ 78.31195083,  90.31719617,  78.26266793,  87.73332667],
               [ 88.57080097, 103.98407893,  87.24103042, 100.96000505],
               [103.1539607 , 118.49715531,  99.10974004, 110.79646607],
               [113.89710322, 118.68656108, 106.50025715, 113.97769944]]).T}

def _func(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_bullish
    cond_1 = cct.is_doji_body
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

(klp_cc_bullish_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval,
          "2022-08-08-v1", "2022-08-08-v1", )
