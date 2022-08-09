# -*- coding: utf-8 -*-
"""Macro for Doji follows a strictly bullish trend.

Created on Mon Aug  8 10:15:12 2022

@author: WaNiNi
"""

import numpy as np
from ._common import _CHT_NAME, _ENG_NAME
from ...common import (COMMON_PARAS, arg_checker, get_candlestick,
                       Candlestick, Macro, NumericTimeSeries,
                       TimeUnit, PlotInfo, Ptype, _TREND_PERIODS)

array = np.array
code = 'klp_cc_strictly_bullish_doji'
name = f'商智K線指標(KLP版)-嚴格上升{_CHT_NAME}({_ENG_NAME})'
description = (f"[嚴格上升趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於嚴格上升趨勢(近期收盤價與最低價皆呈向上趨勢)；\n"
               "2. 實體線長度為零或接近於零。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，市場呈現多空拉鋸的僵局；"
               f"在上升段出現{_CHT_NAME}，表示買盤壓力可能正在減弱，有可能出現反轉。"
               f"但，單獨的{_CHT_NAME}往往不能確定走勢是否會發生反轉，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")
_DEFAULT_SAMPLES = {
    TimeUnit.DAY:
        array([[ 90.5100704 ,  92.02587321,  89.12490727,  89.21549101],
               [ 90.75418753,  93.3376362 ,  90.08167911,  93.26803173],
               [ 93.29236864,  96.54535592,  92.92152583,  96.54535592],
               [ 94.93717016,  95.79811256,  94.50150279,  95.23173233],
               [ 93.60289218,  95.71801474,  93.13649627,  95.09243009],
               [ 98.03369499, 100.88485196,  97.44873679, 100.36543004],
               [103.61675906, 106.0302034 , 102.99182384, 105.77398424],
               [107.34695512, 109.58845184, 106.89334239, 107.73437469],
               [103.69995885, 107.03933669, 102.88085561, 105.9641595 ],
               [106.10818548, 110.2314323 , 106.00452296, 108.57993249],
               [109.36045131, 110.23539601, 108.06098288, 109.48534368]]).T,
    TimeUnit.WEEK:
        array([[ 88.93156032,  96.63693658,  88.93156032,  92.37788609],
               [ 93.49809155, 101.50344588,  92.88625596, 100.96102625],
               [100.01205876, 104.17351384,  98.87795772, 100.84962443],
               [ 97.44497166, 104.21480703,  96.66414382, 101.00533186],
               [100.65619032, 106.89589717,  98.64542229, 106.79396627],
               [107.17238037, 109.09697393, 104.32384727, 107.4461503 ]]).T,
    TimeUnit.MONTH:
        array([[ 83.10786228,  95.5730141 ,  82.26583755,  91.85749261],
               [ 91.05280871,  99.67965172,  82.65526127,  95.40221311],
               [ 98.48868549, 115.84142056,  97.22935015, 113.69633124],
               [114.48326186, 119.18324258, 105.46775866, 114.01580809]]).T}

def _func(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_strictly_bullish
    cond_1 = cct.is_doji_body
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

(klp_cc_strictly_bullish_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval,
          "2022-08-08-v1", "2022-08-08-v1", )
