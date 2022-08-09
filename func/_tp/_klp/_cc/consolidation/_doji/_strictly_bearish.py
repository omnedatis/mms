# -*- coding: utf-8 -*-
"""Macro for Doji follows a strictly bearish trend.

Created on Mon Aug  8 10:15:05 2022

@author: WaNiNi
"""

import numpy as np
from ._common import _CHT_NAME, _ENG_NAME
from ...common import (COMMON_PARAS, arg_checker, get_candlestick,
                      Candlestick, Macro, NumericTimeSeries,
                      TimeUnit, PlotInfo, Ptype, _TREND_PERIODS)
array = np.array
code = 'klp_cc_strictly_bearish_doji'
name = f'商智K線指標(KLP版)-嚴格下降{_CHT_NAME}({_ENG_NAME})'
description = (f"[嚴格下降趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於嚴格下降趨勢(收盤價與最高價呈向下趨勢)；\n"
               "2. 實體線長度為零或接近於零。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，市場呈現多空拉鋸的僵局；"
               f"在下降段出現{_CHT_NAME}，表示賣盤壓力可能正在減弱，有可能出現反轉。"
               f"但，單獨的{_CHT_NAME}往往不能確定走勢是否會發生反轉，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")
_DEFAULT_SAMPLES = {
    TimeUnit.DAY:
        array([[109.1897386 , 109.61497381, 106.3691468 , 107.33351181],
               [107.29624351, 109.58005901, 106.29865097, 108.51581462],
               [112.1163598 , 113.0415974 , 107.69014369, 107.87567887],
               [105.93483736, 106.81793588, 104.09309204, 105.88200216],
               [104.73238732, 106.24421653, 103.39885534, 103.81544569],
               [102.3546368 , 102.71009901, 100.0704614 , 100.8011002 ],
               [100.18935278, 100.84587858,  95.39673603,  95.69101941],
               [ 94.99671741,  95.9364899 ,  93.63475497,  94.2493404 ],
               [ 91.65934582,  92.33493238,  90.97496833,  90.97496833],
               [ 90.17766232,  90.30409808,  88.19765543,  88.19765543],
               [ 88.73043119,  89.16060098,  87.79095603,  88.77944758]]).T,
    TimeUnit.WEEK:
        array([[107.54246501, 112.87273589, 106.14572408, 106.27083291],
               [104.96909431, 106.82333059, 101.42204838, 103.20893305],
               [105.04610394, 105.04610394,  96.91309148, 103.26394464],
               [102.91378771, 104.40018116,  97.81744414,  99.59697608],
               [ 98.84040742, 101.43117859,  89.4057102 ,  89.4057102 ],
               [ 88.79086582,  91.24057588,  87.58722847,  89.04552611]]).T,
    TimeUnit.MONTH:
        array([[127.44893939, 127.44893939, 103.25663856, 106.66940995],
               [107.73218541, 109.66335641,  93.21646802,  97.96937156],
               [ 97.10241768,  99.35583864,  85.75791753,  90.06755737],
               [ 88.74056565,  94.11450998,  82.13061572,  89.32526871]]).T}

def _func(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_strictly_bearish
    cond_1 = cct.is_doji_body
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

(klp_cc_strictly_bearish_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval,
          "2022-08-08-v1", "2022-08-08-v1", )
