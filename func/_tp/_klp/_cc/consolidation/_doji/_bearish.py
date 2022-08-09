# -*- coding: utf-8 -*-
"""Macro for Doji follows a bearish trend.

Created on Mon Aug  8 10:15:05 2022

@author: WaNiNi
"""

import numpy as np
from ._common import _CHT_NAME, _ENG_NAME
from ...common import (COMMON_PARAS, arg_checker, get_candlestick,
                      Candlestick, Macro, NumericTimeSeries,
                      TimeUnit, PlotInfo, Ptype, _TREND_PERIODS)

array = np.array
code = 'klp_cc_bearish_doji'
name = f'商智K線指標(KLP版)-下降{_CHT_NAME}({_ENG_NAME})'
description = (f"[下降趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於下降趨勢(近期收盤價呈向下趨勢)；\n"
               "2. 實體線長度為零或接近於零。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，市場呈現多空拉鋸的僵局；"
               f"在下降段出現{_CHT_NAME}，表示賣盤壓力可能正在減弱，有可能出現反轉。"
               f"但，單獨的{_CHT_NAME}往往不能確定走勢是否會發生反轉，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")
_DEFAULT_SAMPLES = {
    TimeUnit.DAY:
        array([[107.43607949, 107.57060352, 105.52672367, 107.00282211],
               [106.96724228, 106.96724228, 104.36011984, 105.49342952],
               [105.19675028, 106.02314938, 103.65155741, 105.97168061],
               [105.8859552 , 106.47459048, 102.29687058, 102.42254169],
               [101.60497759, 103.84300585, 101.16735752, 101.7896093 ],
               [102.56297409, 102.56297409,  99.47404358, 101.38861042],
               [102.11104782, 102.15981963,  99.60972624, 100.17927006],
               [ 99.47054917,  99.60763733,  97.58919839,  97.95266875],
               [ 96.07562085,  96.20481142,  92.21565997,  92.21565997],
               [ 90.94935621,  91.55943837,  89.83261596,  90.30421913],
               [ 89.38995462,  90.63476756,  88.78855381,  89.50851398]]).T,
    TimeUnit.WEEK:
        array([[117.02272843, 117.02272843, 110.13610794, 113.51532331],
               [109.62624742, 111.57041992, 104.51516533, 105.47369689],
               [102.37197184, 103.6809365 ,  99.76163059,  99.76163059],
               [ 97.72363113,  98.80638819,  92.5625924 ,  94.50240621],
               [ 93.57017449,  95.37911086,  87.65713605,  89.18655945],
               [ 88.83300404,  92.54863746,  85.61639375,  89.15537878]]).T,
    TimeUnit.MONTH:
        array([[119.60909806, 119.60909806, 105.44300349, 108.94908092],
               [110.30866793, 113.99911885,  97.71984567, 101.76234588],
               [103.50780231, 105.5474006 ,  85.18507851,  85.91745046],
               [ 85.79156201,  88.32841153,  82.29673003,  86.02530567]]).T}

def _func(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_bearish
    cond_1 = cct.is_doji_body
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

(klp_cc_bearish_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval,
          "2022-08-08-v1", "2022-08-08-v1", )
