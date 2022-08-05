# -*- coding: utf-8 -*-
"""Macro - Doji.


"""

import numpy as np

from ..common import (COMMON_PARAS, LEADING_LEN, arg_checker, get_candlestick,
                      Candlestick, Macro, NumericTimeSeries,
                      TimeUnit, PlotInfo, Ptype, _TREND_PERIODS)

_ENG_NAME = 'Doji'
_CHT_NAME = '十字線'

# Bullish Doji
code = 'klp_cc_bullish_doji'
name = f'商智K線指標(KLP版)-上升{_CHT_NAME}({_ENG_NAME})'
description = (f"[上升趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於上升趨勢；\n"
               "2. 實體線長度為零或接近於零。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，市場呈現多空拉鋸的僵局；"
               f"在上升段出現{_CHT_NAME}，表示買盤壓力可能正在減弱，有可能出現反轉。"
               f"但，單獨的{_CHT_NAME}往往不能確定走勢是否會發生反轉，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")

_DEFAULT_SAMPLES = {
    'bullish': {
        TimeUnit.DAY:
            np.array([[ 84.4863159 ,  86.84657014,  84.47491911,  86.62607313],
                      [ 87.32264819,  89.8698186 ,  86.04614609,  89.13169785],
                      [ 92.47846503,  93.24392696,  89.61171425,  89.69994661],
                      [ 91.38479328,  92.36591891,  90.41722138,  92.27956827],
                      [ 94.73821716, 100.61688441,  94.73821716,  99.98584462],
                      [101.68412791, 102.28060632,  99.3908225 , 101.4947468 ]]).T,
        TimeUnit.WEEK:
            np.array([[42.2283645 , 42.53122566, 39.96735303, 40.55635767],
                      [40.74422625, 43.55540008, 40.74422625, 42.54930909],
                      [41.03874151, 43.31707736, 40.43633494, 43.31707736],
                      [42.8838677 , 46.25594705, 42.59612551, 45.06039692],
                      [45.35262999, 49.18285807, 44.94002694, 49.18285807],
                      [51.03633277, 52.59480234, 50.28204686, 50.77501327]]).T,
        TimeUnit.MONTH:
            np.array([[222.31779699, 241.39581858, 213.11563675, 240.22599385],
                      [243.06959121, 275.33243345, 219.31091639, 224.27727437],
                      [233.23049921, 266.63732406, 233.23049921, 259.47464583],
                      [261.50212993, 326.57798063, 248.33879467, 308.58156045],
                      [314.91062506, 452.71584215, 306.47200494, 445.93399841],
                      [450.16397962, 481.6504831 , 420.55210148, 449.53041753]]).T},
    'bearish': {
        TimeUnit.DAY:
            np.array([[107.1445515 , 110.90469163, 106.60655735, 110.29025517],
                      [107.53223566, 108.30396955, 105.53534961, 106.13406029],
                      [106.30653842, 108.51345122, 105.52809169, 107.78997529],
                      [105.16722711, 105.72509053, 102.93540736, 104.11402458],
                      [102.8507499 , 103.39992318, 100.55266249, 101.87969375],
                      [ 99.48894611, 101.02796826,  98.690497  ,  99.5711917 ]]).T,
        TimeUnit.WEEK:
            np.array([[108.97918642, 110.41566974, 105.27426778, 106.40983659],
                      [105.49238225, 107.90993216, 102.61425975, 105.05025041],
                      [100.47595813, 101.08560003,  91.21199434,  92.12612942],
                      [ 93.67218882,  99.65320111,  92.23576164,  92.69931374],
                      [ 95.24263036,  95.24263036,  87.30892508,  89.16397757],
                      [ 88.4345157 ,  91.40639334,  86.22999751,  88.03245003]]).T,
        TimeUnit.MONTH:
            np.array([[39.36498546, 39.36498546, 32.56835408, 32.62394686],
                      [32.89941502, 32.89941502, 28.0238041 , 30.36482146],
                      [30.58358852, 31.33863414, 26.90140177, 27.25307794],
                      [27.52645125, 29.24206903, 23.68811081, 24.77975011],
                      [23.93830858, 24.67555546, 20.88951262, 21.44174279],
                      [21.46508683, 22.85152779, 20.13675824, 21.28230393]]).T}}

def _bullish(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_bullish
    cond_1 = cct.is_doji_body
    ret = cond_0 & cond_1
    return ret

def _bullish_macro(market: str, period_type: str):
    tunit = TimeUnit.get(period_type)
    cct = get_candlestick(market, tunit)
    ret = _bullish(cct).to_pandas().rename(f'{cct.name}.IsBullish{_ENG_NAME}')
    return ret

def _bullish_sample(period_type: str):
    tunit = TimeUnit.get(period_type)
    ret = [PlotInfo(Ptype.CANDLE, 'K', _DEFAULT_SAMPLES['bullish'][tunit])]
    return ret

def _bullish_interval(**kwargs):
    return 1 + _TREND_PERIODS[-1]

(klp_cc_bullish_doji
) = Macro(code, name, description, COMMON_PARAS, _bullish_macro,
          arg_checker, _bullish_sample, _bullish_interval)

# Bearish Doji
code = 'klp_cc_bearish_doji'
name = f'商智K線指標(KLP版)-下降{_CHT_NAME}({_ENG_NAME})'
description = (f"[下降趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於下降趨勢；\n"
               "2. 實體線長度為零或接近於零。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，市場呈現多空拉鋸的僵局；"
               f"在下降段出現{_CHT_NAME}，表示賣盤壓力可能正在減弱，有可能出現反轉。"
               f"但，單獨的{_CHT_NAME}往往不能確定走勢是否會發生反轉，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")

def _bearish(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_bearish
    cond_1 = cct.is_doji_body
    ret = cond_0 & cond_1
    return ret

def _bearish_macro(market: str, period_type: str):
    tunit = TimeUnit.get(period_type)
    cct = get_candlestick(market, tunit)
    ret = _bearish(cct).to_pandas().rename(f'{cct.name}.IsBearish{_ENG_NAME}')
    return ret

def _bearish_sample(period_type: str):
    tunit = TimeUnit.get(period_type)
    ret = [PlotInfo(Ptype.CANDLE, 'K', _DEFAULT_SAMPLES['bearish'][tunit])]
    return ret

def _bearish_interval(**kwargs):
    return 1 + _TREND_PERIODS[-1]

(klp_cc_bearish_doji
) = Macro(code, name, description, COMMON_PARAS, _bearish_macro,
          arg_checker, _bearish_sample, _bearish_interval)


