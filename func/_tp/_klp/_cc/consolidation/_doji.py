# -*- coding: utf-8 -*-
"""Macro - Doji.


"""

import numpy as np

from ..common import (COMMON_PARAS, LEADING_LEN, arg_checker, get_candlestick,
                      Candlestick, Macro, NumericTimeSeries,
                      TimeUnit, PlotInfo, Ptype)

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
            np.array([[75.48207395, 77.98480587, 74.80489363, 77.8324632 ],
                      [77.66291632, 79.96655475, 77.15818456, 79.20734238],
                      [79.98030175, 81.4738077 , 78.72464422, 80.41897585],
                      [82.95992913, 84.62770021, 82.95992913, 84.06827553],
                      [86.14753874, 86.59701702, 84.30381149, 84.95626812],
                      [86.31172403, 87.21899214, 85.7440667 , 86.16605868]]),
        TimeUnit.WEEK:
            np.array([[ 91.46687815,  92.54620307,  88.47633535,  92.51393525],
                      [ 95.83405432, 106.05630605,  93.53623503, 105.9374337 ],
                      [108.76122705, 111.87115193, 103.94033072, 106.8787178 ],
                      [109.21270783, 116.74423255, 109.21270783, 113.01149585],
                      [114.74611663, 119.54890125, 113.1248009 , 115.61869273],
                      [118.00763804, 121.43075317, 115.31471967, 117.99319273]]),
        TimeUnit.MONTH:
            np.array([[ 75.72824464,  88.79213053,  75.11696398,  87.30340966],
                      [ 88.31809887,  98.23926931,  88.31809887,  93.12331931],
                      [ 96.28722625, 110.69533732,  95.17598504, 105.62299826],
                      [106.95538836, 127.75306475, 104.50316678, 124.72184484],
                      [125.90294395, 146.05131448, 124.25493068, 142.6425251 ],
                      [144.71555368, 149.88426998, 129.57213255, 145.51681529]])},
    'bearish': {
        TimeUnit.DAY:
            np.array([[112.04227086, 112.81792123, 110.97659952, 112.81792123],
                      [111.46103093, 111.69021512, 108.56526955, 108.78340707],
                      [108.81308085, 108.91739733, 104.76920531, 105.28555031],
                      [101.88847701, 106.13810994, 101.88847701, 106.13810994],
                      [103.60092352, 105.19114654, 102.27022227, 102.36491036],
                      [ 99.79624986, 101.93357416,  99.0331805 ,  99.77868963]]),
        TimeUnit.WEEK:
            np.array([[108.97918642, 110.41566974, 105.27426778, 106.40983659],
                      [105.49238225, 107.90993216, 102.61425975, 105.05025041],
                      [100.47595813, 101.08560003,  91.21199434,  92.12612942],
                      [ 93.67218882,  99.65320111,  92.23576164,  92.69931374],
                      [ 95.24263036,  95.24263036,  87.30892508,  89.16397757],
                      [ 88.4345157 ,  91.40639334,  86.22999751,  88.03245003]]),
        TimeUnit.MONTH:
            np.array([[36.78761049, 37.51003709, 33.67738947, 36.09139617],
                      [36.20294796, 40.1288599 , 34.15467   , 34.15467   ],
                      [35.32754031, 36.44175495, 30.94517581, 32.33528067],
                      [31.22812724, 31.7535821 , 25.8972272 , 26.97526877],
                      [26.27720842, 26.27720842, 21.22638055, 22.35962483],
                      [22.12560108, 23.81965021, 21.37566654, 22.23895376]])}}

def _bullish(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_bullish
    cond_1 = cct.is_doji_body
    ret = cond_0 & cond_1
    return ret

def _macro(market: str, period_type: str):
    tunit = TimeUnit.get(period_type)
    cct = get_candlestick(market, tunit)
    ret = _bullish(cct).to_pandas().rename(f'{cct.name}.IsBullish{_ENG_NAME}')
    return ret

def _sample(period_type: str):
    tunit = TimeUnit.get(period_type)
    ret = [PlotInfo(Ptype.CANDLE, 'K', _DEFAULT_SAMPLES['bullish'][tunit])]
    return ret

def _interval(**kwargs):
    return 1

(klp_cc_bullish_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval)

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

def _macro(market: str, period_type: str):
    tunit = TimeUnit.get(period_type)
    cct = get_candlestick(market, tunit)
    ret = _bearish(cct).to_pandas().rename(f'{cct.name}.IsBearish{_ENG_NAME}')
    return ret

def _sample(period_type: str):
    tunit = TimeUnit.get(period_type)
    ret = [PlotInfo(Ptype.CANDLE, 'K', _DEFAULT_SAMPLES['bearish'][tunit])]
    return ret

def _interval(**kwargs):
    return 1

(klp_cc_bearish_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval)

