# -*- coding: utf-8 -*-
"""Macro - Doji.


"""

import numpy as np

from ..common import (COMMON_PARAS, LEADING_LEN, arg_checker, get_candlestick,
                      Candlestick, Macro, NumericTimeSeries,
                      TimeUnit, PlotInfo, Ptype, _TREND_PERIODS)

_ENG_NAME = 'DragonFlyDoji'
_CHT_NAME = 'T字線'

# Bullish Doji
code = 'klp_cc_bullish_dragonfly_doji'
name = f'商智K線指標(KLP版)-上升{_CHT_NAME}({_ENG_NAME})'
description = (f"[上升趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於上升趨勢；\n"
               "2. 實體線長度為零或接近於零。\n"
               "3. 不存在上影線或上影線長度接近於零。\n"
               "4. 存在很長的下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，一開始賣盤壓力大；"
               "驅使價格持續下跌，但後來買盤力量增強，將價格推回開盤價的位置。"
               f"當{_CHT_NAME}出現在上升段時，表示趨勢可能會發生反轉。")

_DEFAULT_SAMPLES = {
    'bullish': {
        TimeUnit.DAY:
            np.array([[243.74264833, 248.25259183, 239.40885924, 248.25259183],
                      [245.10000708, 253.8066747 , 245.10000708, 252.61666302],
                      [252.87977876, 257.70409007, 249.71488185, 256.53530063],
                      [258.60351355, 259.71269345, 254.17560155, 258.14539655],
                      [261.14329771, 266.6109667 , 260.37287186, 264.15034353],
                      [267.51873746, 268.16932227, 261.73635407, 267.31488368]]).T,
        TimeUnit.WEEK:
            np.array([[23.85545518, 23.86486739, 22.84519314, 23.29554368],
                      [23.30085278, 24.33545442, 23.09317399, 24.2563804 ],
                      [23.72640151, 25.05612254, 23.72640151, 24.66800211],
                      [25.13883262, 26.90506539, 24.27798635, 26.41033109],
                      [25.87687604, 28.24964641, 25.70943092, 27.50527613],
                      [27.86480097, 27.91953861, 25.70061617, 27.7166592 ]]).T,
        TimeUnit.MONTH:
            np.array([[231.83372462, 244.53997177, 218.65630697, 228.05884896],
                      [230.8093272 , 246.74302002, 227.30679232, 246.74302002],
                      [252.18352581, 274.89530141, 241.8136592 , 269.12659403],
                      [272.45683803, 318.37612647, 272.45683803, 305.58245602],
                      [312.23932548, 331.82340092, 295.73057058, 320.09935857],
                      [324.38839431, 325.152554  , 289.41525791, 325.152554  ]]).T},
    'bearish': {
        TimeUnit.DAY:
            np.array([[33.32797354, 33.93145053, 32.81150498, 33.68081061],
                      [32.69352455, 33.16291303, 32.57868509, 33.16291303],
                      [32.57279621, 32.78738251, 31.40488337, 31.50609818],
                      [31.00223009, 31.09400182, 30.27151843, 30.45182089],
                      [30.66809712, 30.66809712, 29.5592498 , 29.70889944],
                      [29.42840671, 29.49097901, 28.84065722, 29.35859573]]).T,
        TimeUnit.WEEK:
            np.array([[100.88468126, 104.18553318,  98.42617786, 103.02657474],
                      [ 98.60665627, 105.35128921,  98.44051908, 102.22649582],
                      [100.55011553, 100.55011553,  91.28777041,  98.160713  ],
                      [ 94.23939651,  94.23939651,  88.24440906,  90.2278368 ],
                      [ 87.34260856,  88.2873077 ,  80.8219208 ,  80.8219208 ],
                      [ 78.87465506,  79.29676416,  73.95320708,  78.87861098]]).T,
        TimeUnit.MONTH:
            np.array([[38.4948513 , 38.74770767, 33.94587838, 34.70477214],
                      [35.31719124, 35.55575368, 29.81468956, 30.53659033],
                      [32.24716339, 32.2805488 , 28.70703964, 28.84309782],
                      [29.33656848, 31.48382415, 27.75656421, 27.81910002],
                      [28.0606064 , 28.61333391, 25.68110252, 27.20404813],
                      [27.11219214, 27.21050041, 24.05273482, 27.07557185]]).T}}

def _bullish(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_bullish
    cond_1 = (cct.is_doji_body &
              cct.is_without_uppershadow &
              cct.is_long_lowershadow)
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

(klp_cc_bullish_dragonfly_doji
) = Macro(code, name, description, COMMON_PARAS, _bullish_macro,
          arg_checker, _bullish_sample, _bullish_interval)

# Bearish Doji
code = 'klp_cc_bearish_dragonfly_doji'
name = f'商智K線指標(KLP版)-下降{_CHT_NAME}({_ENG_NAME})'
description = (f"[下降趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於下降趨勢；\n"
               "2. 實體線長度為零或接近於零。\n"
               "3. 不存在上影線或上影線長度接近於零。\n"
               "4. 存在很長的下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，一開始賣盤壓力大；"
               "驅使價格持續下跌，但後來買盤力量增強，將價格推回開盤價的位置。"
               f"當{_CHT_NAME}出現在下降段時，表示趨勢可能會發生反轉。")

def _bearish(cct: Candlestick) -> NumericTimeSeries:
    cond_0 = cct.shift(1).is_bearish
    cond_1 = (cct.is_doji_body &
              cct.is_without_uppershadow &
              cct.is_long_lowershadow)
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

(klp_cc_bearish_dragonfly_doji
) = Macro(code, name, description, COMMON_PARAS, _bearish_macro,
          arg_checker, _bearish_sample, _bearish_interval)


