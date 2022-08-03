# -*- coding: utf-8 -*-
"""Macro - Doji.


"""

import numpy as np

from ..common import (COMMON_PARAS, LEADING_LEN, arg_checker, get_candlestick,
                      Candlestick, Macro, NumericTimeSeries,
                      TimeUnit, PlotInfo, Ptype)

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
            np.array([[190.3763738 , 194.55094141, 188.92057307, 191.07844663],
                      [194.1255597 , 194.79301044, 189.29533995, 193.83897524],
                      [196.84470727, 202.3791615 , 194.84533248, 201.35469973],
                      [203.38305314, 207.83019049, 202.68758944, 206.28304826],
                      [210.69056958, 214.42725361, 207.3514774 , 214.35701207],
                      [217.42689445, 217.77226733, 211.41264905, 217.14390845]]).T,
        TimeUnit.WEEK:
            np.array([[188.57687584, 210.59712537, 183.64337053, 210.59712537],
                      [216.31441992, 223.56201254, 209.37359975, 219.3103986 ],
                      [217.01829687, 234.69197059, 213.98140179, 224.84000477],
                      [225.67677757, 240.68229315, 225.67677757, 233.38059274],
                      [232.932727  , 245.69647007, 231.64859623, 245.35519678],
                      [244.95241964, 246.90363689, 237.02734084, 245.28298313]]).T,
        TimeUnit.MONTH:
            np.array([[275.17903668, 275.54436687, 234.10685277, 256.19007255],
                      [251.51659674, 281.49851336, 251.51659674, 260.73029023],
                      [259.15841004, 305.55559991, 255.75774999, 305.55559991],
                      [314.7966392 , 375.80998107, 313.48247369, 347.28106911],
                      [356.98205981, 436.90814878, 353.81627313, 418.89023011],
                      [419.30871911, 428.1762607 , 388.5061408 , 424.71341971]]).T},
    'bearish': {
        TimeUnit.DAY:
            np.array([[90.63529412, 91.81073519, 88.94514661, 91.81073519],
                      [92.47067441, 93.3539011 , 90.2869211 , 90.4662409 ],
                      [89.08609212, 90.12162504, 88.14849082, 89.27382895],
                      [88.15644068, 89.00046636, 86.8926421 , 87.88450226],
                      [86.00339301, 86.82494119, 84.91977972, 85.87684748],
                      [83.72804087, 83.84680288, 82.00110789, 83.61066623]]).T,
        TimeUnit.WEEK:
            np.array([[3166.17571104, 3227.1381957 , 2971.95850341, 2980.465229  ],
                      [2951.09490268, 3033.62704051, 2901.01594048, 3033.62704051],
                      [2982.53154096, 3015.70998931, 2877.94335655, 2881.14283283],
                      [2882.0940115 , 3008.42567262, 2834.64269883, 2876.09933747],
                      [2741.14899601, 2825.12721451, 2681.62338818, 2767.44304902],
                      [2757.3324286 , 2769.61571417, 2561.15175904, 2758.17146843]]).T,
        TimeUnit.MONTH:
            np.array([[354.98396184, 363.61941586, 321.17854737, 351.54863498],
                      [358.94784134, 365.02230962, 291.80156776, 291.90058421],
                      [294.99713345, 306.361338  , 278.130422  , 301.48017272],
                      [296.44991377, 315.4260512 , 279.92017048, 279.92017048],
                      [284.86703127, 288.78034666, 256.49029463, 268.24181645],
                      [269.14879986, 270.38553588, 231.80065172, 269.61736049]]).T}}

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
    return 1

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
    return 1

(klp_cc_bearish_dragonfly_doji
) = Macro(code, name, description, COMMON_PARAS, _bearish_macro,
          arg_checker, _bearish_sample, _bearish_interval)


