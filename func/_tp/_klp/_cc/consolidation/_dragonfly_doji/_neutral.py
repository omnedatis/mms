# -*- coding: utf-8 -*-
"""Macro for Drafonfly Doji no matter what the leading trend is.

Created on Mon Aug  8 10:15:12 2022

@author: WaNiNi
"""

import numpy as np
from ._common import _CHT_NAME, _ENG_NAME
from ...common import (COMMON_PARAS, arg_checker, get_candlestick,
                       Candlestick, Macro, NumericTimeSeries,
                       TimeUnit, PlotInfo, Ptype)


code = 'klp_cc_dragonfly_doji'
name = f'商智K線指標(KLP版)-上升{_CHT_NAME}({_ENG_NAME})'
description = (f"[上升趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 實體線長度為零或接近於零。\n"
               "2. 不存在上影線或上影線長度接近於零。\n"
               "3. 存在長下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，一開始賣盤壓力大；"
               "驅使價格持續下跌，但後來買盤力量增強，將價格推回開盤價的位置。"
               f"當{_CHT_NAME}出現在上升段時，表示趨勢可能會發生反轉；"
               f"單獨的{_CHT_NAME}往往不能確定走勢，"
               "還需要更多的訊號才能做進一步的確認。")
_DEFAULT_SAMPLES = {
    TimeUnit.DAY:
        np.array([[100, 100, 95, 100]]).T,
    TimeUnit.WEEK:
        np.array([[100, 100, 90, 100]]).T,
    TimeUnit.MONTH:
        np.array([[100, 100, 80, 100]]).T}

def _func(cct: Candlestick) -> NumericTimeSeries:
    ret = (cct.is_doji_body &
           cct.is_without_uppershadow &
           cct.is_long_lowershadow)
    return ret

def _macro(market: str, period_type: str):
    tunit = TimeUnit.get(period_type)
    cct = get_candlestick(market, tunit)
    ret = _func(cct).to_pandas().rename(f'{cct.name}.{_ENG_NAME}')
    return ret

def _sample(period_type: str):
    tunit = TimeUnit.get(period_type)
    ret = [PlotInfo(Ptype.CANDLE, 'K', _DEFAULT_SAMPLES[tunit])]
    return ret

def _interval(**kwargs):
    return 1

(klp_cc_dragonfly_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval,
          "2022-08-08-v1", "2022-08-08-v1", )
