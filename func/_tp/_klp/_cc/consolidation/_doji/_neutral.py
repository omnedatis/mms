# -*- coding: utf-8 -*-
"""Macro for Doji no matter what the leading trend is.

Created on Mon Aug  8 10:15:12 2022

@author: WaNiNi
"""

import numpy as np
from ._common import _CHT_NAME, _ENG_NAME
from ...common import (COMMON_PARAS, arg_checker, get_candlestick,
                       Candlestick, Macro, NumericTimeSeries,
                       TimeUnit, PlotInfo, Ptype)


code = 'klp_cc_doji'
name = f'商智K線指標(KLP版)-上升{_CHT_NAME}({_ENG_NAME})'
description = (f"[上升趨勢-{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 實體線長度為零或接近於零。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，市場呈現多空拉鋸的僵局；"
               f"但，單獨的{_CHT_NAME}往往不能確定走勢，"
               "還需要更多的訊號才能做進一步的確認。")
_DEFAULT_SAMPLES = {
    TimeUnit.DAY:
        np.array([[100., 102., 98., 100.]]).T,
    TimeUnit.WEEK:
        np.array([[100., 105., 95., 100.]]).T,
    TimeUnit.MONTH:
        np.array([[100., 110., 90., 100.]]).T}

def _func(cct: Candlestick) -> NumericTimeSeries:
    return cct.is_doji_body

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

(klp_cc_doji
) = Macro(code, name, description, COMMON_PARAS, _macro,
          arg_checker, _sample, _interval,
          "2022-08-08-v1", "2022-08-08-v1", )
