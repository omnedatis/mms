# -*- coding: utf-8 -*-
"""White Long-Line.
Created on Mon Sep 19 10:42:06 2022

@author: WaNiNi
"""

import numpy as np

from func._td import NumericTimeSeries, TimeUnit

from ..._common import Candlestick, LeadingTrend, MacroInfo

LeadingTrends = LeadingTrend.type
_ENG_NAME = 'WhiteLongLine'
_CHT_NAME = '大陽線'

description = (f"[{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於指定趨勢(詳見參數`近期趨勢`)；\n"
               "2. 長實體陽線。\n"
               "3. 短上影線。\n"
               "4. 短下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，由買方控制價格；"
               f"在下降段出現{_CHT_NAME}，表示賣盤壓力可能正在減弱，有可能出現反轉；"
               f"在上升段出現{_CHT_NAME}，表示買方力道依舊強勢，漲勢有可能繼續延續，"
               "但若出現於一段較長的上升段後，則價格可能處於危險的高點，有可能出現反轉。"
               f"但，單獨的{_CHT_NAME}往往不能確定市場走勢，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")

array = np.array

_DAY = np.array([[100., 102.5, 99.5, 102.]]).T
_WEEK = np.array([[100., 106, 99., 105.]]).T
_MONTH = np.array([[100., 112., 98., 110.]]).T

_DEFAULT_SAMPLES = {TimeUnit.DAY: _DAY,
                    TimeUnit.WEEK: _WEEK,
                    TimeUnit.MONTH: _MONTH}

def _func(cct: Candlestick) -> NumericTimeSeries:
    ret = (cct.is_long_body & cct.is_white_body &
           cct.is_short_uppershadow & cct.is_short_lowershadow)
    return ret

macro = MacroInfo(symbol=_ENG_NAME, name=_CHT_NAME, description=description,
                  func=_func, interval=1, samples=_DEFAULT_SAMPLES,
                  py_version="2201", db_version="2201")
