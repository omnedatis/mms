# -*- coding: utf-8 -*-
"""White Short-Line.

Created on Mon Sep 19 11:00:15 2022

@author: WaNiNi
"""

import numpy as np

from func._td import NumericTimeSeries, TimeUnit

from ..._common import Candlestick, LeadingTrend, MacroInfo

LeadingTrends = LeadingTrend.type
_ENG_NAME = 'WhiteShortLine'
_CHT_NAME = '小陽線'

description = (f"[{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於指定趨勢(詳見參數`近期趨勢`)；\n"
               "2. 短實體陽線。\n"
               "3. 短上影線。\n"
               "4. 短下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，市場處於多空膠著的狀態；"
               f"{_CHT_NAME}出現，表示市場目前方向未明，趨勢隨時有可能出現變化。"
               f"但，單獨的{_CHT_NAME}往往不能確定市場走勢，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")

array = np.array

_DAY = np.array([[100., 101.5, 99.5, 101.]]).T
_WEEK = np.array([[100., 103, 99., 102.]]).T
_MONTH = np.array([[100., 106., 98., 104.]]).T

_DEFAULT_SAMPLES = {TimeUnit.DAY: _DAY,
                    TimeUnit.WEEK: _WEEK,
                    TimeUnit.MONTH: _MONTH}

def _func(cct: Candlestick) -> NumericTimeSeries:
    ret = (~cct.is_doji_body & cct.is_short_body & cct.is_white_body &
           cct.is_short_uppershadow & cct.is_short_lowershadow)
    return ret

macro = MacroInfo(symbol=_ENG_NAME, name=_CHT_NAME, description=description,
                  func=_func, interval=1, samples=_DEFAULT_SAMPLES,
                  py_version="2201", db_version="2201")
