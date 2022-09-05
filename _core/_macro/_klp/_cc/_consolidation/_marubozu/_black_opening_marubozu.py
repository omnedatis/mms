# -*- coding: utf-8 -*-
"""Common used by Macors in BlackOpeningMarubozu's Series.

Created on Wed Aug 10 09:17:48 2022

@author: WaNiNi
"""

import numpy as np

from func._td import NumericTimeSeries, TimeUnit

from ..._common import Candlestick, LeadingTrend, MacroInfo

LeadingTrends = LeadingTrend.type
_ENG_NAME = 'BlackOpeningMarubozu'
_CHT_NAME = '光頭陰線'

description = (f"[{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於指定趨勢(詳見參數`近期趨勢`)；\n"
               "2. 長實體陽線。\n"
               "3. 不存在上影線或上影線長度接近於零。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，由賣方控制價格；"
               f"在上升段出現{_CHT_NAME}，表示買盤壓力可能正在減弱，有可能出現反轉；"
               f"在下降段出現{_CHT_NAME}，表示賣方力道依舊強勢，跌勢有可能繼續延續，"
               "但若出現於一段較長的下降段後，則市場可能已進入最後的拋售，有可能出現反轉。"
               f"但，單獨的{_CHT_NAME}往往不能確定市場走勢，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")

array = np.array

_DAY = np.array([[100., 100., 97.5, 98.]]).T
_WEEK = np.array([[100., 100., 94., 95.]]).T
_MONTH = np.array([[100., 100., 88., 90.]]).T

_DEFAULT_SAMPLES = {TimeUnit.DAY: _DAY,
                    TimeUnit.WEEK: _WEEK,
                    TimeUnit.MONTH: _MONTH}

def _func(cct: Candlestick) -> NumericTimeSeries:
    ret = cct.is_long_body & cct.is_black_body & cct.is_without_uppershadow
    return ret

macro = MacroInfo(symbol=_ENG_NAME, name=_CHT_NAME, description=description,
                  func=_func, interval=1, samples=_DEFAULT_SAMPLES,
                  py_version="2201", db_version="2201")


