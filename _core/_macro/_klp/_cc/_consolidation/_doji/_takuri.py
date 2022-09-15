# -*- coding: utf-8 -*-
"""Common used by Macors in Takuri Series.
Created on Tue Aug 23 09:54:20 2022

@author: WaNiNi
"""

import numpy as np

from func._td import NumericTimeSeries, TimeUnit

from ..._common import Candlestick, LeadingTrend, MacroInfo

LeadingTrends = LeadingTrend.type
_ENG_NAME = 'Takuri'
_CHT_NAME = '探水竿'

description = (f"[{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於指定趨勢(詳見參數`近期趨勢`)；\n"
               "2. 實體線長度為零或接近於零。\n"
               "3. 不存在上影線或上影線長度接近於零。\n"
               "4. 存在非常長的下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，一開始賣盤壓力很大；"
               "驅使價格持續下跌，但後來買盤力量增強，將價格推回接近開盤價的位置。"
               f"在下降段出現{_CHT_NAME}，表示賣盤壓力可能正在減弱，有可能出現反轉；"
               f"在上升段出現{_CHT_NAME}，表示買盤力量已經開始減弱，"
               "未來上升趨勢有可能趨緩甚至出現反轉。"
               f"但，單獨的{_CHT_NAME}往往不能確定走勢是否會發生反轉，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")

_DAY = np.array([[100., 100., 96., 100.]]).T
_WEEK = np.array([[100., 100., 90., 100.]]).T
_MONTH = np.array([[100., 100., 80., 100.]]).T

_DEFAULT_SAMPLES = {TimeUnit.DAY: _DAY,
                    TimeUnit.WEEK: _WEEK,
                    TimeUnit.MONTH: _MONTH}

def _func(cct: Candlestick) -> NumericTimeSeries:
    ret = (cct.is_doji_body &
           cct.is_without_uppershadow &
           cct.is_verylong_lowershadow)
    return ret

macro = MacroInfo(symbol=_ENG_NAME, name=_CHT_NAME, description=description,
                  func=_func, interval=1, samples=_DEFAULT_SAMPLES,
                  py_version="2201", db_version="2201")
