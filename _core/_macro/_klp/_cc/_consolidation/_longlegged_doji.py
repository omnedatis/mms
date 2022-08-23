# -*- coding: utf-8 -*-
"""Common used by Macors in Long-Legged Doji's Series.

Created on Tue Aug 23 10:17:08 2022

@author: WaNiNi
"""

import numpy as np

from func._td import NumericTimeSeries, TimeUnit

from .._common import Candlestick, LeadingTrend, MacroInfo

LeadingTrends = LeadingTrend.type
_ENG_NAME = 'LongLeggedDoji'
_CHT_NAME = '長腳十字字線'

description = (f"[{_CHT_NAME}({_ENG_NAME})]\n"
               "判定規則：\n"
               "1. 市場處於指定趨勢(詳見參數`近期趨勢`)；\n"
               "2. 實體線長度為零或接近於零。\n"
               "3. 不存在上影線或上影線長度接近於零。\n"
               "4. 存在長下影線。\n"
               "現象說明：\n"
               f"基本上，{_CHT_NAME}的發生，表示在該段時間內，市場交易非常活躍；"
               "驅使價格劇烈波動，但後來還是回到接近開盤價的位置，表示市場目前方向未明。"
               f"在下降段出現{_CHT_NAME}，表示賣盤的優勢地位已經出現動搖，"
               "未來下降趨勢有可能趨緩甚至出現反轉。"
               f"在上升段出現{_CHT_NAME}，表示買盤的優勢地位已經出現動搖，"
               "未來上升趨勢有可能趨緩甚至出現反轉。"
               f"但，單獨的{_CHT_NAME}往往不能確定走勢是否會發生反轉，一般來說，"
               "還需要更多的訊號才能做進一步的確認。")

_DAY = np.array([[100., 102., 98., 100.]]).T
_WEEK = np.array([[100., 105., 95., 100.]]).T
_MONTH = np.array([[100., 110., 90., 100.]]).T

_DEFAULT_SAMPLES = {TimeUnit.DAY: _DAY,
                    TimeUnit.WEEK: _WEEK,
                    TimeUnit.MONTH: _MONTH}

def _func(cct: Candlestick) -> NumericTimeSeries:
    ret = (cct.is_doji_body &
           cct.is_long_uppershadow &
           cct.is_long_lowershadow &
           cct.is_close_to(cct.lowershadow, cct.uppershadow))
    return ret

longlegged_doji = MacroInfo(symbol=_ENG_NAME,
                            name=_CHT_NAME,
                            description=description,
                            func=_func,
                            interval=1,
                            samples=_DEFAULT_SAMPLES,
                            py_version="2201",
                            db_version="2201")
