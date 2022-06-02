# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:05:39 2022

@author: WaNiNi
"""

from enum import IntEnum

import numpy as np

from ..common import gen_macro, RawMacro, MacroParam, ParamType
   
def get_ma(data: np.ndarray, period: int):
    if not isinstance(period, int):
        raise TypeError(f"`period` must be 'int' not '{type(period).__name__}'")
    if period <= 0:
        raise ValueError("`period` must be positive")
    if period == 1:
        return data
    temp = np.cumsum(data)
    ret = np.concatenate([temp[:period] / np.arange(1, period+1),
                          (temp[period:] - temp[:-period]) / period])
    return ret
