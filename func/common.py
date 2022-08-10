# -*- coding: utf-8 -*-
""" Module of classes and functions common used in current package.

問題:
- 此檔案應該被移入 _tp 中。
- 此檔案與 _core中的_macro有重複包裝物件的問題，應予以合併。

Created on Wed Jun  1 16:12:29 2022

@author: WaNiNi

"""

from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple

import numpy as np
import pandas as pd

from ._td import TimeUnit

class Dtype(NamedTuple):
    code: str
    type: type

class ParamEnumElement(NamedTuple):
    code: str
    name: str
    data: Any

class ParamEnumBase(ParamEnumElement, Enum):
    @classmethod
    def get(cls, code: str):
        for each in cls:
            if each.value.code == code:
                return each
        return None

class _PeriodTypes(ParamEnumBase):
    DAY = ParamEnumElement('day', '日', TimeUnit.DAY)
    WEEK = ParamEnumElement('week', '週', TimeUnit.WEEK)
    MONTH = ParamEnumElement('month', '月', TimeUnit.MONTH)

PeriodType = Dtype('period_type', _PeriodTypes)

class ParamType(Dtype, Enum):
    """Parameter Type.

    Members
    -------
    INT, FLOAT, STR.

    methods
    -------
    get: Dtype
        get corresponding `Dtype` of given code.

    See Also
    --------
    Dtype.

    """
    INT = Dtype('int', int)
    FLOAT = Dtype('float', float)
    STR = Dtype('string', str)

    @classmethod
    def get(cls, code: str):
        """search member.

        Parameters
        ----------
        code: str

        Returns
        -------
        Dtype

        """
        for each in cls:
            if each.code == code:
                return each
        return None

class MacroParam(NamedTuple):
    """Parameter of Macro."""
    code: str
    name: str
    desc: str
    dtype: ParamType
    default: Any

    def to_dict(self):
        ret = {'PARAM_CODE': self.code,
               'PARAM_NAME': self.name,
               'PARAM_DESC': self.desc,
               'PARAM_DEFAULT': self.default,
               'PARAM_TYPE': self.dtype.code}
        return ret

class Ptype(Enum):
    CANDLE = 'Candle'
    OP = 'OP'
    HP = 'HP'
    LP = 'LP'
    CP = 'CP'
    MA = 'MA'

class PlotInfo(NamedTuple):
    ptype: Ptype
    title: str
    data: np.ndarray

class Macro(NamedTuple):
    """Macro.

    Notes:
    ------
    `run` must be a function with one default argument and
    a variable number of keyword arguments.

    """
    code: str
    name: str
    desc: str
    params: List[MacroParam]
    run: Callable[..., pd.Series]
    check: Callable[..., Dict[str, str]]
    plot: List[PlotInfo]
    frame: Callable[..., int]
    db_ver: str = ""
    py_ver: str = ""

if __name__ == '__main__':
    assert int is ParamType.get('int').type
    assert float is ParamType.get('float').type
    assert str is ParamType.get('string').type
