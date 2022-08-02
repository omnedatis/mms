# -*- coding: utf-8 -*-
""" Module of classes and functions common used in current package.

Created on Wed Jun  1 16:12:29 2022

@author: WaNiNi

"""

from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple

import numpy as np
import pandas as pd

class Dtype(NamedTuple):
    code: str
    type: type

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
        raise ValueError(f"code: '{code}' was not found")

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
               'PARAM_TYPE': self.dtype.name}
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

if __name__ == '__main__':
    assert int is ParamType.get('int').type
    assert float is ParamType.get('float').type
    assert str is ParamType.get('string').type
