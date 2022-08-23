# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:51:03 2022

@author: WaNiNi
"""

from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import pandas as pd
from const import MacroInfoField, MacroParamEnumField, MacroVersionInfoField
from func.common import Dtype, MacroParam, PlotInfo, ParamEnumBase
from func.common import Macro as _Macro

class Macro(NamedTuple):
    code: str
    name: str
    description: str
    parameters: List[MacroParam]
    macro: Callable
    sample_generator: Callable
    interval_evaluator: Callable
    arg_checker: Callable
    db_version: str = ""
    py_version: str = ""

    def filter_arguments(self, **kwargs) -> Dict[str, Any]:
        ret = {}
        for each in self.parameters:
            key = each.code
            if key not in kwargs:
                raise TypeError(f"{self.code} missing 1 required argument: "
                                f"'{key}'")
            value = kwargs[key]
            if not isinstance(value, each.dtype.type):
                raise TypeError(f"{self.code} {key} must be "
                                f"{each.dtype.type.__name__}, "
                                f"not {type(value).__name__}")
            ret[key] = value
        if len(ret) < len(kwargs):
            for key in kwargs:
                if key not in key:
                    raise TypeError(f"{self.code} got an unexpected argument: "
                                    f"'{key}'")
        return ret

    def evaluate(self, market_id, **kwargs) -> pd.Series:
        kwrags = self.filter_arguments(**kwargs)
        return self.macro(market_id, **kwrags)

    def get_sample(self, **kwargs) -> List[PlotInfo]:
        kwrags = self.filter_arguments(**kwargs)
        return self.sample_generator(**kwrags)

    def get_interval(self, **kwargs) -> int:
        kwrags = self.filter_arguments(**kwargs)
        return self.interval_evaluator(**kwrags)

    def check_arguments(self, **kwargs) -> Dict[str,str]:
        kwargs = self.filter_arguments(**kwargs)
        return self.arg_checker(**kwargs)

    def to_dict(self):
        ret = {MacroInfoField.MACRO_NAME.value: self.name,
               MacroInfoField.MACRO_DESC.value: self.description,
               MacroInfoField.FUNC_CODE.value: self.code,
               'PARAM': [each.to_dict() for each in self.parameters],
               MacroVersionInfoField.CODE_VERSION.value: self.py_version,
               MacroVersionInfoField.INFO_VERSION.value: self.db_version}
        return ret

def gen_macro(recv: _Macro) -> Macro:
    ret = Macro(code=recv.code, name=recv.name, description=recv.desc,
                db_version=recv.db_ver, py_version=recv.py_ver,
                parameters=recv.params, macro=recv.run,
                arg_checker=recv.check,
                sample_generator=recv.plot,
                interval_evaluator=recv.frame)
    return ret


class MacroManagerBase(Macro, Enum):
    @classmethod
    def get(cls, name: str) -> Optional[Macro]:
        if name in cls._member_map_:
            return cls._member_map_[name].value
        return None

    @classmethod
    def dump(cls):
        ret = [each.to_dict() for each in cls]
        return ret

class MacroParaEnumManagerBase(Dtype, Enum):
    @classmethod
    def get(cls, dtype: str, value: str) -> Any:
        for each in cls:
            if each.value.code == dtype:
                return each.value.type.get(value)
        return None

    @classmethod
    def dump(cls):
        values = []
        for each in cls:
            values += [[each.code, element.value.code, element.value.name] for element in each.value.type]
        ret = pd.DataFrame(values, columns=[
            MacroParamEnumField.ENUM_CODE.value,
            MacroParamEnumField.ENUM_VALUE_CODE.value,
            MacroParamEnumField.ENUM_VALUE_NAME.value])
        return ret
