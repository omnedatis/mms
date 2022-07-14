# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:49:39 2022

@author: WaNiNi
"""

from typing import Any, Callable, List, NamedTuple, Optional, Union
from func.common import Macro, MacroParam, ParamType, PlotInfo, Ptype

_Number = Union[int, float]

class LimitedCondition(NamedTuple):
    expr: Callable[[Any, ], bool]
    message: str

    def check(self, value):
        return self.expr(value)


class LimitedVariable(NamedTuple):
    lower: Optional[_Number] = None
    upper: Optional[_Number] = None
    valids: Optional[List[_Number]] = None
    invalids: Optional[List[_Number]] = None

    @classmethod
    def make(cls, lower=None, upper=None, valids=None, invalids=None):
        return cls(lower, upper, valids, invalids)


    def _check_valids(self, value):
        if value not in self.valids:
            return f"有效值為{','.join(self.valids)}"

    def check(self, value):
        if self.valids:
            return value in self.valids
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        if self.invalids and value in self.invalids:
            return False
        return True

    @property
    def message(self):
        if self.valids:
            return f"有效值為{','.join(self.valids)}"
        # range conditions
        if self.lower is None:
            if self.upper is None:
                ret = ""
            else:
                ret = f"不得大於{self.upper}"
        elif self.upper is None:
            ret = f"不得小於於{self.lower}"
        else:
            ret = f"必須介於{self.lower}~{self.upper}"
        if self.invalids:
            msg = f"不得為{','.join(map(str, self.invalids))}"
            if ret == "":
                ret = msg
            else:
                ret = f"{ret}且{msg}"
        return ret
    
class RawMacro(NamedTuple):
    code: str
    name: str
    description: str
    parameters: List[MacroParam]
    macro: Callable
    sample_generator: Callable
    interval_evaluator: Callable
    arg_checker: Callable

    def filter_arguments(self, **kwargs):
        ret = {}
        for each in self.parameters:
            key = each.code
            if key not in kwargs:
                raise TypeError(f"{self.code} missing 1 required argument: '{key}'")
            value = kwargs[key]
            if not isinstance(value, each.dtype.type):
                error_msg = (f"{self.code} {key} must be "
                             f"{each.dtype.type.__name__}, not {type(value).__name__}")
                raise TypeError(error_msg)
            ret[key] = value
        return ret

    def execute(self, market_id, **kwargs):
        kwrags = self.filter_arguments(**kwargs)
        return self.macro(market_id, **kwrags).to_pandas()

    def get_sample(self, **kwargs):
        kwrags = self.filter_arguments(**kwargs)
        return self.sample_generator(**kwrags)

    def get_interval(self, **kwargs):
        kwrags = self.filter_arguments(**kwargs)
        return self.interval_evaluator(**kwrags)

    def check_arguments(self, **kwargs):
        kwargs = self.filter_arguments(**kwargs)
        return self.arg_checker(**kwargs)

class Pattern(NamedTuple):
    macro: Macro
    kwargs: Dict[str, Any]

    def check_arguments(self):
        return self.macro.check_arguments(**self.kwargs)

    def execute(self, market_id: str):
        return self.macro.execute(market_id, **self.kwargs)

    @property
    def sample(self):
        return self.macro.get_sample(**self.kwargs)

    @property
    def interval(self) -> int:
        return self.macro.get_interval(**self.kwargs)

def gen_macro(recv: RawMacro) -> Macro:
    ret = Macro(code=recv.code, name=recv.name, desc=recv.description,
                params=recv.parameters, run=recv.execute,
                check=recv.check_arguments, plot=recv.get_sample,
                frame=recv.get_interval)
    return ret
