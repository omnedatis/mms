# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:49:39 2022

@author: WaNiNi
"""

from typing import Any, Callable, Dict, List, NamedTuple
from func.common import Macro, MacroParam, ParamType

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
