# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:51:03 2022

@author: WaNiNi
"""

from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import pandas as pd

from func.common import MacroParam, PlotInfo
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


def gen_macro(recv: _Macro) -> Macro:
    ret = Macro(code=recv.code, name=recv.name, description=recv.desc,
                parameters=recv.params, macro=recv.run,
                arg_checker=recv.check,
                sample_generator=recv.plot,
                interval_evaluator=recv.frame)
    return ret


class MacroManagerBase(Enum):
    @classmethod
    def get(cls, name: str) -> Optional[Macro]:
        if name in cls._member_map_:
            return cls._member_map_[name].value
        return None
