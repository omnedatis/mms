# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:57:55 2022

@author: WaNiNi
"""

from typing import Any, Dict, NamedTuple, List

import pandas as pd

from .._macro import Macro, MacroManager
from func.common import PlotInfo

class Pattern(NamedTuple):
    pid: str
    macro: Macro
    kwargs: Dict[str, Any]

    def check(self) -> Dict[str, str]:
        return self.macro.check_arguments(**self.kwargs)

    def run(self, market_id: str) -> pd.Series:
        return self.macro.evaluate(market_id, **self.kwargs)

    def plot(self) -> List[PlotInfo]:
        return self.macro.get_sample(**self.kwargs)

    def frame(self) -> int:
        return self.macro.get_interval(**self.kwargs)

    @classmethod
    def make(cls, pid: str, code: str, params: Dict[str, Any]):
        macro = MacroManager.get(code)
        if macro is None:
            raise RuntimeError(f"invalid macro code: '{code}'")
        # No checking for `params`
        return cls(pid=pid, macro=macro, kwargs=params)
