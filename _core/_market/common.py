# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:23:13 2022

@author: WaNiNi
"""

from typing import NamedTuple, Optional

class MarketInfo(NamedTuple):
    mid: str
    mtype: str
    category: str

    @classmethod
    def make(cls, mid: str, mtype: Optional[str], category: Optional[str]):
        return cls(mid, mtype or "", category or "")
