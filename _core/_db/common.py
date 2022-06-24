# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:43:48 2022

@author: WaNiNi
"""

from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from .._pattern import Pattern

class MimosaDBMeta(metaclass=ABCMeta):
    @abstractmethod
    def get_markets(self) -> List[str]:
        pass

    @abstractmethod
    def get_pattern_ids(self) -> List[str]:
        pass

    @abstractmethod
    def get_pattern(self, pattern_id: str) -> Pattern:
        pass

    @abstractmethod
    def get_market_dates(self, market_id: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_future_returns(self, market_id: str, period: int) -> np.ndarray:
        pass

    @abstractmethod
    def get_pattern_values(self, market_id: str, pattern_id: int) -> np.ndarray:
        pass
