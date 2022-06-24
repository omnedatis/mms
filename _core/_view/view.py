# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:15:15 2022

@author: WaNiNi
"""

import datetime
import os
from typing import Dict, List, NamedTuple, Optional

import pandas as pd

from const import (DEFAULT_TRAIN_BEGIN_DATE, DEFAULT_TRAIN_GAP,
                   DEFAULT_ALGORITHM, LOCAL_DB)

from .model import ModelMeta, ModelCategory

def _get_view_dir(view_id: str) -> str:
    return f'{MODEL_DB_PATH}/{view_id}'

MODEL_DB_PATH = f'{LOCAL_DB}/models'
class ViewModel(NamedTuple):
    view_id: str
    markets: Optional[List[str]]
    patterns: List[str]
    predict_period: int
    train_begin_date: datetime.date
    effective_date: datetime.date
    expiration_date: datetime.date
    model: ModelMeta

    @property
    def _id(self) -> str:
        return f'{self.view_id}-{self.effective_date}-{self.predict_period}'

    @property
    def _file(self) -> str:
        return f'{_get_view_dir(self.view_id)}/{self._id}.pkl'

    @property
    def trained_markets(self) -> List[str]:
        return self.model.targets

    def is_trained(self):
        return os.path.exists(self._file)

    def update_markets(self, x_data: Dict[str, pd.DataFrame],
                       y_data: Dict[str, pd.Series]):
        if not self.is_trained():
            raise RuntimeError(f"model is not trained: {self._id}")
        self.model.load(self._file)
        self.model.update_targets(x_data, y_data)
        self.model.dump(self._file)

    def train(self, x_data: Dict[str, pd.DataFrame],
              y_data: Dict[str, pd.Series]):
        self.model.train(x_data, y_data)
        self.model.dump(self._file)

    def predict(self, x_data: Dict[str, pd.DataFrame]
                ) -> Dict[str, pd.DataFrame]:
        if not self.is_trained():
            raise RuntimeError(f"model is not trained: {self._id}")
        self.model.load(self._file)
        ret = self.model.predict(x_data)
        return ret

    @classmethod
    def make(cls, view_id: str, markets: Optional[List[str]],
             patterns: List[str], predict_period: int,
             train_begin_date: datetime.date, effective_date: datetime.date,
             expiration_date: datetime.date, algorithm: str):
        model_cls = ModelCategory.get(algorithm)
        if model_cls is None:
            raise ValueError(f"invalid 'algorithm': {algorithm}")
        model = model_cls()
        return cls(view_id, markets, patterns, predict_period,
                   train_begin_date, effective_date, expiration_date, model)

class View(NamedTuple):
    view_id: str
    patterns: List[str]
    markets: Optional[List[str]]
    train_begin: datetime.date
    train_gap: int
    algorithm: str

    @staticmethod
    def get_dir(view_id):
        return _get_view_dir(view_id)

    @classmethod
    def make(cls, model_id: str, patterns: List[str],
             markets: Optional[List[str]]=None,
             train_begin: Optional[datetime.date]=None,
             train_gap: Optional[int]=None,
             algorithm: Optional[str]=None):
        if train_begin is None:
            train_begin = DEFAULT_TRAIN_BEGIN_DATE
        if train_gap is None or train_gap <= 0:
            train_gap = DEFAULT_TRAIN_GAP
        if algorithm is None:
            algorithm = DEFAULT_ALGORITHM
        return cls(model_id, patterns, markets, train_begin, train_gap, algorithm)

    def get_model(self, period: int, today: datetime.date) -> ViewModel:
        ret = ViewModel.make(view_id=self.view_id,
                             markets=self.markets,
                             patterns=self.patterns,
                             predict_period=period,
                             train_begin_date=self.train_begin,
                             effective_date=self.get_effective_date(today),
                             expiration_date=self.get_expiration_date(today),
                             algorithm=self.algorithm)
        return ret

    def get_effective_date(self, today: datetime.date) -> datetime.date:
        month = (today.month-1) // self.train_gap * self.train_gap + 1
        ret = datetime.date(today.year, month, 1)
        return ret

    def get_expiration_date(self, today: datetime.date) -> datetime.date:
        recv = self.get_effective_date(today)
        year = recv.year
        month = recv.month + self.train_gap
        if month > 12:
            month -= 12
            year += 1
        ret = datetime.date(year, month, 1)
        return ret
