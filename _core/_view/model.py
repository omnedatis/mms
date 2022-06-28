# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:28:55 2022

@author: WaNiNi
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as Dtc
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from const import PredictResultField, Y_LABELS, MIN_Y_SAMPLES, Y_OUTLIER
from utils import pickle_dump, pickle_load

# Note: imports from `const` is bad coupling
# ToDo: remove the dependency from this module to `const`

class Labelization:
    class Info(NamedTuple):
        labels: int
        boundaries: Optional[np.ndarray] = None
        means: Optional[np.ndarray] = None

        def isfitted(self):
            return self.boundaries is not None

    def __init__(self, labels: int=5):
        self._info = self.Info(labels)

    def isfitted(self):
        return self._info.isfitted()

    @property
    def labels(self) -> int:
        return self._info.labels

    @property
    def upper_bounds(self) -> np.ndarray:
        return self._info.boundaries[1:]

    @property
    def lower_bounds(self) -> np.ndarray:
        return self._info.boundaries[:-1]

    @property
    def cutting_points(self) -> np.ndarray:
        return self._info.boundaries[1:-1]

    @property
    def means(self) -> np.ndarray:
        return self._info.means

    def fit(self, data: np.ndarray, outlier: float=0):

        data = data.flatten()
        dlen = len(data)
        data.sort()
        interval = dlen // self.labels

        if self.labels % 2 == 0:
            lpart = []
            rpart = []
            prev_idx, next_idx = 0, interval
            for i in range(self.labels // 2 - 1):
                lpart.append(data[prev_idx: next_idx])
                rpart.append(data[::-1][prev_idx: next_idx][::-1])
                prev_idx, next_idx = next_idx, next_idx + interval
            next_idx = dlen // 2
            lpart.append(data[prev_idx: next_idx])
            rpart.append(data[::-1][prev_idx: next_idx][::-1])
            sections = lpart + rpart[::-1]
        else:
            lpart = []
            rpart = []
            prev_idx, next_idx = 0, interval
            for i in range(self.labels // 2):
                lpart.append(data[prev_idx: next_idx])
                rpart.append(data[::-1][prev_idx: next_idx][::-1])
                prev_idx, next_idx = next_idx, next_idx + interval
            sections = lpart + [data[prev_idx: -prev_idx]] + rpart[::-1]

        means = np.array([each.mean() for each in sections])
        boundaries = [(l[-1] + r[0]) / 2 for l, r in zip(sections[:-1], sections[1:])]

        # Deal with specified outlier
        outlier = min([outlier, 0.25 / self.labels])
        outlier = int(dlen * outlier)
        llimit = data[outlier]
        rlimit = data[-(outlier+1)]
        means[0] = sections[0][outlier:].mean()
        means[-1] = sections[-1][:-outlier].mean()
        boundaries = np.array([llimit] + boundaries + [rlimit])
        self._info = self.Info(self.labels, boundaries, means)

    def fit_transform(self, data: np.ndarray, outlier: float=0):
        self.fit(data, outlier)
        return self.transform(data)

    def transform(self, data):
        if not self.isfitted():
            raise RuntimeError("transform before fitting")
        data = np.broadcast_to(data.T, (self.labels-1, ) + data.T.shape).T
        ret = (data > self.cutting_points).sum(axis=-1)
        return ret

    def label2upperbound(self, labels):
        if not self.isfitted():
            raise RuntimeError("transform before fitting")
        return self.upper_bounds[labels]

    def label2lowerbound(self, labels):
        if not self.isfitted():
            raise RuntimeError("transform before fitting")
        return self.lower_bounds[labels]

    def label2mean(self, labels):
        if not self.isfitted():
            raise RuntimeError("transform before fitting")
        return self.means[labels]

    def dump(self):
        return self._info

    def _make(self, boundaries, means):
        self._info = self.Info(self.labels, boundaries, means)

    @classmethod
    def make(cls, info):
        labels, boundaries, means = info
        if (len(boundaries) != labels + 1 or len(means) != labels):
            raise ValueError(f"inconsistent info: {info}")
        ret = cls(labels)._make(boundaries, means)
        return ret

class ModelResultField(str, Enum):
    UPPER_BOUND = PredictResultField.UPPER_BOUND.value
    LOWER_BOUND = PredictResultField.LOWER_BOUND.value
    MEAN = PredictResultField.PREDICT_VALUE.value


COMMON_DECISION_TREE_PARAMS = {
    'max_depth': 20,
    'criterion': 'entropy',
    'random_state': 0,
    'max_leaf_nodes': 200,
    'min_samples_leaf': 30}


class ModelMeta(metaclass=ABCMeta):
    @abstractmethod
    def _dump(self) -> Any:
        pass

    @abstractmethod
    def _load(self, recv: Any):
        pass

    def dump(self, file):
        pickle_dump(self._dump(), file)

    def load(self, file):
        self._load(pickle_load(file))

    @abstractproperty
    def targets(self) -> List[str]:
        pass

    @abstractmethod
    def update_targets(self, x_data: Dict[str, pd.DataFrame],
                       y_data: Dict[str, pd.Series]):
        pass

    @abstractmethod
    def train(self, x_data: Dict[str, pd.DataFrame],
              y_data: Dict[str, pd.Series]):
        pass

    @abstractmethod
    def predict(self, x_data: Dict[str, pd.DataFrame]
                ) -> Dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def isfitted(self) -> bool:
        pass

class DecisionTreeClassifierModel(ModelMeta):
    class _TrainDataSet(NamedTuple):
        target: str
        x_data: Optional[np.ndarray]
        y_data: Optional[np.ndarray]

    def __init__(self):
        self._model = Dtc(**COMMON_DECISION_TREE_PARAMS)
        self._y_coders: Dict[str, Labelization] = {}

    def _dump(self):
        return {'model': self._model, 'y_coders': self._y_coders}

    def _load(self, recv):
        self._model = recv['model']
        self._y_coders = recv['y_coders']

    @property
    def targets(self) -> List[str]:
        return list(self._y_coders.keys())

    def isfitted(self):
        try:
            check_is_fitted(self._model)
            return True
        except NotFittedError:
            return False

    @classmethod
    def _get_train_data(cls, x_data: Dict[str, pd.DataFrame],
                        y_data: Dict[str, pd.Series],
                        min_samples: int=1) -> List[_TrainDataSet]:
        ret = []
        for key, cur_x in x_data.items():
            cur_y = y_data.get(key)
            if cur_y is None:
                raise ValueError("inconsistent 'x_data' and 'y_data': {key}")
            # Note: without checking consistency between index of `cur_x` and
            #       `cur_y` for performance.
            y_values = cur_y.values
            x_values = cur_x.values
            idxs = ~(np.isnan(x_values).any(axis=1) & np.isnan(y_values))
            if idxs.sum() < min_samples:
                ret.append(cls._TrainDataSet(key, None, None))
                continue
            if idxs.all():
                ret.append(cls._TrainDataSet(key, x_values, y_values))
                continue
            ret.append(cls._TrainDataSet(key, x_values[idxs], y_values[idxs]))
        return ret

    def update_targets(self, x_data: Dict[str, pd.DataFrame],
                       y_data: Dict[str, pd.Series]):
        for each in self._get_train_data(x_data, y_data, MIN_Y_SAMPLES):
            if each.y_data is None:
                self._y_coders[each.target] = None
                continue
            y_coder = Labelization(Y_LABELS)
            y_coder.fit(each.y_data, Y_OUTLIER)
            self._y_coders[each.target] = y_coder

    def train(self, x_data: Dict[str, pd.DataFrame],
              y_data: Dict[str, pd.Series]):
        train_x = []
        train_y = []
        for each in self._get_train_data(x_data, y_data, MIN_Y_SAMPLES):
            if each.y_data is None or each.x_data is None:
                self._y_coders[each.target] = None
                continue
            train_x.append(each.x_data.astype(bool))
            y_coder = Labelization(Y_LABELS)
            train_y.append(y_coder.fit_transform(each.y_data, Y_OUTLIER))
            self._y_coders[each.target] = y_coder
        if train_x and train_y:
            self._model.fit(np.concatenate(train_x, axis=0),
                            np.concatenate(train_y, axis=0))

    def predict(self, x_data: Dict[str, pd.DataFrame]
                ) -> Dict[str, pd.DataFrame]:
        ret = {}
        if self.isfitted():
            columns = [ModelResultField.LOWER_BOUND.value,
                       ModelResultField.UPPER_BOUND.value,
                       ModelResultField.MEAN.value]
            for key, data in x_data.items():
                y_coder = self._y_coders.get(key)
                if y_coder is None:
                    values = np.full((len(data), len(columns)), np.nan)
                    ret[key] = pd.DataFrame(values, index=data.index, columns=columns)
                    continue
                index = data.index
                isnan = np.isnan(data.values).any(axis=1)
                labels = self._model.predict(np.nan_to_num(data.values, nan=0).astype(bool))
                values = np.array([y_coder.label2lowerbound(labels),
                                   y_coder.label2upperbound(labels),
                                   y_coder.label2mean(labels)]).T
                values[isnan] = np.nan
                ret[key] = pd.DataFrame(values, index=index, columns=columns)
        return ret

class ModelCategory(Enum):
    DECISION_TREE_CLASSIFIER = DecisionTreeClassifierModel

    @classmethod
    def get(cls, name: str) -> Optional[ModelMeta]:
        if name in cls._member_map_:
            return cls._member_map_[name].value
        return None
