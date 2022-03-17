# -*- coding: utf-8 -*-
"""Time Series.

This module includes classes and methods for time-series and operations.

Classes
-------
BooleanTimeSeries : A time-series with boolean values.
NumericTimeSeries : A time-series with numeric values.
TimeSeries : A sequence of data points indexed by time.
TimeSeriesRolling : A group of time-series produced by rolling method.

"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from ._groupby import ArrayGroup, HomoArrayGroup
from ._index import TimeIndex, TimeUnit


class TimeSeriesRolling:
    """Group of time-series produced by rolling method.

    Parameters
    ----------
    index : TimeIndex
        Time-index of source time-series.
    values : ArrayGroup
        Group of data array of time-series rolling.
    indices : ArrayGroup
        Group of index array of time-series rolling.
    name : str
        Name of object.

    Notes
    -----
    Objects of this class are only produced by rolling method of TimeSeries
    instance. For better performance, ignore unnecessary dynamic checking.

    Methods
    -------
    to_dict : Dict[np.datetime64, TimeSeries]
        Output group of time-series as dict.

    """
    def __init__(self, index: TimeIndex, values: ArrayGroup,
                 indices: ArrayGroup, name: str):
        self._index = index
        self._values = values
        self._indices = indices
        self._name = name

    def to_dict(self) -> Dict[np.datetime64, 'TimeSeries']:
        """Output group of time-series as dict."""
        keys = self._index.values
        indices = self._indices.values
        values = self._values.values
        masks = self._values.masks
        if self._indices.masks is not None:
            filters = [~each for each in self._indices.masks]
            indices = [each[f] for each, f in zip(indices, filters)]
            values = [each[f] for each, f in zip(values, filters)]
            masks = [each[f] for each, f in zip(masks, filters)]
        ret = {k: TimeSeries(data=d, index=i, masks=m,
                             name=f'{self._name}[{idx}]')
               for idx, (k, d, i, m) in enumerate(zip(keys, values, indices,
                                                      masks))}
        return ret

    def first(self) -> 'TimeSeries':
        """The 1st element of each time-series in group."""
        def _first(values, masks=None):
            if masks is None:
                return values[0]
            else:
                return values[~masks][0]
        index = self._index
        if self._values.masks is None:
            values = np.array([each[0] for each in self._values.values])
        else:
            values = np.array(list(map(_first, self._values.values, self._values.masks)))
        masks = None
        name = f'{self._name}.first()'
        if np.issubdtype(values.dtype, np.number):
            return NumericTimeSeries(values, index=index, name=name, masks=masks)
        if np.issubdtype(values.dtype, np.bool):
            return BooleanTimeSeries(values, index=index, name=name, masks=masks)
        return TimeSeries(values, index=index, name=name, masks=masks)

    def max(self) -> 'TimeSeries':
        """Max of each time-series in group."""
        def _max(values, masks=None):
            if masks is None:
                return values.max()
            else:
                return values[~masks].max()
        index = self._index
        if self._values.masks is None:
            values = np.array([each.max() for each in self._values.values])
            masks = None
        else:
            values = np.array(list(map(_max, self._values.values, self._values.masks)))
            masks = np.array([each.any() for each in self._values.masks])
        name = f'{self._name}.max()'
        if np.issubdtype(values.dtype, np.number):
            return NumericTimeSeries(values, index=index, name=name, masks=masks)
        if np.issubdtype(values.dtype, np.bool):
            return BooleanTimeSeries(values, index=index, name=name, masks=masks)
        return TimeSeries(values, index=index, name=name, masks=masks)

    def min(self) -> 'TimeSeries':
        """Min of each time-series in group."""
        def _min(values, masks=None):
            if masks is None:
                return values.min()
            else:
                return values[~masks].min()
        index = self._index
        if self._values.masks is None:
            values = np.array([each.min() for each in self._values.values])
            masks = None
        else:
            values = np.array(list(map(_min, self._values.values, self._values.masks)))
            masks = np.array([each.any() for each in self._values.masks])
        name = f'{self._name}.min()'
        if np.issubdtype(values.dtype, np.number):
            return NumericTimeSeries(values, index=index, name=name, masks=masks)
        if np.issubdtype(values.dtype, np.bool):
            return BooleanTimeSeries(values, index=index, name=name, masks=masks)
        return TimeSeries(values, index=index, name=name, masks=masks)

    def last(self) -> 'TimeSeries':
        """The last element of each time-series in group."""
        def _first(values, masks=None):
            if masks is None:
                return values[-1]
            else:
                return values[~masks][-1]
        index = self._index
        if self._values.masks is None:
            values = np.array([each[-1] for each in self._values.values])
        else:
            values = np.array(list(map(_first, self._values.values, self._values.masks)))
        masks = None
        name = f'{self._name}.first()'
        if np.issubdtype(values.dtype, np.number):
            return NumericTimeSeries(values, index=index, name=name, masks=masks)
        if np.issubdtype(values.dtype, np.bool):
            return BooleanTimeSeries(values, index=index, name=name, masks=masks)
        return TimeSeries(values, index=index, name=name, masks=masks)

class TimeSeriesSampling(TimeSeriesRolling):
    """Group of time-series produced by rolling method.

    Parameters
    ----------
    index : TimeIndex
        Time-index of source time-series.
    values : HomoArrayGroup
        Group of data array of time-series rolling.
    indices : HomoArrayGroup
        Group of index array of time-series rolling.
    name : str
        Name of object.

    Notes
    -----
    Objects of this class are only produced by rolling method of TimeSeries
    instance. For better performance, ignore unnecessary dynamic checking.

    Methods
    -------
    to_dict : Dict[np.datetime64, TimeSeries]
        Output group of time-series as dict.
    to_pandas : pd.DataFrame
        Output group of time-series as pandas DataFrame.

    See Also
    --------
    TimeSeriesRolling

    """
    def __init__(self, index: TimeIndex, values: HomoArrayGroup,
                 indices: HomoArrayGroup, name: str):
        if not isinstance(values, HomoArrayGroup):
            raise TypeError("'values' must be a 'HomoArrayGroup'")
        if not isinstance(indices, HomoArrayGroup):
            raise TypeError("'indices' must be a 'HomoArrayGroup'")
        super().__init__(index, values, indices, name)

    def to_pandas(self) -> pd.DataFrame:
        """Output group of time-series as pandas DataFrame."""
        index = self._index.values
        values = self._values.values
        masks = self._values.masks
        if masks is not None and masks.any():
            if issubclass(values.dtype.type, np.number):
                values = values.astype(float)
            else:
                values = values.astype(object)
            values[masks] = np.nan
        return pd.DataFrame(values, index=index)

    def first(self) -> 'TimeSeries':
        """The 1st element of each time-series in group."""
        index = self._index
        values = self._values.values[:, 0]
        masks = self._values.masks.any(axis=1)
        name = f'{self._name}.first()'
        return NumericTimeSeries(data=values, index=index,
                                 name=name, masks=masks)

    def max(self) -> 'TimeSeries':
        """Max of each time-series in group."""
        index = self._index
        values = self._values.values.max(axis=1)
        masks = self._values.masks.any(axis=1)
        name = f'{self._name}.max()'
        return NumericTimeSeries(data=values, index=index,
                                 name=name, masks=masks)

    def min(self) -> 'TimeSeries':
        """Min of each time-series in group."""
        index = self._index
        values = self._values.values.min(axis=1)
        masks = self._values.masks.any(axis=1)
        name = f'{self._name}.min()'
        return NumericTimeSeries(data=values, index=index,
                                 name=name, masks=masks)

    def last(self) -> 'TimeSeries':
        """The 1st element of each time-series in group."""
        index = self._index
        values = self._values.values[:, -1]
        masks = self._values.masks.any(axis=1)
        name = f'{self._name}.last()'
        return NumericTimeSeries(data=values, index=index,
                                 name=name, masks=masks)

    def sum(self) -> 'NumericTimeSeries':
        """sum of each time-series in group."""
        index = self._index
        values = self._values.values.sum(axis=1)
        masks = self._values.masks.any(axis=1)
        name = f'{self._name}.sum()'
        return NumericTimeSeries(data=values, index=index,
                                 name=name, masks=masks)

    def all(self) -> 'BooleanTimeSeries':
        """sum of each time-series in group."""
        index = self._index
        values = self._values.values.all(axis=1)
        masks = self._values.masks.any(axis=1)
        name = f'{self._name}.sum()'
        return BooleanTimeSeries(data=values, index=index,
                                 name=name, masks=masks)

    def mean(self) -> 'NumericTimeSeries':
        """Mean of each time-series in group."""
        index = self._index
        values = self._values.values.mean(axis=1)
        masks = self._values.masks.any(axis=1)
        name = f'{self._name}.mean()'
        return NumericTimeSeries(data=values, index=index,
                                 name=name, masks=masks)

class TimeSeries:
    """Time Series.

    A sequence of data points indexed by time.

    Parameters
    ----------
    data : array-like (1-dimensional)
        The data stored in TimeSeries.
    index : array-like or TimeIndex
        The datetime values indexing `data`.
    name : str
        The name given to TimeSeries.
    masks : array-like (1-dimensional), optional
        A boolean array indicating which elements in `data` are N/A. If it is
        not specified, all elements in `data` are vaild.

    Notes
    -----
    `data`, `index`, and `masks`(if it is specified) must have same length.

    Properties
    ----------
    name : str
        The name of TimeSeries.

    Built-in Functions
    ------------------
    len : int
        The length of TimeSeries.

    Methods
    -------
    rename :
        Rename TimeSeries.
    equals :
        Determine if another TimeSeries object equals to self except name.
    isna :
        Return a pandas Series with boolean values indicating which values in
        TimeSeries are N/A.
    dropna :
        Return a new TimeSeries with N/A values removed.
    fillna :
        Fill N/A values using given value or specified method.
    to_pandas : pandas.Series
        Return a copy of TimeSeries as pandas Series.
    shift :
        Shift index by desired number of periods.
    rolling : TimeSeriesRolling
        Rolling group values along index by desired period.
    sampling : TimeSeriesSampling
        Moving Sample values along index by given step.

    Classmethods
    ------------
    make :
        Make a new object from a series.

    """
    def __init__(self, data: Union[np.ndarray, list],
                 index: Union[np.ndarray, list, TimeIndex],
                 name: str, masks: Optional[Union[np.ndarray, list]] = None):
        if len(data) != len(index):
            raise ValueError("inconsistent data length")
        self._values = np.array(data)

        if not isinstance(index, TimeIndex):
            index = TimeIndex(index)
        self._index = index
        self._dtype = self._values.dtype
        self._name = name

        if masks is not None:
            if len(masks) != len(index):
                raise ValueError("inconsistent masks length")
            self._masks = np.array(masks, dtype=bool)
        else:
            self._masks = np.full(len(self._values), False, dtype=bool)

    @property
    def name(self) -> str:
        """The name of TimeSeries."""
        return self._name

    def __len__(self) -> int:
        """The length of TimeSeries."""
        return len(self._index)

    def rename(self, name: str):
        """Rename TimeSeries.

        Arguments
        ---------
        name : str
            New name set to TimeSeries.

        """
        self._name = name

    def equals(self, other: 'TimeSeries') -> bool:
        """Determine if another TimeSeries object equals to self.

        The things that are being compared are index and values.

        Parameters
        ----------
        other : TimeSeries
            The other TimeSeries object to compare against.

        Returns
        -------
        bool
            True if `other` is an TimeSeries and it has the same index and
            values as the calling object; False otherwise.

        """
        if not isinstance(other, TimeSeries):
            raise TypeError("not support comparison between '%s' and '%s'"
                            % (self.__class__.__name__,
                               other.__class__.__name__))
        # pylint: disable=protected-access
        if not self._index.equals(other._index):
            return False
        if not np.array_equal(self._masks, other._masks):
            return False
        return np.array_equal(self._values[~self._masks],
                              other._values[~other._masks])
        # pylint: enable=protected-access

    def isna(self) -> pd.Series:
        """Detect N/A values.

        Return a pandas Series with boolean values indicating if the values in
        TimeSeries are N/A.

        Returns
        -------
        pandas.Series
            Mask of bool values for each element in TimeSeries that indicates
            whether an element is an N/A value.

        """
        values = self._masks
        index = self._index.values
        name = f'{self._name}.isna()'
        return pd.Series(values, index=index, name=name)

    def dropna(self) -> 'TimeSeries':
        """Return a new TimeSeries with N/A values removed."""
        selected = ~self._masks
        values = self._values[selected]
        index = self._index.values[selected]
        name = self._name
        ret = self.__class__(values, index=index, name=name)
        return ret

    def _fillna(self, value: Any) -> 'TimeSeries':
        """Fill N/A values with specified value."""
        values = self._values.copy()
        values[self._masks] = value
        index = self._index
        name = self._name
        ret = self.__class__(values, index=index, name=name)
        return ret

    def _forward_fillna(self) -> 'TimeSeries':
        """Fill N/A values with previous valid value."""
        def get_previous_valid_value(masks):
            """Get index of previous valid value or -1 if does not exist."""
            ret = np.arange(1, len(masks) + 1)
            ret[masks] = 0
            ret = np.maximum.accumulate(ret) - 1
            return ret
        prev_vids = get_previous_valid_value(self._masks)
        values = self._values[prev_vids]
        index = self._index
        name = self._name
        masks = prev_vids < 0
        ret = self.__class__(values, index=index, name=name, masks=masks)
        return ret

    def _backward_fillna(self) -> 'TimeSeries':
        """Fill N/A values with previous valid value."""
        def get_next_valid_value(masks):
            """Get index of next valid value or 0 if does not exist."""
            masks = masks[::-1]
            ret = np.arange(1, len(masks) + 1)
            ret[masks] = 0
            ret = -np.maximum.accumulate(ret)[::-1]
            return ret
        next_vids = get_next_valid_value(self._masks)
        values = self._values[next_vids]
        index = self._index
        name = self._name
        masks = next_vids >= 0
        ret = self.__class__(values, index=index, name=name, masks=masks)
        return ret

    def fillna(self, value: Any = None, method: Optional[str] = None
               ) -> 'TimeSeries':
        """Fill N/A values using given value or specified method.

        Parameters
        ----------
        value : Any, default None.
            Value used to fill N/A values.
        method : {'bfill', 'ffill', None}, defualt is None
            Method used to fill N/A values.
            There are two valid methods are as follows:
            - 'ffill' : propagate last valid observation forward to next valid.
            - 'bfill' : use next valid observation to fill gap.

        Notes
        -----
        Exactly one of the described parameters should be specified.

        Returns
        -------
        TimeSeries:
            A new instance with N/A values filled.

        """
        if value is not None:
            if method is not None:
                raise ValueError("cannot specify both `value` and `method`")
            return self._fillna(value)
        if method is None:
            raise ValueError("must specify a fill `value` or `method`")
        if method == 'ffill':
            return self._forward_fillna()
        if method == 'bfill':
            return self._backward_fillna()
        raise ValueError("invalid method. It should be '%s' or '%s' but get '%s'"
                         % ('ffill', 'bfill', method))

    def shift(self, period: int, punit: Optional[TimeUnit]=None) -> 'TimeSeries':
        """Shift index by desired number of periods.

        Parameters
        ----------
        period : int
            Number of periods to shift. Can be positive, negative or zero.
            The actions corresponding three conditions are descripted as
            follows:
            - If `period` is zero, return a copy.
            - If `period` is set as a positive integr, n, the elements of data
            are shifted forward n positions along index and the first n
            elements are set to be N/A.
            - If `period` is set as a negative integer, -n, the elements of
            data are shifted backward n positions along index and the last n
            elements are set to be N/A.
        punit : TimeUnit
            It is optional. It it is specified, it must be an instance of
            TimeUnit which is a super-unit of or equivalent to the dtype of
            index.

        Returns
        -------
        TimeSeries
            Copy of input object, shifted.

        See Also
        --------
        numpy.roll and pandas.Series.shift.

        """
        if not isinstance(period, int):
            raise TypeError("`period` must be an `int` not '%s'"
                            % (type(period).__name__))
        if period == 0:
            return self.make(self)
        index = self._index
        if punit is None:
            values = np.roll(self._values, period)
            name = f'{self._name}.shift({period})'
            masks = np.roll(self._masks, period)
            if period > 0:
                masks[:period] = True
            else:
                masks[period:] = True
        else:
            samples = abs(period) + 1
            steps = 1 if period > 0 else -1
            name = f'{self._name}.shift({period}, {punit.name})'
            values = self._index.sampling(self._values, samples, steps,
                                          sunit=punit).values[:, 0]
            masks = self._index.sampling(self._masks, samples, steps,
                                         sunit=punit)
            masks = masks.fillna(True).values[:, 0]
        return self.__class__(values, index=index, name=name, masks=masks)

    def rolling(self, period: int, punit: Optional[TimeUnit] = None
                ) -> TimeSeriesRolling:
        """Rolling group values along index by desired period.

        Parameters
        ----------
        period : int
            The period of rolling window.
        punit : TimeUnit
            It is optional. It it is specified, it must be an instance of
            TimeUnit which is a super-unit of or equivalent to the dtype of
            index.

        Returns
        -------
        TimeSeriesRolling

        """
        if punit is None:
            name = f'{self._name}.rolling({period})'
        else:
            if not isinstance(punit, TimeUnit):
                raise ValueError("`punit` must be a member of 'TimeUnit'")
            name = f'{self._name}.rolling({period}, {punit.name})'
        indices = self._index.rolling(self._index.values, period, punit)
        values = self._index.rolling(self._values, period, punit)
        masks = self._index.rolling(self._masks, period, punit)
        values.masks = masks.fillna(True).values
        return TimeSeriesRolling(index=self._index, values=values,
                                 indices=indices, name=name)

    def sampling(self, samples: int, step: int = 1,
                 sunit: Optional[TimeUnit] = None) -> TimeSeriesSampling:
        """Moving Sample values along index by given step.

        Parameters
        ----------
        samples : int
            The number of samples for each time-stamp.
        step : int, default is ``1``
            The step between two samples.
        sunit : TimeUnit
            It is optional. It it is specified, it must be an instance of
            TimeUnit which is a super-unit of or equivalent to the dtype of
            index.

        Returns
        -------
        TimeSeriesSampling

        """
        if sunit is None:
            name = f'{self._name}.sampling({samples}, {step})'
        else:
            if not isinstance(sunit, TimeUnit):
                raise ValueError("`sunit` must be a member of 'TimeUnit'")
            name = f'{self._name}.sampling({samples}, {step}, {sunit.name})'
        indices = self._index.sampling(self._index.values, samples, step, sunit)
        values = self._index.sampling(self._values, samples, step, sunit)
        masks = self._index.sampling(self._masks, samples, step, sunit)
        values.masks = masks.fillna(True).values
        return TimeSeriesSampling(index=self._index, values=values,
                                  indices=indices, name=name)

    def to_pandas(self) -> pd.Series:
        """Return a copy of TimeSeries as pandas Series.

        Returns
        -------
        pandas.Series
            If there is no N/A value, it return Series with same dtype.
            If there are N/A values and its dtype is numpy.number, it return
            Series with dtype as numpy.float and use np.nan to represent NA
            values; otherwise, it return Series with datype as numpy.object
            and use numpy.nan to represent NA values.

        """
        values = self._values
        index = self._index.values
        name = self._name
        if self._masks.any():
            if issubclass(self._dtype.type, np.number):
                values = values.astype(float)
            else:
                values = values.astype(object)
            values[self._masks] = np.nan
        return pd.Series(values, index=index, name=name)

    @classmethod
    def _make_from_pandas(cls, tseries: pd.Series) -> 'TimeSeries':
        # without checking type, it should check by the caller
        values = tseries.values
        index = tseries.index.values
        name = tseries.name
        masks = tseries.isna().values
        return cls(values, index=index, name=name, masks=masks)

    @classmethod
    def _make_from_tseries(cls, tseries: 'TimeSeries') -> 'TimeSeries':
        # without checking type, it should check by the caller
        # pylint: disable=protected-access
        values = tseries._values
        index = tseries._index
        name = tseries._name
        masks = tseries._masks
        # pylint: enable=protected-access
        return cls(values, index=index, name=name, masks=masks)

    @classmethod
    def make(cls, tseries: Union['TimeSeries', pd.Series]) -> 'TimeSeries':
        """Make a new object from a series.

        Parameters
        ----------
        tseries : TimeSeries, pandas.Series

        Returns
        -------
        TimeSeries:

        """
        if isinstance(tseries, pd.Series):
            return cls._make_from_pandas(tseries)
        if isinstance(tseries, TimeSeries):
            return cls._make_from_tseries(tseries)
        raise TypeError("'%s' is not a valid series type"
                        % (tseries.__class__.__name__))


_NumericType = Union[int, float]


class NumericTimeSeries(TimeSeries):
    """Time-series with numeric values.

    Support numeric operator as follows:
    - Unary : Negation(-), Absolute value(abs)
    - Arithmetic : Addition(+), Substraction(-), Multiplication(*), Division(/),
                   Exponentiation(**), Floor division(//), Modulus(%)
    - Comparison : Equal(==), Not equal(!=), Greater than(>), Less than(<),
                   Greater than or equal to(>=), Less than or equal to(<=)

    In addition to unary operators, others are operations between two
    NumericTimeSeries which need with the equilvent index or between a
    NumericTimeSeries and a single number. While comparison operations always
    return a BooleanTimeSeries, other operations always return a
    NumericTimeSeries.

    See Also
    --------
    TimeSeries, BooleanTimeSeries.

    """
    def __init__(self, data: Union[np.ndarray, list],
                 index: Union[np.ndarray, list, TimeIndex],
                 name: str, masks: Optional[Union[np.ndarray, list]] = None):
        super().__init__(data, index=index, name=name, masks=masks)
        if not issubclass(self._dtype.type, np.number):
            raise ValueError("non-numeric values in `data`")

    def __neg__(self) -> 'NumericTimeSeries':
        values = -self._values
        index = self._index
        name = f'-{self._name}'
        masks = self._masks.copy()
        return NumericTimeSeries(values, index=index, name=name, masks=masks)

    def __abs__(self) -> 'NumericTimeSeries':
        values = abs(self._values)
        index = self._index
        name = f'abs({self._name})'
        masks = self._masks.copy()
        return NumericTimeSeries(values, index=index, name=name, masks=masks)

    def log(self) -> 'NumericTimeSeries':
        values = np.log(self._values)
        index = self._index
        name = f'log({self._name})'
        masks = self._masks.copy()
        return NumericTimeSeries(values, index=index, name=name, masks=masks)

    def _arithmetic_op(self, other: Union['NumericTimeSeries', _NumericType],
                       symbol: 'str', func: Callable[[Any, Any], Any]
                       ) -> 'NumericTimeSeries':
        if isinstance(other, NumericTimeSeries):
            # pylint: disable=protected-access
            if not self._index.equals(other._index):
                raise ValueError("inconsistent index")
            values = func(self._values, other._values)
            name = f'{self._name} {symbol} {other._name}'
            masks = self._masks | other._masks
            # pylint: enable=protected-access
        else:
            if not isinstance(other, (int, float)):
                raise TypeError("unsupported operand type(s) for %s: '%s' and '%s'"
                                % (symbol, 'NumericTimeSeries',
                                   other.__class__.__name__))
            values = func(self._values, other)
            name = f'{self._name} {symbol} {other}'
            masks = self._masks.copy()
        index = self._index
        return NumericTimeSeries(values, index=index, name=name, masks=masks)

    def _comparison_op(self, other: Union['NumericTimeSeries', _NumericType],
                       symbol: 'str', func: Callable[[Any, Any], Any]
                       ) -> 'BooleanTimeSeries':
        if isinstance(other, NumericTimeSeries):
            # pylint: disable=protected-access
            if not self._index.equals(other._index):
                raise ValueError("inconsistent index")
            values = func(self._values, other._values)
            name = f'{self._name} {symbol} {other._name}'
            masks = self._masks | other._masks
            # pylint: enable=protected-access
        else:
            if not isinstance(other, (int, float)):
                raise TypeError("unsupported operand type(s) for %s: '%s' and '%s'"
                                % (symbol, 'NumericTimeSeries',
                                   other.__class__.__name__))
            values = func(self._values, other)
            name = f'{self._name} {symbol} {other}'
            masks = self._masks.copy()
        index = self._index
        return BooleanTimeSeries(values, index=index, name=name, masks=masks)

    def __add__(self, other: Union['NumericTimeSeries', _NumericType]
                ) -> 'NumericTimeSeries':
        ret = self._arithmetic_op(other, symbol='+',
                                  func=lambda x, y: x + y)
        return ret

    def __sub__(self, other: Union['NumericTimeSeries', _NumericType]
                ) -> 'NumericTimeSeries':
        ret = self._arithmetic_op(other, symbol='-',
                                  func=lambda x, y: x - y)
        return ret

    def __mul__(self, other: Union['NumericTimeSeries', _NumericType]
                ) -> 'NumericTimeSeries':
        ret = self._arithmetic_op(other, symbol='*',
                                  func=lambda x, y: x * y)
        return ret

    def __truediv__(self, other: Union['NumericTimeSeries', _NumericType]
                    ) -> 'NumericTimeSeries':
        ret = self._arithmetic_op(other, symbol='/',
                                  func=lambda x, y: x / y)
        return ret

    def __pow__(self, other: Union['NumericTimeSeries', _NumericType]
                ) -> 'NumericTimeSeries':
        ret = self._arithmetic_op(other, symbol='**',
                                  func=lambda x, y: x ** y)
        return ret

    def __mod__(self, other: Union['NumericTimeSeries', _NumericType]
                ) -> 'NumericTimeSeries':
        ret = self._arithmetic_op(other, symbol='%',
                                  func=lambda x, y: x % y)
        return ret

    def __floordiv__(self, other: Union['NumericTimeSeries', _NumericType]
                     ) -> 'NumericTimeSeries':
        ret = self._arithmetic_op(other, symbol='//',
                                  func=lambda x, y: x // y)
        return ret

    def __eq__(self,  # type: ignore
               other: Union['NumericTimeSeries', _NumericType]
               ) -> 'BooleanTimeSeries':
        ret = self._comparison_op(other, symbol='==',
                                  func=lambda x, y: x == y)
        return ret

    def __ne__(self,  # type: ignore
               other: Union['NumericTimeSeries', _NumericType]
               ) -> 'BooleanTimeSeries':
        ret = self._comparison_op(other, symbol='!=',
                                  func=lambda x, y: x != y)
        return ret

    def __gt__(self, other: Union['NumericTimeSeries', _NumericType]
               ) -> 'BooleanTimeSeries':
        ret = self._comparison_op(other, symbol='>',
                                  func=lambda x, y: x > y)
        return ret

    def __lt__(self, other: Union['NumericTimeSeries', _NumericType]
               ) -> 'BooleanTimeSeries':
        ret = self._comparison_op(other, symbol='<',
                                  func=lambda x, y: x < y)
        return ret

    def __ge__(self, other: Union['NumericTimeSeries', _NumericType]
               ) -> 'BooleanTimeSeries':
        ret = self._comparison_op(other, symbol='>=',
                                  func=lambda x, y: x >= y)
        return ret

    def __le__(self, other: Union['NumericTimeSeries', _NumericType]
               ) -> 'BooleanTimeSeries':
        ret = self._comparison_op(other, symbol='<=',
                                  func=lambda x, y: x <= y)
        return ret

    @classmethod
    def max(cls, *args: 'NumericTimeSeries') -> 'NumericTimeSeries':
        """Get max of given objects of NumericTimeSeries."""
        if len(args) <= 0:
            raise ValueError("no object to compare")
        if not isinstance(args[0], cls):
            raise TypeError("max function only accepts 'NumericTimeSeries'")
        values = [args[0]._values]
        masks = [args[0]._masks]
        names = [args[0]._name]
        index = args[0]._index
        for each in args[1:]:
            if not isinstance(each, cls):
                raise TypeError("max function only accepts 'NumericTimeSeries'")
            if not index.equals(each._index):
                raise ValueError("inconsistent index")
            values.append(each._values)
            masks.append(each._masks)
            names.append(each._name)
        return cls(np.array(values).max(axis=0), index=index,
                   name=f'max({",".join(names)})',
                   masks=np.array(masks).any(axis=0))

    @classmethod
    def min(cls, *args: 'NumericTimeSeries') -> 'NumericTimeSeries':
        """Get min of given objects of NumericTimeSeries."""
        if len(args) <= 0:
            raise ValueError("no object to compare")
        if not isinstance(args[0], cls):
            raise TypeError("min function only accepts 'NumericTimeSeries'")
        values = [args[0]._values]
        masks = [args[0]._masks]
        names = [args[0]._name]
        index = args[0]._index
        for each in args[1:]:
            if not isinstance(each, cls):
                raise TypeError("min function only accepts 'NumericTimeSeries'")
            if not index.equals(each._index):
                raise ValueError("inconsistent index")
            values.append(each._values)
            masks.append(each._masks)
            names.append(each._name)
        return cls(np.array(values).min(axis=0), index=index,
                   name=f'min({",".join(names)})',
                   masks=np.array(masks).any(axis=0))

    @classmethod
    def average(cls, *args: 'NumericTimeSeries') -> 'NumericTimeSeries':
        """Get average of given objects of NumericTimeSeries."""
        if len(args) <= 0:
            raise ValueError("no object to compare")
        if not isinstance(args[0], cls):
            raise TypeError("average function only accepts 'NumericTimeSeries'")
        values = [args[0]._values]
        masks = [args[0]._masks]
        names = [args[0]._name]
        index = args[0]._index
        for each in args[1:]:
            if not isinstance(each, cls):
                raise TypeError("average function only accepts 'NumericTimeSeries'")
            if not index.equals(each._index):
                raise ValueError("inconsistent index")
            values.append(each._values)
            masks.append(each._masks)
            names.append(each._name)
        return cls(np.array(values).mean(axis=0), index=index,
                   name=f'average({",".join(names)})',
                   masks=np.array(masks).any(axis=0))


class BooleanTimeSeries(TimeSeries):
    """Time-series with boolean values.

    Support logical operations as follows:
    - Unary : NOT(~)
    - Binary : AND(&), OR(|), XOR(^).

    While NOT operates on a single BooleanTimeSeries, AND, OR, and XOR all
    operate on two BooleanTimeSeries which need with the equivalent index.
    All these opertaions return a BooleanTimeSeries. It does not support Equal
    and Not equal operations because they can be realized simply by logical
    operations. For example, x == y is equivalent to x ^ ~y and x != y is
    equivalent to x ^ y.

    See Also
    --------
    TimeSeries.

    """
    def __init__(self, data: Union[np.ndarray, list],
                 index: Union[np.ndarray, list, TimeIndex],
                 name: str, masks: Optional[Union[np.ndarray, list]] = None):
        super().__init__(data, index=index, name=name, masks=masks)
        if not issubclass(self._dtype.type, np.bool_):
            raise ValueError("non-numeric values in data")

    def __invert__(self) -> 'BooleanTimeSeries':
        values = ~self._values
        index = self._index
        name = f'~{self._name}'
        masks = self._masks.copy()
        return BooleanTimeSeries(values, index=index, name=name, masks=masks)

    def _logical_op(self, other: 'BooleanTimeSeries',
                    symbol: 'str', func: Callable[[Any, Any], Any]
                    ) -> 'BooleanTimeSeries':
        if not isinstance(other, BooleanTimeSeries):
            raise TypeError("unsupported operand type(s) for %s: '%s' and '%s'"
                            % (symbol, 'BooleanTimeSeries',
                               other.__class__.__name__))
        # pylint: disable=protected-access
        if not self._index.equals(other._index):
            raise ValueError("inconsistent index")
        values = func(self._values, other._values)
        index = self._index
        name = f'{self._name} {symbol} {other._name}'
        masks = self._masks | other._masks
        # pylint: enable=protected-access
        return BooleanTimeSeries(values, index=index, name=name, masks=masks)

    def __and__(self, other: 'BooleanTimeSeries') -> 'BooleanTimeSeries':
        ret = self._logical_op(other, symbol='&',
                               func=lambda x, y: x & y)
        return ret

    def __or__(self, other: 'BooleanTimeSeries') -> 'BooleanTimeSeries':
        ret = self._logical_op(other, symbol='|',
                               func=lambda x, y: x | y)
        return ret

    def __xor__(self, other: 'BooleanTimeSeries') -> 'BooleanTimeSeries':
        ret = self._logical_op(other, symbol='^',
                               func=lambda x, y: x ^ y)
        return ret

    @classmethod
    def all(cls, *args: 'BooleanTimeSeries') -> 'BooleanTimeSeries':
        """Get all of given objects of BooleanTimeSeries."""
        if len(args) <= 0:
            raise ValueError("no object to compare")
        if not isinstance(args[0], cls):
            raise TypeError("all function only accepts 'BooleanTimeSeries'")
        values = [args[0]._values]
        masks = [args[0]._masks]
        names = [args[0]._name]
        index = args[0]._index
        for each in args[1:]:
            if not isinstance(each, cls):
                raise TypeError("all function only accepts 'BooleanTimeSeries'")
            if not index.equals(each._index):
                raise ValueError("inconsistent index")
            values.append(each._values)
            masks.append(each._masks)
            names.append(each._name)
        return cls(np.array(values).all(axis=0), index=index,
                   name=f'all({",".join(names)})',
                   masks=np.array(masks).any(axis=0))

    @classmethod
    def any(cls, *args: 'BooleanTimeSeries') -> 'BooleanTimeSeries':
        """Get any of given objects of BooleanTimeSeries."""
        if len(args) <= 0:
            raise ValueError("no object to compare")
        if not isinstance(args[0], cls):
            raise TypeError("any function only accepts 'BooleanTimeSeries'")
        values = [args[0]._values]
        masks = [args[0]._masks]
        names = [args[0]._name]
        index = args[0]._index
        for each in args[1:]:
            if not isinstance(each, cls):
                raise TypeError("any function only accepts 'BooleanTimeSeries'")
            if not index.equals(each._index):
                raise ValueError("inconsistent index")
            values.append(each._values)
            masks.append(each._masks)
            names.append(each._name)
        return cls(np.array(values).any(axis=0), index=index,
                   name=f'any({",".join(names)})',
                   masks=np.array(masks).any(axis=0))

    def to_pandas(self) -> pd.Series:
        """Return a copy of TimeSeries as pandas Series.

        Returns
        -------
        pandas.Series
            If there is no N/A value, it return Series with same dtype.
            If there are N/A values and its dtype is numpy.number, it return
            Series with dtype as numpy.float and use np.nan to represent NA
            values; otherwise, it return Series with datype as numpy.object
            and use numpy.nan to represent NA values.

        """
        values = self._values.astype(float)
        index = self._index.values
        name = self._name
        values[self._masks] = np.nan
        return pd.Series(values, index=index, name=name)