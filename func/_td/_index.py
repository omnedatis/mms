# -*- coding: utf-8 -*-
"""Time Index.

This module includes classes and methods for time-index and operations.

Enumerators
-----------
TimeUnit : Definitions of unit of time.

Classes
-------
TimeIndex : Immutable ndarray of datetimes, used as index of timing data.

"""

from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Optional, Union

import numpy as np

from ._groupby import ArrayGroup, HeteroArrayGroup, HomoArrayGroup, moving_sampling


class _TimeUnit(NamedTuple):
    """Unit of Time.

    Methods
    -------
    is_superunit : Determine the time-unit is a super-unit of another numpy
                   dtype of time.
    is_subunit : Determine the time-unit is a sub-unit of another numpy
                 dtype of time.
    is_equivalent : Determine the time-unit is equivalent to another numpy
                    dtype of time.
    encode : Trans an array of datetimes to the integer representation of
             this time-unit.
    decode : Trans the input array of integers to the numpy datetime
             representation of this time-unit.

    """
    name: str
    prefix: str
    dtype: np.dtype
    extra_encoder: Optional[Callable[[np.ndarray], np.ndarray]] = None
    extra_decoder: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def is_superunit(self, other: np.dtype) -> bool:
        """Determine the time-unit is a super-unit of another dtype of time.

        Parameters
        ----------
        other : numpy.dtype

        Returns
        -------
        bool
            Return ``True`` if the time-unit is a super-unit of the given dtype
            of time; otherwise return ``False``.

        """
        # Only used in current module, ignore dynamic type checking.
        if self.dtype == other:
            return self.extra_encoder is not None
        return self.dtype < other

    def is_subunit(self, other: np.dtype) -> bool:
        """Determine the time-unit is a sub-unit of another dtype of time.

        Parameters
        ----------
        other : numpy.dtype

        Returns
        -------
        bool
            Return ``True`` if the time-unit is a sub-unit of the given dtype
            of time; otherwise return ``False``.

        """
        # Only used in current module, ignore dynamic type checking.
        return self.dtype > other

    def is_equivalent(self, other: np.dtype) -> bool:
        """Determine the time-unit is equivalent to another dtype of time.

        Parameters
        ----------
        other : numpy.dtype

        Returns
        -------
        bool
            Return ``True`` if the time-unit is equivalent to the given dtype
            of time; otherwise return ``False``.

        """
        # Only used in current module, ignore dynamic type checking.
        return self.dtype == other and self.extra_encoder is None

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Encoder of time-unit.

        Trans the input array of datetimes to the integer representation of
        this time-unit.

        Parameters
        ----------
        values : numpy.ndarray
            The input array of datetimes.

        Returns
        -------
        numpy.ndarray
            The output array of integers.

        """
        # Only used in current module, ignore dynamic type checking.
        ret = values.astype(self.dtype).astype(int)
        if self.extra_encoder:
            # pylint: disable=not-callable
            ret = self.extra_encoder(ret)
            # pylint:enable=not-callable
        return ret

    def decode(self, values: np.ndarray) -> np.ndarray:
        """Decoder of time-unit.

        Trans the input array of integers to the numpy datetime representation
        of this time-unit.

        Parameters
        ----------
        values : numpy.ndarray
            The input array of integers.

        Returns
        -------
        numpy.ndarray
            The output array of datetimes.

        """
        # Only used in current module, ignore dynamic type checking.
        if self.extra_decoder:
            # pylint: disable=not-callable
            values = self.extra_decoder(values)
            # pylint:enable=not-callable
        ret = values.astype(self.dtype)
        return ret


def _encode2week(values: np.ndarray) -> np.ndarray:
    # Only used in current module, ignore dynamic type checking.
    # Because the integer representation of 1970-01-01 is 0 but
    # its weekday is 4(Thirsday), we shift intger representation
    # of dates with 3.
    ret = (values + 3) // 7  # 7 days per week
    return ret


def _decode_from_week(values: np.ndarray) -> np.ndarray:
    # Only used in current module, ignore dynamic type checking.
    # inverse of `_encode2week`
    ret = values * 7 - 3  # 7 days per week
    return ret


def _encode2quarter(values: np.ndarray) -> np.ndarray:
    # Only used in current module, ignore dynamic type checking.
    ret = values // 3  # 3 months per quarter
    return ret


def _decode_from_quarter(values: np.ndarray) -> np.ndarray:
    # Only used in current module, ignore dynamic type checking.
    # inverse of `_encode2quarter`
    ret = values * 3  # 3 months per quarter
    return ret


class TimeUnit(_TimeUnit, Enum):
    """Definitions of unit of time.

    Members
    -------
    SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, YEAR.

    """
    SECOND = _TimeUnit('second', 's', np.dtype('datetime64[s]'))
    MINUTE = _TimeUnit('minute', 'm', np.dtype('datetime64[m]'))
    HOUR = _TimeUnit('hour', 'h', np.dtype('datetime64[h]'))
    DAY = _TimeUnit('day', 'D', np.dtype('datetime64[D]'))
    WEEK = _TimeUnit('week', 'W', np.dtype('datetime64[D]'),
                     extra_encoder=_encode2week,
                     extra_decoder=_decode_from_week)
    MONTH = _TimeUnit('month', 'M', np.dtype('datetime64[M]'))
    QUARTER = _TimeUnit('quarter', 'Q', np.dtype('datetime64[M]'),
                        extra_encoder=_encode2quarter,
                        extra_decoder=_decode_from_quarter)
    YEAR = _TimeUnit('year', 'Y', np.dtype('datetime64[Y]'))

    @classmethod
    def get(cls, prefix: str):
        for each in cls:
            if each.prefix == prefix or each.name == prefix:
                return each
            
        raise ValueError(f"prefix: '{prefix}' was not found")

class _IndexGroupBy:
    def __init__(self, index: np.ndarray, unit: TimeUnit):
        # Only used in current module, ignore dynamic type checking.
        values = unit.encode(index)
        is_changed = values[1:] != values[:-1]
        is_begin = np.concatenate([[True], is_changed])
        is_end = np.concatenate([is_changed, [True]])
        self._group_id = np.cumsum(is_begin) - 1
        self._begin_idx = np.where(is_begin)[0]  # only 1-D
        self._end_idx = np.where(is_end)[0]  # only 1-D

    def rolling(self, values: np.ndarray, period: int) -> HeteroArrayGroup:
        """Rolling group values along index by desired period.

        Parameters
        ----------
        values : np.ndarray
        period : int

        Returns
        -------
        HeteroArrayGroup

        """
        # Only used in current module, ignore dynamic type checking.
        if period > 0:
            # group backward
            end_idx = np.arange(1, len(values) + 1)
            begin_idx = self._begin_idx
            if period > 1:
                begin_idx = np.concatenate([np.full(period - 1, 0), begin_idx])
            begin_idx = begin_idx[self._group_id]
        else:
            # group forward
            begin_idx = np.arange(len(values))
            end_idx = self._end_idx + 1
            if period < -1:
                offset = -period - 1
                end_idx = np.concatenate([end_idx[offset:],
                                          np.full(offset, end_idx[-1])])
            end_idx = end_idx[self._group_id]
        ret = [values[bidx:eidx] for bidx, eidx in zip(begin_idx, end_idx)]
        return HeteroArrayGroup(ret)

    def sampling(self, values: np.ndarray, samples: int, step: int
                 ) -> HomoArrayGroup:
        """Moving sample values along index by given step.

        Parameters
        ----------
        values : numpy.ndarray
            Values to sample.
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
        HomoArrayGroup

        """
        # Only used in current module, ignore dynamic type checking.
        if step > 0:
            # samples backward
            ret = moving_sampling(values[self._end_idx], samples, step)
            ret_values = ret.values[self._group_id]
            ret_masks = ret.masks[self._group_id]
            ret_values[:, -1] = values
        else:
            # samples forward
            ret = moving_sampling(values[self._begin_idx], samples, step)
            ret_values = ret.values[self._group_id]
            ret_masks = ret.masks[self._group_id]
            ret_values[:, 0] = values
        return HomoArrayGroup(ret_values, ret_masks)


class TimeIndex:
    """Time Index.

    Immutable sequence of datetimes, used as index of timing data.

    Parameters
    ----------
    data : array-like (1-dimensional)
        An array or list of datetimes.
    sort : bool
        If ``True``, sort `data`; otherwise do not. Default is ``True``.

    Properties
    ----------
    values : numpy.ndarray
        A copy of TimeIndex as a numpy array of datetimes.

    Built-in Functions
    ------------------
    len : int
        The length of TimeIndex.

    Methods
    -------
    equals : bool
        Determine if another TimeIndex object equals to self.
    rolling : ArrayGroup
        Rolling group values along index by desired period.
    sampling : HomoArrayGroup
        Moving Sample values along index by given step.

    Class Methods
    -------------
    combine : TimeIndex
        Combine TimeIndex objects as a new TimeIndex object.

    """
    def __init__(self, data: Union[np.ndarray, list],
                 sort: bool = True):
        # check `data` and set values
        if isinstance(data, list):
            # `data` is a list, trans it to a numpy array of datetimes
            self._values = np.array(data, dtype=np.datetime64)
        else:
            if not isinstance(data, np.ndarray):
                # `data` is neither a list nor a numpy array, raise an exception
                raise TypeError("`data` must be a list or numpy array not '%s'"
                                % (type(data).__name__))
            # `data` is a numpy array
            # Check its dtype, if it is np.datetime64, copy it;
            # otherwise raise an exception.
            if data.dtype.type is not np.datetime64:
                raise ValueError("dtype of `data` must be np.datetime64 not '%s'"
                                 % (data.dtype.type.__name__))
            self._values = data.copy()

        if sort:
            self._values.sort()

        # chche for `_groupby`
        self._cache: Dict['str', _IndexGroupBy] = {}

    @property
    def values(self) -> np.ndarray:
        """
        Return a copy of TimeIndex as a numpy array of datetimes.

        Returns
        -------
        numpy.ndarray

        """
        return self._values.copy()

    def __len__(self) -> int:
        """Return the length of TimeIndex."""
        return len(self._values)

    def equals(self, other: 'TimeIndex') -> bool:
        """Determine if another TimeIndex object equals to self.

        The things that are being compared are:
        - the elements inside the TimeIndex object, and
        - the order of the elements inside the TimeIndex object.

        Parameters
        ----------
        other : TimeIndex
            The other TimeIndex object to compare against.

        Returns
        -------
        bool
            ``True`` if `other` is an TimeIndex and it has the same elements
            and the same order as the calling object; ``False`` otherwise.

        """
        if not isinstance(other, TimeIndex):
            return False
        if self is other:
            return True
        # pylint: disable=protected-access
        return np.array_equal(self._values, other._values)
        # pylint: enable=protected-access

    @classmethod
    def combine(cls, indexs: List['TimeIndex']):
        """Combine TimeIndex objects as a new TimeIndex object.

        Parameters
        ----------
        indexs : List[Time Index]
            A list of TimeIndex objects to combine.

        Returns
        -------
        TimeIndex
            A new TimeIndex object including all unique elements which belongs
            to at least one of TimeIndex objects in `indexs`.

        """
        values = [each._values for each in indexs]
        ret = cls(np.unique(np.concatenate(values, axis=0)))
        return ret

    def _groupby(self, unit: TimeUnit) -> _IndexGroupBy:
        """Group index by desired unit of time.

        Parameters
        ----------
        unit : TimeUnit

        Returns
        -------
        _IndexGroupBy

        """
        # Only used in current class, ignore dynamic type checking.
        # It is more efficient to use name of time-unit as key in cache instead
        # of time-unit itself. It is no problem because there will never be two
        # time-units with same name.
        if unit.name not in self._cache:
            self._cache[unit.name] = _IndexGroupBy(self._values, unit)
        return self._cache[unit.name]

    def rolling(self, values: np.ndarray, period: int,
                punit: Optional[TimeUnit] = None
                ) -> ArrayGroup:
        """Rolling group values along index by desired period.

        Parameters
        ----------
        values : numpy.ndarray
            Values to rolling.
        period : int
            The period of rolling window.
        punit : TimeUnit
            It is optional. It it is specified, it must be an instance of
            TimeUnit which is a super-unit of or equivalent to the dtype of
            index.

        Returns
        -------
        ArrayGroup
            If `punit` is not specified or `punit` is the same as the
            time-unit of index, return an instance of HomoArrayGroup;
            otherwsie return an instance of HeteroArrayGroup.

        """
        # `values` must be a numpy array with the same length as index.
        if not isinstance(values, np.ndarray):
            raise TypeError("`values` must be an numpy array not '%s'"
                            % type(values).__name__)
        if len(values) != len(self._values):
            raise ValueError("inconsistent length of `values`")
        # `period` must be a non-zero integer.
        if not isinstance(period, int):
            raise TypeError("`period` must be an integer not '%s'"
                            % type(period).__name__)
        if period == 0:
            raise ValueError("`period` must be either positive or negative"
                             " not zero")
        # `punit` could not be specified. If it is, it must be a TimeUnit which
        # is a superunit of or equivalent to the dtype of index.
        if punit is not None:
            if not isinstance(punit, TimeUnit):
                raise ValueError("`punit` must be a member of 'TimeUnit'")
            if punit.is_subunit(self._values.dtype):
                raise ValueError("it is impossible to upsample from '%s' to '%s'"
                                 % (self._values.dtype.name, punit.dtype.name))
            if punit.is_superunit(self._values.dtype):
                # `punit` is a super-unit of dtype of index
                return self._groupby(punit).rolling(values, period)
        # `punit` is not specified or is an equivalent unit to dtype of index
        if period > 0:
            return moving_sampling(values, period, 1)
        return moving_sampling(values, -period, -1)

    def sampling(self, values: np.ndarray, samples: int, step: int = 1,
                 sunit: Optional[TimeUnit] = None
                 ) -> HomoArrayGroup:
        """Moving Sample values along index by given step.

        Parameters
        ----------
        values : numpy.ndarray
            Values to sample.
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
        HomoArrayGroup

        """
        # `values` must be a numpy array with the same length as index.
        if not isinstance(values, np.ndarray):
            raise TypeError("`values` must be an numpy array not '%s'"
                            % type(values).__name__)
        if len(values) != len(self._values):
            raise ValueError("inconsistent length of `values`")
        # `samples` must be a positive integer.
        if not isinstance(samples, int):
            raise TypeError("`samples` must be an integer not '%s'"
                            % type(samples).__name__)
        if samples <= 0:
            raise ValueError("`samples` must be a positive integer")
        # `step` must be a non-zero integer.
        if not isinstance(step, int):
            raise TypeError("`step` must be an integer not '%s'"
                            % type(step).__name__)
        if step == 0:
            raise ValueError("`step` could be positive or negative integer"
                             " not zero")
        # `sunit` could not be specified. If it is, it must be a TimeUnit which
        # is a superunit of or equivalent to the dtype of index.
        if sunit is not None:
            if not isinstance(sunit, TimeUnit):
                raise ValueError("`sunit` must be a member of 'TimeUnit'")
            if sunit.is_subunit(self._values.dtype):
                raise ValueError("it is impossible to upsample from '%s' to '%s'"
                                 % (self._values.dtype.name, sunit.dtype.name))
        if samples == 1:
            return HomoArrayGroup(values.reshape((-1, 1)))
        if sunit is None or sunit.is_equivalent(self._values.dtype):
            # `sunit` is not specified or is an equivalent unit with time-unit
            # of index
            return moving_sampling(values, samples, step)
        # `sunit` is a super-unit of time-unit of index
        return self._groupby(sunit).sampling(values, samples, step)
