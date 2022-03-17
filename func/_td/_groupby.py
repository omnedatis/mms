# -*- coding: utf-8 -*-
"""Utilities for GroupBy.

All classes and methods in this module are only used in current package. All
unnecessary dynamic checkings are ignored.

"""

from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional

import numpy as np


class ArrayGroup(metaclass=ABCMeta):
    """Abstract class of group of numpy arrays with masks.

    Only used in current package; there is no protection for properties access.

    Properties
    ----------
    values :
        Group of data arrays.
    masks :
        If it is set, it must be a group of boolean arrays in which each entry
        used to indicate whether the corresponding entry of data arrays in
        `values` is N/A or not. Otherwise, all entries in `values` are
        available.

    AbstractMethods
    ---------------
    equals : bool
        Determine if another ArrayGroup object equals to self.
    fillna :
        Fill N/A values by given value.

    """
    def __init__(self, values, masks=None):
        self.values = values
        self.masks = masks

    @abstractmethod
    def equals(self, other: Any) -> bool:
        """Determine if another ArrayGroup object equals to self.

        The things that are being compared are:
        - the valid entries of data arrays in `values`, and
        - the boolean arrays in `masks`.

        Parameters
        ----------
        other : ArrayGroup
            The other ArrayGroup object to compare against.

        Returns
        -------
        bool
            ``True`` if `other` is an ArrayGroup and it has the same valid
            values of the calling object; ``False`` otherwise.

        """

    @abstractmethod
    def fillna(self, value: Any) -> 'ArrayGroup':
        """Fill N/A values by given value.

        Parameters
        ----------
        value : Any
            Value used to fill N/A values.

        Returns
        -------
        ArrayGroup:
            A new instance with N/A values filled.

        """


class HomoArrayGroup(ArrayGroup):
    """Group of numpy arrays of same length.

    Only used in current package; there is no protection for properties access.

    Properties
    ----------
    values : numpy.array
        A 2-D array, in which the rows are data arrays of group.
    masks : numpy.array
        If it is set, it shoud be a 2-D boolean array, in which each entry
        used to indicate whether the corresponding entry in `values` is N/A
        or not. Otherwise, all entries in `values` are available.

    Methods
    -------
    equals : bool
        Determine if another HomoArrayGroup object equals to self.
    fillna :
        Fill N/A values by given value.

    See Also
    --------
    ArrayGroup

    """
    def __init__(self, values: np.ndarray, masks: Optional[np.ndarray] = None):
        # pylint: disable=useless-super-delegation
        # Unnecessary init method for static type checking(mypy)
        super().__init__(values, masks)
        # pylint: enable=useless-super-delegation

    def equals(self, other: Any) -> bool:
        """Determine if another HomoArrayGroup object equals to self.

        The things that are being compared are:
        - the valid entries of data arrays in `values`, and
        - the boolean arrays in `masks`.

        Parameters
        ----------
        other : HomoArrayGroup
            The other HomoArrayGroup object to compare against.

        Returns
        -------
        bool
            ``True`` if `other` is an HomoArrayGroup and it has the same valid
            values of the calling object; ``False`` otherwise.

        """
        if isinstance(other, HomoArrayGroup):
            if self.masks is None:
                if other.masks is None:
                    return np.array_equal(self.values, other.values)
            else:
                if other.masks is not None:
                    cond_1 = np.array_equal(self.masks, other.masks)
                    # pylint: disable=invalid-unary-operand-type
                    cond_2 = np.array_equal(self.values[~self.masks],
                                            other.values[~other.masks])
                    # pylint: enable=invalid-unary-operand-type
                    return cond_1 and cond_2
        return False

    def fillna(self, value: Any) -> 'HomoArrayGroup':
        """Fill N/A values by given value.

        Parameters
        ----------
        value : Any
            Value used to fill N/A values.

        Returns
        -------
        HomoArrayGroup:
            A new instance with N/A values filled.

        """
        values = self.values.copy()
        if self.masks is not None:
            values[self.masks] = value
        return HomoArrayGroup(values)


class HeteroArrayGroup(ArrayGroup):
    """Group of numpy arrays of various length.

    Only used in current package; there is no protection for properties access.

    Properties
    ----------
    values : List[numpy.array]
        A list of 1-D arrays, in which all arrays are data arrays of group.
    masks : List[numpy.array]
        If it is set, it should be a list of 1-D boolean arrays, in which each
        entry used to indicate whether the corresponding entry in `values` is
        N/A or not. Otherwise, all entries in `values` are available.

    Methods
    -------
    equals : bool
        Determine if another HeteroArrayGroup object equals to self.
    fillna :
        Fill N/A values by given value.

    See Also
    --------
    ArrayGroup

    """
    def __init__(self, values: List[np.ndarray],
                 masks: Optional[List[np.ndarray]] = None):
        # pylint: disable=useless-super-delegation
        # Unnecessary init method for static type checking(mypy)
        super().__init__(values, masks)
        # pylint: enable=useless-super-delegation

    def equals(self, other: Any) -> bool:
        """Determine if another HeteroArrayGroup object equals to self.

        The things that are being compared are:
        - the valid entries of data arrays in `values`, and
        - the boolean arrays in `masks`.

        Parameters
        ----------
        other : HeteroArrayGroup
            The other HeteroArrayGroup object to compare against.

        Returns
        -------
        bool
            ``True`` if `other` is an HeteroArrayGroup and it has the same
            valid values of the calling object; ``False`` otherwise.

        """
        if isinstance(other, HeteroArrayGroup):
            if self.masks is None:
                if other.masks is None:
                    for tar, ref in zip(self.values, other.values):
                        if np.array_equal(tar, ref):
                            continue
                        return False
                    return True
            else:
                if other.masks is not None:
                    for tar_v, ref_v, tar_m, ref_m in zip(self.values,
                                                          other.values,
                                                          self.masks,
                                                          other.masks):
                        cond_1 = np.array_equal(tar_m, ref_m)
                        cond_2 = np.array_equal(tar_v[~tar_m],
                                                ref_v[~ref_m])
                        if cond_1 and cond_2:
                            continue
                        return False
                    return True
        return False

    def fillna(self, value: Any) -> 'HeteroArrayGroup':
        """Fill N/A values by given value.

        Parameters
        ----------
        value : Any
            Value used to fill N/A values.

        Returns
        -------
        HeteroArrayGroup:
            A new instance with N/A values filled.

        """
        values = [each.copy() for each in self.values]
        if self.masks is not None:
            for idx, masks in enumerate(self.masks):
                values[idx][masks] = value
        return HeteroArrayGroup(values)


def _rolling_block(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    # Only used in current module, ignore dynamic type checking.
    shape = (len(values) - rows * cols + 1, rows, cols)
    strides = values.strides + (values.strides[-1] * cols, values.strides[-1])
    ret = np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides)
    return ret


def moving_sampling(values: np.ndarray, samples: int, step: int
                    ) -> HomoArrayGroup:
    """Moving sampling.

    Parameters
    ----------
    values : numpy.ndarray
        Values to sample.
    samples : int
        The number of samples.
    step : int, default is ``1``
        The step between two samples.

    Returns
    -------
    HomoArrayGroup

    """
    paddings = samples * abs(step) - 1
    if paddings <= 0:
        return HomoArrayGroup(values.reshape(-1, 1))
    if step > 0:
        # samples backward
        ex_values = np.concatenate([np.full(paddings, values[0]), values])
        ex_masks = np.concatenate([np.full(paddings, True),
                                   np.full(len(values), False)])
        ret_values = _rolling_block(ex_values, samples, step)[:, :, -1]
        ret_masks = _rolling_block(ex_masks, samples, step)[:, :, -1]
    else:
        # samples forward
        ex_values = np.concatenate([values, np.full(paddings, values[-1])])
        ex_masks = np.concatenate([np.full(len(values), False),
                                   np.full(paddings, True)])
        ret_values = _rolling_block(ex_values, samples, -step)[:, :, 0]
        ret_masks = _rolling_block(ex_masks, samples, -step)[:, :, 0]
    return HomoArrayGroup(ret_values, ret_masks)
