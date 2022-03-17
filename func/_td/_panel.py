# -*- coding: utf-8 -*-
"""Panel Data.

This module includes classes and methods for panel data and operations.

Classes
-------
PanelData : A collection of time-series with a common time-index.

"""

from ._series import TimeSeries


class PanelData:
    """Panel Data.

    A collection of time-series with a common time-index.

    Parameters
    ----------
    name : str
        The name of the panel data.
    **kwargs : dict of {str, TimeSeries}
        A set of Named arguments corresponds to each element in the collection
        of the panel data.

    Attributes
    ----------
    name : str, read-only
        The name of the panel data.

    Built-in Functions
    ------------------
    len : int
        The length of the panel data.

    Methods
    -------
    rename
        Alter the name of the panel data.
    dropna : PanelData
        Return a new instance with all records containing N/A values removed.
    ffill : PanelData
        Return a new instance with N/A values filled by available value prior
        to the N/A.
    bfill : PanelData
        Return a new instance with N/A values filled by next available
        value.
    to_pandas : pandas.DataFrame
        Return a `pandas.DataFrame` instance representing the panel data.
    shift :
        Shift index by desired number of periods.

    """
    def __init__(self, name: str, **kwargs: TimeSeries):
        if not isinstance(name, str):
            raise TypeError(f"`name` must be 'str' not '{type(name).__name__}'")
        if len(kwargs) <= 0:
            raise TypeError("`PanelData` takes at least one additional named"
                            " argument")
        self._name = name
        self._index = None
        self._series = {}
        for key, ts_ in kwargs:
            if not isinstance(ts_, TimeSeries):
                raise TypeError(f"extra named argument `{key}` must be `TimeSeries`"
                                f" not {type(ts_).__name__}")
            if self._index is None:
                self._index = ts_.index
            elif not self._index.equals(ts_.index):
                raise ValueError("all extra named arguments msut be instances"
                                 " of `TimeSeries` with common index")
            self._series[key] = ts_
            
    def dropna(self):
        is_na = np.array(

        
        

