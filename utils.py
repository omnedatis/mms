# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

@author: Jeff
"""

import datetime
import logging
import os
import pickle
import threading as mt
import traceback
from typing import Any, List, Union, Dict

import numpy as np
import pandas as pd

def mkdir(path: str):
    """make dir recursively.

    Parameters
    ----------
    path: str
        Path of directory to be made.

    """
    if not os.path.exists(path):
        os.makedirs(path)

def pickle_dump(data: Any, file: str):
    """dump object to file by pickle.

    Parameters
    ----------
    data: object
        Object to be dump.
    file: str
        Filename where the data is dump to.

    """
    mkdir(os.path.dirname(file))
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def pickle_load(file: str) -> Any:
    """load object from file by pickle.

    Parameters
    ----------
    file: str
        Filename where the data is loaded from.

    Returns
    -------
    object
        The object loaded from given file.

    """
    with open(file, 'rb') as f:
        ret = pickle.load(f)
    return ret

def esp2str(esp:Exception) -> str:
    """Trans exception to string."""
    return f'{type(esp).__name__}: {str(esp)}'

def fast_concat(data: List[Union[pd.Series, pd.DataFrame]]) -> pd.DataFrame:
    """Concat Pandas's Series and DataFrames with common index fastly."""
    index = data[0].index
    columns = []
    values = []
    for each in data:
        if isinstance(each, pd.DataFrame):
            columns += list(each.columns)
            values += list(each.values.T)
        else:
            columns.append(each.name)
            values.append(each.values)
    return pd.DataFrame(np.array(values).T, columns=columns, index=index)

def dict_equals(dict_a: Dict, dict_b: Dict) -> bool:
    """判斷兩個 dict 內容是否完全相等

    Parameters
    ----------
    dict_a: Dict
        字典 A
    dict_b: Dict
        字典 B

    Returns
    -------
    result: bool
        傳入的兩個字典內容是否完全相等
    """
    result = True
    if len(dict_a) != len(dict_b):
        result = False
    for key, value in dict_a.items():
        if key not in dict_b:
            result = False
            break
        if dict_b[key] != value:
            result = False
            break
    return result

class _CacheElement:
    def __init__(self, prev: int, next_: int, key: str, value: Any):
        self._prev = prev
        self._next = next_
        self._key = key
        self._value = value

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value: Any):
        self._value = value

    @property
    def prev(self) -> int:
        return self._prev

    @prev.setter
    def prev(self, key: int):
        self._prev = key

    @property
    def next_(self) -> int:
        return self._next

    @next_.setter
    def next_(self, key: int):
        self._next = key

class Cache:
    # 這邊建立了一個記憶體空間做快取,
    # 並強迫快取僅能在該空間中保存
    #
    # 使用dict紀錄key與節點的對應關係
    # 使用link list管理資料間的置換順序
    # 使用list作為cache buffer，提高link-list操作的速度
    # 若要使用FIFO置換策略，`lru`設定為False，
    # 否則將採用預設的LRU置換策略
    #
    # Examples:
    #     c = Cache(3)
    #     c['a'] = 1
    #     print(c.keys())
    #     c['b'] = 2
    #     print(c.keys())
    #     print(c['a'])
    #     print(c.keys())
    #     c['c'] = 3
    #     c['d'] = 4
    #     print(c.keys())

    def __init__(self, size: int, lru: bool = True):
        self._size = size
        self._lru = lru
        self._map: Dict[str, int] = {}
        self._buffer = [None for i in range(size)]
        self._head = -1
        self._tail = -1

    def clear(self):
        self._map = {}
        self._head = -1
        self._tail = -1
        # Note: clean buffer to free memory sapce
        self._buffer = [None for i in range(self._size)]

    def __contains__(self, key) -> bool:
        return key in self._map

    def __getitem__(self, key) -> Any:
        return self.get(key)

    def __setitem__(self, key, value):
        if key in self:
            self.update(key, value)
        else:
            self.add(key, value)

    def reset(self, size: int):
        self._size = size
        self.clear()

    def _set_tail(self, idx: int):
        if idx == self._tail:
            return
        tar = self._buffer[idx]
        if idx == self._head:
            self._head = tar.next_
            self._buffer[tar.next_].prev = -1
        else:
            self._buffer[tar.next_].prev = tar.prev
            self._buffer[tar.prev].next_ = tar.next_
        self._buffer[self._tail].next_ = idx
        tar.prev = self._tail
        tar.next_ = -1
        self._tail = idx

    def update(self, key: str, value: Any):
        idx = self._map[key]
        self._buffer[idx].value = value
        if self._lru:
            self._set_tail(idx)

    def get(self, key: str) -> Any:
        if key not in self._map:
            return None
        idx = self._map[key]
        if self._lru:
            self._set_tail(idx)
        return self._buffer[idx].value

    def add(self, key: str, value: Any):
        if key in self._map:
            return self.update(key, value)
        if len(self._map) >= self._size:  # It's full
            head = self._buffer[self._head]
            del self._map[head.key]
            self._map[key] = self._head
            self._buffer[self._head] = _CacheElement(head.prev, head.next_, key, value)
            self._set_tail(self._head)
        else:
            idx = len(self._map)
            if idx == 0:  # It's empty
                self._head = idx
                self._tail = idx
                self._buffer[idx] = _CacheElement(-1, -1, key, value)
            else:  # It's neither empty nor full
                self._buffer[self._tail].next_ = idx
                self._buffer[idx] = _CacheElement(self._tail, -1, key, value)
                self._tail = idx
            self._map[key] = idx

    def keys(self) -> List[str]:
        ret = []
        idx = self._head
        while idx >= 0:
            cur = self._buffer[idx]
            ret.append(cur.key)
            idx = cur.next_
        return ret

    def info(self):
        logging.info(f"""
            head: {self._buffer[self._head].key if self._head >= 0 else ""};
            tail: {self._buffer[self._tail].key if self._tail >= 0 else ""};
            keys: {self.keys()}
        """)

def datetime64d2int(recv: np.ndarray):
    return recv.astype(int)

def int2datetime64d(recv: np.ndarray):
    return recv.astype('datetime64[D]')

def extend_working_dates(recv: np.ndarray, length: int) -> np.ndarray:
    """Extend given dates with designated working dates.

    Parameters
    ----------
    recv: np.ndarray
        The dates to be extended.
    length: int
        The number of working dates to be added into `recv`.
        - if length is positive, extend `length` working dates forward.
        - if length is zero, do nothing.
        - if length is negative, extend `-length` working dates backward.

    Returns
    -------
    np.ndarray
        The extended dates.

    """
    if length == 0:
        return recv
    if length > 0:
        base = datetime64d2int(recv[-1])
        weekday = (base + 3) % 7
        if weekday >= 4:
            paddings = np.arange(length)
            paddings = paddings // 5 * 7 + paddings % 5 + (7 - weekday)
        else:
            offset = 4 - weekday
            if length > offset:
                paddings = np.arange(length - offset)
                paddings = paddings // 5 * 7 + paddings % 5 + (offset + 3)
                paddings = np.concatenate([np.arange(offset) + 1, paddings],
                                          axis=0)
            else:
                paddings = np.arange(length) + 1
        ret = np.concatenate([recv,
                              int2datetime64d(paddings + base)], axis=0)
    else:
        base = datetime64d2int(recv[0])
        weekday = (base + 3) % 7
        length = -length
        if weekday > 4:
            paddings = np.arange(length)
            paddings = paddings // 5 * 7 + paddings % 5 + (weekday - 4)
        else:
            offset = weekday
            if length > offset:
                paddings = np.arange(length - offset)
                paddings = paddings // 5 * 7 + paddings % 5 + (offset + 3)
                paddings = np.concatenate([np.arange(offset) + 1, paddings],
                                          axis=0)
            else:
                paddings = np.arange(length) + 1
        ret = np.concatenate([int2datetime64d(base - paddings[::-1]),
                              recv], axis=0)
    return ret


class CatchableTread:

    def __init__(self, target, args=None, name=None):
        self._args = args
        self._target = target
        self.esp = None
        self._thread = mt.Thread(name=name, target=self._run)

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            if self._args is not None:
                self._target(*self._args)
            else:
                self._target()
        except Exception as esp:
            logging.error(traceback.format_exc())
            self.esp = traceback.format_exc()

    def join(self):
        self._thread.join()

def print_error_info(esp):
    try:
        raise esp
    except:
        logging.error(traceback.format_exc())

class Wtimer:
    def __init__(self):
        self._t0 = datetime.datetime.now()

    def stop(self) -> int:
        now = datetime.datetime.now()
        ret = (now - self._t0).total_seconds()
        self._t0 = None
        return ret

    def start(self):
        if self._t0 is not None:
            raise RuntimeError("Timer is active")
        self._t0 = datetime.datetime.now()

def singleton(cls):
    """ Decorator, used to define classes on singleton pattern. """
    instances = {}
    def _wrapper():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return _wrapper


class ThreadController:
    """多執行緒執行控制器

    在多執行緒環境下，支援外部程序控制(提前中止)其他執行緒的執行

    Properties
    ----------
    isactive: bool
        True, if this controller is switch-on; otherwise, Flase.

    Methods
    -------
    switch_off:
        Turn-off the switch of this controller.

    """

    def __init__(self):
        self._state = True

    @property
    def isactive(self) -> bool:
        return self._state

    def switch_off(self):
        self._state = False
