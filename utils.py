# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

@author: Jeff
"""
import pickle
import os
from typing import Any, List, Union
import pandas as pd
import numpy as np

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
