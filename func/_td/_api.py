# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:38:25 2021

@author: WaNiNi
"""

import json
from typing import List, Optional, Union

import pandas as pd
import numpy as np

PREDICT_PERIODS = [5, 10, 20, 40, 60, 120]

def fast_concat(data: List[Union[pd.Series, pd.DataFrame]]):
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

def api_get_vinfo() -> List[dict]:
    raise NotImplementedError()

def api_get_markets() -> List[str]:
    raise NotImplementedError()

def api_get_data(markets: Optional[List[str]] = None,
                 date: Optional[str] = None):
    raise NotImplementedError()

def api_get_pinfo():
    raise NotImplementedError()

def get_markets() -> List[str]:
    try:
        return api_get_markets()
    except NotImplementedError:
        return ['TWSE'] + [f'TWSE_{i}'for i in range(200)]

def get_pinfo():
    try:
        return api_get_pinfo()
    except NotImplementedError:
        ret = []
        for each in json.load(open('_local_db/doutze_cc_p.json', 'r', encoding="utf-8")):
            ret.append({'id': each['id'],
                        'func': each['func'],
                        'params': each['params']})
        for each in json.load(open('_local_db/jack_ma_p.json', 'r', encoding="utf-8")):
            ret.append({'id': each['id'],
                        'func': each['func'],
                        'params': each['params']})
        for each in json.load(open('_local_db/stone_ma_p.json', 'r', encoding="utf-8")):
            ret.append({'id': each['id'],
                        'func': each['func'],
                        'params': each['params']})
        return ret

def get_vinfo():
    try:
        return api_get_vinfo()
    except NotImplementedError:
        ret = []
        pids = [each['id'] for each in json.load(open('_local_db/doutze_cc_p.json', 'r', encoding="utf-8"))]
        ret.append({'model_id': 'NTWL',
                    'markets': None,
                    'patterns': pids,
                    'train_begin': None,
                    'train_gap': 3})
        pids = [each['id'] for each in json.load(open('_local_db/jack_ma_p.json', 'r', encoding="utf-8"))]
        ret.append({'model_id': 'JackMA',
                    'markets': None,
                    'patterns': pids,
                    'train_begin': None,
                    'train_gap': 3})
        pids = [each['id'] for each in json.load(open('_local_db/stone_ma_p.json', 'r', encoding="utf-8"))]
        ret.append({'model_id': 'StoneMA',
                    'markets': None,
                    'patterns': pids,
                    'train_begin': None,
                    'train_gap': 3})
        pids = ([each['id'] for each in json.load(open('_local_db/doutze_cc_p.json', 'r', encoding="utf-8"))] + 
                [each['id'] for each in json.load(open('_local_db/jack_ma_p.json', 'r', encoding="utf-8"))] + 
                [each['id'] for each in json.load(open('_local_db/stone_ma_p.json', 'r', encoding="utf-8"))])
        ret = []
        idxs = np.arange(len(pids))        
        for i in range(10):
            for k in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100,
                      120, 150, 200]:
                np.random.shuffle(idxs)
                ret.append({'model_id': f'P{k}_{i}',
                            'markets': None,
                            'patterns': [pids[i] for i in idxs[:k]],
                            'train_begin': None,
                            'train_gap': 3})
        import random
        random.shuffle(ret)
        return ret
