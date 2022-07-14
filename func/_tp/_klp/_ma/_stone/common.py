# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:04:59 2022

@author: WaNiNi
"""

import numpy as np

from ..common import (get_ma, RawMacro, LimitedCondition, LimitedVariable,
                      MacroParam, ParamType, PlotInfo, Ptype)

MAX_MA_PERIOD = 2520
MAX_SMAPLES = 252

def fix_values(values, sp, lp, n, sign, n_cover=0):
    for i in range(n):
        base = values[i: i+sp].mean()
        tar = values[i+sp: i+lp].mean()
        if sign >= 0 and base > tar:
            if i < n_cover:
                values[sp+n_cover: ] += (base - tar) * (lp - sp) / (lp - sp - n_cover + i)
            else:
                values[i+sp: ] += (base - tar)
        if sign <= 0 and base < tar:
            if i < n_cover:
                values[sp+n_cover: ] -= (tar - base) * (lp - sp) / (lp - sp - n_cover + i)
            else:
                values[i+sp: ] -= (tar - base)
    return values

def gen_cps4arranged_mas(periods, idx, n, sign=1, mu=0.002, sigma=0.005):
    if idx == 0:
        ret = np.cumprod(1+ np.random.normal(-mu * sign, sigma, n + periods[-1] - 1))
    elif idx == len(periods)-1:
        ret = np.cumprod(1+ np.random.normal(mu * sign, sigma, n + periods[-1] - 1))
    else:
        heads = np.random.normal(mu * sign, sigma, periods[idx-1])
        tails = np.random.normal(-mu * sign, sigma, periods[-1] - periods[idx-1])
        mids = np.random.normal(0, sigma, n - 1)
        ret = np.cumprod(1+ np.concatenate([heads, mids, tails]))
    for i, cur_p in enumerate(periods[1:]):
        prev_p = periods[i]
        if i < idx:
            ret = fix_values(ret, prev_p, cur_p, n, sign, 0)
        else:
            if i == idx and i > 0:
                ret = fix_values(ret, prev_p, cur_p, n, -sign, n - 1)
            else:
                ret = fix_values(ret, prev_p, cur_p, n, -sign, 0)
    ret = 100 * ret[::-1]
    if ret.min() <= 0:
        ret = ret - ret.min() + 100
    return ret

def gen_cps4arranged_mas_by_random_test(periods, idx, n, sign=1, sigma=0.01, max_times=100):
    def get_ma(cps, period):
        if period <= 1:
            return cps
        temp = np.cumsum(cps, axis=1)
        ret = np.concatenate([temp[:,:period] / np.arange(1, period+1),
                              (temp[:,period:] - temp[:,:-period]) / period], axis=1)
        return ret
    N = 10000
    L = periods[-1] + n - 1
    for i in range(max_times):
        cps = np.cumprod(1 + np.random.normal(0, sigma, (N, L)), axis=1) * 100
        mas = [get_ma(cps, p)[:,-n:] for p in periods]
        conditions = []
        for i in range(idx):
            if sign > 0:
                conditions.append((mas[i] <= mas[i+1]).all(axis=1))
            else:
                conditions.append((mas[i] >= mas[i+1]).all(axis=1))
        for i in range(idx+1, len(periods)):
            if sign > 0:
                conditions.append((mas[i] <= mas[i-1]).all(axis=1))
            else:
                conditions.append((mas[i] >= mas[i-1]).all(axis=1))
        conditions = np.array(conditions).all(axis=0)
        if conditions.any():
            return cps[conditions][0]
    raise RuntimeError('try over times')

def gen_cps4arranged_mas_by_random(periods, idx, m, n, sign=1, sigma=0.01, batch_size=10000, max_times=10):
    def get_ma(cps, period):
        if period <= 1:
            return cps
        temp = np.cumsum(cps, axis=1)
        ret = np.concatenate([temp[:,:period] / np.arange(1, period+1),
                              (temp[:,period:] - temp[:,:-period]) / period], axis=1)
        return ret
    N = batch_size
    L = periods[-1] + m - 1
    for i in range(max_times):
        cps = np.cumprod(1 + np.random.normal(0, sigma, (N, L)), axis=1) * 100
        mas = [get_ma(cps, p)[:,-m:] for p in periods]
        conditions = []
        for i in range(idx):
            if sign > 0:
                conditions.append(mas[i] <= mas[i+1])
            else:
                conditions.append(mas[i] >= mas[i+1])
        for i in range(idx+1, len(periods)):
            if sign > 0:
                conditions.append(mas[i] <= mas[i-1])
            else:
                conditions.append(mas[i] >= mas[i-1])
        conditions = np.array(conditions).all(axis=0).sum(axis=1) >= n
        if conditions.any():
            return cps[conditions][0]
    return None

def check_cps4arranged_mas(cps, periods, idx, m, n, sign=1):
    mas = [get_ma(cps, p).round(2)[-m:] for p in periods]
    ret = []
    for i in range(idx):
        if sign > 0:
            ret.append(mas[i] <= mas[i+1])
        else:
            ret.append(mas[i] >= mas[i+1])
    for i in range(idx+1, len(periods)):
        if sign > 0:
            ret.append(mas[i] <= mas[i-1])
        else:
            ret.append(mas[i] >= mas[i-1])
    assert np.array(ret).all(axis=0).sum() >= n
