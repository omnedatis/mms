import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable

import pandas as pd
from _core._macro.common import Macro, MacroTags
from func.common import MacroParam, ParamType, PlotInfo, Ptype, PeriodType
from func._td._index import TimeUnit
from func._ti import TechnicalIndicator as TI
from func._tp._sakata._moke_candle import MokeCandle, KType

code = 'wj011'
name = '酒田戰法指標(WJ版)-實體長黑線'
description = """

> 趨勢向下

## 型態說明

1. 實體陰線
2. 實體長度相對於線圖上其他 K 線較長
3. 有短的上下影線

## 未來趨勢

若是於下跌趨勢發生，那麼會持續向下；若是於上漲趨勢發生，則是反轉向下訊號。

## 現象解釋

### 傳統解釋

股價開高走低，收盤價遠低於開盤價，代表賣氣強盛，賣盤在強力主導行情。

### 心理面解釋

強力的賣氣使得股價下跌，雖然於開盤時有些許買壓，但不久後仍是賣壓大勝，使得股價直到收
盤時一直下跌，並且下跌幅度巨大。這代表著大部分資金的投資人強烈看跌，並且帶動多數投資
人引發強烈賣氣，若在上漲區段時將會是反轉訊號，下跌區段時則是使股價持續下跌。

### 備註

是否帶動多數投資人仍需要看成交量的關係較為準確。
"""
params = [
    MacroParam(
        code='period_type',
        name='K線週期',
        desc='希望以哪種 K 線週期來偵測實體長黑線',
        dtype=PeriodType,
        default=PeriodType.type.DAY)
]
db_ver = '2022090501'
py_ver = '2022090501'
tags = [MacroTags.PRICE]

def func(market_id:str, **kwargs) -> pd.Series:
    """計算並取得指定市場 ID 中的歷史資料, 每個日期是否有發生實體長黑線的序列

    判斷規則:
    1. 實體陰線
    2. 實體長度相對於線圖上其他 K 線較長
    3. 有短的上下影線

    Parameters
    ----------
    market_id: str
        市場 ID
    period_type: PeriodType
        [day | week | month]
        取得 K 線資料時需轉換為哪個時間單位做偵測(日K, 週K, 月K)

    Returns
    -------
    result: pd.Series
        市場各歷史時間點是否有發生實體長黑線序列

    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj011'")
    candle = TI.Candle(market_id, period_type)
    period_type_to_period = {
        TimeUnit.DAY: 50,
        TimeUnit.WEEK: 10,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    # 1. 實體陰線
    is_black = candle.close < candle.open
    cond_1 = is_black
    # 2. 實體長度相對於線圖上其他 K 線較長
    b_roll = candle.body.rolling(period, period_type)
    b_mean = b_roll.mean()
    b_std = b_roll.std()
    b_threshold = b_mean + b_std * 2
    cond_2 = (candle.body >= b_threshold)
    # 3. 有短的上下影線
    usa_ratio = candle.upper_shadow/candle.amplitude
    lsa_ratio = candle.lower_shadow/candle.amplitude
    cond_3 = (usa_ratio <= 0.1) & (lsa_ratio <= 0.1)

    cond = cond_1 & cond_2 & cond_3
    result = cond.to_pandas()
    return result

def check(**kwargs) -> Dict[str, str]:
    """參數正確性檢查式, 用於判斷參數是否正常, 檢查項目為
    1. 天期型態(日週月)必須為合法格式

    Parameters
    ----------
    period_type : str
        天期型態.

    Returns
    -------
    results: Dict[str, str]
        錯誤訊息字典, 內容為參數與其對應的錯誤原因

    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj011'")

    results = {}
    try:
        period_type = kwargs['period_type'].data
    except ValueError as esp:
        results['period_type'] = f"invalid argument '{kwargs['period_type']}'"
    return results

def plot(**kwargs) -> List[PlotInfo]:
    """wj011 的範例圖製作函式

    判斷規則:
    1. 實體陰線
    2. 實體長度相對於線圖上其他 K 線較長
    3. 有短的上下影線

    Parameters
    ----------
    period_type: str
        [day | week | month]
        取得 K 線資料時需轉換為哪個時間單位做偵測(日K, 週K, 月K)

    Returns
    -------
    results: List[PlotInfo]
        畫圖所使用的多條序列與序列名稱
    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj011'")
    data = [MokeCandle.make(KType.BLACK_LONG_TINY_EQUAL_SHADOW)]
    data = np.array(data)
    result = [PlotInfo(
        ptype=Ptype.CANDLE,
        title=f"K線",
        data=data.T)]
    return result

def frame(**kwargs) -> int:
    """取得繪製現象發生焦點的範圍大小

    Parameters
    ----------
    period_type: str
        [day | week | month]
        取得 K 線資料時需轉換為哪個時間單位做偵測(日K, 週K, 月K)

    Returns
    -------
    results: int
        現象發生時的範圍大小
    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj011'")
    return 1


wj011 = Macro(code=code, name=name, description=description, parameters=params,
        macro=func, sample_generator=plot, interval_evaluator=frame, arg_checker=check,
        db_version=db_ver, py_version=py_ver, tags=tags)
