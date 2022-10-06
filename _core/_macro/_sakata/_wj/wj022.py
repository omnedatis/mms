import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable

import pandas as pd
from _core._macro._sakata._wj import wj017, wj011
from _core._macro.common import Macro, MacroTags
from _core._macro._sakata._moke_candle import MokeCandle, KType
from func.common import MacroParam, ParamType, PlotInfo, Ptype, PeriodType
from func._td._index import TimeUnit
from func._ti import TechnicalIndicator as TI


code = 'wj022'
name = '酒田戰法指標(WJ版)-空頭吞噬'
description = """

> 先漲後跌，由多轉空的反轉型態

## 型態說明

1. 先前市場趨勢向上
2. 第一個 K 線是短實體陽線
3. 第二個 K 線是較長的實體黑線
4. 第二個 K 線的實體完全吞噬前一個 K 線實體的紅色實體

## 未來趨勢

發生後，則代表市場即將向下反轉。

## 現象解釋

### 傳統解釋

第一天的實體短陽線代表原本的上升力道變弱，第二天雖然開高，但接著一路下挫至低於前日開
盤價，代表賣壓已經形成，多頭失去了動能，而空頭的力道正在增強

### 心理面解釋

實體短陽線代表買氣的下降，並且第二日的長黑線代表賣壓遠超於買壓，使得原先的上漲趨勢完
全停止，刺激不久前入場或融資的投資者，進一步造成賣壓的持續上升，導致最終形成反轉向下
趨勢。

### 備註

需根據市場性質進一步判斷其他現象。
"""
params = [
    MacroParam(
        code='period_type',
        name='K線週期',
        desc='希望以哪種 K 線週期來偵測空頭吞噬',
        dtype=PeriodType,
        default=PeriodType.type.DAY)
]
db_ver = '2022100301'
py_ver = '2022100301'
tags = [MacroTags.PRICE]

def func(market_id:str, **kwargs) -> pd.Series:
    """計算並取得指定市場 ID 中的歷史資料, 每個日期是否有發生空頭吞噬的序列

    判斷規則:
    1. 先前市場趨勢向上
    2. 第一個 K 線是短實體陽線
    3. 第二個 K 線是較長的實體黑線
    4. 第二個 K 線的實體完全吞噬前一個 K 線實體的紅色實體

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
        市場各歷史時間點是否有發生空頭吞噬序列

    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj022'")
    candle = TI.Candle(market_id, period_type)
    period_type_to_period = {
        TimeUnit.DAY: 50,
        TimeUnit.WEEK: 10,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    # 1. 型態發生前會出現上升區段
    ma_5 = TI.MA(market_id, 5, period_type)
    ma_10 = TI.MA(market_id, 10, period_type)
    ma_diff = (ma_5 - ma_10).rolling(period, period_type).min()
    cond_1 = (ma_diff > 0)

    # 2. 第一個 K 線是短實體陽線
    raise_long_body = wj017.macro(market_id, **kwargs).shift(1)
    cond_2 = raise_long_body & (raise_long_body==raise_long_body)
    
    # 3. 第二個 K 線是較長的實體黑線
    down_long_body = wj011.macro(market_id, **kwargs)
    cond_3 = down_long_body & (down_long_body==down_long_body)
    
    # 4. 第二個 K 線的實體完全吞噬前一個 K 線實體的紅色實體
    cond_4 = (candle.open > candle.shift(1).close) & (candle.close < candle.shift(1).open).to_pandas()
    cond_4 = cond_4 & (cond_4==cond_4)
    result = cond_1.to_pandas() & cond_2 & cond_3 & cond_4
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
                           "'wj022'")

    results = {}
    try:
        period_type = kwargs['period_type'].data
    except ValueError as esp:
        results['period_type'] = f"invalid argument '{kwargs['period_type']}'"
    return results

def plot(**kwargs) -> List[PlotInfo]:
    """wj022 的範例圖製作函式

    判斷規則:
    1. 先前市場趨勢向上
    2. 第一個 K 線是短實體陽線
    3. 第二個 K 線是較長的實體黑線
    4. 第二個 K 線的實體完全吞噬前一個 K 線實體的紅色實體

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
                           "'wj021'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    rand_size = np.cumsum(np.abs(np.random.normal(5, 1, period+2)))
    rand_size[-1] = rand_size[-1]-10
    rand_size[-2] = rand_size[-1]
    data = [
        MokeCandle.make(KType.DOJI_FOUR_PRICE) for _ in range(period)
    ]+[MokeCandle.make(KType.WHITE_SHORT), MokeCandle.make(KType.BLACK_LONG)]
    data = np.array(data) + rand_size.reshape((len(rand_size), 1))
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
                           "'wj022'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    return period


wj022 = Macro(code=code, name=name, description=description, parameters=params,
        macro=func, sample_generator=plot, interval_evaluator=frame, arg_checker=check,
        db_version=db_ver, py_version=py_ver, tags=tags)
