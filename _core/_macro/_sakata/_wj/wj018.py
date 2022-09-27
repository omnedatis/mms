import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable

import pandas as pd
from _core._macro.common import Macro, MacroTags
from _core._macro._sakata._moke_candle import MokeCandle, KType
from func.common import MacroParam, ParamType, PlotInfo, Ptype, PeriodType
from func._td._index import TimeUnit
from func._ti import TechnicalIndicator as TI

code = 'wj018'
name = '酒田戰法指標(WJ版)-短黑燭'
description = """

> 趨勢不明

## 型態說明

1. 實體陰線
2. 實體較短
3. 有上下影線
4. 上下影線皆比實體短

## 未來趨勢

發生後，市場趨勢處於不明的階段，任何情況都有可能會發生。

## 現象解釋

### 傳統解釋

股價有可能是開高走低，也有可能是開低走高，最終收在低於開盤價相對較高的位置。

### 心理面解釋

賣氣稍高於買氣，但不足以形成強烈趨勢，使得發生買氣隨時可以超越賣氣的不明趨勢狀態。

### 備註

需根據市場性質進一步判斷其他現象。
"""
params = [
    MacroParam(
        code='period_type',
        name='K線週期',
        desc='希望以哪種 K 線週期來偵測短黑燭',
        dtype=PeriodType,
        default=PeriodType.type.DAY)
]
db_ver = '2022091901'
py_ver = '2022091901'
tags = [MacroTags.PRICE]

def func(market_id:str, **kwargs) -> pd.Series:
    """計算並取得指定市場 ID 中的歷史資料, 每個日期是否有發生短黑燭的序列

    判斷規則:
    1. 實體陰線
    2. 實體較短
    3. 有上下影線
    4. 上下影線皆比實體短

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
        市場各歷史時間點是否有發生短黑燭序列

    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj018'")
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
    # 2. 實體較短
    b_roll = candle.body.rolling(period, period_type)
    b_mean = b_roll.mean()
    b_std = b_roll.std()
    b_threshold = b_mean - b_std * 0.5
    cond_2 = (candle.body <= b_threshold)
    # 3. 上下影線皆比實體短
    ultb = candle.upper_shadow < candle.body
    lltb = candle.lower_shadow < candle.body
    cond_3 = ultb & lltb

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
                           "'wj018'")

    results = {}
    try:
        period_type = kwargs['period_type'].data
    except ValueError as esp:
        results['period_type'] = f"invalid argument '{kwargs['period_type']}'"
    return results

def plot(**kwargs) -> List[PlotInfo]:
    """wj018 的範例圖製作函式

    判斷規則:
    1. 實體陰線
    2. 實體較短
    3. 有上下影線
    4. 上下影線皆比實體短

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
                           "'wj018'")
    data = [MokeCandle.make(KType.BLACK_TINY_SHORT_EQUAL_SHADOW)]
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
                           "'wj018'")
    return 1


wj018 = Macro(code=code, name=name, description=description, parameters=params,
        macro=func, sample_generator=plot, interval_evaluator=frame, arg_checker=check,
        db_version=db_ver, py_version=py_ver, tags=tags)
