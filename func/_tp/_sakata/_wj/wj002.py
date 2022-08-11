import random
import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable

import pandas as pd
from func._tp._ma import _stone as tp
from func.common import Macro, MacroParam, ParamType, PlotInfo, Ptype, PeriodType
from func._td._index import TimeUnit
from func._ti import TechnicalIndicator as TI
from func._tp._sakata._moke_candle import MokeCandle, KType

code = 'wj002'
name = '商智酒田戰法指標(WJ版)-高浪線'
description = """

> 趨勢的盤整

## 型態說明

1. 型態發生前會有上升或下降區段，或處於市場頂部或底部
2. 實體線很短
3. 上下影線很長

## 未來趨勢

進入盤整階段

## 現象解釋

### 傳統解釋

市場依照原先趨勢上漲或下跌，不論盤中如何劇烈震盪，於盤後還是會回到開盤價附近。這代表
市場當前已失去方向感，不論受到哪方向的刺激仍無法使其持續，處於多空不明的階段。

### 心理面解釋

原本市場投資人已經對於趨勢有一個共識，因此使得價格持續上漲或下跌，該趨勢可能中途受到
了某個劇烈的影響，使得持有大量資金的投資人改變對市場的看法，導致趨勢持續時受到劇烈阻
擋，在大量的交易中產生了盤中價格劇烈震盪，最終由於市場看法打平，停留在開盤價附近。

### 備註

這個現象暗示了某個持有市場大量資金的投資人的看法轉變，並不一定是針對市場做出了反向看
法，只能說明當前市場有大量資金的轉換發生。
"""
params = [
    MacroParam(
        code='period_type',
        name='K線週期',
        desc='希望以哪種 K 線週期來偵測高浪線',
        dtype=PeriodType,
        default=PeriodType.type.DAY)
]
db_ver = '20220810-v1'
py_ver = '20220810-v1'

def func(market_id:str, **kwargs) -> pd.Series:
    """計算並取得指定市場 ID 中的歷史資料, 每個日期是否有發生吊人線的序列

    判斷規則:
    1. 型態發生前會有上升或下降區段
    2. 實體線很短
    3. 上下影線很長

    Parameters
    ----------
    market_id: str
        市場 ID
    period_type: str
        [day | week | month]
        取得 K 線資料時需轉換為哪個時間單位做偵測(日K, 週K, 月K)

    Returns
    -------
    result: pd.Series
        市場各歷史時間點是否有發生高浪線序列

    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj002'")
    candle = TI.Candle(market_id, period_type)
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    # 1. 型態發生前會有上升或下降區段
    ma_5 = TI.MA(market_id, 5, period_type)
    ma_10 = TI.MA(market_id, 10, period_type)
    ma_diff = (ma_5 - ma_10).rolling(period, period_type)
    cond_1 = (ma_diff.min() > 0) | ((ma_diff.max() < 0))
    # 2. 實體線很短
    ba_ratio = candle.body/candle.amplitude
    cond_2 = (ba_ratio < 0.2)
    # 3. 上下影線很長
    lsb_ratio = candle.lower_shadow/candle.body
    usb_ratio = candle.upper_shadow/candle.body
    cond_3 = (lsb_ratio >= 1.6) & (usb_ratio >= 1.6)
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
                           "'wj002'")

    results = {}
    try:
        period_type = kwargs['period_type'].data
    except ValueError as esp:
        results['period_type'] = f"invalid argument '{kwargs['period_type']}'"
    return results

def plot(**kwargs) -> List[PlotInfo]:
    """wj002 的範例圖製作函式

    判斷規則:
    1. 型態發生前會有上升或下降區段，或處於市場頂部或底部
    2. 實體線很短
    3. 上下影線很長

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
                           "'wj002'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    pos_neg_rand = np.random.randint(0,2)*2 - 1
    candles = [
        MokeCandle.make(KType.WHITE_TINY_LONG_EQUAL_SHADOW),
        MokeCandle.make(KType.BLACK_TINY_LONG_EQUAL_SHADOW)
    ]
    rand_size = pos_neg_rand * np.cumsum(np.abs(np.random.normal(5, 1, period+1)))
    data = [
        MokeCandle.make(KType.DOJI_FOUR_PRICE) for _ in range(period)
    ]+[random.choice(candles)]
    data = np.array(data) + rand_size.reshape((len(rand_size), 1))
    result = [PlotInfo(
        ptype=Ptype.CANDLE,
        title=f"高浪線_{kwargs['period_type']}",
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
                           "'wj002'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    return period

wj002 = Macro(code=code, name=name, desc=description, params=params,
        run=func, check=check, plot=plot, frame=frame,
        db_ver=db_ver, py_ver=py_ver)
