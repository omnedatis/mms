import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable

import pandas as pd
from func._tp._ma import _stone as tp
from func.common import Macro, MacroParam, ParamType, PlotInfo, Ptype, PeriodType
from func._td._index import TimeUnit
from func._ti import TechnicalIndicator as TI
from func._tp._sakata._moke_candle import MokeCandle, KType

code = 'wj006'
name = '商智酒田戰法指標(WJ版)-槌子'
description = """

> 趨勢反轉向上

## 型態說明

1. 型態發生前會有下降區段
2. 短實體陽線
3. 沒有上影線
4. 下影線很長, 長度為實體長度的兩倍以上

## 未來趨勢

若是於下跌趨勢發生，則是反轉向上訊號。

## 現象解釋

### 傳統解釋

在下跌趨勢發生時，開盤後維持下跌，但盤中開始出現大量買壓，並且買壓超過賣壓一直到收盤
。這表示開始出現大量買氣，使得原先的下跌趨勢可能面臨向上反轉。

### 心理面解釋

在下跌區段中，盤中出現了大量的買氣，不僅造成下跌的中段開始反彈，更是使得收盤價高於開
盤價。這樣的買氣會增加放空投資人的風險，使其開始擔心放空部位是否已造成過大的風險，進
一步開始考量平倉，促成下跌區段的停止，進而造成股價反轉向上。

### 備註

當發生槌子時，可能是造成趨勢反轉向上的前兆，可以觀察到市場買壓已高於賣壓，但仍需觀察
數日以確保該現象能夠發酵，進而確認反轉訊號。若是有多個反轉向上的訊號於附近發生，那麼
則有很大的可能可以確定為反轉向上訊號。
"""
params = [
    MacroParam(
        code='period_type',
        name='K線週期',
        desc='希望以哪種 K 線週期來偵測槌子',
        dtype=PeriodType,
        default=PeriodType.type.DAY)
]
db_ver = '2022082201'
py_ver = '2022082201'

def func(market_id:str, **kwargs) -> pd.Series:
    """計算並取得指定市場 ID 中的歷史資料, 每個日期是否有發生槌子的序列

    判斷規則:
    1. 型態發生前會有下降區段
    2. 短實體陽線
    3. 沒有上影線
    4. 下影線很長, 長度為實體長度的兩倍以上

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
        市場各歷史時間點是否有發生槌子序列

    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj006'")
    candle = TI.Candle(market_id, period_type)
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    # 1. 型態發生前會有下降區段
    ma_5 = TI.MA(market_id, 5, period_type)
    ma_10 = TI.MA(market_id, 10, period_type)
    ma_diff = (ma_5 - ma_10).rolling(period, period_type)
    cond_1 = (ma_diff.min() > 0)
    # 2. 短實體陽線
    is_white = candle.close > candle.open
    ba_ratio = candle.body/candle.amplitude
    cond_2 = is_white & (ba_ratio > 0) & (ba_ratio < 0.2)
    # 3. 沒有上影線
    usa_ratio = candle.upper_shadow/candle.amplitude
    cond_3 = usa_ratio == 0
    # 4. 下影線很長, 長度為實體長度的兩倍以上
    lsb_ratio = candle.lower_shadow/candle.body
    cond_4 = lsb_ratio >= 2

    cond = cond_1 & cond_2 & cond_3 & cond_4
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
                           "'wj006'")

    results = {}
    try:
        period_type = kwargs['period_type'].data
    except ValueError as esp:
        results['period_type'] = f"invalid argument '{kwargs['period_type']}'"
    return results

def plot(**kwargs) -> List[PlotInfo]:
    """wj006 的範例圖製作函式

    判斷規則:
    1. 型態發生前會有下降區段
    2. 短實體陽線
    3. 沒有上影線
    4. 下影線很長, 長度為實體長度的兩倍以上

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
                           "'wj006'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    rand_size = (-1) * np.cumsum(np.abs(np.random.normal(5, 1, period+1)))
    rand_size[-1] = rand_size[-1]-10
    data = [
        MokeCandle.make(KType.DOJI_FOUR_PRICE) for _ in range(period)
    ]+[MokeCandle.make(KType.WHITE_HAMMER)]
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
                           "'wj006'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    return period


wj006 = Macro(code=code, name=name, desc=description, params=params,
        run=func, check=check, plot=plot, frame=frame,
        db_ver=db_ver, py_ver=py_ver)
