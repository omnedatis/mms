import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable

import pandas as pd
from func._tp._ma import _stone as tp
from func.common import Macro, MacroParam, ParamType, PlotInfo, Ptype, PeriodType
from func._td._index import TimeUnit
from func._ti import TechnicalIndicator as TI
from func._tp._sakata._moke_candle import MokeCandle, KType

code = 'wj004'
name = '商智酒田戰法指標(WJ版)-傘型線'
description = """

> 趨勢的鏡像反轉

## 型態說明

1. 型態發生前會有上升或下降區段
2. 沒有實體線
3. 沒有上影線
4. 下影線很長

## 未來趨勢

若是於上漲趨勢發生，則是反轉向下訊號；若是於下跌趨勢發生，則是反轉向上訊號。

## 現象解釋

### 傳統解釋

當發生在上升趨勢時，代表賣壓強過買壓，使得盤中價格大幅下跌，雖然最終收盤時將價格拉回
至開盤價，但仍凸顯出賣壓已開始強過買壓，因此為向下反轉訊號；當發生在下跌趨勢時，代表
持續的賣壓無法壓制買壓，使得最終收盤時價格落於開盤價，意味著買壓已開始強過賣壓，因此
為向上反轉訊號。

### 心理面解釋

當傘型線發生在上漲趨勢時，持續上漲的價格被突然的賣壓打斷，使得盤中價格下降，雖然因為
買氣還在並且仍然強烈，因此使得價格回升，但那些較近期才持有的投資人會開始對於市場的信
心產生動搖，使得賣壓持續上升，最終導致反轉向下（這樣的過程類似於吊人線）；當傘型線發
生在下跌趨勢時，持續下跌的價格起初仍持續下探，但突然的強烈買壓導致價格大幅回彈，最終
導致價格收於開盤價附近，這使得短期賣空者風險大幅上升，發生比起再上漲區段的傘型線更為
強烈且快速的反彈，導致市場反轉向上。

### 備註

與吊人線相同，傘型線發生後若發生吊人線或傘型線，則反轉趨勢會更加明顯。此外，若是於下
跌區段發生傘型線，那麼會有著比起上漲區段更加強烈的反轉訊號。
"""
params = [
    MacroParam(
        code='period_type',
        name='K線週期',
        desc='希望以哪種 K 線週期來偵測傘型線',
        dtype=PeriodType,
        default=PeriodType.type.DAY)
]
db_ver = '2022081001'
py_ver = '2022081001'

def func(market_id:str, **kwargs) -> pd.Series:
    """計算並取得指定市場 ID 中的歷史資料, 每個日期是否有發生傘型線的序列

    判斷規則:
    1. 型態發生前會有上升或下降區段
    2. 沒有實體線
    3. 沒有上影線
    4. 下影線很長

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
        市場各歷史時間點是否有發生傘型線序列

    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj004'")
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
    # 2. 沒有實體線
    ba_ratio = candle.body/candle.amplitude
    cond_2 = (ba_ratio == 0)
    # 3. 沒有上影線
    usa_ratio = candle.upper_shadow/candle.amplitude
    cond_3 = usa_ratio == 0
    # 4. 下影線很長
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
                           "'wj004'")

    results = {}
    try:
        period_type = kwargs['period_type'].data
    except ValueError as esp:
        results['period_type'] = f"invalid argument '{kwargs['period_type']}'"
    return results

def plot(**kwargs) -> List[PlotInfo]:
    """wj004 的範例圖製作函式

    判斷規則:
    1. 型態發生前會有上升或下降區段
    2. 沒有實體線
    3. 沒有上影線
    4. 下影線很長

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
                           "'wj004'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    pos_neg_rand = np.random.randint(0,2)*2 - 1
    rand_size = pos_neg_rand * np.cumsum(np.abs(np.random.normal(5, 1, period+1)))
    rand_size[-1] = rand_size[-1]-10
    data = [
        MokeCandle.make(KType.DOJI_FOUR_PRICE) for _ in range(period)
    ]+[MokeCandle.make(KType.DOJI_UMBRELLA)]
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
                           "'wj004'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    return period


wj004 = Macro(code=code, name=name, desc=description, params=params,
        run=func, check=check, plot=plot, frame=frame,
        db_ver=db_ver, py_ver=py_ver)
