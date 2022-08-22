import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable

import pandas as pd
from func._tp._ma import _stone as tp
from func.common import Macro, MacroParam, ParamType, PlotInfo, Ptype, PeriodType
from func._td._index import TimeUnit
from func._ti import TechnicalIndicator as TI
from func._tp._sakata._moke_candle import MokeCandle, KType

code = 'wj005'
name = '商智酒田戰法指標(WJ版)-倒狀傘型線'
description = """

> 趨勢的鏡像反轉

## 型態說明

1. 型態發生前會有上升或下降區段
2. 沒有實體線
3. 沒有下影線
4. 上影線很長

## 未來趨勢

若是於上漲趨勢發生，則是反轉向下訊號；若是於下跌趨勢發生，則是反轉向上訊號。

## 現象解釋

### 傳統解釋

當發生在上升趨勢時，代表買壓僅持續至盤中，後續開始低於賣壓，是多轉空的訊號；當發生在
下降區段時，則是賣壓無法完全高於買壓，使得盤中價格上升，雖後續賣壓仍高於買壓，但已經
出現了反彈的預兆，是空轉多的訊號。

### 心理面解釋

當倒狀傘型線發生在上漲趨勢時，持續上漲的趨勢於盤中後便遭到巨大的賣壓壓回，雖然賣壓最
終與買壓達到平衡，使得最終收盤價落於開盤價附近，但仍代表市場中持有大量資金的投資人已
經有半數開始持空頭看法，因此為向下反轉訊號；當倒狀傘型線發生在下跌趨勢時，代表下跌趨
勢在盤中以前受到大量的買氣影響，使得價格一路飆升，雖然無法持續至收盤，但仍對於持空頭
的投資人帶來巨大的壓力，進而增加額外的風險，使得空頭部分投資人承受壓力，進一步造成市
場向上反彈。

### 備註

當倒狀傘型線發生後，若是後續發生其他的反彈訊號時，因為會使得持有空頭部位的投資人需要
承擔的風險持續放大，因此應該會使得該反轉型態更加確定。
"""
params = [
    MacroParam(
        code='period_type',
        name='K線週期',
        desc='希望以哪種 K 線週期來偵測倒狀傘型線',
        dtype=PeriodType,
        default=PeriodType.type.DAY)
]
db_ver = '2022082201'
py_ver = '2022082201'

def func(market_id:str, **kwargs) -> pd.Series:
    """計算並取得指定市場 ID 中的歷史資料, 每個日期是否有發生倒狀傘型線的序列

    判斷規則:
    1. 型態發生前會有上升或下降區段
    2. 沒有實體線
    3. 沒有下影線
    4. 上影線很長

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
        市場各歷史時間點是否有發生倒狀傘型線序列

    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj005'")
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
    # 3. 沒有下影線
    lsa_ratio = candle.lower_shadow/candle.amplitude
    cond_3 = lsa_ratio == 0
    # 4. 上影線很長
    usb_ratio = candle.upper_shadow/candle.body
    cond_4 = usb_ratio >= 2

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
                           "'wj005'")

    results = {}
    try:
        period_type = kwargs['period_type'].data
    except ValueError as esp:
        results['period_type'] = f"invalid argument '{kwargs['period_type']}'"
    return results

def plot(**kwargs) -> List[PlotInfo]:
    """wj005 的範例圖製作函式

    判斷規則:
    1. 型態發生前會有上升或下降區段
    2. 沒有實體線
    3. 沒有下影線
    4. 上影線很長

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
                           "'wj005'")
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
    ]+[MokeCandle.make(KType.DOJI_INVERSE_UMBRELLA)]
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
                           "'wj005'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    return period


wj005 = Macro(code=code, name=name, desc=description, params=params,
        run=func, check=check, plot=plot, frame=frame,
        db_ver=db_ver, py_ver=py_ver)
