import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable

import pandas as pd
from func._tp._ma import _stone as tp
from func.common import Macro, MacroParam, ParamType, PlotInfo, Ptype, PeriodType
from func._td._index import TimeUnit
from func._ti import TechnicalIndicator as TI
from func._tp._sakata._moke_candle import MokeCandle, KType

code = 'wj001'
name = '商智酒田戰法指標(WJ版)-吊人線'
description = """

> 先漲後跌，由多轉空的反轉型態

## 型態說明
1. 型態發生前會有上升區段，或處於市場頂部
2. 開盤價在當天價格高點
3. 當日形成一個短實體黑 K 棒
4. 下影線很長，至少要有實體長度的兩倍以上
5. 沒有上影線

## 未來趨勢
反轉向下

## 現象解釋
### 傳統解釋
繼原先的上漲趨勢，開盤時持續向上開高，盤中面對大量的賣出壓力，使得價格大幅下降，即使
買氣仍持續，但仍敵不過賣壓，最後雖收在開盤價附近的價位，但仍為黑線。

### 心理面解釋
認為當前價格已達最高的投資人比例開始占多數，因此有急著趁高價脫手的現象發生，使得開高
後遇到大幅賣壓。當這些想脫手的投資人賣完後，仍有認為會續漲的投資人買進，但由於這些投
資人的人數沒有多到將先前的跌幅補足（但仍足夠多到可以將盤面拉回開盤價附近），因此最終
收在開盤價附近。

酒田戰法認為，價格已達最高價的投資人會在該價位中保持一定比例，這會使得價格無法向上繼
續提升，進一步的讓更多原本不認為已達最高價的投資人開始認為真的已達最高價，並開始使得
認為這個價位為最高價的投資人比例持續提升，進一步造成盤面開始下跌，因此該現象將會是一
個高點反轉訊號。

### 備註
這邊認為吊人線所造成的反轉現象會根據公司的股性不同而有所改變，反轉後的跌幅也會有所改
變。

股性價格本身就易受影響的股票，當現象發生後會發生反轉的時間點會較快，下跌的幅度也較高
；而價格本身不易受影響的股票則是會需要更多的時間來讓投資人相信價位已達高點，並且使其
緩慢下跌。

期間內越多次的吊人線應該會使得反轉現象更有機會發生
"""
params = [
    MacroParam(
        code='period_type',
        name='K線週期',
        desc='希望以哪種 K 線週期來偵測吊人線',
        dtype=PeriodType,
        default=PeriodType.type.DAY)
]
db_ver = '2022081001'
py_ver = '2022081001'


def func(market_id:str, **kwargs) -> pd.Series:
    """計算並取得指定市場 ID 中的歷史資料, 每個日期是否有發生吊人線的序列

    判斷規則:
    1. 型態發生前會有上升區段，或處於市場頂部
    2. 開盤價在當天價格高點
    3. 當日形成一個短實體黑 K 棒
    4. 下影線很長，至少要有實體長度的兩倍以上
    5. 沒有上影線

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
        市場各歷史時間點是否有發生吊人線序列

    """
    try:
        period_type = kwargs['period_type'].data
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'wj001'")
    candle = TI.Candle(market_id, period_type)
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    # 1. 型態發生前會出現上升區段，或處於市場頂部
    ma_5 = TI.MA(market_id, 5, period_type)
    ma_10 = TI.MA(market_id, 10, period_type)
    ma_diff = (ma_5 - ma_10).rolling(period, period_type).min()
    cond_1 = (ma_diff > 0)
    # 2. 開盤價位於當天價格的高點
    # avg_ampt = (candle.amplitude).rolling(period, period_type).mean()
    oh_diff_ratio = (candle.high - candle.open)/candle.amplitude
    cond_2 = oh_diff_ratio < 0.1
    # 3. 當日形成短實體黑 K 棒
    ba_ratio = candle.body/candle.amplitude
    cond_3 = (ba_ratio < 0.2) & (candle.close < candle.open)
    # 4. 下影線超過實體線長度 2 倍以上
    lsb_ratio = candle.lower_shadow/candle.body
    cond_4 = lsb_ratio >= 2
    # 5. 沒有上影線
    usa_ratio = candle.upper_shadow/candle.amplitude
    cond_5 = usa_ratio < 0.1
    cond = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
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
                           "'wj001'")

    results = {}
    try:
        period_type = kwargs['period_type'].data
    except ValueError as esp:
        results['period_type'] = f"invalid argument '{kwargs['period_type']}'"
    return results

def plot(**kwargs) -> List[PlotInfo]:
    """wj001 的範例圖製作函式

    判斷規則:
    1. 型態發生前會有上升區段，或處於市場頂部
    2. 開盤價在當天價格高點
    3. 當日形成一個短實體黑 K 棒
    4. 下影線很長，至少要有實體長度的兩倍以上
    5. 沒有上影線

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
                           "'wj001'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    rand_size = np.cumsum(np.abs(np.random.normal(5, 1, period+1)))
    rand_size[-1] = rand_size[-1]-10
    data = [
        MokeCandle.make(KType.DOJI_FOUR_PRICE) for _ in range(period)
    ]+[MokeCandle.make(KType.BLACK_HAMMER)]
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
                           "'wj001'")
    period_type_to_period = {
        TimeUnit.DAY: 10,
        TimeUnit.WEEK: 3,
        TimeUnit.MONTH: 3
    }
    period = period_type_to_period[period_type]
    return period


wj001 = Macro(code=code, name=name, desc=description, params=params,
        run=func, check=check, plot=plot, frame=frame, 
        db_ver=db_ver, py_ver=py_ver)
