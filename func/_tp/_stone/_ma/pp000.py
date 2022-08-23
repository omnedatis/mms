import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable
from func._tp._ma import _stone as tp
from func.common import Macro, MacroParam, ParamType, PlotInfo, Ptype

code = 'stone_pp000'
name = 'MA指標-PP000'
func = tp.stone_pp000
description = func.__doc__
params = [
    MacroParam(code='target_period', name='目標均線天期',
               desc='用於比較的 MA 計算時所使用的天期',
               dtype=ParamType.INT, default=5),
    MacroParam(code='base_period', name='基準均線天期',
               desc='用於當作比較基準的 MA 計算時所使用的天期',
               dtype=ParamType.INT, default=10)
]
db_ver = '2022082301'
py_ver = '2022081001'

def check(**kwargs) -> Dict[str, str]:
    """參數正確性檢查式, 用於判斷參數是否正常, 檢查項目為
    1. 兩個 MA 天期皆必須為正整數
    2. 兩個 MA 天期不可相等

    Parameters
    ----------
    target_period : int
        目標均線天期(t).
    base_period : int
        基準均線天期(o).

    Returns
    -------
    results: Dict[str, str]
        錯誤訊息字典, 內容為參數與其對應的錯誤原因

    """
    def add_message(msgs: Dict[str, str], param_name: str, msg: str):
        if param_name in msgs:
            msgs[param_name] += msg
        msgs[param_name] = msg

    def _is_positive_integer(value: int) -> bool:
        if isinstance(value, int) and value > 0:
            return True
        return False
    try:
        target_period = kwargs['target_period']
        base_period = kwargs['base_period']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp000'")
    results = {}
    # 每個值都必須為正整數
    if not _is_positive_integer(target_period):
        add_message(results, 'target_period', "目標均線天期必須要為正整數; ")
    if not _is_positive_integer(base_period):
        add_message(results, 'base_period', "基準均線天期必須要為正整數; ")
    if target_period == base_period:
        add_message(results, 'target_period', "目標天期與基底天期相等, 請修正為不同數值; ")
        add_message(results, 'base_period', "目標天期與基底天期相等, 請修正為不同數值; ")
    # 均線天期需要小於 2520 
    if target_period > 2520:
        add_message(results, 'target_period', "目標均線天期必須少於 2520")
    if base_period > 2520:
        add_message(results, 'base_period', "基準均線天期必須少於 2520")
    return results


def plot(**kwargs) -> List[PlotInfo]:
    """pp000 的範例圖製作函式

    pp000 規則：
        MA (t) 大於 MA (b).

    Parameters
    ----------
    target_period : int
        目標均線天期(t).
    base_period : int
        基準均線天期(o).

    Returns
    -------
    results: Dict[str, List[float]]
        畫圖所使用的多條序列與序列名稱
    """
    def get_cp_series(size: int, fix_func: Optional[Callable[..., np.ndarray]], **kwargs) -> np.ndarray:
        """取得指定修正式下的收盤價

        Parameters
        ----------
        size: int
            序列長度
        fix_func: FunctionType
            收盤價修正式

        Returns
        -------
        cps: np.ndarray
            收盤價序列

        """
        cps = np.random.normal(0, 1, size).cumsum(axis=0)
        # 移除負值
        cps = cps - np.min(cps) + 1
        if fix_func is not None:
            cps = fix_func(cps, **kwargs)
        return cps

    def get_ma_series(cps: np.ndarray, period: int) -> np.ndarray:
        """ 根據指定的天期, 產生指定的 ma 序列

        Parameters
        ----------
        cps: np.ndarray
            用於計算 MA 的收盤價序列
        period: int
            MA 天期

        Returns
        -------
        result: np.ndarray:
            MA 序列
        """
        if len(cps) < period:
            raise Exception('計算 MA 用的序列長度不足')
        new_shape = (period, cps.shape[0] - period + 1)
        new_strides = (*cps.strides, *cps.strides)
        return np.mean(
            np.lib.stride_tricks.as_strided(
                cps, shape=new_shape, strides=new_strides).copy(), axis=0)

    def get_ma_lines(size: int, periods: List[int], fix_func: Optional[Callable[..., np.ndarray]], **kwargs) -> Dict[str, np.ndarray]:
        """ 取得指定規則下的多組 MA 序列

        Parameters
        ----------
        size: int 
            每個序列的長度
        periods: List[int]
            每組 MA 的計算天期
        fix_func: Callable[..., np.ndarray]
            收盤價的修正式, 用於調整收盤價使最終結果線圖符合預期

        Returns
        -------
        results: Dict[str, np.ndarray]
            多組 MA 序列
        """
        cps = get_cp_series(size + max(periods) - 1, fix_func, **kwargs)
        if isinstance(cps, str):
            return cps
        mas = {f"MA {period}": get_ma_series(
            cps, period)[-size:] for period in periods}
        return mas

    def fix_func(cps: np.ndarray, **kwargs) -> np.ndarray:
        """
        (x+y)/(n+k) > x/n
        """
        cps = cps.copy()
        target_period = kwargs['target_period']
        base_period = kwargs['base_period']
        k = abs(base_period-target_period)
        diff = 0
        if target_period < base_period:
            # 短天期大於長天期
            y = cps[-base_period:-target_period].sum(axis=0)
            x = cps[-target_period:].sum(axis=0)
            limit = (x * k / target_period)
            if y >= limit:
                diff = y - limit + 1
            # 將多餘的部分從區段中移除
            cps[-base_period:-target_period] -= diff/k
        elif target_period > base_period:
            # 長天期大於短天期
            y = cps[-target_period:-base_period].sum(axis=0)
            x = cps[-base_period:].sum(axis=0)
            limit = (x * k / base_period)
            if y <= limit:
                diff = limit - y + 1
            # 將多餘的部分從區段中補上
            cps[-target_period:-base_period] += diff/k
        return cps
    try:
        target_period = kwargs['target_period']
        base_period = kwargs['base_period']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp000 plot'")

    plot_size = int(max([target_period, base_period]) * 2)

    result = get_ma_lines(
        plot_size, periods=[target_period, base_period], fix_func=fix_func, **kwargs)
    result = [
        PlotInfo(ptype=Ptype.MA, title=key, data=value) 
        for key, value in result.items()]
    return result


def frame(**kwargs) -> int:
    """取得繪製現象發生焦點的範圍大小

    Parameters
    ----------
    target_period : int
        目標均線天期(t).
    base_period : int
        基準均線天期(o).

    Returns
    -------
    results: int
        現象發生時的範圍大小
    """
    results = 1
    return results


stone_pp000 = Macro(code=code, name=name, desc=description, params=params,
        run=func, check=check, plot=plot, frame=frame,
        db_ver=db_ver, py_ver=py_ver)
