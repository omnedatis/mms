import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable
from func._tp._ma import _stone as tp
from func._tp._klp._ma._stone.common import get_ma, MacroParam, ParamType, gen_cps4arranged_mas
from func.common import Macro, MacroParam, ParamType, PlotInfo, Ptype

code = 'stone_pp002'
name = '商智MA指標-PP002'
func = tp.stone_pp002
description = func.__doc__
params = [
    MacroParam('base_period_1', '基底均線天期(t1)', '基底均線天期(t1)', 
               ParamType.INT, 1),
    MacroParam('base_period_2', '基底均線天期(t2)', '基底均線天期(t2)', 
               ParamType.INT, 5),
    MacroParam('base_period_3', '基底均線天期(t3)', '基底均線天期(t3)', 
               ParamType.INT, 10),
    MacroParam('base_period_4', '基底均線天期(t4)', '基底均線天期(t4)', 
               ParamType.INT, 20),
    MacroParam('base_period_5', '基底均線天期(t5)', '基底均線天期(t5)', 
               ParamType.INT, 60),
    MacroParam('base_period_6', '基底均線天期(t6)', '基底均線天期(t6)', 
               ParamType.INT, 120),
    MacroParam('base_period_7', '基底均線天期(t7)', '基底均線天期(t7)', 
               ParamType.INT, 250),
    MacroParam(code='target_period', name='目標均線天期',
               desc='用於比較的 MA 計算時所使用的天期',
               dtype=ParamType.INT, default=40),
    MacroParam(code='statistical_duration', name='判斷過去幾天內的天數(n)',
               desc='判斷過去幾天內連續發生的天數(n)', 
               dtype=ParamType.INT, default=10)
]
db_ver = '20220810-v1'
py_ver = '20220810-v1'

def check(**kwargs) -> Dict[str, str]:
    """參數正確性檢查式, 用於判斷參數是否正常, 檢查項目為
    1. 所有 MA 天期皆必須為正整數
    2. 發生天數必須為正整數
    3. 所有 MA 天期不可相等

    Parameters
    ----------
    target_period : int
        目標均線天期(t).
    base_period : int
        基準均線天期(b).
    statistical_duration : int
        判斷過去幾天內的天數(n).
    min_occurence : int
        判斷發生事件是否大於或等於幾次的次數(o).

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
        base_periods = [kwargs[f'base_period_{idx}'] for idx in range(1, 8)]
        statistical_duration = kwargs['statistical_duration']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp001'")
    results = {}
    # 每個值都必須為正整數
    if not _is_positive_integer(target_period):
        add_message(results, 'target_period', "目標均線天期必須要為正整數; ")
    for i, base_period in enumerate(base_periods):
        if not _is_positive_integer(base_period):
            add_message(results, f'base_period_{i+1}', f"基準均線天期(t{i+1})必須要為正整數; ")
    if not _is_positive_integer(statistical_duration):
        add_message(results, 'statistical_duration', "判斷天數必須要為正整數; ")
    if len(set(base_periods)) != len(base_periods):
        for base_period in set(base_periods):
            idxs = np.argwhere(np.array(base_periods) == base_period)
            if len(idxs) > 0:
                for idx in idxs:
                    add_message(results, f'base_period_{idx[0]+1}', f"基礎均線天期發生重複, 請重新設定;")
    if len(set([target_period]+base_periods)) != (len(base_periods) + 1):
        add_message(results, 'target_period', "目標均線天期與基礎均線天期重複, 請重新設定; ")
        idxs = np.argwhere(np.array(base_periods) == target_period)
        if len(idxs) > 0:
            for idx in idxs:
                add_message(results, f'base_period_{idx[0]+1}', f"基礎均線天期發生重複, 請重新設定; ")
    # 均線天期需要小於 2520 
    if target_period > 2520:
        add_message(results, 'target_period', "目標均線天期必須少於 2520; ")
    for i, base_period in enumerate(base_periods):
        if base_period > 2520:
            add_message(results, f'base_period_{i+1}', f"基準均線天期(t{i+1})必須少於 2520; ")
    # 判斷天期需要少於 252
    if statistical_duration > 252:
        add_message(results, 'statistical_duration', "判斷天數必須要少於 252 日; ")
    return results


def plot(**kwargs) -> List[PlotInfo]:
    """pp002 的畫圖規則

    規則：
        過去 n 天內的MA (t) 都大於各個基準比較對象 MA (b1), MA(b2), …, MA(b7).

    Parameters
    ----------
    target_period : int
        目標均線天期(t).
    base_period_1 : int
        基準均線天期(b1).
    base_period_2 : int
        基準均線天期(b2).
    base_period_3 : int
        基準均線天期(b3).
    base_period_4 : int
        基準均線天期(b4).
    base_period_5 : int
        基準均線天期(b5).
    base_period_6 : int
        基準均線天期(b6).
    base_period_7 : int
        基準均線天期(b7).
    statistical_duration : int
        判斷過去幾天內的天數(n).

    Returns
    -------
    result: str | np.ndarray
        需調整說明或示意圖資訊
    """
    
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

    try:
        target_period = kwargs['target_period']
        base_periods = [kwargs[f'base_period_{idx}'] for idx in range(1, 8)]
        statistical_duration = kwargs['statistical_duration']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp001 plot'")

    plot_size = int(max([max([target_period, base_periods[-1]]) * 2, statistical_duration]))

    periods = base_periods + [target_period]
    periods.sort()
    trough_index = periods.index(target_period)
    cps_tails = gen_cps4arranged_mas(periods, trough_index, statistical_duration, sign=1)
    cps_heads = np.random.normal(0, 1, plot_size + max(periods) - 1 - len(cps_tails))
    for i in range(len(cps_heads)):
        if i == 0:
            cps_heads[len(cps_heads) - 1 - i] = cps_tails[0] - cps_heads[len(cps_heads) - 1 - i]
            continue
        cps_heads[len(cps_heads) - 1 - i] = (cps_heads[len(cps_heads) - 1 - i + 1] - cps_heads[len(cps_heads) - 1 - i])
    cps = np.concatenate([cps_heads, cps_tails], axis=0)
    result = {f"MA {period}": get_ma_series(
            cps, period)[-plot_size:] for period in periods}
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
        基準均線天期(b).
    statistical_duration : int
        判斷過去幾天內的天數(n).
    min_occurence : int
        判斷發生事件是否大於或等於幾次的次數(o).

    Returns
    -------
    result: int
        現象發生時的範圍大小
    """
    result = kwargs['statistical_duration']
    return result

stone_pp002 = Macro(code=code, name=name, desc=description, params=params,
                    run=func, check=check, plot=plot, frame=frame,
                    db_ver=db_ver, py_ver=py_ver)
