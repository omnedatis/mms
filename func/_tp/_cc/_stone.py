# -*- coding: utf-8 -*-

from typing import Union

from .._context import TechnicalIndicator as TI
from .._context import (_CandleStick, BooleanTimeSeries,
                        NumericTimeSeries, MarketData, ts_min, ts_max)

class StoneCandleStick(_CandleStick):
    """Stone's CandleStick."""

    @classmethod
    def make(cls, candle: _CandleStick):
        """Generate instance from another."""
        data = MarketData(candle.close.log() - candle.prev_close.log(),
                          candle.low.log() - candle.prev_close.log(),
                          candle.high.log() - candle.prev_close.log(),
                          candle.open.log() - candle.prev_close.log())
        return cls(data, name=candle._name, tunit=candle._tunit)

    @property
    def positive_amplitude(self):
        recv = self.high - self.open
        recv.rename(f'{self._name}.PositiveAmplitude')
        return recv

    @property
    def negative_amplitude(self):
        recv = self.open - self.low
        recv.rename(f'{self._name}.NegativeAmplitude')
        return recv

    def is_ignored_body(self, body_ignore_ratio: float):
        """ignored body."""
        recv = self.body < body_ignore_ratio
        recv.rename(f'{self._name}.IsIgnoredBody')
        return recv

    def is_long_body(self, long_body_ratio: float):
        """long body."""
        recv = self.body > self.amplitude * long_body_ratio
        recv.rename(f'{self._name}.IsLongBody')
        return recv

    def is_middle_body(self, long_body_ratio: float, middle_body_ratio: float):
        """middle body."""
        cond_1 = self.body < self.amplitude * long_body_ratio
        cond_2 = self.body > self.amplitude * middle_body_ratio
        recv = cond_1 & cond_2
        recv.rename(f'{self._name}.IsMiddleBody')
        return recv

    def is_strong_positive(self, strong_positive_body_ratio: float):
        """strong positive."""
        recv = self.close - self.open > strong_positive_body_ratio
        recv.rename(f'{self._name}.IsStrongPositive')
        return recv

    def is_middle_positive(self, strong_positive_body_ratio: float,
                           middle_positive_body_ratio: float):
        """middle positive."""
        temp = self.close - self.open
        cond_1 = temp < strong_positive_body_ratio
        cond_2 = temp > middle_positive_body_ratio
        recv = cond_1 & cond_2
        recv.rename(f'{self._name}.IsMiddlePositive')
        return recv

    def is_strong_negative(self, strong_negative_body_ratio: float):
        """strong negative."""
        recv = self.open - self.close > strong_negative_body_ratio
        recv.rename(f'{self._name}.IsStrongNegative')
        return recv

    def is_middle_negative(self, strong_negative_body_ratio: float,
                           middle_negative_body_ratio: float):
        """middle negative."""
        temp = self.open - self.close
        cond_1 = temp < strong_negative_body_ratio
        cond_2 = temp > middle_negative_body_ratio
        recv = cond_1 & cond_2
        recv.rename(f'{self._name}.IsMiddleNegative')
        return recv

    def is_ignored_amplitude(self, amplitude_ignore_ratio: float):
        """ignored amplitude."""
        recv = self.amplitude < amplitude_ignore_ratio
        recv.rename(f'{self._name}.IsIgnoredAmplitude')
        return recv

    def is_ignored_lowershadow(self, shadow_ignore_ratio: float):
        """ignored lower-shadow."""
        recv = self.lower_shadow < shadow_ignore_ratio
        recv.rename(f'{self._name}.IsIgnoredLowerShadow')
        return recv

    def is_ignored_uppershadow(self, shadow_ignore_ratio: float):
        """ignored upper-shadow."""
        recv = self.upper_shadow < shadow_ignore_ratio
        recv.rename(f'{self._name}.IsIgnoredUpperShadow')
        return recv

    def is_long_lowershadow(self, long_shadow_ratio: float):
        """long lower-shadow."""
        recv = self.lower_shadow > self.upper_shadow * long_shadow_ratio
        recv.rename(f'{self._name}.IsLongLowerShadow')
        return recv

    def is_long_uppershadow(self, long_shadow_ratio: float):
        """long upper-shadow."""
        recv = self.upper_shadow > self.lower_shadow * long_shadow_ratio
        recv.rename(f'{self._name}.IsLongUpperShadow')
        return recv

    def is_doji(self, body_ignore_ratio: float,
                shadow_ignore_ratio: float) -> BooleanTimeSeries:
        """is doji."""
        cond_1 = self.is_ignored_body(body_ignore_ratio=body_ignore_ratio)
        cond_2 = ~self.is_ignored_lowershadow(shadow_ignore_ratio=shadow_ignore_ratio)
        cond_3 = ~self.is_ignored_uppershadow(shadow_ignore_ratio=shadow_ignore_ratio)
        recv = cond_1 & cond_2 & cond_3
        recv.rename(f'{self._name}.IsDoji')
        return recv

def get_candle(market_id: str, candle_period: int) -> StoneCandleStick:
    if candle_period == 1:
        name = f'{market_id}.CandleStick'
    else:
        name = f'{market_id}.{candle_period}CandleStick'
    ret = StoneCandleStick.make(TI.Candle(market_id, candle_period))
    return ret

def stone_kp001(market_id: str, **kwargs):
    """KP001 : 十字線 且 長下影.

    規則：
         1. 十字線
         2. 長下影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    shadow_ignore_ratio : float
        忽略影線臨界值.
    long_shadow_ratio: float
        長下影線臨界值
    body_ignore_ratio : float
        忽略實體臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        shadow_ignore_ratio = kwargs['shadow_ignore_ratio']
        long_shadow_ratio = kwargs['long_shadow_ratio']
        body_ignore_ratio = kwargs['body_ignore_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp001'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp001': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.is_doji(body_ignore_ratio=body_ignore_ratio,
                         shadow_ignore_ratio=shadow_ignore_ratio)
    cond_2 = cct.is_long_lowershadow(long_shadow_ratio=long_shadow_ratio)
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp001({kwargs})')
    return ret

def stone_kp002(market_id: str, **kwargs):
    """KP002 : 十字線 且 長上影.

    規則：
         1. 十字線
         2. 長上影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    shadow_ignore_ratio : float
        忽略影線臨界值.
    long_shadow_ratio: float
        長上影線臨界值
    body_ignore_ratio : float
        忽略實體臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        shadow_ignore_ratio = kwargs['shadow_ignore_ratio']
        long_shadow_ratio = kwargs['long_shadow_ratio']
        body_ignore_ratio = kwargs['body_ignore_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp002'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp002': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.is_doji(body_ignore_ratio=body_ignore_ratio,
                         shadow_ignore_ratio=shadow_ignore_ratio)
    cond_2 = cct.is_long_uppershadow(long_shadow_ratio=long_shadow_ratio)
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp002({kwargs})')
    return ret

def stone_kp003(market_id: str, **kwargs):
    """KP003 : 水平線.

    規則：
         1. 水平線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    amplitude_ignore_ratio : float
        忽略振幅臨界值.
    body_ignore_ratio : float
        忽略實體臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        amplitude_ignore_ratio = kwargs['amplitude_ignore_ratio']
        body_ignore_ratio = kwargs['body_ignore_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp003'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp003': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.is_ignored_body(body_ignore_ratio=body_ignore_ratio)
    cond_2 = cct.is_ignored_amplitude(amplitude_ignore_ratio=amplitude_ignore_ratio)
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp003({kwargs})')
    return ret

def stone_kp004(market_id: str, **kwargs):
    """KP004 : 大陽線.

    規則：
         1. 大陽線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    long_body_ratio : float
        長實體臨界值.
    strong_positive_body_ratio : float
        劇烈漲幅臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        long_body_ratio = kwargs['long_body_ratio']
        strong_positive_body_ratio = kwargs['strong_positive_body_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp004'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp004': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.is_strong_positive(strong_positive_body_ratio=strong_positive_body_ratio)
    cond_2 = cct.is_long_body(long_body_ratio=long_body_ratio)
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp004({kwargs})')
    return ret

def stone_kp005(market_id: str, **kwargs):
    """KP005 : 大陰線.

    規則：
         1. 大陰線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    long_body_ratio : float
        長實體臨界值.
    strong_negative_body_ratio : float
        劇烈跌幅臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        long_body_ratio = kwargs['long_body_ratio']
        strong_negative_body_ratio = kwargs['strong_negative_body_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp005'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp005': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.is_strong_negative(strong_negative_body_ratio=strong_negative_body_ratio)
    cond_2 = cct.is_long_body(long_body_ratio=long_body_ratio)
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp005({kwargs})')
    return ret

def stone_kp006(market_id: str, **kwargs):
    """KP006 : 小陽線.

    規則：
         1. 小陽線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    long_body_ratio : float
        長實體臨界值.
    middle_body_ratio : float
        中等長度實體臨界值.
    strong_positive_body_ratio : float
        劇烈漲幅臨界值.
    middle_positive_body_ratio : float
        正向實體臨界值.
    """
    try:
        candle_period = kwargs['candle_period']
        long_body_ratio = kwargs['long_body_ratio']
        middle_body_ratio = kwargs['middle_body_ratio']
        strong_positive_body_ratio = kwargs['strong_positive_body_ratio']
        middle_positive_body_ratio = kwargs['middle_positive_body_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp006'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp006': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.is_middle_positive(strong_positive_body_ratio=strong_positive_body_ratio,
                                    middle_positive_body_ratio=middle_positive_body_ratio)
    cond_2 = cct.is_middle_body(long_body_ratio=long_body_ratio,
                                middle_body_ratio=middle_body_ratio)
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp006({kwargs})')
    return ret

def stone_kp007(market_id: str, **kwargs):
    """KP007 : 小陰線.

    規則：
         1. 小陰線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    long_body_ratio : float
        長實體臨界值.
    middle_body_ratio : float
        中等長度實體臨界值.
    strong_negative_body_ratio : float
        劇烈跌幅臨界值.
    middle_negative_body_ratio : float
        負向實體臨界值.
    """
    try:
        candle_period = kwargs['candle_period']
        long_body_ratio = kwargs['long_body_ratio']
        middle_body_ratio = kwargs['middle_body_ratio']
        strong_negative_body_ratio = kwargs['strong_negative_body_ratio']
        middle_negative_body_ratio = kwargs['middle_negative_body_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp007'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp007': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.is_middle_negative(strong_negative_body_ratio=strong_negative_body_ratio,
                                    middle_negative_body_ratio=middle_negative_body_ratio)
    cond_2 = cct.is_middle_body(long_body_ratio=long_body_ratio,
                                middle_body_ratio=middle_body_ratio)
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp007({kwargs})')
    return ret

def stone_kp011(market_id: str, **kwargs):
    """KP011.

    規則：
         在最近n日內發生o次十字線且長下影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    shadow_ignore_ratio : float
        忽略影線臨界值.
    long_shadow_ratio: float
        長下影線臨界值
    body_ignore_ratio : float
        忽略實體臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        shadow_ignore_ratio = kwargs['shadow_ignore_ratio']
        long_shadow_ratio = kwargs['long_shadow_ratio']
        body_ignore_ratio = kwargs['body_ignore_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp011'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp011': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1a = cct.is_doji(body_ignore_ratio=body_ignore_ratio,
                         shadow_ignore_ratio=shadow_ignore_ratio)
    cond_1b = cct.is_long_lowershadow(long_shadow_ratio=long_shadow_ratio)
    cond_1 = cond_1a & cond_1b
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp011({kwargs})')
    return ret

def stone_kp012(market_id: str, **kwargs):
    """KP012.

    規則：
         在最近n日內發生o次十字線且長上影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    shadow_ignore_ratio : float
        忽略影線臨界值.
    long_shadow_ratio: float
        長上影線臨界值
    body_ignore_ratio : float
        忽略實體臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        shadow_ignore_ratio = kwargs['shadow_ignore_ratio']
        long_shadow_ratio = kwargs['long_shadow_ratio']
        body_ignore_ratio = kwargs['body_ignore_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp012'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp012': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1a = cct.is_doji(body_ignore_ratio=body_ignore_ratio,
                         shadow_ignore_ratio=shadow_ignore_ratio)
    cond_1b = cct.is_long_uppershadow(long_shadow_ratio=long_shadow_ratio)
    cond_1 = cond_1a & cond_1b
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp012({kwargs})')
    return ret

def stone_kp013(market_id: str, **kwargs):
    """KP013.

    規則：
         最近 n 日內有發生水平線且天數大於或等於 o 次

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    amplitude_ignore_ratio : float
        忽略振幅臨界值.
    body_ignore_ratio : float
        忽略實體臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        amplitude_ignore_ratio = kwargs['amplitude_ignore_ratio']
        body_ignore_ratio = kwargs['body_ignore_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp013'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp013': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1a = cct.is_ignored_body(body_ignore_ratio=body_ignore_ratio)
    cond_1b = cct.is_ignored_amplitude(amplitude_ignore_ratio=amplitude_ignore_ratio)
    cond_1 = cond_1a & cond_1b
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp013({kwargs})')
    return ret

def stone_kp014(market_id: str, **kwargs):
    """KP014.

    規則：
         1. 在最近n日內發生o次十字線且長下影線
         2. 在最近n日內發生o次水平線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    shadow_ignore_ratio : float
        忽略影線臨界值.
    long_shadow_ratio: float
        長下影線臨界值
    amplitude_ignore_ratio : float
        忽略振幅臨界值.
    body_ignore_ratio : float
        忽略實體臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        shadow_ignore_ratio = kwargs['shadow_ignore_ratio']
        long_shadow_ratio = kwargs['long_shadow_ratio']
        amplitude_ignore_ratio = kwargs['amplitude_ignore_ratio']
        body_ignore_ratio = kwargs['body_ignore_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp014'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp014': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1a = cct.is_doji(body_ignore_ratio=body_ignore_ratio,
                         shadow_ignore_ratio=shadow_ignore_ratio)
    cond_1b = cct.is_long_lowershadow(long_shadow_ratio=long_shadow_ratio)
    cond_1 = cond_1a & cond_1b
    cond_1 = cond_1.sampling(n_days, candle_period).sum() >= o_times
    cond_2a = cct.is_ignored_body(body_ignore_ratio=body_ignore_ratio)
    cond_2b = cct.is_ignored_amplitude(amplitude_ignore_ratio=amplitude_ignore_ratio)
    cond_2 = cond_2a & cond_2b
    cond_2 = cond_2.sampling(n_days, candle_period).sum() >= o_times
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp014({kwargs})')
    return ret

def stone_kp015(market_id: str, **kwargs):
    """KP015.

    規則：
         1. 在最近n日內發生o次十字線且長上影線
         2. 在最近n日內發生o次水平線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    shadow_ignore_ratio : float
        忽略影線臨界值.
    long_shadow_ratio: float
        長上影線臨界值
    amplitude_ignore_ratio : float
        忽略振幅臨界值.
    body_ignore_ratio : float
        忽略實體臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        shadow_ignore_ratio = kwargs['shadow_ignore_ratio']
        long_shadow_ratio = kwargs['long_shadow_ratio']
        amplitude_ignore_ratio = kwargs['amplitude_ignore_ratio']
        body_ignore_ratio = kwargs['body_ignore_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp015'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp015': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1a = cct.is_doji(body_ignore_ratio=body_ignore_ratio,
                         shadow_ignore_ratio=shadow_ignore_ratio)
    cond_1b = cct.is_long_uppershadow(long_shadow_ratio=long_shadow_ratio)
    cond_1 = cond_1a & cond_1b
    cond_1 = cond_1.sampling(n_days, candle_period).sum() >= o_times
    cond_2a = cct.is_ignored_body(body_ignore_ratio=body_ignore_ratio)
    cond_2b = cct.is_ignored_amplitude(amplitude_ignore_ratio=amplitude_ignore_ratio)
    cond_2 = cond_2a & cond_2b
    cond_2 = cond_2.sampling(n_days, candle_period).sum() >= o_times
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp015({kwargs})')
    return ret

def stone_kp016(market_id: str, **kwargs):
    """KP016.

    規則：
         1. 在最近n日內發生o次大陽線
         2. 在最近n日內發生o次水平線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    long_body_ratio : float
        長實體臨界值.
    strong_positive_body_ratio : float
        劇烈漲幅臨界值.
    amplitude_ignore_ratio : float
        忽略振幅臨界值.
    body_ignore_ratio : float
        忽略實體臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        long_body_ratio = kwargs['long_body_ratio']
        strong_positive_body_ratio = kwargs['strong_positive_body_ratio']
        amplitude_ignore_ratio = kwargs['amplitude_ignore_ratio']
        body_ignore_ratio = kwargs['body_ignore_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp016'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp016': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1a = cct.is_strong_positive(strong_positive_body_ratio=strong_positive_body_ratio)
    cond_1b = cct.is_long_body(long_body_ratio=long_body_ratio)
    cond_1 = cond_1a & cond_1b
    cond_1 = cond_1.sampling(n_days, candle_period).sum() >= o_times
    cond_2a = cct.is_ignored_body(body_ignore_ratio=body_ignore_ratio)
    cond_2b = cct.is_ignored_amplitude(amplitude_ignore_ratio=amplitude_ignore_ratio)
    cond_2 = cond_2a & cond_2b
    cond_2 = cond_2.sampling(n_days, candle_period).sum() >= o_times
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp016({kwargs})')
    return ret

def stone_kp017(market_id: str, **kwargs):
    """KP017.

    規則：
         1. 在最近n日內發生o次大陰線
         2. 在最近n日內發生o次水平線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    long_body_ratio : float
        長實體臨界值.
    strong_negative_body_ratio : float
        劇烈跌幅臨界值.
    amplitude_ignore_ratio : float
        忽略振幅臨界值.
    body_ignore_ratio : float
        忽略實體臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        long_body_ratio = kwargs['long_body_ratio']
        strong_negative_body_ratio = kwargs['strong_negative_body_ratio']
        amplitude_ignore_ratio = kwargs['amplitude_ignore_ratio']
        body_ignore_ratio = kwargs['body_ignore_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp017'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp017': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1a = cct.is_strong_negative(strong_negative_body_ratio=strong_negative_body_ratio)
    cond_1b = cct.is_long_body(long_body_ratio=long_body_ratio)
    cond_1 = cond_1a & cond_1b
    cond_1 = cond_1.sampling(n_days, candle_period).sum() >= o_times
    cond_2a = cct.is_ignored_body(body_ignore_ratio=body_ignore_ratio)
    cond_2b = cct.is_ignored_amplitude(amplitude_ignore_ratio=amplitude_ignore_ratio)
    cond_2 = cond_2a & cond_2b
    cond_2 = cond_2.sampling(n_days, candle_period).sum() >= o_times
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp017({kwargs})')
    return ret

def stone_kp018(market_id: str, **kwargs):
    """KP018.

    規則：
         1. 在最近n日內發生o次大陰線
         2. 當日發生大陽線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    long_body_ratio : float
        長實體臨界值.
    strong_negative_body_ratio : float
        劇烈跌幅臨界值.
    strong_positive_body_ratio : float
        劇烈漲幅臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        long_body_ratio = kwargs['long_body_ratio']
        strong_negative_body_ratio = kwargs['strong_negative_body_ratio']
        strong_positive_body_ratio = kwargs['strong_positive_body_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp018'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp018': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1a = cct.is_strong_negative(strong_negative_body_ratio=strong_negative_body_ratio)
    cond_b = cct.is_long_body(long_body_ratio=long_body_ratio)
    cond_1 = cond_1a & cond_b
    cond_1 = cond_1.sampling(n_days, candle_period).sum() >= o_times
    cond_2a = cct.is_strong_positive(strong_positive_body_ratio=strong_positive_body_ratio)
    cond_2 = cond_2a & cond_b
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.stone_kp018({kwargs})')
    return ret

def stone_kp019(market_id: str, **kwargs):
    """KP019.

    規則：
         1. 在最近n日內發生o次大陽線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    long_body_ratio : float
        長實體臨界值.
    strong_positive_body_ratio : float
        劇烈漲幅臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        long_body_ratio = kwargs['long_body_ratio']
        strong_positive_body_ratio = kwargs['strong_positive_body_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp019'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp019': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1a = cct.is_strong_positive(strong_positive_body_ratio=strong_positive_body_ratio)
    cond_1b = cct.is_long_body(long_body_ratio=long_body_ratio)
    cond_1 = cond_1a & cond_1b
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp019({kwargs})')
    return ret

def stone_kp020(market_id: str, **kwargs):
    """KP020.

    規則：
         1. 在最近n日內發生o次大陰線

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    long_body_ratio : float
        長實體臨界值.
    strong_negative_body_ratio : float
        劇烈跌幅臨界值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        long_body_ratio = kwargs['long_body_ratio']
        strong_negative_body_ratio = kwargs['strong_negative_body_ratio']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp020'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp020': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1a = cct.is_strong_negative(strong_negative_body_ratio=strong_negative_body_ratio)
    cond_1b = cct.is_long_body(long_body_ratio=long_body_ratio)
    cond_1 = cond_1a & cond_1b
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp020({kwargs})')
    return ret

def stone_kp030a(market_id: str, **kwargs):
    """KP030a.

    規則：
         1. 最近 n 日發生正振幅數值部分小於判斷值的天數大於 o 日

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    ratio_threshold : float
        振幅判斷值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        ratio_threshold = kwargs['ratio_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp030a'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp030a': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.positive_amplitude < ratio_threshold
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp030a({kwargs})')
    return ret

def stone_kp030b(market_id: str, **kwargs):
    """KP030b.

    規則：
         1. 最近 n 日發生正振幅數值部分大於判斷值的天數大於 o 日

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    ratio_threshold : float
        振幅判斷值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        ratio_threshold = kwargs['ratio_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp030b'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp030b': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.positive_amplitude > ratio_threshold
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp030b({kwargs})')
    return ret

def stone_kp040a(market_id: str, **kwargs):
    """KP040a.

    規則：
         1. 最近 n 日發生負振幅數值部分小於判斷值的天數大於 o 日

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    ratio_threshold : float
        振幅判斷值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        ratio_threshold = kwargs['ratio_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp040a'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp040a': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.negative_amplitude < ratio_threshold
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp040a({kwargs})')
    return ret

def stone_kp040b(market_id: str, **kwargs):
    """KP040b.

    規則：
         1. 最近 n 日發生負振幅數值部分大於判斷值的天數大於 o 日

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    ratio_threshold : float
        振幅判斷值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        ratio_threshold = kwargs['ratio_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp040b'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp040b': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.negative_amplitude > ratio_threshold
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp040b({kwargs})')
    return ret

def stone_kp050a(market_id: str, **kwargs):
    """KP050a.

    規則：
         1. 最近 n 日發生上影線數值部分小於判斷值的天數大於 o 日

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    ratio_threshold : float
        上影線判斷值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        ratio_threshold = kwargs['ratio_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp050a'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp050a': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.upper_shadow < ratio_threshold
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp050a({kwargs})')
    return ret

def stone_kp050b(market_id: str, **kwargs):
    """KP050b.

    規則：
         1. 最近 n 日發生上影線數值部分大於判斷值的天數大於 o 日

    Arguments
    ---------
    market_id : string
        目標市場ID
    candle_period : int
        K線週期
    n_days : int
        近期參考天數(n)
    o_times : int
        近期事件發生次數(o)
    ratio_threshold : float
        上影線判斷值.

    """
    try:
        candle_period = kwargs['candle_period']
        n_days = kwargs['n_days']
        o_times = kwargs['o_times']
        ratio_threshold = kwargs['ratio_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_kp050b'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'stone_kp050b': {esp}")
    cct = get_candle(market_id, candle_period)
    cond_1 = cct.upper_shadow > ratio_threshold
    ret = cond_1.sampling(n_days, candle_period).sum() >= o_times
    ret.rename(f'{market_id}.stone_kp050b({kwargs})')
    return ret

