# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, NamedTuple, Union

from .._context import TechnicalIndicator as TI
from .._context import (TimeUnit, _CandleStick, BooleanTimeSeries,
                        NumericTimeSeries, MarketData, ts_min, ts_max, MD_CACHE)

class ArgPrototype(NamedTuple):
    symbol: str
    name: str
    rule: str
    checker: Callable[[Any], bool]

def is_non_negative_number(value) -> str:
    return isinstance(value, (int, float)) and value >= 0

def doutze_ccp_check(f):
    def check(kwargs):
        ret = {}
        for key, value in kwargs.items():
            if not DOUTZE_CC_CHCEKERS[key].checker(value):
                ret[key] = DOUTZE_CC_CHCEKERS[key].rule
        return ret
    f.check = check
    return f

DOUTZE_CC_CHCEKERS = {}
DOUTZE_CC_CHCEKERS['shadow_ignore_threshold'
                   ] = ArgPrototype(symbol='shadow_ignore_threshold',
                                    name='忽略影線臨界值',
                                    rule='必須為非負實數',
                                    checker=is_non_negative_number)
DOUTZE_CC_CHCEKERS['violent_threshold'
                   ] = ArgPrototype(symbol='violent_threshold',
                                    name='劇烈漲跌臨界值(%)',
                                    rule='必須為非負實數',
                                    checker=is_non_negative_number)
DOUTZE_CC_CHCEKERS['close_zero_threshold'
                   ] = ArgPrototype(symbol='close_zero_threshold',
                                    name='持平臨界值(%)',
                                    rule='必須為非負實數',
                                    checker=is_non_negative_number)
DOUTZE_CC_CHCEKERS['long_shadow_threshold'
                   ] = ArgPrototype(symbol='long_shadow_threshold',
                                    name='長影線臨界值',
                                    rule='必須為非負實數',
                                    checker=is_non_negative_number)
DOUTZE_CC_CHCEKERS['body_close_threshold'
                   ] = ArgPrototype(symbol='body_close_threshold',
                                    name='兩實體線長度相近臨界值',
                                    rule='必須為非負實數',
                                    checker=is_non_negative_number)

def _is_close(tar: NumericTimeSeries, ref: NumericTimeSeries,
              close_zero_threshold: float) -> BooleanTimeSeries:
    #return abs(tar / ref - 1) < close_zero_threshold
    return abs(tar - ref) < ref * (close_zero_threshold / 100)

class DoutzeCandleStick(_CandleStick):
    """Doutze's CandleStick."""

    @classmethod
    def make(cls, candle: _CandleStick, name: str):
        """Generate instance from another."""
        return cls(candle._data, name=name, tunit=candle._tunit)

    @property
    def real_body(self) -> NumericTimeSeries:
        """Real Body."""
        recv = abs(self.close - self.open)
        recv.rename(f'{self._name}.RealBOdy')
        return recv

    @property
    def body_rate(self) -> NumericTimeSeries:
        """Body rate."""
        recv = abs(self.close / self.open - 1) * 100
        recv.rename(f'{self._name}.BodyRate')
        return recv

    @property
    def amplitude_rate(self) -> NumericTimeSeries:
        """Amplitude rate."""
        recv = (self.high / self.low - 1) * 100
        recv.rename(f'{self._name}.AmplitudeRate')
        return recv

    def ignored_amplitude(self, close_zero_threshold: float
                          ) -> BooleanTimeSeries:
        """Ignored Amplitude.

        Parameters
        ----------
        close_zero_threshold: float
            If the `amplitude_rate` is less than `close_zero_threshold`(%),
            it can be ignored.

        """
        recv = self.amplitude_rate < close_zero_threshold
        recv.rename(f'{self._name}.IgnoredAmplitude')
        return recv

    def long_body(self, violent_threshold: float) -> BooleanTimeSeries:
        """Long Body.

        Parameters
        ----------
        violent_threshold: float
            If the `body_rate` is larger than `violent_threshold`(%),
            it is long body.

        """
        recv = self.body_rate > violent_threshold
        recv.rename(f'{self._name}.LongBody')
        return recv

    def short_body(self, violent_threshold: float,
                   close_zero_threshold: float) -> BooleanTimeSeries:
        """Short Body.

        See Also
        --------
        ignored_body, long_body

        """
        recv = ~(self.long_body(violent_threshold) |
                 self.ignored_body(close_zero_threshold))
        recv.rename(f'{self._name}.ShortBody')
        return recv

    def ignored_body(self, close_zero_threshold: float) -> BooleanTimeSeries:
        """Ignored Body.

        Parameters
        ----------
        close_zero_threshold: float
            If the `body_rate` is less than `close_zero_threshold`(%),
            it can be ignored.

        """
        recv = self.body_rate < close_zero_threshold
        recv.rename(f'{self._name}.IgnoredBody')
        return recv

    @property
    def lower_shadow_rate(self) -> NumericTimeSeries:
        """Lower Shadow rate."""
        return self.lower_shadow / self.body

    @property
    def upper_shadow_rate(self) -> NumericTimeSeries:
        """Upper Shadow rate."""
        return self.upper_shadow / self.body

    def long_lower_shadow(self, long_shadow_threshold: float
                          ) -> BooleanTimeSeries:
        """Long Lower-Shadow.

        Parameters
        ----------
        long_shadow_threshold: float
            If the `lower_shadow_rate` is larger than `long_shadow_threshold`,
            it is long lower-shadow.

        """
        #recv = self.lower_shadow_rate > long_shadow_threshold
        recv = self.lower_shadow > self.body * long_shadow_threshold
        recv.rename(f'{self._name}.LongLowerShadow')
        return recv

    def short_lower_shadow(self, long_shadow_threshold: float,
                           shadow_ignore_threshold: float) -> BooleanTimeSeries:
        """Short Lower-Shadow.

        See Also
        --------
        ignored_lower_shadow, long_lower_shadow

        """
        recv = ~(self.long_lower_shadow(long_shadow_threshold) |
                 self.ignored_lower_shadow(shadow_ignore_threshold))
        recv.rename(f'{self._name}.ShortLowerShadow')
        return recv

    def ignored_lower_shadow(self, shadow_ignore_threshold: float
                             ) -> BooleanTimeSeries:
        """Ignored Lower-Shadow.

        Parameters
        ----------
        shadow_ignore_threshold: float
            If the `lower_shadow_rate` is less than `shadow_ignore_threshold`,
            it can be ignored.

        """
        recv = self.lower_shadow <= self.body * shadow_ignore_threshold
        recv.rename(f'{self._name}.IgnoredLowerShadow')
        return recv

    def long_upper_shadow(self, long_shadow_threshold: float
                          ) -> BooleanTimeSeries:
        """Long Upper-Shadow.

        Parameters
        ----------
        long_shadow_threshold: float
            If the `upper_shadow_rate` is larger than `long_shadow_threshold`,
            it is long upper-shadow.

        """
        #recv = self.upper_shadow_rate > long_shadow_threshold
        recv = self.upper_shadow > self.body * long_shadow_threshold
        recv.rename(f'{self._name}.LongUpperShadow')
        return recv

    def short_upper_shadow(self, long_shadow_threshold: float,
                           shadow_ignore_threshold: float) -> BooleanTimeSeries:
        """Short Upper-Shadow.

        See Also
        --------
        ignored_upper_shadow, long_upper_shadow

        """
        recv = ~(self.long_upper_shadow(long_shadow_threshold) |
                 self.ignored_upper_shadow(shadow_ignore_threshold))
        recv.rename(f'{self._name}.ShortUpperShadow')
        return recv

    def ignored_upper_shadow(self, shadow_ignore_threshold: float
                             ) -> BooleanTimeSeries:
        """Ignored Upper-Shadow.

        Parameters
        ----------
        shadow_ignore_threshold: float
            If the `upper_shadow_rate` is less than `shadow_ignore_threshold`,
            it can be ignored.

        """
        recv = self.upper_shadow <= self.body * shadow_ignore_threshold
        recv.rename(f'{self._name}.IgnoredUpperShadow')
        return recv

    def is_white(self) -> BooleanTimeSeries:
        """White Real-Body"""
        recv = self.close > self.open
        recv.rename(f'{self._name}.IsWhite')
        return recv

    def is_black(self) -> BooleanTimeSeries:
        recv = self.close < self.open
        recv.rename(f'{self._name}.IsBlack')
        return recv

    def white_body(self, close_zero_threshold: float) -> BooleanTimeSeries:
        """White Real-Body.

        Parameters
        ----------
        close_zero_threshold: float
            If the `body_rate` is less than `close_zero_threshold`(%),
            it can be ignored.

        """
        recv = (self.is_white() &
                ~self.ignored_body(close_zero_threshold=close_zero_threshold))
        recv.rename(f'{self._name}.WhiteBody')
        return recv

    def black_body(self, close_zero_threshold: float) -> BooleanTimeSeries:
        """Black Real-Body.

        Parameters
        ----------
        close_zero_threshold: float
            If the `body_rate` is less than `close_zero_threshold`(%),
            it can be ignored.

        """
        recv = (self.is_black() &
                ~self.ignored_body(close_zero_threshold=close_zero_threshold))
        recv.rename(f'{self._name}.BlackBody')
        return recv

    def doji(self, close_zero_threshold: float,
             shadow_ignore_threshold: float) -> BooleanTimeSeries:
        """Doji.

        ignorable real-body, unignorable lower-shadow and unignorable upper-shadow.

        See Also
        --------
        ignored_body, ignored_upper_shadow, ignored_lower_shadow.

        """
        recv = (self.ignored_body(close_zero_threshold=close_zero_threshold) &
                ~self.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)&
                ~self.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold))
        recv.rename(f'{self._name}.Doji')
        return recv

def get_candle(market_id: str, period_type: Union[int, TimeUnit]) -> _CandleStick:
    if period_type == TimeUnit.DAY:
        name = f'{market_id}.DailyDoutzeCandleStick'
    elif period_type == TimeUnit.WEEK:
        name = f'{market_id}.WeeklyDoutzeCandleStick'
    elif period_type == TimeUnit.MONTH:
        name = f'{market_id}.MonthlyDoutzeCandleStick'
    else:
        name = f'{market_id}.DoutzeCandleStick({period_type})'
    if name not in MD_CACHE:
        MD_CACHE[name] = DoutzeCandleStick.make(TI.Candle(market_id, period_type), name)
    return MD_CACHE[name]

def _is_long_white(cct: DoutzeCandleStick,
                   shadow_ignore_threshold: float,
                   violent_threshold: float):
    cond_1 = cct.is_white()
    cond_2 = cct.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_long_white(market_id: str, **kwargs):
    """long_white.

    規則：
        大陽線；
        1. 陽線
        2. 長實體線
        3. 無上影線(可忽略的上影線)
        4. 無下影線(可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_long_white'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_long_white': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_long_white(cct,
                         shadow_ignore_threshold=shadow_ignore_threshold,
                         violent_threshold=violent_threshold)
    ret.rename(f'{market_id}.doutze_long_white({kwargs})')
    return ret

def _is_white_closing_marubozu(cct: DoutzeCandleStick,
                               shadow_ignore_threshold: float,
                               violent_threshold: float):
    cond_1 = cct.is_white()
    cond_2 = cct.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = ~cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_white_closing_marubozu(market_id: str, **kwargs):
    """white_closing_marubozu.

    規則：
        光頭陽線；
        1. 陽線
        2. 長實體線
        3. 無上影線(可忽略的上影線)
        4. 有下影線(不可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_white_closing_marubozu'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_white_closing_marubozu': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_white_closing_marubozu(cct,
                                     shadow_ignore_threshold=shadow_ignore_threshold,
                                     violent_threshold=violent_threshold)
    ret.rename(f'{market_id}.doutze_white_closing_marubozu({kwargs})')
    return ret

def _is_white_hammer(cct: DoutzeCandleStick,
                     shadow_ignore_threshold: float,
                     violent_threshold: float,
                     close_zero_threshold: float):
    cond_1 = cct.is_white()
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    cond_3 = cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = ~cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_white_hammer(market_id: str, **kwargs):
    """white_hammer.

    規則：
        陽線槌子線；
        1. 陽線
        2. 短實體線
        3. 無上影線(可忽略的上影線)
        4. 有下影線(不可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).
	close_zero_threshold : float
        持平臨界值(r3).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_white_hammer'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_white_hammer': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_white_hammer(cct,
                           shadow_ignore_threshold=shadow_ignore_threshold,
                           violent_threshold=violent_threshold,
                           close_zero_threshold=close_zero_threshold)
    ret.rename(f'{market_id}.doutze_white_hammer({kwargs})')
    return ret

def _is_white_opening_marubozu(cct: DoutzeCandleStick,
                               shadow_ignore_threshold: float,
                               violent_threshold: float):
    cond_1 = cct.is_white()
    cond_2 = cct.long_body(violent_threshold=violent_threshold)
    cond_3 = ~cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_white_opening_marubozu(market_id: str, **kwargs):
    """white_opening_marubozu.

    規則：
        光腳陽線；
        1. 陽線
        2. 長實體線
        3. 有上影線(不可忽略的上影線)
        4. 無下影線(可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_white_opening_marubozu'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_white_opening_marubozu': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_white_opening_marubozu(cct,
                                     shadow_ignore_threshold=shadow_ignore_threshold,
                                     violent_threshold=violent_threshold)
    ret.rename(f'{market_id}.doutze_white_opening_marubozu({kwargs})')
    return ret

def _is_white_inverse_hammer(cct: DoutzeCandleStick,
                             shadow_ignore_threshold: float,
                             violent_threshold: float,
                             close_zero_threshold: float):
    cond_1 = cct.is_white()
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    cond_3 = ~cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_white_inverse_hammer(market_id: str, **kwargs):
    """white_inverse_hammer.

    規則：
        倒陽線槌子線；
        1. 陽線
        2. 短實體線
        3. 有上影線(不可忽略的上影線)
        4. 無下影線(可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).
	close_zero_threshold : float
        持平臨界值(r3).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_white_inverse_hammer'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_white_inverse_hammer': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_white_inverse_hammer(cct,
                                   shadow_ignore_threshold=shadow_ignore_threshold,
                                   violent_threshold=violent_threshold,
                                   close_zero_threshold=close_zero_threshold)
    ret.rename(f'{market_id}.doutze_white_inverse_hammer({kwargs})')
    return ret

def _is_short_white(cct: DoutzeCandleStick,
                             shadow_ignore_threshold: float,
                             violent_threshold: float,
                             close_zero_threshold: float,
                             long_shadow_threshold: float):
    cond_1 = cct.is_white()
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    # ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    # cond_3 = ~cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    # cond_4 = ~cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_5a = (cct.long_upper_shadow(long_shadow_threshold=long_shadow_threshold) &
               cct.long_lower_shadow(long_shadow_threshold=long_shadow_threshold))
    cond_5b = (cct.short_upper_shadow(long_shadow_threshold=long_shadow_threshold,
                                     shadow_ignore_threshold=shadow_ignore_threshold) &
               cct.short_lower_shadow(long_shadow_threshold=long_shadow_threshold,
                                     shadow_ignore_threshold=shadow_ignore_threshold))
    cond_5 = cond_5a | cond_5b
    # ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret = cond_1 & cond_2 & cond_5
    return ret

@doutze_ccp_check
def doutze_short_white(market_id: str, **kwargs):
    """short_white.

    規則：
        小陽線；
        1. 陽線
        2. 短實體線
        3. 有上影線(不可忽略的上影線)
        4. 有下影線(不可忽略的下影線)
        5. 上下影線不能一長一短
        -> (長上影線 且 長下影線) 或 (短上影線 且 短下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).
	close_zero_threshold : float
        持平臨界值(r3).
	long_shadow_threshold : float
        長影線臨界值(r4).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_short_white'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_short_white': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_short_white(cct,
                          shadow_ignore_threshold=shadow_ignore_threshold,
                          violent_threshold=violent_threshold,
                          close_zero_threshold=close_zero_threshold,
                          long_shadow_threshold=long_shadow_threshold)
    ret.rename(f'{market_id}.doutze_short_white({kwargs})')
    return ret

def _is_long_upper_shadow_short_white(cct: DoutzeCandleStick,
                                      shadow_ignore_threshold: float,
                                      violent_threshold: float,
                                      close_zero_threshold: float,
                                      long_shadow_threshold: float):
    cond_1 = cct.is_white()
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    cond_3 = cct.long_upper_shadow(long_shadow_threshold=long_shadow_threshold)
    cond_4 = cct.short_lower_shadow(long_shadow_threshold=long_shadow_threshold,
                                    shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_long_upper_shadow_short_white(market_id: str, **kwargs):
    """long_upper_shadow_short_white.

    規則：
        上影線較長的小陽線；
        1. 陽線
        2. 短實體線
        3. 長上影線
        4. 短下影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).
	close_zero_threshold : float
        持平臨界值(r3).
	long_shadow_threshold : float
        長影線臨界值(r4).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_long_upper_shadow_short_white'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_long_upper_shadow_short_white': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_long_upper_shadow_short_white(
            cct,
            shadow_ignore_threshold=shadow_ignore_threshold,
            violent_threshold=violent_threshold,
            close_zero_threshold=close_zero_threshold,
            long_shadow_threshold=long_shadow_threshold)
    ret.rename(f'{market_id}.doutze_long_upper_shadow_short_white({kwargs})')
    return ret

def _is_long_lower_shadow_short_white(cct: DoutzeCandleStick,
                                      shadow_ignore_threshold: float,
                                      violent_threshold: float,
                                      close_zero_threshold: float,
                                      long_shadow_threshold: float):
    cond_1 = cct.is_white()
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    cond_3 = cct.short_upper_shadow(long_shadow_threshold=long_shadow_threshold,
                                    shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = cct.long_lower_shadow(long_shadow_threshold=long_shadow_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_long_lower_shadow_short_white(market_id: str, **kwargs):
    """long_lower_shadow_short_white.

    規則：
        下影線較長的小陽線；
        1. 陽線
        2. 短實體線
        3. 短上影線
        4. 長下影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).
	close_zero_threshold : float
        持平臨界值(r3).
	long_shadow_threshold : float
        長影線臨界值(r4).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_long_lower_shadow_short_white'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_long_lower_shadow_short_white': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_long_lower_shadow_short_white(
            cct,
            shadow_ignore_threshold=shadow_ignore_threshold,
            violent_threshold=violent_threshold,
            close_zero_threshold=close_zero_threshold,
            long_shadow_threshold=long_shadow_threshold)
    ret.rename(f'{market_id}.doutze_long_lower_shadow_short_white({kwargs})')
    return ret

def _is_long_black(cct: DoutzeCandleStick,
                   shadow_ignore_threshold: float,
                   violent_threshold: float):
    cond_1 = cct.is_black()
    cond_2 = cct.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_long_black(market_id: str, **kwargs):
    """long_black.

    規則：
        大陰線；
        1. 陰線
        2. 長實體線
        3. 無上影線(可忽略的上影線)
        4. 無下影線(可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_long_black'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_long_black': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_long_black(cct,
                         shadow_ignore_threshold=shadow_ignore_threshold,
                         violent_threshold=violent_threshold)
    ret.rename(f'{market_id}.doutze_long_black({kwargs})')
    return ret

def _is_black_opening_marubozu(cct: DoutzeCandleStick,
                               shadow_ignore_threshold: float,
                               violent_threshold: float):
    cond_1 = cct.is_black()
    cond_2 = cct.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = ~cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_black_opening_marubozu(market_id: str, **kwargs):
    """black_opening_marubozu.

    規則：
        光頭陰線；
        1. 陰線
        2. 長實體線
        3. 無上影線(可忽略的上影線)
        4. 有下影線((不可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_black_opening_marubozu'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_black_opening_marubozu': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_black_opening_marubozu(cct,
                                     shadow_ignore_threshold=shadow_ignore_threshold,
                                     violent_threshold=violent_threshold)
    ret.rename(f'{market_id}.doutze_black_opening_marubozu({kwargs})')
    return ret

def _is_black_hammer(cct: DoutzeCandleStick,
                     shadow_ignore_threshold: float,
                     violent_threshold: float,
                     close_zero_threshold: float):
    cond_1 = cct.is_black()
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    cond_3 = cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = ~cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_black_hammer(market_id: str, **kwargs):
    """black_hammer.

    規則：
        陰線槌子線；
        1. 陰線
        2. 短實體線
        3. 無上影線(可忽略的上影線)
        4. 有下影線(不可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).
	close_zero_threshold : float
        持平臨界值(r3).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_black_hammer'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_black_hammer': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_black_hammer(cct,
                           shadow_ignore_threshold=shadow_ignore_threshold,
                           violent_threshold=violent_threshold,
                           close_zero_threshold=close_zero_threshold)
    ret.rename(f'{market_id}.doutze_black_hammer({kwargs})')
    return ret

def _is_black_closing_marubozu(cct: DoutzeCandleStick,
                               shadow_ignore_threshold: float,
                               violent_threshold: float):
    cond_1 = cct.is_black()
    cond_2 = cct.long_body(violent_threshold=violent_threshold)
    cond_3 = ~cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_black_closing_marubozu(market_id: str, **kwargs):
    """black_closing_marubozu.

    規則：
        光腳陰線；
        1. 陰線
        2. 長實體線
        3. 有上影線(不可忽略的上影線)
        4. 無下影線(可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_black_closing_marubozu'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_black_closing_marubozu': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_black_closing_marubozu(cct,
                                     shadow_ignore_threshold=shadow_ignore_threshold,
                                     violent_threshold=violent_threshold)
    ret.rename(f'{market_id}.doutze_black_closing_marubozu({kwargs})')
    return ret

def _is_black_inverse_hammer(cct: DoutzeCandleStick,
                             shadow_ignore_threshold: float,
                             violent_threshold: float,
                             close_zero_threshold: float):
    cond_1 = cct.is_black()
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    cond_3 = ~cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_black_inverse_hammer(market_id: str, **kwargs):
    """black_inverse_hammer.

    規則：
        倒陰線槌子線；
        1. 陰線
        2. 短實體線
        3. 有上影線(不可忽略的上影線)
        4. 無下影線(可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).
	close_zero_threshold : float
        持平臨界值(r3).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_black_inverse_hammer'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_black_inverse_hammer': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_black_inverse_hammer(cct,
                                   shadow_ignore_threshold=shadow_ignore_threshold,
                                   violent_threshold=violent_threshold,
                                   close_zero_threshold=close_zero_threshold)
    ret.rename(f'{market_id}.doutze_black_inverse_hammer({kwargs})')
    return ret

def _is_short_black(cct: DoutzeCandleStick,
                             shadow_ignore_threshold: float,
                             violent_threshold: float,
                             close_zero_threshold: float,
                             long_shadow_threshold: float):
    cond_1 = cct.is_black()
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    # condition 3 and 4 are not requried, since condition 5 exclude the possibility
    # cond_3 = ~cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    # cond_4 = ~cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_5a = (cct.long_upper_shadow(long_shadow_threshold=long_shadow_threshold) &
               cct.long_lower_shadow(long_shadow_threshold=long_shadow_threshold))
    cond_5b = (cct.short_upper_shadow(long_shadow_threshold=long_shadow_threshold,
                                     shadow_ignore_threshold=shadow_ignore_threshold) &
               cct.short_lower_shadow(long_shadow_threshold=long_shadow_threshold,
                                     shadow_ignore_threshold=shadow_ignore_threshold))
    cond_5 = cond_5a | cond_5b

    # ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret = cond_1 & cond_2 & cond_5
    return ret

@doutze_ccp_check
def doutze_short_black(market_id: str, **kwargs):
    """short_black.

    規則：
        小陰線；
        1. 陰線
        2. 短實體線
        3. 有上影線(不可忽略的上影線)
        4. 有下影線(不可忽略的下影線)
        5. 上下影線不能一長一短
        -> (長上影線 且 長下影線) 或 (短上影線 且 短下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).
	close_zero_threshold : float
        持平臨界值(r3).
	long_shadow_threshold : float
        長影線臨界值(r4).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_short_black'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_short_black': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_short_black(cct,
                          shadow_ignore_threshold=shadow_ignore_threshold,
                          violent_threshold=violent_threshold,
                          close_zero_threshold=close_zero_threshold,
                          long_shadow_threshold=long_shadow_threshold)
    ret.rename(f'{market_id}.doutze_short_black({kwargs})')
    return ret

def _is_long_upper_shadow_short_black(cct: DoutzeCandleStick,
                                      shadow_ignore_threshold: float,
                                      violent_threshold: float,
                                      close_zero_threshold: float,
                                      long_shadow_threshold: float):
    cond_1 = cct.is_black()
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    cond_3 = cct.long_upper_shadow(long_shadow_threshold=long_shadow_threshold)
    cond_4 = cct.short_lower_shadow(long_shadow_threshold=long_shadow_threshold,
                                    shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_long_upper_shadow_short_black(market_id: str, **kwargs):
    """long_upper_shadow_short_black.

    規則：
        上影線較長的小陰線；
        1. 陰線
        2. 短實體線
        3. 長上影線
        4. 短下影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).
	close_zero_threshold : float
        持平臨界值(r3).
	long_shadow_threshold : float
        長影線臨界值(r4).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_long_upper_shadow_short_black'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_long_upper_shadow_short_black': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_long_upper_shadow_short_black(
            cct,
            shadow_ignore_threshold=shadow_ignore_threshold,
            violent_threshold=violent_threshold,
            close_zero_threshold=close_zero_threshold,
            long_shadow_threshold=long_shadow_threshold)
    ret.rename(f'{market_id}.doutze_long_upper_shadow_short_black({kwargs})')
    return ret

def _is_long_lower_shadow_short_black(cct: DoutzeCandleStick,
                                      shadow_ignore_threshold: float,
                                      violent_threshold: float,
                                      close_zero_threshold: float,
                                      long_shadow_threshold: float):
    cond_1 = cct.is_black()
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    cond_3 = cct.short_upper_shadow(long_shadow_threshold=long_shadow_threshold,
                                    shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4 = cct.long_lower_shadow(long_shadow_threshold=long_shadow_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4
    return ret

@doutze_ccp_check
def doutze_long_lower_shadow_short_black(market_id: str, **kwargs):
    """long_lower_shadow_short_black.

    規則：
        下影線較長的小陰線；
        1. 陰線
        2. 短實體線
        3. 短上影線
        4. 長下影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值(r2).
	close_zero_threshold : float
        持平臨界值(r3).
	long_shadow_threshold : float
        長影線臨界值(r4).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_long_lower_shadow_short_black'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_long_lower_shadow_short_black': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_long_lower_shadow_short_black(
            cct,
            shadow_ignore_threshold=shadow_ignore_threshold,
            violent_threshold=violent_threshold,
            close_zero_threshold=close_zero_threshold,
            long_shadow_threshold=long_shadow_threshold)
    ret.rename(f'{market_id}.doutze_long_lower_shadow_short_black({kwargs})')
    return ret

def _is_four_price_doji(cct: DoutzeCandleStick,
                        close_zero_threshold: float):
    ret = cct.ignored_amplitude(close_zero_threshold=close_zero_threshold)
    return ret

@doutze_ccp_check
def doutze_four_price_doji(market_id: str, **kwargs):
    """four_price_doji.

    規則：
        一字線；
        1. 無震幅(可忽略的震幅)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
	close_zero_threshold : float
        持平臨界值(r1).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_four_price_doji'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_four_price_doji': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_four_price_doji(cct, close_zero_threshold=close_zero_threshold)
    ret.rename(f'{market_id}.doutze_four_price_doji({kwargs})')
    return ret

def _is_umbrella(cct: DoutzeCandleStick,
                 shadow_ignore_threshold: float,
                 close_zero_threshold: float):
    cond_1 = cct.ignored_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = ~cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3
    return ret

@doutze_ccp_check
def doutze_umbrella(market_id: str, **kwargs):
    """umbrella.

    規則：
        T字線；
        1. 無實體線(可忽略的實體線)
        2. 無上影線(可忽略的上影線)
        3. 有下影線(不可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
	close_zero_threshold : float
        持平臨界值(r2).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_umbrella'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_umbrella': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_umbrella(cct, shadow_ignore_threshold=shadow_ignore_threshold,
                       close_zero_threshold=close_zero_threshold)
    ret.rename(f'{market_id}.doutze_umbrella({kwargs})')
    return ret

def _is_inverse_umbrella(cct: DoutzeCandleStick,
                         shadow_ignore_threshold: float,
                         close_zero_threshold: float):
    cond_1 = cct.ignored_body(close_zero_threshold=close_zero_threshold)
    cond_2 = ~cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3
    return ret

@doutze_ccp_check
def doutze_inverse_umbrella(market_id: str, **kwargs):
    """inverse_umbrella.

    規則：
        墓碑線/倒T字線；
        1. 無實體線(可忽略的實體線)
        2. 有上影線(不可忽略的上影線)
        3. 無下影線(可忽略的下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
	close_zero_threshold : float
        持平臨界值(r2).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_inverse_umbrella'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_inverse_umbrella': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_inverse_umbrella(cct,
                               shadow_ignore_threshold=shadow_ignore_threshold,
                               close_zero_threshold=close_zero_threshold)
    ret.rename(f'{market_id}.doutze_inverse_umbrella({kwargs})')
    return ret

def _is_doji(cct: DoutzeCandleStick,
             shadow_ignore_threshold: float,
             close_zero_threshold: float,
             long_shadow_threshold: float):
    cond_1 = cct.doji(close_zero_threshold=close_zero_threshold,
                      shadow_ignore_threshold=shadow_ignore_threshold)
    cond_2a = (cct.long_upper_shadow(long_shadow_threshold=long_shadow_threshold) &
               cct.long_lower_shadow(long_shadow_threshold=long_shadow_threshold))
    cond_2b = (cct.short_upper_shadow(long_shadow_threshold=long_shadow_threshold,
                                     shadow_ignore_threshold=shadow_ignore_threshold) &
               cct.short_lower_shadow(long_shadow_threshold=long_shadow_threshold,
                                     shadow_ignore_threshold=shadow_ignore_threshold))
    cond_2 = cond_2a | cond_2b
    ret = cond_1 & cond_2
    return ret

@doutze_ccp_check
def doutze_doji(market_id: str, **kwargs):
    """doji.

    規則：
        十字線；
        1. 無實體線(可忽略的實體線)
        2. 有上影線(不可忽略的上影線)
        3. 有下影線(不可忽略的下影線)
        4. 上下影線不能一長一短
        -> (長上影線 且 長下影線) 或 (短上影線 且 短下影線)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
	close_zero_threshold : float
        持平臨界值(r2).
	long_shadow_threshold : float
        長影線臨界值(r3).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_doji'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_doji': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_doji(cct, shadow_ignore_threshold=shadow_ignore_threshold,
                   close_zero_threshold=close_zero_threshold,
                   long_shadow_threshold=long_shadow_threshold)
    ret.rename(f'{market_id}.doutze_doji({kwargs})')
    return ret

def _is_long_lower_shadow_doji(cct: DoutzeCandleStick,
                               shadow_ignore_threshold: float,
                               close_zero_threshold: float,
                               long_shadow_threshold: float):
    cond_1 = cct.ignored_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct.short_upper_shadow(long_shadow_threshold=long_shadow_threshold,
                                    shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = cct.long_lower_shadow(long_shadow_threshold=long_shadow_threshold)
    ret = cond_1 & cond_2 & cond_3
    return ret

@doutze_ccp_check
def doutze_long_lower_shadow_doji(market_id: str, **kwargs):
    """long_lower_shadow_doji.

    規則：
        下影線較長的十字線；
        1. 無實體線(可忽略的實體線)
        2. 短上影線
        3. 長下影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
	close_zero_threshold : float
        持平臨界值(r2).
	long_shadow_threshold : float
        長影線臨界值(r3).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_long_lower_shadow_doji'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_long_lower_shadow_doji': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_long_lower_shadow_doji(cct,
                                     shadow_ignore_threshold=shadow_ignore_threshold,
                                     close_zero_threshold=close_zero_threshold,
                                     long_shadow_threshold=long_shadow_threshold)
    ret.rename(f'{market_id}.doutze_long_lower_shadow_doji({kwargs})')
    return ret

def _is_long_upper_shadow_doji(cct: DoutzeCandleStick,
                               shadow_ignore_threshold: float,
                               close_zero_threshold: float,
                               long_shadow_threshold: float):
    cond_1 = cct.ignored_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct.long_upper_shadow(long_shadow_threshold=long_shadow_threshold)
    cond_3 = cct.short_lower_shadow(long_shadow_threshold=long_shadow_threshold,
                                    shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3
    return ret

@doutze_ccp_check
def doutze_long_upper_shadow_doji(market_id: str, **kwargs):
    """long_upper_shadow_doji.

    規則：
        上影線較長的十字線；
        1. 無實體線(可忽略的實體線)
        2. 長上影線
        3. 短下影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
	close_zero_threshold : float
        持平臨界值(r2).
	long_shadow_threshold : float
        長影線臨界值(r3).

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_long_upper_shadow_doji'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_long_upper_shadow_doji': {esp}")
    cct = get_candle(market_id, period_type)
    ret = _is_long_upper_shadow_doji(cct,
                                     shadow_ignore_threshold=shadow_ignore_threshold,
                                     close_zero_threshold=close_zero_threshold,
                                     long_shadow_threshold=long_shadow_threshold)
    ret.rename(f'{market_id}.doutze_long_upper_shadow_doji({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_rainy(market_id: str, **kwargs):
    """bearish_rainy.

    規則：
        大雨傾盆；
        1. 昨日大陽線
        2. 今日大陰線
        3. 昨日最低價 > 今日收盤價
        4. 昨日收盤價 > 今日開盤價
        5. 今日燭身需插入昨日燭身一半以上
        -> (昨日開盤價 + 昨日收盤價) / 2 <= 今日開盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.

    See Also
    --------
    doutze_long_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_rainy'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_rainy': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_white(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_long_black(cct,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_3 = cct_1.low > cct.close
    cond_4 = cct_1.close > cct.open
    cond_5 = (cct_1.open + cct_1.close) / 2 <= cct.open
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bearish_rainy({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_sunny(market_id: str, **kwargs):
    """bullish_sunny.

    規則：
        旭日東昇；
        1. 昨日大陰線
        2. 今日大陽線
        3. 昨日最高價 > 今日開盤價
        4. 昨日收盤價 < 今日開盤價
        5. 今日燭身需插入昨日燭身一半以上
        -> (昨日開盤價 + 昨日收盤價) / 2 >= 今日開盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.

    See Also
    --------
    doutze_long_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_sunny'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_sunny': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_black(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_long_white(cct,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_3 = cct_1.high > cct.open
    cond_4 = cct_1.close < cct.open
    cond_5 = (cct_1.open + cct_1.close) / 2 >= cct.open
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bullish_sunny({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_dark_cloud_cover(market_id: str, **kwargs):
    """bearish_dark_cloud_cover.

    規則：
        烏雲罩頂；
        1. 昨日大陽線
        2. 今日大陰線
        3. 昨日最高價 < 今日開盤價
        4. 昨日開盤價 < 今日收盤價
        5. 今日燭身需插入昨日燭身一半以上
        -> (昨日開盤價 + 昨日收盤價) / 2 >= 今日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.

    See Also
    --------
    doutze_long_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_dark_cloud_cover'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_dark_cloud_cover': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_white(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_long_black(cct,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_3 = cct_1.high < cct.open
    cond_4 = cct_1.open < cct.close
    cond_5 = (cct_1.open + cct_1.close) / 2 >= cct.close
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bearish_dark_cloud_cover({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_piercing(market_id: str, **kwargs):
    """bullish_piercing.

    規則：
        曙光乍現；
        1. 昨日大陰線
        2. 今日大陽線
        3. 昨日最低價 > 今日開盤價
        4. 昨日開盤價 > 今日收盤價
        5. 今日燭身需插入昨日燭身一半以上
        -> (昨日開盤價 + 昨日收盤價) / 2 <= 今日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.

    See Also
    --------
    doutze_long_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_piercing'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_piercing': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_black(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_long_white(cct,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_3 = cct_1.low > cct.open
    cond_4 = cct_1.open > cct.close
    cond_5 = (cct_1.open + cct_1.close) / 2 <= cct.close
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bullish_piercing({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_harami(market_id: str, **kwargs):
    """bearish_harami.

    規則：
        頂部母子線/空頭母子；
        1. 昨日大陽線
        2. 今日小陰線
        3. 昨日開盤價 <= 今日收盤價
        4. 昨日收盤價 >= 今日開盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.
    long_shadow_threshold : float
        長影線臨界值.

    See Also
    --------
    doutze_short_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_harami'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_harami': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_white(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_short_black(cct,
                             shadow_ignore_threshold=shadow_ignore_threshold,
                             violent_threshold=violent_threshold,
                             close_zero_threshold=close_zero_threshold,
                             long_shadow_threshold=long_shadow_threshold)
    cond_3 = cct_1.open <= cct.close
    cond_4 = cct_1.close >= cct.open
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bearish_harami({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_harami(market_id: str, **kwargs):
    """bullish_harami.

    規則：
        底部母子線/多頭母子；
        1. 昨日大陰線
        2. 今日小陽線
        3. 昨日收盤價 <= 今日開盤價
        4. 昨日開盤價 >= 今日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.
    long_shadow_threshold : float
        長影線臨界值.

    See Also
    --------
    doutze_short_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_harami'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_harami': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_black(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_short_white(cct,
                             shadow_ignore_threshold=shadow_ignore_threshold,
                             violent_threshold=violent_threshold,
                             close_zero_threshold=close_zero_threshold,
                             long_shadow_threshold=long_shadow_threshold)
    cond_3 = cct_1.close <= cct.open
    cond_4 = cct_1.open >= cct.close
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bullish_harami({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_engulfing(market_id: str, **kwargs):
    """bearish_engulfing.

    規則：
        頂部穿頭破腳/吞噬；
        1. 昨日陰線
        2. 今日陽線
        3. 昨日最高價 <= 今日最高價
        4. 昨日最低價 >= 今日最低價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
	close_zero_threshold : float
        持平臨界值.

    See Also
    --------
    doutze_short_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_engulfing'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_engulfing': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = cct_1.black_body(close_zero_threshold)
    cond_2 = cct.white_body(close_zero_threshold)
    cond_3 = cct_1.high <= cct.high
    cond_4 = cct_1.low >= cct.low
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bearish_engulfing({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_engulfing(market_id: str, **kwargs):
    """bullish_engulfing.

    規則：
        底部穿頭破腳/多頭吞噬；
        1. 昨日陽線
        2. 今日陰線
        3. 昨日最高價 <= 今日最高價
        4. 昨日最低價 >= 今日最低價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
	close_zero_threshold : float
        持平臨界值.

    See Also
    --------
    doutze_short_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_engulfing'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_engulfing': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = cct_1.white_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct.black_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct_1.high <= cct.high
    cond_4 = cct_1.low >= cct.low
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bullish_engulfing({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_separating_lines(market_id: str, **kwargs):
    """bullish_separating_lines.

    規則：
        看漲分手線；
        1. 昨日大陰線
        2. 今日大陽線
        3. 昨日開盤價 <= 今日最低價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.

    See Also
    --------
    doutze_long_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_separating_lines'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_separating_lines': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_black(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_long_white(cct,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_3 = cct_1.open <= cct.low
    ret = cond_1 & cond_2 & cond_3
    ret.rename(f'{market_id}.doutze_bullish_separating_lines({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_separating_lines(market_id: str, **kwargs):
    """bearish_separating_lines.

    規則：
        看跌分手線；
        1. 昨日大陽線
        2. 今日大陰線
        3. 昨日開盤價 <= 今日最高價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.

    See Also
    --------
    doutze_long_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_separating_lines'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_separating_lines': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_white(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_long_black(cct,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_3 = cct_1.open <= cct.high
    ret = cond_1 & cond_2 & cond_3
    ret.rename(f'{market_id}.doutze_bearish_separating_lines({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_harami_cross(market_id: str, **kwargs):
    """bearish_harami_cross.

    規則：
        頂部十字胎；
        1. 昨日大陽線
        2. 今日十字線
        3. 昨日最高價 >= 今日最高價
        4. 昨日最低價 <= 今日最低價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.
    long_shadow_threshold : float
        長影線臨界值.

    See Also
    --------
    doutze_doji, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_harami_cross'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_harami_cross': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_white(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_doji(cct, shadow_ignore_threshold=shadow_ignore_threshold,
                      close_zero_threshold=close_zero_threshold,
                      long_shadow_threshold=long_shadow_threshold)
    cond_3 = cct_1.high >= cct.high
    cond_4 = cct_1.low <= cct.low
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bearish_harami_cross({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_harami_cross(market_id: str, **kwargs):
    """bullish_harami_cross.

    規則：
        底部十字胎；
        1. 昨日大陰線
        2. 今日十字線
        3. 昨日最高價 >= 今日最高價
        4. 昨日最低價 <= 今日最低價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.
    long_shadow_threshold : float
        長影線臨界值.

    See Also
    --------
    doutze_doji, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_harami_cross'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_harami_cross': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_black(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_doji(cct, shadow_ignore_threshold=shadow_ignore_threshold,
                      close_zero_threshold=close_zero_threshold,
                      long_shadow_threshold=long_shadow_threshold)
    cond_3 = cct_1.high >= cct.high
    cond_4 = cct_1.low <= cct.low
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bullish_harami_cross({kwargs})')
    return ret

####################### ↑↑↑ code review 2021-11-03 ↑↑↑ #######################
@doutze_ccp_check
def doutze_bullish_covered_lines(market_id: str, **kwargs):
    """bullish_covered_lines.

    規則：
        看漲覆蓋線；
        1. 昨日大陰線
        2. 今日大陽線
        3. 昨日收盤價 > 今日開盤價
        4. 今日燭身需插入昨日燭身一半以上
        -> (昨日開盤價 + 昨日收盤價) / 2 <= 今日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.

    See Also
    --------
    doutze_long_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_covered_lines'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_covered_lines': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_black(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_long_white(cct,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_3 = cct_1.close > cct.open
    cond_4 = (cct_1.open + cct_1.close) / 2 <= cct.close
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bullish_covered_lines({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_covered_lines(market_id: str, **kwargs):
    """bearish_covered_lines.

    規則：
        看跌覆蓋線；
        1. 昨日大陽線
        2. 今日大陰線
        3. 昨日開盤價 > 今日收盤價
        4. 今日燭身需插入昨日燭身一半以上
        -> (昨日開盤價 + 昨日收盤價) / 2 <= 今日開盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.

    See Also
    --------
    doutze_long_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_covered_lines'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_covered_lines': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_white(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_long_black(cct,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_3 = cct_1.open > cct.close
    cond_4 = (cct_1.open + cct_1.close) / 2 <= cct.open
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bearish_covered_lines({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_meeting_lines(market_id: str, **kwargs):
    """bullish_meeting_lines.

    規則：
        看漲約會線/好友反攻；
        1. 昨日大陰線
        2. 今日大陽線
        3. 昨日收盤價接近今日收盤價(兩者的差接近0)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    See Also
    --------
    doutze_long_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_meeting_lines'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_meeting_lines': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_black(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_long_white(cct,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_3 = _is_close(cct_1.close, cct.close,
                       close_zero_threshold=close_zero_threshold)
    ret = cond_1 & cond_2 & cond_3
    ret.rename(f'{market_id}.doutze_bullish_meeting_lines({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_meeting_lines(market_id: str, **kwargs):
    """bearish_meeting_lines.

    規則：
        看跌約會線/淡友反攻；
        1. 昨日大陽線
        2. 今日大陰線
        3. 昨日收盤價接近今日收盤價(兩者的差接近0)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值(r).

    See Also
    --------
    doutze_long_black, doutze_long_white

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_meeting_lines'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_meeting_lines': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_white(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2 = _is_long_black(cct,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_3 = _is_close(cct_1.close, cct.close,
                       close_zero_threshold=close_zero_threshold)
    ret = cond_1 & cond_2 & cond_3
    ret.rename(f'{market_id}.doutze_bearish_meeting_lines({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_tweezer_bottoms(market_id: str, **kwargs):
    """bearish_tweezer_bottoms.

    規則：
        平底；
        1. 昨日大陰線
        2. 今日十字線或陽線槌子線
        3. 昨日最低價 <= 今日最低價
        4. (昨日最高價 + 昨日最低價) / 2 >= 今日最高價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.
    long_shadow_threshold : float
        長影線臨界值.

    See Also
    --------
    doutze_doji, doutze_long_white, doutze_white_hammer

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_tweezer_bottoms'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_tweezer_bottoms': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_black(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2a = _is_doji(cct, shadow_ignore_threshold=shadow_ignore_threshold,
                       close_zero_threshold=close_zero_threshold,
                       long_shadow_threshold=long_shadow_threshold)
    cond_2b = _is_white_hammer(cct,
                               shadow_ignore_threshold=shadow_ignore_threshold,
                               violent_threshold=violent_threshold,
                               close_zero_threshold=close_zero_threshold)
    cond_2 = cond_2a | cond_2b
    cond_3 = cct_1.low <= cct.low
    cond_4 = (cct_1.high + cct_1.low) / 2 >= cct.high
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bearish_tweezer_bottoms({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_tweezer_bottoms(market_id: str, **kwargs):
    """bullish_tweezer_bottoms.

    規則：
        平頂；
        1. 昨日大陽線
        2. 今日十字線或陰線槌子線
        3. 昨日最高價 >= 今日最高價
        4. (昨日最高價 + 昨日最低價) / 2 <= 今日最低價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.
    long_shadow_threshold : float
        長影線臨界值.

    See Also
    --------
    doutze_doji, doutze_long_white, doutze_black_hammer

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_tweezer_bottoms'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_tweezer_bottoms': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_white(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2a = _is_doji(cct, shadow_ignore_threshold=shadow_ignore_threshold,
                       close_zero_threshold=close_zero_threshold,
                       long_shadow_threshold=long_shadow_threshold)
    cond_2b = _is_black_hammer(cct,
                               shadow_ignore_threshold=shadow_ignore_threshold,
                               violent_threshold=violent_threshold,
                               close_zero_threshold=close_zero_threshold)
    cond_2 = cond_2a | cond_2b
    cond_3 = cct_1.high >= cct.high
    cond_4 = (cct_1.high + cct_1.low) / 2 <= cct.low
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bullish_tweezer_bottoms({kwargs})')
    return ret

################# ↑↑↑ doutze code review 2021-11-06 ↑↑↑ ####################
@doutze_ccp_check
def doutze_bearish_three_outside_down(market_id: str, **kwargs):
    """bearish_three_outside_down.

    規則：
        外側三日下跌；
        1. 兩日前陽線
        2. 昨日陰線
        3. 今日陰線
        4. 昨日陰線吞噬兩日前陽線燭身
        => 昨日開盤價 >= 兩日前收盤價 且 昨日收盤價 <= 兩日前開盤價
        5. 今日收盤價低於昨日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_three_outside_down'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_three_outside_down': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.white_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct_1.black_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.black_body(close_zero_threshold=close_zero_threshold)
    cond_4a = cct_1.open >= cct_2.close
    cond_4b = cct_1.close <= cct_2.open
    cond_4 = cond_4a & cond_4b
    cond_5 = cct.close < cct_1.close
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bearish_three_outside_down({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_three_inside_down(market_id: str, **kwargs):
    """bearish_three_inside_down.

    規則：
        內困三日翻黑；
        1. 兩日前劇烈陽線
        2. 昨日陰線
        3. 今日陰線
        4. 昨日陰線被前兩日前陽線燭身吞噬
        => 昨日開盤價 <= 兩日前收盤價 且 昨日收盤價 >= 兩日前開盤價
		5. 今日收盤價 < 昨日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
	violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_three_inside_down'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_three_inside_down': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.black_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.black_body(close_zero_threshold=close_zero_threshold)
    cond_4a = cct_1.open <= cct_2.close
    cond_4b = cct_1.close >= cct_2.open
    cond_4 = cond_4a & cond_4b
    cond_5 = cct.close < cct_1.close
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bearish_three_inside_down({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_evening_star(market_id: str, **kwargs):
    """bearish_evening_star.

    規則：
        夜星；
        1. 兩日前劇烈陽線
        2. 昨日陰線或陽線
        3. 今日劇烈陰線
        4. 昨日陰或陽線與兩日前和今日有實體上跳空
        => min(昨日收盤價, 昨日開盤價) > 兩日前收盤價 且
		   min(昨日收盤價, 昨日開盤價) > 今日開盤價
		5. 昨日為短實體線
		6. 今日燭身需插入兩日前燭身一半以上
		=> 今日的收盤價 <= (兩日前開盤價 + 兩日前收盤價) / 2

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
	violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_evening_star'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_evening_star': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2a = cct_1.black_body(close_zero_threshold=close_zero_threshold)
    cond_2b = cct_1.white_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cond_2a | cond_2b
    cond_3 = cct.is_black() & cct.long_body(violent_threshold=violent_threshold)
    cond_4a = ts_min(cct_1.open, cct_1.close) > cct_2.close
    cond_4b = ts_min(cct_1.open, cct_1.close) > cct.open
    cond_4 = cond_4a & cond_4b
    cond_5 = cct_1.short_body(violent_threshold=violent_threshold,
                              close_zero_threshold=close_zero_threshold)
    cond_6 = cct.close <= (cct_2.open + cct_2.close) / 2
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6
    ret.rename(f'{market_id}.doutze_bearish_evening_star({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_evening_doji_star(market_id: str, **kwargs):
    """bearish_evening_doji_star.

    規則：
        夜星十字；
        1. 兩日前劇烈陽線
        2. 昨日十字線(實體線可忽略且同時存在上下影線)
        3. 今日劇烈陰線
        4. 昨日十字線與兩日前和今日有實體上跳空
        => min(昨日收盤價, 昨日開盤價) > 兩日前收盤價 且
		   min(昨日收盤價, 昨日開盤價) > 今日開盤價
		5. 今日收盤價需插入兩日前陽線燭身一半以上
		=> 今日的收盤價 <= (兩日前開盤價 + 兩日前收盤價) / 2

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_evening_doji_star'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_evening_doji_star': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.doji(close_zero_threshold=close_zero_threshold,
                        shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = cct.is_black() & cct.long_body(violent_threshold=violent_threshold)
    cond_4a = ts_min(cct_1.open, cct_1.close) > cct_2.close
    cond_4b = ts_min(cct_1.open, cct_1.close) > cct.open
    cond_4 = cond_4a & cond_4b
    cond_5 = cct.close <= (cct_2.open + cct_2.close) / 2
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bearish_evening_doji_star({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_abandoned_baby(market_id: str, **kwargs):
    """bearish_abandoned_baby.

    規則：
        夜星棄嬰；
        1. 兩日前劇烈陽線
        2. 昨日十字線(實體線可忽略且同時存在上下影線)
        3. 今日劇烈陰線
        4. 昨日十字線與兩日前和今日有上跳空(最低價大於兩日前與今日最高價)
        => 昨日最低價 > 兩日前最高價 且
		   昨日最低價 > 今日最高價
		5. 今日收盤價需插入兩日前陽線燭身一半以上
		=> 今日的收盤價 <= (兩日前開盤價 + 兩日前收盤價) / 2

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_abandoned_baby'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_abandoned_baby': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.doji(close_zero_threshold=close_zero_threshold,
                        shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = cct.is_black() & cct.long_body(violent_threshold=violent_threshold)
    cond_4a = cct_1.low > cct_2.high
    cond_4b = cct_1.low > cct.high
    cond_4 = cond_4a & cond_4b
    cond_5 = cct.close <= (cct_2.open + cct_2.close) / 2
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bearish_abandoned_baby({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_tri_star(market_id: str, **kwargs):
    """bearish_tri_star.

    規則：
        空頭三星；
        1. 兩日前十字線(實體線可忽略且同時存在上下影線)
        2. 昨日十字線(實體線可忽略且同時存在上下影線)
        3. 今日十字線(實體線可忽略且同時存在上下影線)
        4. 昨日十字線與兩日前和今日有上跳空(最低價大於兩日前與今日最高價)
        => 昨日最低價 > 兩日前最高價 且
		   昨日最低價 > 今日最高價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_tri_star'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_tri_star': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.doji(close_zero_threshold=close_zero_threshold,
                        shadow_ignore_threshold=shadow_ignore_threshold)
    cond_2 = cct_1.doji(close_zero_threshold=close_zero_threshold,
                        shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = cct.doji(close_zero_threshold=close_zero_threshold,
                      shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4a = cct_1.low > cct_2.high
    cond_4b = cct_1.low > cct.high
    cond_4 = cond_4a & cond_4b
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bearish_tri_star({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_identical_three_crows(market_id: str, **kwargs):
    """bearish_identical_three_crows.

    規則：
        疊影三鴉；
        1. 兩日前劇烈陰線
        2. 昨日劇烈陰線
        3. 今日劇烈陰線
        4. 昨日開盤價與兩日前收盤價相近
		5. 今日開盤價與昨日收盤價相近

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_identical_three_crows'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_identical_three_crows': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_4 = _is_close(cct_1.open, cct_2.close,
                       close_zero_threshold=close_zero_threshold)
    cond_5 = _is_close(cct.open, cct_1.close,
                       close_zero_threshold=close_zero_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bearish_identical_three_crows({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_upside_gap_two_crows(market_id: str, **kwargs):
    """bearish_upside_gap_two_crows.

    規則：
        雙鴉躍空；
        1. 兩日前劇烈陽線
        2. 昨日陰線
        3. 今日陰線
        4. 昨日最低價大於兩日前最高價(上跳空)
        5. 今日開盤價大於昨日最高價
        6. 今日收盤價低於昨日收盤價
        7. 今日收盤價大於第一日最高價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_upside_gap_two_crows'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_upside_gap_two_crows': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.black_body(close_zero_threshold)
    cond_3 = cct.black_body(close_zero_threshold)
    cond_4 = cct_1.low > cct_2.high
    cond_5 = cct.open > cct_1.high
    cond_6 = cct.close < cct_1.close
    cond_7 = cct.close > cct_2.high
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7
    ret.rename(f'{market_id}.doutze_bearish_upside_gap_two_crows({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_two_crows(market_id: str, **kwargs):
    """bearish_two_crows.

    規則：
        雙飛烏鴉；
        1. 兩日前劇烈陽線
        2. 昨日陰線
        3. 今日陰線
        4. 昨日最低價大於兩日前最高價(上跳空)
        5. 今日開盤價小於昨日開盤價
        6. 今日開盤價大於昨日收盤價
        7. 今日收盤價小於兩日前收盤價
        8. 今日收盤價大於兩日前開盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_two_crows'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_two_crows': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.black_body(close_zero_threshold)
    cond_3 = cct.black_body(close_zero_threshold)
    cond_4 = cct_1.low > cct_2.high
    cond_5 = cct.open < cct_1.open
    cond_6 = cct.open > cct_1.close
    cond_7 = cct.close < cct_2.close
    cond_8 = cct.close > cct_2.open
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 & cond_8
    ret.rename(f'{market_id}.doutze_bearish_two_crows({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_three_black_crows(market_id: str, **kwargs):
    """bearish_three_black_crows.

    規則：
        三飛烏鴉；
        1. 兩日前劇烈陰線
        2. 昨日劇烈陰線
        3. 今日劇烈陰線
        4. 昨日開盤價落於兩日前燭身
		=> 昨日開盤價 < 兩日前開盤價 且 昨日開盤價 > 兩日前收盤價
        5. 昨日收盤價與昨日最低價相近
        6. 今日開盤價落於昨日燭身
		=> 今日開盤價 < 昨日開盤價 且 今日開盤價 > 昨日收盤價
        7. 今日收盤價與今日最低價相近
        8. 兩日前實體與昨日實體長度相近
        9. 昨日實體與今日實體長度相近
        10.今日實體與兩日前實體長度相近

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.
	body_close_threshold : float
		兩實體線長度相近臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        body_close_threshold = kwargs['body_close_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_three_black_crows'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_three_black_crows': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.is_black() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.is_black() & cct.long_body(violent_threshold=violent_threshold)
    cond_4 = (cct_1.open < cct_2.open) & (cct_1.open > cct_2.close)
    cond_5 = _is_close(cct_1.close, cct_1.low,
                       close_zero_threshold=close_zero_threshold)
    cond_6 = (cct.open < cct_1.open) & (cct.open > cct_1.close)
    cond_7 = _is_close(cct.close, cct.low,
                       close_zero_threshold=close_zero_threshold)
    #cond_8 = abs(cct_2.real_body / cct_1.real_body - 1) <= body_close_threshold
    #cond_9 = abs(cct_1.real_body / cct.real_body - 1) <= body_close_threshold
    #cond_10 = abs(cct.real_body / cct_2.real_body - 1) <= body_close_threshold
    cond_8 = abs(cct_2.real_body - cct_1.real_body) <= cct_1.real_body * body_close_threshold
    cond_9 = abs(cct_1.real_body - cct.real_body) <= cct.real_body * body_close_threshold
    cond_10 = abs(cct.real_body - cct_2.real_body) <= cct_2.real_body * body_close_threshold
    ret = (cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 &
           cond_8 & cond_9 & cond_10)
    ret.rename(f'{market_id}.doutze_bearish_three_black_crows({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_advance_block(market_id: str, **kwargs):
    """bearish_advance_block.

    規則：
        大敵當前；
        1. 兩日前劇烈陽線
        2. 昨日陽線
        3. 今日陽線
        4. 昨日收盤價大於兩日前收盤價
        5. 昨日燭身小於兩日前燭身
        6. 今日收盤價大於昨日收盤價
        7. 今日燭身小於昨日燭身
        8. 今日有長上影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.
	long_shadow_threshold : float
		長影線臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_advance_block'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_advance_block': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.white_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.white_body(close_zero_threshold=close_zero_threshold)
    cond_4 = cct_1.close > cct_2.close
    cond_5 = cct_1.real_body < cct_2.real_body
    cond_6 = cct.close > cct_1.close
    cond_7 = cct.real_body < cct_1.real_body
    cond_8 = cct.long_upper_shadow(long_shadow_threshold=long_shadow_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 & cond_8
    ret.rename(f'{market_id}.doutze_bearish_advance_block({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_deliberation_block(market_id: str, **kwargs):
    """bearish_deliberation_block.

    規則：
        步步為營；
        1. 兩日前劇烈陽線
        2. 昨日劇烈陽線
        3. 今日陽線
        4. 昨日收盤價大於兩日前收盤價
        5. 昨日燭身與兩日前燭身相似
        6. 今日收盤價大於昨日收盤價
        7. 今日實體線存在但很小 => 短實體線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.
	body_close_threshold : float
		兩實體線長度相近臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        body_close_threshold = kwargs['body_close_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_deliberation_block'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_deliberation_block': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.is_white() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.white_body(close_zero_threshold=close_zero_threshold)
    cond_4 = cct_1.close > cct_2.close
    #cond_5 = abs(cct_1.real_body / cct_2.real_body - 1) <= body_close_threshold
    cond_5 = abs(cct_1.real_body - cct_2.real_body) <= cct_2.real_body * body_close_threshold
    cond_6 = cct.close > cct_1.close
    cond_7 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7
    ret.rename(f'{market_id}.doutze_bearish_deliberation_block({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_shooting_star(market_id: str, **kwargs):
    """bearish_shooting_star.

    規則：
        射擊之星；
        1. 長上影線
		2. 短實體線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.
	long_shadow_threshold : float
		長影線臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_shooting_star'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_shooting_star': {esp}")
    cct = get_candle(market_id, period_type)
    cond_1 = cct.long_upper_shadow(long_shadow_threshold=long_shadow_threshold)
    cond_2 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    ret = cond_1 & cond_2
    ret.rename(f'{market_id}.doutze_bearish_shooting_star({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_doji_star(market_id: str, **kwargs):
    """bearish_doji_star.

    規則：
        頂部星形十字；
        1. 昨日劇烈陽線
        2. 今日十字線
        3. 今日最低價高於昨日最高價(上跳空)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
	shadow_ignore_threshold : float
		忽略影線臨界值.
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_doji_star'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_doji_star': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = cct_1.is_white() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_2 = cct.doji(close_zero_threshold=close_zero_threshold,
                      shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = cct.low > cct_1.high
    ret = cond_1 & cond_2 & cond_3
    ret.rename(f'{market_id}.doutze_bearish_doji_star({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_kicking(market_id: str, **kwargs):
    """bearish_kicking.

    規則：
        看跌反衝/空頭反撲
        1. 昨日陽線
        2. 今日陰線
        3. 今日最高價小於昨日最低價(下跳空)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_kicking'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_kicking': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = cct_1.white_body(close_zero_threshold)
    cond_2 = cct.black_body(close_zero_threshold)
    cond_3 = cct.high < cct_1.low
    ret = cond_1 & cond_2 & cond_3
    ret.rename(f'{market_id}.doutze_bearish_kicking({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_three_outside_up(market_id: str, **kwargs):
    """bullish_three_outside_up.

    規則：
        外側三日上升；
        1. 兩日前陰線
        2. 昨日陽線
        3. 今日陽線
        4. 昨日陽線吞噬兩日前陰線燭身
        => 昨日收盤價 >= 兩日前開盤價 且 昨日開盤價 <= 兩日前收盤價
        5. 今日收盤價高於昨日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_three_outside_up'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_three_outside_up': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.black_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct_1.white_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.white_body(close_zero_threshold=close_zero_threshold)
    cond_4a = cct_1.close >= cct_2.open
    cond_4b = cct_1.open <= cct_2.close
    cond_4 = cond_4a & cond_4b
    cond_5 = cct.close > cct_1.close
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bullish_three_outside_up({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_three_inside_up(market_id: str, **kwargs):
    """bullish_three_inside_up.

    規則：
        內困三日翻紅；
        1. 兩日前劇烈陰線
        2. 昨日陽線
        3. 今日陽線
        4. 昨日陽線被兩日前陰線燭身吞噬
        => 昨日收盤價 <= 兩日前開盤價 且 昨日開盤價 >= 兩日前收盤價
		5. 今日收盤價 > 昨日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
	violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_three_inside_up'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_three_inside_up': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.white_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.white_body(close_zero_threshold=close_zero_threshold)
    cond_4a = cct_1.close <= cct_2.open
    cond_4b = cct_1.open >= cct_2.close
    cond_4 = cond_4a & cond_4b
    cond_5 = cct.close > cct_1.close
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bullish_three_inside_up({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_morning_star(market_id: str, **kwargs):
    """bullish_morning_star.

    規則：
        晨星；
        1. 兩日前劇烈陰線
        2. 昨日陰線或陽線(See 5.)
        3. 今日劇烈陽線
        4. 昨日陰或陽線與兩日前和今日有實體下跳空
        => max(昨日收盤價, 昨日開盤價) < 兩日前收盤價 且
		   max(昨日收盤價, 昨日開盤價) < 今日開盤價
		5. 昨日為短實體線
		6. 今日燭身需插入兩日前燭身一半以上
		=> 今日的收盤價 >= (兩日前開盤價 + 兩日前收盤價) / 2

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
	violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_morning_star'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_morning_star': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    # 條件2被條件5包含
    # cond_2a = cct_1.black_body(close_zero_threshold=close_zero_threshold)
    # cond_2b = cct_1.white_body(close_zero_threshold=close_zero_threshold)
    # cond_2 = cond_2a | cond_2b
    cond_3 = cct.is_white() & cct.long_body(violent_threshold=violent_threshold)
    cond_4a = ts_max(cct_1.open, cct_1.close) < cct_2.close
    cond_4b = ts_max(cct_1.open, cct_1.close) < cct.open
    cond_4 = cond_4a & cond_4b
    cond_5 = cct_1.short_body(violent_threshold=violent_threshold,
                              close_zero_threshold=close_zero_threshold)
    cond_6 = cct.close >= (cct_2.open + cct_2.close) / 2
    # ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6
    ret = cond_1 & cond_3 & cond_4 & cond_5 & cond_6
    ret.rename(f'{market_id}.doutze_bullish_morning_star({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_morning_doji_star(market_id: str, **kwargs):
    """bullish_morning_doji_star.

    規則：
        晨星十字；
        1. 兩日前劇烈陰線
        2. 昨日十字線(實體線可忽略且同時存在上下影線)
        3. 今日劇烈陽線
        4. 昨日十字線與兩日前和今日有實體下跳空
        => max(昨日收盤價, 昨日開盤價) < 兩日前收盤價 且
		   max(昨日收盤價, 昨日開盤價) < 今日開盤價
		5. 今日收盤價需插入兩日前陰線燭身一半以上
		=> 今日的收盤價 >= (兩日前開盤價 + 兩日前收盤價) / 2

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_morning_doji_star'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_morning_doji_star': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.doji(close_zero_threshold=close_zero_threshold,
                        shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = cct.is_white() & cct.long_body(violent_threshold=violent_threshold)
    cond_4a = ts_max(cct_1.open, cct_1.close) < cct_2.close
    cond_4b = ts_max(cct_1.open, cct_1.close) < cct.open
    cond_4 = cond_4a & cond_4b
    cond_5 = cct.close >= (cct_2.open + cct_2.close) / 2
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bullish_morning_doji_star({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_abandoned_baby_bottom(market_id: str, **kwargs):
    """bullish_abandoned_baby_bottom.

    規則：
        晨星棄嬰；
        1. 兩日前劇烈陰線
        2. 昨日十字線(實體線可忽略且同時存在上下影線)
        3. 今日劇烈陽線
        4. 昨日十字線與兩日前和今日有下跳空(最高價小於兩日前與今日最低價)
        => 昨日最高價 < 兩日前最低價 且
		   昨日最高價 < 今日最低價
		5. 今日收盤價需插入兩日前陰線燭身一半以上
		=> 今日的收盤價 >= (兩日前開盤價 + 兩日前收盤價) / 2

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	violent_threshold : float
        劇烈漲跌臨界值.
	close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_abandoned_baby_bottom'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_abandoned_baby_bottom': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.doji(close_zero_threshold=close_zero_threshold,
                        shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = cct.is_white() & cct.long_body(violent_threshold=violent_threshold)
    cond_4a = cct_1.high < cct_2.low
    cond_4b = cct_1.high < cct.low
    cond_4 = cond_4a & cond_4b
    cond_5 = cct.close >= (cct_2.open + cct_2.close) / 2
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bullish_abandoned_baby_bottom({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_tri_star(market_id: str, **kwargs):
    """bullish_tri_star.

    規則：
        多頭三星；
        1. 兩日前十字線(實體線可忽略且同時存在上下影線)
        2. 昨日十字線(實體線可忽略且同時存在上下影線)
        3. 今日十字線(實體線可忽略且同時存在上下影線)
        4. 昨日十字線與兩日前和今日有下跳空(最高價小於兩日前與今日最低價)
        => 昨日最高價 < 兩日前最低價 且
		   昨日最高價 < 今日最低價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
	close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_tri_star'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_tri_star': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.doji(close_zero_threshold=close_zero_threshold,
                        shadow_ignore_threshold=shadow_ignore_threshold)
    cond_2 = cct_1.doji(close_zero_threshold=close_zero_threshold,
                        shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = cct.doji(close_zero_threshold=close_zero_threshold,
                      shadow_ignore_threshold=shadow_ignore_threshold)
    cond_4a = cct_1.high < cct_2.low
    cond_4b = cct_1.high < cct.low
    cond_4 = cond_4a & cond_4b
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bullish_tri_star({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_three_white_samurai(market_id: str, **kwargs):
    """bullish_three_white_samurai.

    規則：
        三白武士；
        1. 兩日前劇烈陽線
        2. 昨日劇烈陽線
        3. 今日劇烈陽線
        4. 昨日開盤價與兩日前收盤價相近
        5. 昨日收盤價與昨日最高價相近
        6. 今日開盤價與昨日收盤價相近
        7. 今日收盤價與今日最高價相近

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_three_white_samurai'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_three_white_samurai': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.is_white() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.is_white() & cct.long_body(violent_threshold=violent_threshold)
    cond_4 = _is_close(cct_1.open, cct_2.close,
                       close_zero_threshold=close_zero_threshold)
    cond_5 = _is_close(cct_1.close, cct_1.high,
                       close_zero_threshold=close_zero_threshold)
    cond_6 = _is_close(cct.open, cct_1.close,
                       close_zero_threshold=close_zero_threshold)
    cond_7 = _is_close(cct.close, cct.high,
                       close_zero_threshold=close_zero_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7
    ret.rename(f'{market_id}.doutze_bullish_three_white_samurai({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_three_white_soldiers(market_id: str, **kwargs):
    """bullish_three_white_soldiers.

    規則：
        三白兵；
        1. 兩日前劇烈陽線
        2. 昨日劇烈陽線
        3. 今日劇烈陽線
        4. 昨日開盤價與兩日前收盤價相近
        5. 今日開盤價與昨日收盤價相近

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_three_white_soldiers'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_three_white_soldiers': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.is_white() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.is_white() & cct.long_body(violent_threshold=violent_threshold)
    cond_4 = _is_close(cct_1.open, cct_2.close,
                       close_zero_threshold=close_zero_threshold)
    cond_5 = _is_close(cct.open, cct_1.close,
                       close_zero_threshold=close_zero_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bullish_three_white_soldiers({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_homing_pigeon(market_id: str, **kwargs):
    """bullish_homing_pigeon.

    規則：
        飛鴿歸巢；
        1. 昨日劇烈陰線
        2. 今日陰線
        3. 今日最高價小於昨日開盤價
        4. 今日最低價大於昨日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_homing_pigeon'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_homing_pigeon': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = cct_1.is_black() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_2 = cct.black_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.high < cct_1.open
    cond_4 = cct.low > cct_1.close
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bullish_homing_pigeon({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_inversed_hammer(market_id: str, **kwargs):
    """bullish_inversed_hammer.

    規則：
        倒狀槌子；
        1. 昨日大陰線
        2. 今日倒陰或陽線槌子線
        3. 今日實體小(see 2.)
        4. 今日與昨日發生實體下跳空
        => max(今日收盤價, 今日開盤價) < 昨日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    See Also
    --------

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_inversed_hammer'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_inversed_hammer': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = _is_long_black(cct_1,
                            shadow_ignore_threshold=shadow_ignore_threshold,
                            violent_threshold=violent_threshold)
    cond_2a = _is_black_inverse_hammer(cct,
                                       shadow_ignore_threshold=shadow_ignore_threshold,
                                       violent_threshold=violent_threshold,
                                       close_zero_threshold=close_zero_threshold)
    cond_2b = _is_white_inverse_hammer(cct,
                                       shadow_ignore_threshold=shadow_ignore_threshold,
                                       violent_threshold=violent_threshold,
                                       close_zero_threshold=close_zero_threshold)
    cond_2 = cond_2a | cond_2b
    # condition already satisfied by 2
    # cond_3 = cct.short_body(violent_threshold=violent_threshold,
    #                         close_zero_threshold=close_zero_threshold)
    cond_4 = ts_max(cct.open, cct.close) < cct_1.close
    # ret = cond_1 & cond_2 & cond_3 & cond_4
    ret = cond_1 & cond_2 & cond_4
    ret.rename(f'{market_id}.doutze_bullish_inversed_hammer({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_doji_star(market_id: str, **kwargs):
    """bullish_doji_star.

    規則：
        底部星形十字；
        1. 昨日劇烈陰線
        2. 今日任一種十字線(十字線、上影線較長的十字線或下影線較長的十字線)
        3. 今日最高價低於昨日最低價(下跳空)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    See Also
    --------

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_doji_star'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_doji_star': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = cct_1.is_black() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_2 = cct.doji(close_zero_threshold=close_zero_threshold,
                      shadow_ignore_threshold=shadow_ignore_threshold)
    cond_3 = cct.high < cct_1.low
    ret = cond_1 & cond_2 & cond_3
    ret.rename(f'{market_id}.doutze_bullish_doji_star({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_stick_sandwich(market_id: str, **kwargs):
    """bullish_stick_sandwich.

    規則：
        三明治；
        1. 兩日前陰線
        2. 昨日陽線
        3. 今日陰線
        4. 昨日無上影線
        5. 昨日開盤價大於兩日前收盤價
        6. 昨日收盤價接近昨日最高價
        7. 今日開盤價大於昨日收盤價
        8. 今日收盤價小於兩日前收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
    close_zero_threshold : float
        持平臨界值.

    See Also
    --------

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_stick_sandwich'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_stick_sandwich': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.black_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct_1.white_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.black_body(close_zero_threshold=close_zero_threshold)
    cond_4 = cct_1.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_5 = cct_1.open > cct_2.close
    cond_6 = _is_close(cct_1.close, cct_1.high,
                       close_zero_threshold=close_zero_threshold)
    cond_7 = cct.open > cct_1.close
    cond_8 = cct.close < cct_2.close
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 & cond_8
    ret.rename(f'{market_id}.doutze_bullish_stick_sandwich({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_unique_three_river_bottom(market_id: str, **kwargs):
    """bullish_unique_three_river_bottom.

    規則：
        獨特三河床；
        1. 兩日前劇烈陰線
        2. 昨日陰線(See 4.)
        3. 今日陽線
        4. 昨日為陰線槌子線
        5. 昨日開盤價大於第一日開盤價
        6. 昨日最低價低於第一日最低價
        7. 昨日無上影線(See 4.)
        8. 今日開盤價低於昨日收盤價(see 3., 10.)
        9. 今日開盤價高於昨日最低價
        10.今日收盤價低於第二日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_unique_three_river_bottom'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_unique_three_river_bottom': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.white_body(close_zero_threshold=close_zero_threshold)
    cond_4 = _is_black_hammer(cct_1,
                              shadow_ignore_threshold=shadow_ignore_threshold,
                              violent_threshold=violent_threshold,
                              close_zero_threshold=close_zero_threshold)
    cond_5 = cct_1.open > cct_2.open
    cond_6 = cct_1.low > cct_2.low
    # condition 3 and 10 include condition 8
    # cond_8 = cct.open < cct_1.close
    cond_9 = cct.open > cct_1.low
    cond_10 = cct.close < cct_1.close
    # ret = cond_1 & cond_3 & cond_4 & cond_5 & cond_6 & cond_8 & cond_9 & cond_10
    ret = cond_1 & cond_3 & cond_4 & cond_5 & cond_6 & cond_9 & cond_10
    ret.rename(f'{market_id}.doutze_bullish_unique_three_river_bottom({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_three_stars_in_the_south(market_id: str, **kwargs):
    """bullish_three_stars_in_the_south.

    規則：
        南方三星；
        1. 兩日前長陰線
        2. 昨日長陰線
        3. 今日短陰線
        4. 兩日前無上影線
        5. 兩日前長下影線
        6. 昨日無上影線
        7. 昨日長下影線
        8. 昨日最高價低於兩日前最高價
        9. 昨日最低價高於兩日前最低價
        10.今日開盤價低於昨日最高價
        11.今日收盤價高於昨日最低價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.
	long_shadow_threshold : float
        長影線臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_three_stars_in_the_south'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_three_stars_in_the_south': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.is_black() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.is_black() & cct.short_body(violent_threshold=violent_threshold,
                                             close_zero_threshold=close_zero_threshold)
    cond_4 = cct_2.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_5 = cct_2.long_lower_shadow(long_shadow_threshold=long_shadow_threshold)
    cond_6 = cct_1.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_7 = cct_1.long_lower_shadow(long_shadow_threshold=long_shadow_threshold)
    cond_8 = cct_1.high < cct_2.high
    cond_9 = cct_1.low > cct_2.low
    cond_10 = cct.open < cct_1.high
    cond_11 = cct.close > cct_1.low
    ret = (cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 &
           cond_8 & cond_9 & cond_10 & cond_11)
    ret.rename(f'{market_id}.doutze_bullish_three_stars_in_the_south({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_concealing_baby_swallow(market_id: str, **kwargs):
    """bullish_concealing_baby_swallow.

    規則：
        閨中乳燕；
        1. 三日前劇烈陰線
        2. 兩日前劇烈陰線
        3. 昨日陰線
        4. 今日陰線
        5. 昨日開盤價小於兩日前最低價
        6. 昨日最高價高於兩日前收盤價
        7. 今日開盤價大於昨日最高價
        8. 今日收盤價低於第三日最低價
        9. 今日最低價低於三日前最低價
        10.今日最低價低於兩日前最低價
        11.今日最低價低於昨日最低價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_concealing_baby_swallow'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_concealing_baby_swallow': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cct_3 = cct.shift(3)
    cond_1 = cct_3.is_black() & cct_3.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_3 = cct_1.black_body(close_zero_threshold)
    cond_4 = cct.black_body(close_zero_threshold)
    cond_5 = cct_1.open < cct_2.low
    cond_6 = cct_1.high > cct_2.close
    cond_7 = cct.open > cct_1.high
    cond_8 = cct.close < cct_1.low
    cond_9 = cct.low < cct_3.low
    cond_10 = cct.low < cct_2.low
    cond_11 = cct.low < cct_1.low
    ret = (cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 &
           cond_8 & cond_9 & cond_10 & cond_11)
    ret.rename(f'{market_id}.doutze_bullish_concealing_baby_swallow({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_kicking(market_id: str, **kwargs):
    """bullish_kicking.

    規則：
        看漲反衝/多頭反撲；
        1. 昨日陰線
        2. 今日陽線
        3. 今日最低價大於昨日最高價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_kicking'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_kicking': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = cct_1.black_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct.white_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.low > cct_1.high
    ret = cond_1 & cond_2 & cond_3
    ret.rename(f'{market_id}.doutze_bullish_kicking({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_ladder_bottom(market_id: str, **kwargs):
    """bullish_ladder_bottom.

    規則：
        梯底；
        1. 四日前陰線
        2. 三日前陰線
        3. 兩日前陰線
        4. 昨日短陰線
        5. 今日陽線
        6. 四日前無上影線
        7. 三日前無上影線
        8. 三日前收盤價低於四日前收盤價
        9. 三日前最低價低於四日前最低價
        10.兩日前無上影線
        11.兩日前收盤價低於三日前收盤價
        12.兩日前最低價低於三日前最低價
        13.昨日長上影線
        14.昨日無下影線
        15.昨日收盤價低於兩日前收盤價
        16.昨日最高價高於兩日前最高價
        17.今日開盤價大於昨日開盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.
	long_shadow_threshold : float
        長影線臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_ladder_bottom'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_ladder_bottom': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cct_3 = cct.shift(3)
    cct_4 = cct.shift(4)
    cond_1 = cct_4.black_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct_3.black_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct_2.black_body(close_zero_threshold=close_zero_threshold)
    cond_4 = cct_1.is_black() & cct_1.short_body(violent_threshold=violent_threshold,
                                                 close_zero_threshold=close_zero_threshold)
    cond_5 = cct.white_body(close_zero_threshold=close_zero_threshold)
    cond_6 = cct_4.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_7 = cct_3.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_8 = cct_3.close < cct_4.close
    cond_9 = cct_3.low < cct_4.low
    cond_10 = cct_2.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_11 = cct_2.close < cct_3.close
    cond_12 = cct_2.low < cct_3.low
    cond_13 = cct_1.long_upper_shadow(long_shadow_threshold=long_shadow_threshold)
    cond_14 = cct_1.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_15 = cct_1.close < cct_2.close
    cond_16 = cct_1.high > cct_2.high
    cond_17 = cct.open > cct_1.open
    ret = (cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 & cond_8 & cond_9 &
           cond_10 & cond_11 & cond_12 & cond_13 & cond_14 & cond_15 & cond_16 & cond_17)
    ret.rename(f'{market_id}.doutze_bullish_ladder_bottom({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_matching_low(market_id: str, **kwargs):
    """bullish_matching_low.

    規則：
        低價配；
        1. 昨日劇烈陰線
        2. 今日陰線
        3. 今日開盤價大於昨日收盤價
        4. 今日收盤價接近昨日收盤價
        5. 今日有上影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_matching_low'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_matching_low': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = cct_1.is_black() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_2 = cct.black_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.open > cct_1.close
    cond_4 = _is_close(cct.close, cct_1.close,
                       close_zero_threshold=close_zero_threshold)
    cond_5 = ~cct.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bullish_matching_low({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_downside_gap_two_crows(market_id: str, **kwargs):
    """bullish_downside_gap_two_crows.

    規則：
        顛倒的雙鴉躍空；
        1. 兩日前劇烈陰線
        2. 昨日陽線
        3. 今日陽線
        4. 昨日最高價小於兩日前最低價(下跳空)
        5. 今日開盤價小於昨日最低價
        6. 今日收盤價大於昨日收盤價
        7. 今日收盤價小於第一日最低價，未將缺口填補

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_downside_gap_two_crows'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_downside_gap_two_crows': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.white_body(close_zero_threshold)
    cond_3 = cct.white_body(close_zero_threshold)
    cond_4 = cct_1.high < cct_2.low
    cond_5 = cct.open < cct_1.low
    cond_6 = cct.close > cct_1.close
    cond_7 = cct.close < cct_2.low
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7
    ret.rename(f'{market_id}.doutze_bullish_downside_gap_two_crows({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_two_crows(market_id: str, **kwargs):
    """bullish_two_crows.

    規則：
        顛倒的雙飛烏鴉；
        1. 兩日前劇烈陰線
        2. 昨日陽線
        3. 今日陽線
        4. 昨日最高價小於兩日前最低價(下跳空)
        5. 今日開盤價大於昨日開盤價
        6. 今日開盤價小於昨日收盤價
        7. 今日收盤價大於兩日前收盤價
        8. 今日收盤價小於兩日前開盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_two_crows'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_two_crows': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.white_body(close_zero_threshold)
    cond_3 = cct.white_body(close_zero_threshold)
    cond_4 = cct_1.high < cct_2.low
    cond_5 = cct.open > cct_1.open
    cond_6 = cct.open < cct_1.close
    cond_7 = cct.close > cct_2.close
    cond_8 = cct.close < cct_2.open
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 & cond_8
    ret.rename(f'{market_id}.doutze_bullish_two_crows({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_advance_block(market_id: str, **kwargs):
    """bullish_advance_block.

    規則：
        顛倒的大敵當前；
        1. 兩日前劇烈陰線
        2. 昨日陰線
        3. 今日陰線
        4. 昨日收盤價小於兩日前收盤價
        5. 昨日燭身小於兩日前燭身
        6. 今日收盤價小於昨日收盤價
        7. 今日燭身小於昨日燭身
        8. 今日有長下影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.
	long_shadow_threshold : float
		長影線臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_advance_block'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_advance_block': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.black_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.black_body(close_zero_threshold=close_zero_threshold)
    cond_4 = cct_1.close < cct_2.close
    cond_5 = cct_1.real_body < cct_2.real_body
    cond_6 = cct.close < cct_1.close
    cond_7 = cct.real_body < cct_1.real_body
    cond_8 = cct.long_lower_shadow(long_shadow_threshold=long_shadow_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 & cond_8
    ret.rename(f'{market_id}.doutze_bullish_advance_block({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bullish_deliberation_block(market_id: str, **kwargs):
    """bullish_deliberation_block.

    規則：
        顛倒的步步為營；
        1. 兩日前劇烈陰線
        2. 昨日劇烈陰線
        3. 今日陰線
        4. 昨日收盤價小於兩日前收盤價
        5. 昨日燭身與兩日前燭身相似
        6. 今日收盤價小於昨日收盤價
        7. 今日實體線存在但很小

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.
	body_close_threshold : float
		兩實體線長度相近臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        body_close_threshold = kwargs['body_close_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bullish_deliberation_block'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bullish_deliberation_block': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_black() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.is_black() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.black_body(close_zero_threshold=close_zero_threshold)
    cond_4 = cct_1.close < cct_2.close
    #cond_5 = abs(cct_1.real_body / cct_2.real_body - 1) <= body_close_threshold
    cond_5 = abs(cct_1.real_body - cct_2.real_body) <= cct_2.real_body * body_close_threshold
    cond_6 = cct.close < cct_1.close
    cond_7 = cct.short_body(violent_threshold=violent_threshold,
                            close_zero_threshold=close_zero_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7
    ret.rename(f'{market_id}.doutze_bullish_deliberation_block({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_unique_three_river_bottom(market_id: str, **kwargs):
    """bearish_unique_three_river_bottom.

    規則：
        顛倒的獨特三河床；
        1. 兩日前劇烈陽線
        2. 昨日陽線(See 4.)
        3. 今日陰線
        4. 昨日為倒陽線槌子線
        5. 昨日開盤價小於第一日開盤價
        6. 昨日最高價大於第一日最高價
        7. 昨日無下影線(See 4.)
        8. 今日開盤價高於昨日收盤價(See 3., 10.)
        9. 今日開盤價低於昨日最高價
        10.今日收盤價高於第二日收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_unique_three_river_bottom'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_unique_three_river_bottom': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.black_body(close_zero_threshold=close_zero_threshold)
    cond_4 = _is_white_inverse_hammer(cct_1,
                                      shadow_ignore_threshold=shadow_ignore_threshold,
                                      violent_threshold=violent_threshold,
                                      close_zero_threshold=close_zero_threshold)
    cond_5 = cct_1.open < cct_2.open
    cond_6 = cct_1.high > cct_2.high
    # cond 3 and cond 10 include cond 8
    # cond_8 = cct.open > cct_1.close
    cond_9 = cct.open < cct_1.high
    cond_10 = cct.close > cct_1.close
    # ret = cond_1 & cond_3 & cond_4 & cond_5 & cond_6 & cond_8 & cond_9 & cond_10
    ret = cond_1 & cond_3 & cond_4 & cond_5 & cond_6 & cond_9 & cond_10
    ret.rename(f'{market_id}.doutze_bearish_unique_three_river_bottom({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_stick_sandwich(market_id: str, **kwargs):
    """bearish_stick_sandwich.

    規則：
        顛倒的三明治；
        1. 兩日前陽線
        2. 昨日陰線
        3. 今日陽線
        4. 昨日無下影線
        5. 昨日開盤價小於兩日前收盤價
        6. 昨日收盤價接近昨日最低價
        7. 今日開盤價小於昨日收盤價
        8. 今日收盤價大於兩日前收盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
    close_zero_threshold : float
        持平臨界值.

    See Also
    --------

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_stick_sandwich'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_stick_sandwich': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.white_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct_1.black_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.white_body(close_zero_threshold=close_zero_threshold)
    cond_4 = cct_1.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_5 = cct_1.open < cct_2.close
    cond_6 = _is_close(cct_1.close, cct_1.low,
                       close_zero_threshold=close_zero_threshold)
    cond_7 = cct.open < cct_1.close
    cond_8 = cct.close > cct_2.close
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 & cond_8
    ret.rename(f'{market_id}.doutze_bearish_stick_sandwich({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_three_stars_in_the_south(market_id: str, **kwargs):
    """bearish_three_stars_in_the_south.

    規則：
        顛倒的南方三星；
        1. 兩日前長陽線
        2. 昨日長陽線
        3. 今日短陽線
        4. 兩日前無下影線
        5. 兩日前長上影線
        6. 昨日無下影線
        7. 昨日長上影線
        8. 昨日最低價高於兩日前最低價
        9. 昨日最高價低於兩日前最高價
        10.今日開盤價高於昨日最低價
        11.今日收盤價低於昨日最高價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值(r1).
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.
	long_shadow_threshold : float
        長影線臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_three_stars_in_the_south'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_three_stars_in_the_south': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cond_1 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_1.is_white() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_3 = cct.is_white() & cct.short_body(violent_threshold=violent_threshold,
                                             close_zero_threshold=close_zero_threshold)
    cond_4 = cct_2.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_5 = cct_2.long_upper_shadow(long_shadow_threshold=long_shadow_threshold)
    cond_6 = cct_1.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_7 = cct_1.long_upper_shadow(long_shadow_threshold=long_shadow_threshold)
    cond_8 = cct_1.low > cct_2.low
    cond_9 = cct_1.high < cct_2.high
    cond_10 = cct.open > cct_1.low
    cond_11 = cct.close < cct_1.high
    ret = (cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 &
           cond_8 & cond_9 & cond_10 & cond_11)
    ret.rename(f'{market_id}.doutze_bearish_three_stars_in_the_south({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_concealing_baby_swallow(market_id: str, **kwargs):
    """bearish_concealing_baby_swallow.

    規則：
        顛倒的閨中乳燕；
        1. 三日前劇烈陽線
        2. 兩日前劇烈陽線
        3. 昨日陽線
        4. 今日陽線
        5. 昨日開盤價大於兩日前最低價
        6. 昨日最低價低於兩日前收盤價
        7. 今日開盤價低於昨日最低價
        8. 今日收盤價大於昨日最高價
        9. 今日最高價大於三日前最高價
        10.今日最高價大於兩日前最高價
        11.今日最高價大於昨日最高價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_concealing_baby_swallow'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_concealing_baby_swallow': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cct_3 = cct.shift(3)
    cond_1 = cct_3.is_white() & cct_3.long_body(violent_threshold=violent_threshold)
    cond_2 = cct_2.is_white() & cct_2.long_body(violent_threshold=violent_threshold)
    cond_3 = cct_1.white_body(close_zero_threshold)
    cond_4 = cct.white_body(close_zero_threshold)
    cond_5 = cct_1.open > cct_2.low
    cond_6 = cct_1.low < cct_2.close
    cond_7 = cct.open < cct_1.low
    cond_8 = cct.close > cct_1.high
    cond_9 = cct.high > cct_3.high
    cond_10 = cct.high > cct_2.high
    cond_11 = cct.high > cct_1.high
    ret = (cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 &
           cond_8 & cond_9 & cond_10 & cond_11)
    ret.rename(f'{market_id}.doutze_bearish_concealing_baby_swallow({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_homing_pigeon(market_id: str, **kwargs):
    """bearish_homing_pigeon.

    規則：
        顛倒的飛鴿歸巢；
        1. 昨日劇烈陽線
        2. 今日陽線
        3. 今日最高價小於昨日收盤價
        4. 今日最低價大於昨日開盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_homing_pigeon'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_homing_pigeon': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = cct_1.is_white() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_2 = cct.white_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.high < cct_1.close
    cond_4 = cct.low > cct_1.open
    ret = cond_1 & cond_2 & cond_3 & cond_4
    ret.rename(f'{market_id}.doutze_bearish_homing_pigeon({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_ladder_bottom(market_id: str, **kwargs):
    """bearish_ladder_bottom.

    規則：
        顛倒的梯底；
        1. 四日前陽線
        2. 三日前陽線
        3. 兩日前陽線
        4. 昨日短陽線
        5. 今日陰線
        6. 四日前無下影線
        7. 三日前無下影線
        8. 三日前收盤價高於四日前收盤價
        9. 三日前最高價高於四日前最高價
        10.兩日前無下影線
        11.兩日前收盤價高於三日前收盤價
        12.兩日前最高價高於三日前最高價
        13.昨日長下影線
        14.昨日無上影線
        15.昨日收盤價高於兩日前收盤價
        16.昨日最低價低於兩日前最低價
        17.今日開盤價小於昨日開盤價

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.
	long_shadow_threshold : float
        長影線臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
        long_shadow_threshold = kwargs['long_shadow_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_ladder_bottom'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_ladder_bottom': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cct_2 = cct.shift(2)
    cct_3 = cct.shift(3)
    cct_4 = cct.shift(4)
    cond_1 = cct_4.white_body(close_zero_threshold=close_zero_threshold)
    cond_2 = cct_3.white_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct_2.white_body(close_zero_threshold=close_zero_threshold)
    cond_4 = cct_1.is_white() & cct_1.short_body(violent_threshold=violent_threshold,
                                                 close_zero_threshold=close_zero_threshold)
    cond_5 = cct.black_body(close_zero_threshold=close_zero_threshold)
    cond_6 = cct_4.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_7 = cct_3.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_8 = cct_3.close > cct_4.close
    cond_9 = cct_3.high > cct_4.high
    cond_10 = cct_2.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_11 = cct_2.close > cct_3.close
    cond_12 = cct_2.high > cct_3.high
    cond_13 = cct_1.long_lower_shadow(long_shadow_threshold=long_shadow_threshold)
    cond_14 = cct_1.ignored_upper_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    cond_15 = cct_1.close > cct_2.close
    cond_16 = cct_1.low < cct_2.low
    cond_17 = cct.open < cct_1.open
    ret = (cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 & cond_8 & cond_9 &
           cond_10 & cond_11 & cond_12 & cond_13 & cond_14 & cond_15 & cond_16 & cond_17)
    ret.rename(f'{market_id}.doutze_bearish_ladder_bottom({kwargs})')
    return ret

@doutze_ccp_check
def doutze_bearish_matching_low(market_id: str, **kwargs):
    """bearish_matching_low.

    規則：
        顛倒的低價配；
        1. 昨日劇烈陽線
        2. 今日陽線
        3. 今日開盤價低於昨日收盤價
        4. 今日收盤價接近昨日收盤價
        5. 今日有下影線

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    shadow_ignore_threshold : float
        忽略影線臨界值.
    violent_threshold : float
        劇烈漲跌臨界值.
    close_zero_threshold : float
        持平臨界值.

    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        shadow_ignore_threshold = kwargs['shadow_ignore_threshold']
        violent_threshold = kwargs['violent_threshold']
        close_zero_threshold = kwargs['close_zero_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'doutze_bearish_matching_low'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'doutze_bearish_matching_low': {esp}")
    cct = get_candle(market_id, period_type)
    cct_1 = cct.shift(1)
    cond_1 = cct_1.is_white() & cct_1.long_body(violent_threshold=violent_threshold)
    cond_2 = cct.white_body(close_zero_threshold=close_zero_threshold)
    cond_3 = cct.open < cct_1.close
    cond_4 = _is_close(cct.close, cct_1.close,
                       close_zero_threshold=close_zero_threshold)
    cond_5 = ~cct.ignored_lower_shadow(shadow_ignore_threshold=shadow_ignore_threshold)
    ret = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    ret.rename(f'{market_id}.doutze_bearish_matching_low({kwargs})')
    return ret

####################### ↑↑↑ code review 2021-11-30 ↑↑↑ #######################
