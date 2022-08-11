# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 08:21:41 2022

@author: WaNiNi
"""

from abc import ABCMeta, abstractproperty
import datetime
from enum import Enum
import random
from typing import Callable, Dict, List, NamedTuple, Union

import numpy as np
import pandas as pd

from func._td import (# global variables and consts
                      MD_CACHE,
                      # functions
                      get_market_data, set_market_data_provider,
                      # classes
                      BooleanTimeSeries, MarketData, MarketDataProvider,
                      NumericTimeSeries, TimeUnit)
from func.common import (Dtype, MacroParam, ParamEnumBase, ParamEnumElement,
                         PeriodType, PlotInfo, Ptype)

from ...common import Macro

_PY_VERSION = '22-v1'
_DB_VERSION = '22-v1'
_PeriodTypes = PeriodType.type

class _LeadingTrends(ParamEnumBase):
    NONE = ParamEnumElement('none', '無', 0)
    BULLISH = ParamEnumElement('bullish', '上升趨勢', 1)
    BEARISH = ParamEnumElement('bearish', '下降趨勢', -1)
    STRICTLY_BULLISH = ParamEnumElement('strictly_bullish', '嚴格上升趨勢', 2)
    STRICTLY_BEARISH = ParamEnumElement('strictly_bearish', '嚴格下降趨勢', -2)

LeadingTrend = Dtype('klp_cc_leading_trend', _LeadingTrends)

class _CandleStickBase(metaclass=ABCMeta):
    @abstractproperty
    def open_(self) -> NumericTimeSeries:
        """Open."""

    @abstractproperty
    def high(self) -> NumericTimeSeries:
        """High."""

    @abstractproperty
    def low(self) -> NumericTimeSeries:
        """Low."""

    @abstractproperty
    def close(self) -> NumericTimeSeries:
        """Close."""

    @abstractproperty
    def realbody(self) -> NumericTimeSeries:
        """RealBody."""

    @abstractproperty
    def bodytop(self) -> NumericTimeSeries:
        """BodyTop."""

    @abstractproperty
    def bodybottom(self) -> NumericTimeSeries:
        """BodyBottom."""

    @abstractproperty
    def uppershadow(self) -> NumericTimeSeries:
        """UpperShadow."""

    @abstractproperty
    def amplitude(self) -> NumericTimeSeries:
        """Amplitude."""

    @abstractproperty
    def shadows(self) -> NumericTimeSeries:
        """Shadows."""

    def get_open(self) -> NumericTimeSeries:
        """get Open."""
        return self.open_

    def get_high(self) -> NumericTimeSeries:
        """get High."""
        return self.high

    def get_low(self) -> NumericTimeSeries:
        """get Low."""
        return self.high

    def get_close(self) -> NumericTimeSeries:
        """get Close."""
        return self.close

    def get_realbody(self) -> NumericTimeSeries:
        """get RealBody := abs(Close - Open) """
        return abs(self.close - self.open_)

    def get_bodytop(self) -> NumericTimeSeries:
        """get BodyTop := Close >= Open ? Close : Open """
        return NumericTimeSeries.max(self.close, self.open_)

    def get_bodybottom(self) -> NumericTimeSeries:
        """get BodyBottom := Close >= Open ? Open : Close """
        return NumericTimeSeries.min(self.close, self.open_)

    def get_uppershadow(self) -> NumericTimeSeries:
        """get UpperShadow := High - BodyTop """
        return self.high - self.bodytop

    def get_lowershadow(self) -> NumericTimeSeries:
        """get LowerShadow := BodyBottom - Low """
        return self.bodybottom - self.low

    def get_amplitude(self) -> NumericTimeSeries:
        """get Amplitude := High - Low """
        return self.high - self.low

    def get_shadows(self) -> NumericTimeSeries:
        """get Shadows := Amplitude - realbody """
        return self.amplitude - self.realbody


class _CandlestickProperty(NamedTuple):
    name: str
    func: Callable[[_CandleStickBase, ], NumericTimeSeries]

class _CandlestickProperties(_CandlestickProperty, Enum):
    OPEN = _CandlestickProperty('Open', _CandleStickBase.get_open)
    HIGH = _CandlestickProperty('High', _CandleStickBase.get_high)
    LOW = _CandlestickProperty('Low', _CandleStickBase.get_low)
    CLOSE = _CandlestickProperty('Close', _CandleStickBase.get_close)
    BODYTOP = _CandlestickProperty('BodyTop', _CandleStickBase.get_bodytop)
    BODYBOTTOM = _CandlestickProperty('BodyBottom',
                                      _CandleStickBase.get_bodybottom)
    REALBODY = _CandlestickProperty('RealBody', _CandleStickBase.get_realbody)
    UPPERSHADOW = _CandlestickProperty('UpperShadow',
                                       _CandleStickBase.get_uppershadow)
    LOWERSHADOW = _CandlestickProperty('LowerShadow',
                                       _CandleStickBase.get_lowershadow)
    SHADOWS = _CandlestickProperty('Shadows', _CandleStickBase.get_shadows)
    AMPLITUDE = _CandlestickProperty('Amplitude',
                                     _CandleStickBase.get_amplitude)


def _moving_average(data: NumericTimeSeries, period: int, tunit: TimeUnit
                    ) -> NumericTimeSeries:
    if period <= 1:
        return data
    return data.sampling(period, sunit=tunit).mean()

def _moving_sum(data: NumericTimeSeries, period: int, tunit: TimeUnit
                ) -> NumericTimeSeries:
    if period <= 1:
        return data
    return data.sampling(period, sunit=tunit).sum()

def _moving_max(data: NumericTimeSeries, lperiod: int, speriod: int,
                tunit: TimeUnit) -> NumericTimeSeries:
    if lperiod <= 1:
        return data
    def func(sample):
        if speriod == 0:
            return sample
        return sample[:-speriod].max(axis=0)
    return data.sampling(lperiod, sunit=tunit).apply(func, 'MH')

def _moving_min(data: NumericTimeSeries, lperiod: int, speriod: int,
                tunit: TimeUnit) -> NumericTimeSeries:
    if lperiod <= 1:
        return data
    def func(sample):
        if speriod == 0:
            return sample
        return sample[:-speriod].min(axis=0)
    return data.sampling(lperiod, sunit=tunit).apply(func, 'ML')

class _StaticMethods(Enum):
    MEAN = _moving_average


class _CandlestickSetting(NamedTuple):
    name: str
    feature: _CandlestickProperties
    period: Union[int, Dict[TimeUnit, int]]
    factor: float
    method: _StaticMethods = _StaticMethods.MEAN

_DAY = TimeUnit.DAY
_WEEK = TimeUnit.WEEK
_MONTH = TimeUnit.MONTH

class CandlestickSettings(_CandlestickSetting, Enum):
    NEAR = _CandlestickSetting('Near / Close To',
                               _CandlestickProperties.AMPLITUDE,
                               #5,
                               {_DAY: 30, _WEEK: 20, _MONTH: 12},
                               0.2,
                               _StaticMethods.MEAN)

    FAR_FROM = _CandlestickSetting('Far From',
                                   _CandlestickProperties.AMPLITUDE,
                                   #5,
                                   {_DAY: 30, _WEEK: 20, _MONTH: 12},
                                   0.6,
                                   _StaticMethods.MEAN)

    EQUAL_TO = _CandlestickSetting('Equal To',
                                   _CandlestickProperties.AMPLITUDE,
                                   #5,
                                   {_DAY: 30, _WEEK: 20, _MONTH: 12},
                                   0.05,
                                   _StaticMethods.MEAN)

    # consistent with equal-to because doji-body means close equals to open
    DOJI_BODY = _CandlestickSetting('Doji Body / Very-Short Body',
                                    _CandlestickProperties.AMPLITUDE,
                                    #10,
                                    {_DAY: 30, _WEEK: 20, _MONTH: 12},
                                    0.05,
                                    _StaticMethods.MEAN)

    SHORT_BODY = _CandlestickSetting('Short Body',
                                     _CandlestickProperties.REALBODY,
                                     #10,
                                     {_DAY: 30, _WEEK: 20, _MONTH: 12},
                                     1.0,
                                     _StaticMethods.MEAN)

    LONG_BODY = _CandlestickSetting('Long Body',
                                    _CandlestickProperties.REALBODY,
                                    #10,
                                    {_DAY: 30, _WEEK: 20, _MONTH: 12},
                                    1.0,
                                    _StaticMethods.MEAN)

    VERY_LONG_BODY = _CandlestickSetting('Very-Long Body',
                                         _CandlestickProperties.REALBODY,
                                         #10,
                                         {_DAY: 30, _WEEK: 20, _MONTH: 12},
                                         3.0,
                                         _StaticMethods.MEAN)

    WITHOUT_SHADOW = _CandlestickSetting('Without / Very-Short Shadow',
                                         _CandlestickProperties.AMPLITUDE,
                                         #10,
                                         {_DAY: 30, _WEEK: 20, _MONTH: 12},
                                         0.05,
                                         _StaticMethods.MEAN)

    SHORT_SHADOW = _CandlestickSetting('Short Shadow',
                                       _CandlestickProperties.SHADOWS,
                                       #10,
                                       {_DAY: 30, _WEEK: 20, _MONTH: 12},
                                       0.5,
                                       _StaticMethods.MEAN)

    LONG_SHADOW = _CandlestickSetting('Long Shadow',
                                      _CandlestickProperties.REALBODY,
                                      0,
                                      1.0,
                                      _StaticMethods.MEAN)

    VERY_LONG_SHADOW = _CandlestickSetting('Very-Long Shadow',
                                           _CandlestickProperties.REALBODY,
                                           0,
                                           2.0,
                                           _StaticMethods.MEAN)

LEADING_TREND_PERIODS = {TimeUnit.DAY: [1, 3, 6, 10],
                         TimeUnit.WEEK: [1, 2, 3, 5],
                         TimeUnit.MONTH: [1,2,3]}

class Candlestick(_CandleStickBase):
    def __init__(self, data: MarketData, name: str, tunit: TimeUnit):
        self._check_data(data)
        if not isinstance(name, str):
            raise ValueError(f"invalid `name`: {name}")
        if not isinstance(tunit, TimeUnit):
            raise ValueError(f"invalid `tunit`: {tunit}")
        self._properties = {_CandlestickProperties.OPEN: data.op,
                            _CandlestickProperties.HIGH: data.hp,
                            _CandlestickProperties.LOW: data.lp,
                            _CandlestickProperties.CLOSE: data.cp}
        self._tunit = tunit
        self._name = self._default_name(name, tunit)
        self._settings = {}
        self._derivatives = {}
        self._patterns = {}


    @property
    def values(self):
        ret = np.array([self.open_.to_pandas().values,
                        self.high.to_pandas().values,
                        self.low.to_pandas().values,
                        self.close.to_pandas().values]).T
        return ret

    @classmethod
    def _check_data(cls, data):
        if not isinstance(data, MarketData):
            raise ValueError(f"invalid `data`: {data}")
        if data.op is None:
            raise ValueError("`data` without `op`")
        if data.hp is None:
            raise ValueError("`data` without `hp`")
        if data.lp is None:
            raise ValueError("`data` without `lp`")

    @classmethod
    def _default_name(cls, name: str, tunit: TimeUnit) -> str:
        if tunit is TimeUnit.DAY:
            prefix = ''
        elif tunit is TimeUnit.WEEK:
            prefix = 'Weekly'
        elif tunit is TimeUnit.MONTH:
            prefix = 'Monthly'
        else:
            raise ValueError("invalid `tunit`: {tunit}")
        return f"{name}.{prefix}CandleStick"

    @property
    def name(self) -> str:
        """Name."""
        return self._name

    def _get_property(self, prop: _CandlestickProperties):
        if prop not in self._properties:
            self._properties[prop] = prop.func(self)
        ret = self._properties[prop]
        ret.rename(f'{self._name}.{prop.name}')
        return ret

    @property
    def open_(self) -> NumericTimeSeries:
        """Open."""
        return self._get_property(_CandlestickProperties.OPEN)

    @property
    def high(self) -> NumericTimeSeries:
        """High."""
        return self._get_property(_CandlestickProperties.HIGH)

    @property
    def low(self) -> NumericTimeSeries:
        """Low."""
        return self._get_property(_CandlestickProperties.LOW)

    @property
    def close(self) -> NumericTimeSeries:
        """Close."""
        return self._get_property(_CandlestickProperties.CLOSE)

    @property
    def realbody(self) -> NumericTimeSeries:
        """RealBody."""
        return self._get_property(_CandlestickProperties.REALBODY)

    @property
    def bodytop(self) -> NumericTimeSeries:
        """BodyTop."""
        return self._get_property(_CandlestickProperties.BODYTOP)

    @property
    def bodybottom(self) -> NumericTimeSeries:
        """BodyBottom."""
        return self._get_property(_CandlestickProperties.BODYBOTTOM)

    @property
    def uppershadow(self) -> NumericTimeSeries:
        """UpperShadow."""
        return self._get_property(_CandlestickProperties.UPPERSHADOW)

    @property
    def lowershadow(self) -> NumericTimeSeries:
        """LowerShadow."""
        return self._get_property(_CandlestickProperties.LOWERSHADOW)

    @property
    def amplitude(self) -> NumericTimeSeries:
        """Amplitude."""
        return self._get_property(_CandlestickProperties.AMPLITUDE)

    @property
    def shadows(self) -> NumericTimeSeries:
        """Shadows."""
        return self._get_property(_CandlestickProperties.SHADOWS)

    def _get_setting(self, setting: _CandlestickSetting) -> NumericTimeSeries:
        if setting.name not in self._settings:
            feature = self._get_property(setting.feature)
            tunit = self._tunit
            period = setting.period
            if isinstance(period, dict):
                period = period[tunit]
            factor = setting.factor
            ret = setting.method(feature, period, tunit) * factor
            if setting.feature == _CandlestickProperties.REALBODY and period <= 1:
                additional = self._get_setting(CandlestickSettings.DOJI_BODY) * factor
                ret = NumericTimeSeries.max(ret, additional)
            self._settings[setting.name] = ret
        ret = self._settings[setting.name]
        ret.rename(f'{self._name}.{setting.name}')
        return ret

    def _get_mas(self) -> List[NumericTimeSeries]:
        periods = LEADING_TREND_PERIODS[self._tunit]
        cur_p = periods[0]
        cur_s = _moving_sum(self.close, cur_p, self._tunit)
        ma_ = cur_s / cur_p
        ma_.rename(f'{self._name}.MA[0:{cur_p}]')
        ret = [ma_]
        prev_p, prev_s = cur_p, cur_s
        for cur_p in periods[1:]:
            cur_s = _moving_sum(self.close, cur_p, self._tunit)
            ma_ = (cur_s - prev_s) / (cur_p - prev_p)
            ma_.rename(f'{self._name}.MA[{prev_p}:{cur_p}]')
            ret.append(ma_)
            prev_p, prev_s = cur_p, cur_s
        return ret

    def _get_mhs(self) -> List[NumericTimeSeries]:
        periods = LEADING_TREND_PERIODS[self._tunit]
        cur_p = periods[0]
        mh_ = _moving_max(self.high, cur_p, 0, self._tunit)
        mh_.rename(f'{self._name}.MH[0:{cur_p}]')
        ret = [mh_]
        prev_p = cur_p
        for cur_p in periods[1:]:
            mh_ = _moving_max(self.high, cur_p, prev_p, self._tunit)
            mh_.rename(f'{self._name}.MH[{prev_p}:{cur_p}]')
            ret.append(mh_)
            prev_p = cur_p
        return ret

    def _get_mls(self) -> List[NumericTimeSeries]:
        periods = LEADING_TREND_PERIODS[self._tunit]
        cur_p = periods[0]
        ml_ = _moving_min(self.low, cur_p, 0, self._tunit)
        ml_.rename(f'{self._name}.ML[0:{cur_p}]')
        ret = [ml_]
        prev_p = cur_p
        for cur_p in periods[1:]:
            ml_ = _moving_min(self.low, cur_p, prev_p, self._tunit)
            ml_.rename(f'{self._name}.ml[{prev_p}:{cur_p}]')
            ret.append(ml_)
            prev_p = cur_p
        return ret

    @property
    def _mas(self) -> List[NumericTimeSeries]:
        pkey = 'MAs'
        if pkey not in self._properties:
            self._properties[pkey] = self._get_mas()
        return self._properties[pkey]

    @property
    def _mls(self) -> List[NumericTimeSeries]:
        pkey = 'MLs'
        if pkey not in self._properties:
            self._properties[pkey] = self._get_mls()
        return self._properties[pkey]

    @property
    def _mhs(self) -> List[NumericTimeSeries]:
        pkey = 'MHs'
        if pkey not in self._properties:
            self._properties[pkey] = self._get_mhs()
        return self._properties[pkey]

    def _is_bullish(self) -> BooleanTimeSeries:
        conds = [sma >= lma for sma, lma in zip(self._mas[:-1], self._mas[1:])]
        # conds = ([sma >= lma for sma, lma
        #           in zip(self._mas[:-1], self._mas[1:])] +
        #          [sml >= lml for sml, lml
        #           in zip(self._mls[:-1], self._mls[1:])] +
        #          [smh >= lmh for smh, lmh
        #           in zip(self._mhs[:-1], self._mhs[1:])])
        ret = conds[0]
        for each in conds[1:]:
            ret = ret & each
        return ret

    def _is_bearish(self) -> BooleanTimeSeries:
        conds = [sma <= lma for sma, lma in zip(self._mas[:-1], self._mas[1:])]
        # conds = ([sma >= lma for sma, lma
        #           in zip(self._mas[:-1], self._mas[1:])] +
        #          [sml <= lml for sml, lml
        #           in zip(self._mls[:-1], self._mls[1:])] +
        #          [smh <= lmh for smh, lmh
        #           in zip(self._mhs[:-1], self._mhs[1:])])
        ret = conds[0]
        for each in conds[1:]:
            ret = ret & each
        return ret

    def _is_strictly_bullish(self) -> BooleanTimeSeries:
        #conds = [sma >= lma for sma, lma in zip(self._mas[:-1], self._mas[1:])]
        # conds = ([sma >= lma for sma, lma
        #           in zip(self._mas[:-1], self._mas[1:])] +
        #          [sml >= lml for sml, lml
        #           in zip(self._mls[:-1], self._mls[1:])] +
        #          [smh >= lmh for smh, lmh
        #           in zip(self._mhs[:-1], self._mhs[1:])])
        conds = ([sma >= lma for sma, lma
                  in zip(self._mas[:-1], self._mas[1:])] +
                  [sml >= lml for sml, lml
                  in zip(self._mls[:-1], self._mls[1:])])
        ret = conds[0]
        for each in conds[1:]:
            ret = ret & each
        return ret

    def _is_strictly_bearish(self) -> BooleanTimeSeries:
        #conds = [sma <= lma for sma, lma in zip(self._mas[:-1], self._mas[1:])]
        # conds = ([sma >= lma for sma, lma
        #           in zip(self._mas[:-1], self._mas[1:])] +
        #          [sml <= lml for sml, lml
        #           in zip(self._mls[:-1], self._mls[1:])] +
        #          [smh <= lmh for smh, lmh
        #           in zip(self._mhs[:-1], self._mhs[1:])])
        conds = ([sma <= lma for sma, lma
                  in zip(self._mas[:-1], self._mas[1:])] +
                  [smh <= lmh for smh, lmh
                  in zip(self._mhs[:-1], self._mhs[1:])])
        ret = conds[0]
        for each in conds[1:]:
            ret = ret & each
        return ret

    @property
    def is_bullish(self):
        pkey = 'IsBullish'
        if pkey not in self._patterns:
            self._patterns[pkey] = self._is_bullish()
        return self._patterns[pkey]

    @property
    def is_bearish(self):
        pkey = 'IsBearish'
        if pkey not in self._patterns:
            self._patterns[pkey] = self._is_bearish()
        return self._patterns[pkey]

    @property
    def is_strictly_bullish(self):
        pkey = 'IsStrictlyBullish'
        if pkey not in self._patterns:
            self._patterns[pkey] = self._is_strictly_bullish()
        return self._patterns[pkey]

    @property
    def is_strictly_bearish(self):
        pkey = 'IsStrictlyBearish'
        if pkey not in self._patterns:
            self._patterns[pkey] = self._is_strictly_bearish()
        return self._patterns[pkey]

    def leading_with(self, leading_trend: _LeadingTrends):
        if leading_trend == _LeadingTrends.BEARISH:
            return self.is_bearish
        if leading_trend == _LeadingTrends.BULLISH:
            return self.is_bullish
        if leading_trend == _LeadingTrends.STRICTLY_BEARISH:
            return self.is_strictly_bearish
        if leading_trend == _LeadingTrends.STRICTLY_BULLISH:
            return self.is_strictly_bullish
        return self.close == self.close

    @property
    def near_tolerance(self) -> NumericTimeSeries:
        """The tolerance for 'Near / Close To'. """
        return self._get_setting(CandlestickSettings.NEAR)

    @property
    def far_threshold(self) -> NumericTimeSeries:
        """The threshold for 'Far From / Far-Less Than/ Far-Greater Than'. """
        return self._get_setting(CandlestickSettings.FAR_FROM)

    @property
    def equal_tolerance(self) -> NumericTimeSeries:
        """The tolerance for 'Equal To'. """
        return self._get_setting(CandlestickSettings.EQUAL_TO)

    @property
    def _doji_body_tolerance(self) -> NumericTimeSeries:
        """The tolerance for 'Doji / Very-Short Body'. """
        return self._get_setting(CandlestickSettings.DOJI_BODY)

    @property
    def _short_body_threshold(self) -> NumericTimeSeries:
        """The threshold for 'Short Body'. """
        return self._get_setting(CandlestickSettings.SHORT_BODY)

    @property
    def _long_body_threshold(self) -> NumericTimeSeries:
        """The threshold for 'Long Body'. """
        return self._get_setting(CandlestickSettings.LONG_BODY)

    @property
    def _verylong_body_threshold(self) -> NumericTimeSeries:
        """The threshold for 'Very-Long Body'. """
        return self._get_setting(CandlestickSettings.VERY_LONG_BODY)

    @property
    def _without_shadow_tolerance(self) -> NumericTimeSeries:
        """The tolerance for 'Without / Very-Short Shadow'. """
        return self._get_setting(CandlestickSettings.WITHOUT_SHADOW)

    @property
    def _short_shadow_threshold(self) -> NumericTimeSeries:
        """The threshold for 'Short Shadow'. """
        return self._get_setting(CandlestickSettings.SHORT_SHADOW)

    @property
    def _long_shadow_threshold(self) -> NumericTimeSeries:
        """The threshold for 'Long Shadow'. """
        ret = NumericTimeSeries.min(
            self._get_setting(CandlestickSettings.LONG_SHADOW),
            self._long_body_threshold)
        return ret

    @property
    def _verylong_shadow_threshold(self) -> NumericTimeSeries:
        """The threshold for 'Very-Long Shadow'. """
        ret = NumericTimeSeries.min(
            self._get_setting(CandlestickSettings.VERY_LONG_SHADOW),
            self._verylong_body_threshold)
        return ret

    @classmethod
    def _make(cls, open_: NumericTimeSeries, high: NumericTimeSeries,
              low: NumericTimeSeries, close: NumericTimeSeries,
              name: str, tunit: TimeUnit) -> 'Candlestick':
        data = MarketData(cp=close, lp=low, hp=high, op=open_)
        ret = cls(data, "", tunit)
        ret._name = name
        return ret

    def shift(self, period: int) -> 'Candlestick':
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"invalid `period`: {period}")
        ckey = f'shift({period})'
        if ckey not in self._derivatives:
            open_ = self.open_.shift(period, self._tunit)
            high = self.high.shift(period, self._tunit)
            low = self.low.shift(period, self._tunit)
            close = self.close.shift(period, self._tunit)
            name = f'{self._name}[t-{period}]'
            self._derivatives[ckey] = self._make(open_, high, low, close,
                                                 name, self._tunit)
        return self._derivatives[ckey]


    @property
    def is_white_body(self) -> BooleanTimeSeries:
        """Close is greater than Open."""
        pkey = 'IsWhiteBody'
        if pkey not in self._patterns:
            ret = self.close > self.open_
            ret.rename(f'{self._name}.IsWhiteBody')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_black_body(self) -> BooleanTimeSeries:
        """Open is greater than Close."""
        pkey = 'IsBlackBody'
        if pkey not in self._patterns:
            ret = self.open_ > self.close
            ret.rename(f'{self._name}.IsBlackBody')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_doji_body(self) -> BooleanTimeSeries:
        """Without RealBody or With Very-Short RealBody."""
        pkey = 'IsDojiBody'
        if pkey not in self._patterns:
            tolerance = self._doji_body_tolerance
            realbody = self.realbody
            ret = realbody <= tolerance
            ret.rename(f'{self._name}.IsDojiBody')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    def _is_short_body(self) -> BooleanTimeSeries:
        threshold = self._short_body_threshold
        realbody = self.realbody
        ret = realbody < threshold
        return ret

    def _is_long_body(self) -> BooleanTimeSeries:
        threshold = self._long_body_threshold
        realbody = self.realbody
        ret = realbody >= threshold
        return ret

    def _is_verylong_body(self) -> BooleanTimeSeries:
        threshold = self._verylong_body_threshold
        realbody = self.realbody
        ret = realbody >= threshold
        return ret

    @property
    def is_short_body(self) -> BooleanTimeSeries:
        """With Short RealBody."""
        pkey = 'IsShortBody'
        if pkey not in self._patterns:
            # ret = (self._is_short_body() | self.is_doji_body) &
            #        ~self._is_long_body()) <- Strictly
            ret = self._is_short_body() | self.is_doji_body # only for this version
            ret.rename(f'{self._name}.IsShortBody')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_long_body(self) -> BooleanTimeSeries:
        """With Long RealBody."""
        pkey = 'IsLongBody'
        if pkey not in self._patterns:
            # ret = self._is_long_body() & ~self._is_short_body() <- Strictly
            ret = self._is_long_body() # only for this version
            ret.rename(f'{self._name}.IsLongBody')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_verylong_body(self) -> BooleanTimeSeries:
        """With Very-Long RealBody."""
        pkey = 'IsVeryLongBody'
        if pkey not in self._patterns:
            # ret = self.is_long_body & self._is_verylong_body() <- Strictly
            ret = self._is_verylong_body() # only for this version
            ret.rename(f'{self._name}.IsVeryLongBody')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    def _is_without_shadow(self, shadow):
        tolerance = self._without_shadow_tolerance
        return shadow <= tolerance

    def _is_short_shadow(self, shadow):
        threshold = self._short_shadow_threshold
        return shadow < threshold

    def _is_long_shadow(self, shadow):
        threshold = self._long_shadow_threshold
        return shadow >= threshold

    def _is_verylong_shadow(self, shadow):
        threshold = self._verylong_shadow_threshold
        return shadow >= threshold

    @property
    def is_without_uppershadow(self):
        """Without Uppershadow or With Very-Short UpperShadow."""
        pkey = 'IsWithoutUpperShadow'
        if pkey not in self._patterns:
            ret = self._is_without_shadow(self.uppershadow)
            ret.rename(f'{self._name}.IsWithoutUpperShadow')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_without_lowershadow(self):
        """Without Lowershadow or With Very-Short UpperShadow."""
        pkey = 'IsWithoutLowerShadow'
        if pkey not in self._patterns:
            ret = self._is_without_shadow(self.lowershadow)
            ret.rename(f'{self._name}.IsWithoutLowerShadow')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_short_uppershadow(self):
        """With Short Uppershadow."""
        pkey = 'IsShortUpperShadow'
        if pkey not in self._patterns:
            ret = ((self._is_short_shadow(self.uppershadow) |
                    self._is_without_shadow(self.uppershadow)) &
                    ~self._is_long_shadow(self.uppershadow))
            ret.rename(f'{self._name}.IsShortUpperShadow')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_short_lowershadow(self):
        """With Short Lowershadow."""
        pkey = 'IsShortLowerShadow'
        if pkey not in self._patterns:
            ret = ((self._is_short_shadow(self.lowershadow) |
                    self._is_without_shadow(self.lowershadow)) &
                    ~self._is_long_shadow(self.lowershadow))
            ret.rename(f'{self._name}.IsShortLowerShadow')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_long_uppershadow(self):
        """With Long Uppershadow."""
        pkey = 'IsLongUpperShadow'
        if pkey not in self._patterns:
            ret = (self._is_long_shadow(self.uppershadow) &
                   ~self.is_short_uppershadow)
            ret.rename(f'{self._name}.IsLongUpperShadow')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_long_lowershadow(self):
        """With Long Lowershadow."""
        pkey = 'IsLongLowerShadow'
        if pkey not in self._patterns:
            ret = (self._is_long_shadow(self.lowershadow) &
                   ~self.is_short_lowershadow)
            ret.rename(f'{self._name}.IsLongLowerShadow')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_verylong_uppershadow(self):
        """With Very-Long Uppershadow."""
        pkey = 'IsVeryLongUpperShadow'
        if pkey not in self._patterns:
            ret = (self._is_verylong_shadow(self.uppershadow) &
                   self.is_long_uppershadow)
            ret.rename(f'{self._name}.IsVeryLongUpperShadow')
            self._patterns[pkey] = ret
        return self._patterns[pkey]

    @property
    def is_verylong_lowershadow(self):
        """With Very-Long Lowershadow."""
        pkey = 'IsVeryLongLowerShadow'
        if pkey not in self._patterns:
            ret = (self._is_verylong_shadow(self.lowershadow) &
                   self.is_long_lowershadow)
            ret.rename(f'{self._name}.IsVeryLongLowerShadow')
            self._patterns[pkey] = ret
        return self._patterns[pkey]


def _get_candlestick(market_id: str, tunit: TimeUnit) -> Candlestick:
    data = get_market_data(market_id)
    if tunit is not TimeUnit.DAY:
        data = MarketData(cp=data.cp,
                          op=data.op.rolling(1, tunit).first(),
                          hp=data.hp.rolling(1, tunit).max(),
                          lp=data.lp.rolling(1, tunit).min())
    ret = Candlestick(data, market_id, tunit)
    return ret

def get_candlestick(market_id: str, tunit: TimeUnit) -> Candlestick:
    key = f'KLP.Candlestick.{market_id}.{tunit.name}'
    if key not in MD_CACHE:
        MD_CACHE[key] = _get_candlestick(market_id, tunit)
    return MD_CACHE[key]


COMMON_PARAS = [
    MacroParam('period_type', 'K線週期',
               'K線週期: 日("day")、週("week")、月("month")',
               PeriodType, _PeriodTypes.DAY),
    MacroParam('leading_trend', '近期趨勢',
               '近期趨勢: 市場在現象發生前的近期趨勢',
               LeadingTrend, _LeadingTrends.NONE)]

def arg_checker(period_type: _PeriodTypes,
                leading_trend: _LeadingTrends) -> Dict[str, str]:
    ret = {}
    if period_type not in _PeriodTypes:
        ret['period_type'] = "無效的K線週期"
    if leading_trend not in _LeadingTrends:
        ret['leading_trend'] = "無效的近期趨勢"
    return ret

def get_lookback_length(settings: List[CandlestickSettings], tunit: TimeUnit,
                        with_prefix_trend: bool=True) -> int:
    ret = 0
    for each in settings:
        if isinstance(each.period, dict):
            ret = max([ret, each.period[tunit]])
        else:
            ret = max([ret, each.period])
    if with_prefix_trend and ret < LEADING_TREND_PERIODS[tunit][-1]:
        ret = LEADING_TREND_PERIODS[tunit][-1]
    return ret

_SIGMA_TUNIT_MAP = {TimeUnit.DAY: 0.02,
                    TimeUnit.WEEK: 0.05,
                    TimeUnit.MONTH: 0.1}
_MU_TUNIT_MAP = {TimeUnit.DAY: 0.0005,
                 TimeUnit.WEEK: 0.0025,
                 TimeUnit.MONTH: 0.01}

def _gen_candlesticks(size: int, tunit: TimeUnit, sign: int):
    sigma = _SIGMA_TUNIT_MAP[tunit] / np.sqrt(100)
    mu = ((1 + _MU_TUNIT_MAP[tunit]) ** 0.01 - 1) * sign
    cr = np.concatenate([np.random.normal(_MU_TUNIT_MAP[TimeUnit.DAY] * sign,
                                          _SIGMA_TUNIT_MAP[TimeUnit.DAY], (size, 1)),
                         np.random.normal(mu, sigma, (size, 100))], axis=1).flatten()
    cr = np.cumprod(1 + cr).round(3).reshape(size, 101) * 100
    op = cr[:, 0]
    cp = cr[:, -1]
    hp = cr.max(axis=1)
    lp = cr.min(axis=1)
    assert (lp <= op).all()
    assert (lp <= cp).all()
    assert (hp >= op).all()
    assert (hp >= cp).all()
    dates = (np.arange(size) - size // 2
             ).astype(f'datetime64[{tunit.prefix.upper()}]')
    """
    index = TimeIndex(dates)
    data = MarketData(cp=NumericTimeSeries(cp, index, 'CP'),
                      op=NumericTimeSeries(op, index, 'OP'),
                      hp=NumericTimeSeries(hp, index, 'HP'),
                      lp=NumericTimeSeries(lp, index, 'LP'))
    ret = Candlestick(data, 'random', tunit.DAY)
    """
    ret = pd.DataFrame(np.array([op, hp, lp, cp]).T, index=dates,
                       columns=['OP', 'HP', 'LP', 'CP'])
    return ret

class _RandomMarketDataProvider(MarketDataProvider):
    def get_ohlc(self, market_id: str) -> pd.DataFrame:
        """Get OHLC data of the designated market.

        Parameters
        ----------
        market_id: str
            ID of the designated market.

        Returns
        -------
        Pandas' DataFrame
            A Pandas' DataFrame consisted of four floating time-series with
            names, `OP`, `HP`, `LP`, and `CP`.

        """
        trend, size, period_type = market_id.split('_')[:3]
        tunit = TimeUnit.get(period_type)
        size = int(size)
        if trend == 'bullish':
            return _gen_candlesticks(size, tunit, 1)
        if trend == 'bearish':
            return _gen_candlesticks(size, tunit, -1)
        return _gen_candlesticks(size, tunit, 0)

_MAX_RETRY_TIMES = 100
def get_sample(macro: Callable, period_type: _PeriodTypes,
               leading_trend: _LeadingTrends) -> np.ndarray:
    sign = leading_trend.data
    interval = macro.get_interval(period_type=period_type, leading_trend=leading_trend)
    set_market_data_provider(_RandomMarketDataProvider())
    for i in range(_MAX_RETRY_TIMES):
        seed = f'{datetime.datetime.now()}_{random.random()}'
        if sign > 0:
            mid = f'bullish_4000_{period_type.value.data.name}_{seed}'
        if sign < 0:
            mid = f'bearish_4000_{period_type.value.data.name}_{seed}'
        if sign == 0:
            mid = f'flat_4000_{period_type.value.data.name}_{seed}'
        cct = get_candlestick(mid, TimeUnit.DAY)
        for idx in np.argwhere(macro.evaluate(mid, period_type=_PeriodTypes.DAY,
                                              leading_trend=leading_trend,
                                              ).values > 0).flatten().tolist():
            if idx >= interval:
                return cct.values[idx-interval+1: idx+1]
    raise TimeoutError()

class MacroInfo(NamedTuple):
    symbol: str
    name: str
    description: str
    func: Callable
    interval: int
    samples: Dict[str, Dict[str, np.ndarray]]
    py_version: str
    db_version: str

    def _macro(self, market: str, period_type: _PeriodTypes,
               leading_trend: _LeadingTrends):
        tunit = period_type.value.data
        cct = get_candlestick(market, tunit)
        ret = self.func(cct)
        if leading_trend != _LeadingTrends.NONE:
            ret = ret & cct.shift(self.interval).leading_with(leading_trend)
        return ret.to_pandas().rename(f'{cct.name}.{self.symbol}')

    def _sample(self, period_type: _PeriodTypes,
                leading_trend: _LeadingTrends):
        tunit = period_type.value.data
        ret = [PlotInfo(Ptype.CANDLE, 'K', self.samples[leading_trend][tunit])]
        return ret

    def _interval(self, period_type: _PeriodTypes,
                  leading_trend: _LeadingTrends):
        if leading_trend == _LeadingTrends.NONE:
            return self.interval
        tunit = period_type.value.data
        return self.interval + LEADING_TREND_PERIODS[tunit][-1]

    def to_macro(self, code: str) -> Macro:
        name = f'商智K線指標(KLP版)-{self.name}({self.symbol})'
        ret = Macro(code, name, self.description,
                    COMMON_PARAS, self._macro,
                    self._sample, self._interval, arg_checker,
                    f'{_PY_VERSION}_{self.py_version}',
                    f'{_DB_VERSION}_{self.db_version}', )
        return ret
