# -*- coding: utf-8 -*-
"""Technical Indicators."""

from enum import Enum
from typing import Union

from ._td import (get_cp, get_hp, get_lp, get_op,
                  MarketData, NumericTimeSeries, TimeUnit, MD_CACHE)

class _CandleStick:
    def __init__(self, data: MarketData, name: str, tunit: Union[int, TimeUnit]):
        self._data = data
        self._name = name
        self._tunit = tunit

    def _shift(self, period: int) -> '_CandleStick':
        name = f'{self._name}.shift({period})'
        if name not in MD_CACHE:      
            if isinstance(self._tunit, TimeUnit):
                op_ = self.open.shift(period, self._tunit)
                hp_ = self.high.shift(period, self._tunit)
                lp_ = self.low.shift(period, self._tunit)
                cp_ = self.close.shift(period, self._tunit)
            else:
                period = period * self._tunit
                op_ = self.open.shift(period)
                hp_ = self.high.shift(period)
                lp_ = self.low.shift(period)
                cp_ = self.close.shift(period)
            ret = self.__class__(MarketData(cp=cp_, lp=lp_, hp=hp_, op=op_),
                                 name=name, tunit=self._tunit)
            MD_CACHE[name] = ret
        return MD_CACHE[name]  

    def shift(self, period: int) -> '_CandleStick':
        ret = self._shift(period)
        return ret

    @property
    def open(self) -> NumericTimeSeries:
        """Open of _CandleStick."""
        recv = self._data.op
        recv.rename(f'{self._name}.Open')
        return recv

    @property
    def high(self) -> NumericTimeSeries:
        """High of _CandleStick."""
        recv = self._data.hp
        recv.rename(f'{self._name}.High')
        return recv

    @property
    def low(self) -> NumericTimeSeries:
        """Low of _CandleStick."""
        recv = self._data.lp
        recv.rename(f'{self._name}.Low')
        return recv

    @property
    def close(self) -> NumericTimeSeries:
        """close of _CandleStick."""
        recv = self._data.cp
        recv.rename(f'{self._name}.Close')
        return recv

    @property
    def prev_open(self) -> NumericTimeSeries:
        """Previous Open of _CandleStick."""
        recv = self.shift(1).open
        recv.rename(f'{self._name}.PrevOpen')
        return recv

    @property
    def prev_high(self) -> NumericTimeSeries:
        """Previous High of _CandleStick."""
        recv = self.shift(1).high
        recv.rename(f'{self._name}.PrevHigh')
        return recv

    @property
    def prev_low(self) -> NumericTimeSeries:
        """Previous Low of _CandleStick."""
        recv = self.shift(1).low
        recv.rename(f'{self._name}.PrevLow')
        return recv

    @property
    def prev_close(self) -> NumericTimeSeries:
        """Previous Close of _CandleStick."""
        recv = self.shift(1).close
        recv.rename(f'{self._name}.PrevClose')
        return recv

    @property
    def amplitude(self):
        result = self.high - self.low
        result.rename(f'{self._name}.Amplitude')
        return result
    
    @property
    def body(self):
        result = abs(self.close - self.open)
        result.rename(f'{self._name}.Body')
        return result

    @property
    def lower_shadow(self):
        result = NumericTimeSeries.min(self.close, self.open) - self.low
        result.rename(f'{self._name}.LowerShadow')
        return result

    @property
    def upper_shadow(self):
        result = self.high - NumericTimeSeries.max(self.close, self.open)
        result.rename(f'{self._name}.UpperShadow')
        return result

def _ma(market_id: str, period: int, period_type: TimeUnit = TimeUnit.DAY,
        ) -> NumericTimeSeries:
    name = f'{market_id}.MA({period}{period_type.name})'
    cp_ = get_cp(market_id)
    result = cp_.sampling(period, sunit=period_type).mean()
    result.rename(name)
    return result

def _return(market_id: str, period: int, period_type: TimeUnit = TimeUnit.DAY,
            ) -> NumericTimeSeries:
    name = f'{market_id}.Return({period}{period_type.name})'
    cp_ = get_cp(market_id)
    result = cp_ / cp_.shift(period, punit=period_type) - 1
    result.rename(name)
    return result

def _future_return(market_id: str, period: int, period_type: TimeUnit = TimeUnit.DAY,
                   ) -> NumericTimeSeries:
    name = f'{market_id}.FutureReturn({period}{period_type.name})'
    result = _return(market_id, period, period_type)
    result = result.shift(-period, punit=period_type)
    result.rename(name)
    return result

def _candle(market_id: str, period_type: Union[int, TimeUnit]) -> _CandleStick:
    if period_type == TimeUnit.DAY:
        name = f'{market_id}.DailyCandleStick'
    elif period_type == TimeUnit.WEEK:
        name = f'{market_id}.WeeklyCandleStick'
    elif period_type == TimeUnit.MONTH:
        name = f'{market_id}.MonthlyCandleStick'
    else:
        name = f'{market_id}.CandleStick({period_type})'
    if isinstance(period_type, TimeUnit):
        op_ = get_op(market_id)
        hp_ = get_hp(market_id)
        lp_ = get_lp(market_id)
        cp_ = get_cp(market_id)
        if period_type is not TimeUnit.DAY:
            op_ = op_.rolling(1, period_type).first()
            hp_ = hp_.rolling(1, period_type).max()
            lp_ = lp_.rolling(1, period_type).min()
    else:
        op_ = get_op(market_id).sampling(period_type).first()
        hp_ = get_hp(market_id).sampling(period_type).max()
        lp_ = get_lp(market_id).sampling(period_type).min()
        cp_ = get_cp(market_id).sampling(period_type).last()
    result = _CandleStick(MarketData(cp=cp_, lp=lp_, hp=hp_, op=op_),
                          name=name, tunit=period_type)
    return result

class TechnicalIndicator(Enum):
    """Enumerator of technical indicators.

    Members
    -------
    MA, Candle

    """
    MA = _ma
    Candle = _candle
    RETURN = _return
