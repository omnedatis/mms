from enum import Enum
from typing import NamedTuple
import numpy as np

class TCandle(NamedTuple):
    name: str = None
    ud: int = None
    ampt: float = None
    body: float = None
    t_offset: float = None

    @classmethod
    def make(cls, name=None, ud=None, ampt=None, body=None, t_offset=None):
        return cls(name, ud, ampt, body, t_offset)

class KType(Enum):
    WHITE_LONG = TCandle.make(name="大陽線", ud=1, ampt=10.0, body=1.0, t_offset=0.0)
    WHITE_LONG_TINY_EQUAL_SHADOW = TCandle.make(name="上下影線都短的大陽線", ud=1, ampt=10.0, body=0.9, t_offset=0.05)
    WHITE_CLOSING_MARUBOZU = TCandle.make(name="光頭陽線", ud=1, ampt=10.0, body=0.7, t_offset=0.0)
    WHITE_HAMMER = TCandle.make(name="陽線槌子線", ud=1, ampt=10.0, body=0.3, t_offset=0.0)
    WHITE_OPENING_MARUBOZU = TCandle.make(name="光腳陽線", ud=1, ampt=10.0, body=0.7, t_offset=0.3)
    WHITE_INVERSE_HAMMER = TCandle.make(name="倒陽線槌子線", ud=1, ampt=10.0, body=0.3, t_offset=0.7)
    WHITE_SHORT = TCandle.make(name="小陽線", ud=1, ampt=10.0, body=0.4, t_offset=0.3)
    WHITE_SHORT_LONG_UPPER_SHADOW = TCandle.make(name="上影線較長的小陽線", ud=1, ampt=10.0, body=0.4, t_offset=0.4)
    WHITE_SHORT_LONG_LOWER_SHADOW = TCandle.make(name="下影線較長的小陽線", ud=1, ampt=10.0, body=0.4, t_offset=0.2)
    WHITE_TINY_LONG_UPPER_SHADOW = TCandle.make(name="上影線較長的極小陽線", ud=1, ampt=10.0, body=0.2, t_offset=0.6)
    WHITE_TINY_LONG_LOWER_SHADOW = TCandle.make(name="下影線較長的極小陽線", ud=1, ampt=10.0, body=0.2, t_offset=0.2)
    WHITE_TINY_LONG_EQUAL_SHADOW = TCandle.make(name="上下影線都長的極小陽線", ud=1, ampt=10.0, body=0.1, t_offset=0.45)
    BLACK_LONG = TCandle.make(name="大陰線", ud=0, ampt=10.0, body=1.0, t_offset=0.0)
    BLACK_CLOSING_MARUBOZU = TCandle.make(name="光頭陰線", ud=0, ampt=10.0, body=0.7, t_offset=0.0)
    BLACK_HAMMER = TCandle.make(name="陰線槌子線", ud=0, ampt=10.0, body=0.3, t_offset=0.0)
    BLACK_OPENING_MARUBOZU = TCandle.make(name="光腳陰線", ud=0, ampt=10.0, body=0.7, t_offset=0.3)
    BLACK_INVERSE_HAMMER = TCandle.make(name="倒陰線槌子線", ud=0, ampt=10.0, body=0.3, t_offset=0.7)
    BLACK_SHORT = TCandle.make(name="小陰線", ud=0, ampt=10.0, body=0.4, t_offset=0.3)
    BLACK_SHORT_LONG_UPPER_SHADOW = TCandle.make(name="上影線較長的小陰線", ud=0, ampt=10.0, body=0.4, t_offset=0.4)
    BLACK_SHORT_LONG_LOWER_SHADOW = TCandle.make(name="下影線較長的小陰線", ud=0, ampt=10.0, body=0.4, t_offset=0.2)
    BLACK_TINY_LONG_UPPER_SHADOW = TCandle.make(name="上影線較長的極小陰線", ud=0, ampt=10.0, body=0.2, t_offset=0.6)
    BLACK_TINY_LONG_LOWER_SHADOW = TCandle.make(name="下影線較長的極小陰線", ud=0, ampt=10.0, body=0.2, t_offset=0.2)
    BLACK_TINY_LONG_EQUAL_SHADOW = TCandle.make(name="上下影線都長的極小陰線", ud=0, ampt=10.0, body=0.1, t_offset=0.45)
    DOJI_FOUR_PRICE = TCandle.make(name="一字線", ud=0, ampt=0.0, body=0.0, t_offset=0.0)
    DOJI_UMBRELLA = TCandle.make(name="T字線", ud=0, ampt=10.0, body=0.0, t_offset=0.0)
    DOJI_INVERSE_UMBRELLA = TCandle.make(name="墓碑線", ud=0, ampt=10.0, body=0.0, t_offset=1.0)
    DOJI = TCandle.make(name="十字線", ud=0, ampt=10.0, body=0.0, t_offset=0.5)
    DOJI_LONG_UPPER_SHADOW = TCandle.make(name="上影線較長的十字線", ud=0, ampt=10.0, body=0.0, t_offset=0.8)
    DOJI_LONG_LOWER_SHADOW = TCandle.make(name="下影線較長的十字線", ud=0, ampt=10.0, body=0.0, t_offset=0.2)

class MokeCandle:
    @classmethod
    def make(cls, ktype: KType) -> np.array:
        # 陰陽線
        ud = ktype.value.ud
        # 全長
        ampt = ktype.value.ampt
        # 燭身長
        body = ampt * ktype.value.body
        # 燭身相對頂部位置
        top_offset = ktype.value.t_offset * ampt

        high = ampt
        low = 0
        if top_offset < 0:
            body += top_offset
            if body < 0:
                body = 0
        elif body + top_offset > ampt:
            body -= body + top_offset - ampt
        elif top_offset >= ampt:
            body = 0
        uds = [ampt - top_offset, ampt - top_offset - body]
        ohlc = [uds[1], high, low, uds[0]] if ud == 1 else [uds[0], high, low, uds[1]]
        return np.array(ohlc)
