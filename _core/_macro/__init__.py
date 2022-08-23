from .common import gen_macro, Macro, MacroManagerBase, MacroParaEnumManagerBase
from ._klp import (KlpCcLeadingTrend, klp_cc_doji, klp_cc_dragonfly_doji,
                   klp_cc_gravestone_doji, klp_cc_takuri,
                   klp_cc_longlegged_doji, klp_cc_rickshaw_man)

from func._tp import (
    jack_ma_order_up, jack_ma_order_thick, jack_ma_order_down,
    jack_ma_through_price_down, jack_ma_through_price_up,
    jack_ma_through_ma_down_trend, jack_ma_through_ma_up_trend,
    stone_pp000, stone_pp001, stone_pp002, stone_pp003,
    stone_pp004, stone_pp005, stone_pp006, stone_pp007,
    stone_pp008, stone_pp009,
    wj001, wj002, wj003, wj004,
    PeriodType)

class MacroManager(MacroManagerBase):
    jack_ma_order_up = gen_macro(jack_ma_order_up)
    jack_ma_order_thick = gen_macro(jack_ma_order_thick)
    jack_ma_order_down = gen_macro(jack_ma_order_down)
    jack_ma_through_price_down = gen_macro(jack_ma_through_price_down)
    jack_ma_through_price_up = gen_macro(jack_ma_through_price_up)
    jack_ma_through_ma_down_trend = gen_macro(jack_ma_through_ma_down_trend)
    jack_ma_through_ma_up_trend = gen_macro(jack_ma_through_ma_up_trend)
    stone_pp000 = gen_macro(stone_pp000)
    stone_pp001 = gen_macro(stone_pp001)
    stone_pp002 = gen_macro(stone_pp002)
    stone_pp003 = gen_macro(stone_pp003)
    stone_pp004 = gen_macro(stone_pp004)
    stone_pp005 = gen_macro(stone_pp005)
    stone_pp006 = gen_macro(stone_pp006)
    stone_pp007 = gen_macro(stone_pp007)
    stone_pp008 = gen_macro(stone_pp008)
    stone_pp009 = gen_macro(stone_pp009)
    wj001 = gen_macro(wj001)
    wj002 = gen_macro(wj002)
    wj003 = gen_macro(wj003)
    wj004 = gen_macro(wj004)
    klp_cc_doji = klp_cc_doji
    klp_cc_dragonfly_doji = klp_cc_dragonfly_doji
    klp_cc_gravestone_doji = klp_cc_gravestone_doji
    klp_cc_takuri = klp_cc_takuri
    klp_cc_longlegged_doji = klp_cc_longlegged_doji
    klp_cc_rickshaw_man =  klp_cc_rickshaw_man

class MacroParaEnumManager(MacroParaEnumManagerBase):
    PERIOD_TYPE = PeriodType
    KLP_CC_LEADING_TREND = KlpCcLeadingTrend
