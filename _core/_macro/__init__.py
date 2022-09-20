from .common import (gen_macro, Macro, MacroManagerBase,
                     MacroParaEnumManagerBase, MacroTags)
from ._klp import (KlpCcLeadingTrend, klp_cc_doji, klp_cc_dragonfly_doji,
                   klp_cc_gravestone_doji, klp_cc_takuri,
                   klp_cc_longlegged_doji, klp_cc_rickshaw_man,
                   klp_cc_black_marubozu, klp_cc_white_marubozu,
                   klp_cc_black_opening_marubozu, klp_cc_white_opening_marubozu,
                   klp_cc_black_closing_marubozu, klp_cc_white_closing_marubozu)
from ._sakata._wj import (wj011, wj012, wj013, wj014, wj015, wj016, wj017, wj018,
                    wj019, wj020)
from func._tp import (
    jack_ma_order_up, jack_ma_order_thick, jack_ma_order_down,
    jack_ma_through_price_down, jack_ma_through_price_up,
    jack_ma_through_ma_down_trend, jack_ma_through_ma_up_trend,
    stone_pp000, stone_pp001, stone_pp002, stone_pp003,
    stone_pp004, stone_pp005, stone_pp006, stone_pp007,
    stone_pp008, stone_pp009,
    wj001, wj002, wj003, wj004, wj005, wj006, wj007, wj008,
    wj009, wj010, 
    PeriodType)

class MacroManager(MacroManagerBase):
    jack_ma_order_up = gen_macro(jack_ma_order_up, [MacroTags.PRICE])
    jack_ma_order_thick = gen_macro(jack_ma_order_thick, [MacroTags.PRICE])
    jack_ma_order_down = gen_macro(jack_ma_order_down, [MacroTags.PRICE])
    jack_ma_through_price_down = gen_macro(jack_ma_through_price_down,
                                           [MacroTags.PRICE])
    jack_ma_through_price_up = gen_macro(jack_ma_through_price_up,
                                         [MacroTags.PRICE])
    jack_ma_through_ma_down_trend = gen_macro(jack_ma_through_ma_down_trend,
                                              [MacroTags.PRICE])
    jack_ma_through_ma_up_trend = gen_macro(jack_ma_through_ma_up_trend,
                                            [MacroTags.PRICE])
    stone_pp000 = gen_macro(stone_pp000, [MacroTags.PRICE])
    stone_pp001 = gen_macro(stone_pp001, [MacroTags.PRICE])
    stone_pp002 = gen_macro(stone_pp002, [MacroTags.PRICE])
    stone_pp003 = gen_macro(stone_pp003, [MacroTags.PRICE])
    stone_pp004 = gen_macro(stone_pp004, [MacroTags.PRICE])
    stone_pp005 = gen_macro(stone_pp005, [MacroTags.PRICE])
    stone_pp006 = gen_macro(stone_pp006, [MacroTags.PRICE])
    stone_pp007 = gen_macro(stone_pp007, [MacroTags.PRICE])
    stone_pp008 = gen_macro(stone_pp008, [MacroTags.PRICE])
    stone_pp009 = gen_macro(stone_pp009, [MacroTags.PRICE])
    wj001 = gen_macro(wj001, [MacroTags.PRICE])
    wj002 = gen_macro(wj002, [MacroTags.PRICE])
    wj003 = gen_macro(wj003, [MacroTags.PRICE])
    wj004 = gen_macro(wj004, [MacroTags.PRICE])
    wj005 = gen_macro(wj005, [MacroTags.PRICE])
    wj006 = gen_macro(wj006, [MacroTags.PRICE])
    wj007 = gen_macro(wj007, [MacroTags.PRICE])
    wj008 = gen_macro(wj008, [MacroTags.PRICE])
    wj009 = gen_macro(wj009, [MacroTags.PRICE])
    wj010 = gen_macro(wj010, [MacroTags.PRICE])
    wj011 = wj011
    wj012 = wj012
    wj013 = wj013
    wj014 = wj014
    wj015 = wj015
    wj016 = wj016
    wj017 = wj017
    wj018 = wj018
    wj019 = wj019
    wj020 = wj020
    klp_cc_doji = klp_cc_doji
    klp_cc_dragonfly_doji = klp_cc_dragonfly_doji
    klp_cc_gravestone_doji = klp_cc_gravestone_doji
    klp_cc_takuri = klp_cc_takuri
    klp_cc_longlegged_doji = klp_cc_longlegged_doji
    klp_cc_rickshaw_man =  klp_cc_rickshaw_man
    klp_cc_black_marubozu = klp_cc_black_marubozu
    klp_cc_white_marubozu = klp_cc_white_marubozu
    klp_cc_black_opening_marubozu = klp_cc_black_opening_marubozu
    #klp_cc_white_opening_marubozu = klp_cc_white_opening_marubozu
    #klp_cc_black_closing_marubozu = klp_cc_black_closing_marubozu
    #klp_cc_white_closing_marubozu = klp_cc_white_closing_marubozu

class MacroParaEnumManager(MacroParaEnumManagerBase):
    PERIOD_TYPE = PeriodType
    KLP_CC_LEADING_TREND = KlpCcLeadingTrend
