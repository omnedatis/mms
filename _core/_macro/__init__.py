from .common import gen_macro, Macro, MacroManagerBase

from func._tp import (jack_ma_order_up, jack_ma_order_thick, jack_ma_order_down,
                      jack_ma_through_price_down, jack_ma_through_price_up,
                      jack_ma_through_ma_down_trend, jack_ma_through_ma_up_trend,
                      stone_pp000, stone_pp001, stone_pp002, stone_pp003,
                      stone_pp004, stone_pp005, stone_pp006, stone_pp007,
                      stone_pp008, stone_pp009,
                      klp_cc_bullish_doji,
                      klp_cc_bearish_doji,
                      klp_cc_bullish_dragonfly_doji,
                      klp_cc_bearish_dragonfly_doji,
                      wj001)

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
    klp_cc_bullish_doji = gen_macro(klp_cc_bullish_doji)
    klp_cc_bearish_doji = gen_macro(klp_cc_bearish_doji)
    klp_cc_bullish_dragonfly_doji = gen_macro(klp_cc_bullish_dragonfly_doji)
    klp_cc_bearish_dragonfly_doji = gen_macro(klp_cc_bearish_dragonfly_doji)
    wj001 = gen_macro(wj001)
