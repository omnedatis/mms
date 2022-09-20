from ._common import LeadingTrend as KlpCcLeadingTrend
from ._consolidation._doji import (doji, dragonfly_doji, gravestone_doji,
                                   takuri, longlegged_doji, rickshaw_man)
from ._consolidation._marubozu import (black_marubozu, white_marubozu,
                                       black_opening_marubozu, white_opening_marubozu,
                                       black_closing_marubozu, white_closing_marubozu)
from ._consolidation._highwave import black_highwave, white_highwave
from ._consolidation._longline import black_longline, white_longline
from ._consolidation._shortline import black_shortline, white_shortline
from ._consolidation._spinning_top import black_spinning_top, white_spinning_top

klp_cc_doji = doji.to_macro('klp_cc_doji')
klp_cc_dragonfly_doji = dragonfly_doji.to_macro('klp_cc_dragonfly_doji')
klp_cc_gravestone_doji = gravestone_doji.to_macro('klp_cc_gravestone_doji')
klp_cc_takuri = takuri.to_macro('klp_cc_takuri')
klp_cc_longlegged_doji = longlegged_doji.to_macro('klp_cc_longlegged_doji')
klp_cc_rickshaw_man = rickshaw_man.to_macro('klp_cc_rickshaw_man')
klp_cc_black_marubozu = black_marubozu.to_macro('klp_cc_black_marubozu')
klp_cc_white_marubozu = white_marubozu.to_macro('klp_cc_white_marubozu')
klp_cc_black_opening_marubozu = black_opening_marubozu.to_macro('klp_cc_black_opening_marubozu')
klp_cc_white_opening_marubozu = white_opening_marubozu.to_macro('klp_cc_white_opening_marubozu')
klp_cc_black_closing_marubozu = black_closing_marubozu.to_macro('klp_cc_black_closing_marubozu')
klp_cc_white_closing_marubozu = white_closing_marubozu.to_macro('klp_cc_white_closing_marubozu')
klp_cc_black_highwave = black_highwave.to_macro('klp_cc_black_highwave')
klp_cc_white_highwave = white_highwave.to_macro('klp_cc_white_highwave')
klp_cc_black_longline = black_longline.to_macro('klp_cc_black_longline')
klp_cc_white_longline = white_longline.to_macro('klp_cc_white_longline')
klp_cc_black_shortline = black_shortline.to_macro('klp_cc_black_shortline')
klp_cc_white_shortline = white_shortline.to_macro('klp_cc_white_shortline')
klp_cc_black_spinning_top = black_spinning_top.to_macro('klp_cc_black_spinning_top')
klp_cc_white_spinning_top = white_spinning_top.to_macro('klp_cc_white_spinning_top')


__all__ = ['KlpCcLeadingTrend']
__all__ += ['klp_cc_doji', 'klp_cc_dragonfly_doji']
__all__ += ['klp_cc_gravestone_doji', 'klp_cc_takuri',
            'klp_cc_longlegged_doji', 'klp_cc_rickshaw_man']
__all__ += ['klp_cc_black_marubozu', 'klp_cc_white_marubozu',
            'klp_cc_black_opening_marubozu', 'klp_cc_white_opening_marubozu',
            'klp_cc_black_closing_marubozu', 'klp_cc_white_closing_marubozu']
__all__ += ['klp_cc_black_highwave', 'klp_cc_white_highwave',
            'klp_cc_black_longline', 'klp_cc_white_longline',
            'klp_cc_black_shortline', 'klp_cc_white_shortline',
            'klp_cc_black_spinning_top', 'klp_cc_white_spinning_top']
