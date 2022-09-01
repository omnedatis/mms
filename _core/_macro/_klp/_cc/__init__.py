from ._common import LeadingTrend as KlpCcLeadingTrend
from ._consolidation._doji import doji
from ._consolidation._dragonfly_doji import dragonfly_doji
from ._consolidation._gravestone_doji import gravestone_doji
from ._consolidation._takuri import takuri
from ._consolidation._longlegged_doji import longlegged_doji
from ._consolidation._rickshaw_man import rickshaw_man

klp_cc_doji = doji.to_macro('klp_cc_doji')
klp_cc_dragonfly_doji = dragonfly_doji.to_macro('klp_cc_dragonfly_doji')
klp_cc_gravestone_doji = gravestone_doji.to_macro('klp_cc_gravestone_doji')
klp_cc_takuri = takuri.to_macro('klp_cc_takuri')
klp_cc_longlegged_doji = longlegged_doji.to_macro('klp_cc_longlegged_doji')
klp_cc_rickshaw_man = rickshaw_man.to_macro('klp_cc_rickshaw_man')

__all__ = ['KlpCcLeadingTrend']
__all__ += ['klp_cc_doji', 'klp_cc_dragonfly_doji']
__all__ += ['klp_cc_gravestone_doji', 'klp_cc_takuri',
            'klp_cc_longlegged_doji', 'klp_cc_rickshaw_man']
