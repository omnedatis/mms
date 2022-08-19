from ._common import LeadingTrend as KlpCcLeadingTrend
from ._consolidation._doji import doji
from ._consolidation._dragonfly_doji import dragonfly_doji

klp_cc_doji = doji.to_macro('klp_cc_doji')
klp_cc_dragonfly_doji = dragonfly_doji.to_macro('klp_cc_dragonfly_doji')

__all__ = ['KlpCcLeadingTrend']
__all__ += ['klp_cc_doji', 'klp_cc_dragonfly_doji']
