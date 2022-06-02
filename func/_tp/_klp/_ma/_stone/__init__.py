from ..common import gen_macro
from .pp003 import stone_pp003 as _stone_pp003
from .pp004 import stone_pp004 as _stone_pp004
from .pp005 import stone_pp005 as _stone_pp005
from .pp006 import stone_pp006 as _stone_pp006
from .pp007 import stone_pp007 as _stone_pp007
from .pp008 import stone_pp008 as _stone_pp008
from .pp009 import stone_pp009 as _stone_pp009

stone_pp003 = gen_macro(_stone_pp003)
stone_pp004 = gen_macro(_stone_pp004)
stone_pp005 = gen_macro(_stone_pp005)
stone_pp006 = gen_macro(_stone_pp006)
stone_pp007 = gen_macro(_stone_pp007)
stone_pp008 = gen_macro(_stone_pp008)
stone_pp009 = gen_macro(_stone_pp009)

__all__ = ['stone_pp003', 'stone_pp004', 'stone_pp005', 'stone_pp006',
           'stone_pp007', 'stone_pp008', 'stone_pp009']
