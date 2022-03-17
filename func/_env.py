# -*- coding: utf-8 -*-
"""Module for execution environment.

Methods
-------
set_mode : Set execution mode.
get_mode : Get execution mode.

"""

_mode = 'test'

def set_mode(mode: str):
    """Set ececution mode.

    Parameters
    ----------
    mode : {'test', 'dev', 'prod'}

    """
    global _mode
    _mode = mode

def get_mode() -> str:
    return _mode
