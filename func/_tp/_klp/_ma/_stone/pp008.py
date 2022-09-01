import numpy as np

from func._tp._ma import _stone as tp
from .common import get_ma, RawMacro, MacroParam, ParamType, PlotInfo, Ptype
from .common import (gen_cps4arranged_mas, gen_cps4arranged_mas_by_random,
                    check_cps4arranged_mas)
from .common import MAX_MA_PERIOD, MAX_SMAPLES, LimitedVariable

code = 'stone_pp008'
name = 'MA指標-PP008'
macro = tp.stone_pp008
description = macro.__doc__
parameters = [
    MacroParam('period_1', '目標均線天期(t1)', '目標均線天期(t1)',
               ParamType.INT, 1),
    MacroParam('period_2', '目標均線天期(t2)', '目標均線天期(t2)',
               ParamType.INT, 5),
    MacroParam('period_3', '目標均線天期(t3)', '目標均線天期(t3)',
               ParamType.INT, 10),
    MacroParam('period_4', '目標均線天期(t4)', '目標均線天期(t4)',
               ParamType.INT, 20),
    MacroParam('period_5', '目標均線天期(t5)', '目標均線天期(t5)',
               ParamType.INT, 40),
    MacroParam('period_6', '目標均線天期(t6)', '目標均線天期(t6)',
               ParamType.INT, 60),
    MacroParam('period_7', '目標均線天期(t7)', '目標均線天期(t7)',
               ParamType.INT, 120),
    MacroParam('period_8', '目標均線天期(t8)', '目標均線天期(t8)',
               ParamType.INT, 250),
    MacroParam('ridge_index', '指定第幾個天期為波峰天期(k)',
               '指定第幾個天期為波峰天期(k)', ParamType.INT, 3),
    MacroParam('statistical_duration', '判斷過去幾天內的天數(n)',
               '判斷過去幾天內的天數(n)', ParamType.INT, 10),
    MacroParam('min_occurence', '判斷發生事件是否大於或等於幾次的次數(o)',
               '判斷發生事件是否大於或等於幾次的次數(o)', ParamType.INT, 5)]

def arg_checker(period_1, period_2, period_3, period_4, period_5,
                period_6, period_7, period_8, ridge_index,
                statistical_duration, min_occurence):
    ret = {}
    periods = [period_1, period_2, period_3, period_4, period_5,
               period_6, period_7, period_8]
    checker = LimitedVariable(lower=0, upper=7)
    if not checker.check(ridge_index):
        ret['ridge_index'] = checker.message

    lower = 1
    for idx, value in enumerate(periods):
        checker = LimitedVariable(lower=lower, upper=MAX_MA_PERIOD)
        if not checker.check(value):
            ret[f'period_{idx+1}'] = checker.message
        lower = value + 1
        if lower > MAX_MA_PERIOD:
            lower = None

    # check `statistical_duration` and `min_occurence`
    if 0 < ridge_index < 7:
        upper = min([periods[ridge_index+1] - periods[ridge_index],
                     MAX_SMAPLES])
    else:
        upper = MAX_SMAPLES
    checker = LimitedVariable(lower=1, upper=upper)
    if not checker.check(statistical_duration):
        ret['statistical_duration'] = checker.message

    checker = LimitedVariable(lower=1, upper=statistical_duration)
    if not checker.check(min_occurence):
        ret['min_occurence'] = checker.message

    return ret

def sample_generator(period_1, period_2, period_3, period_4, period_5,
                     period_6, period_7, period_8, ridge_index,
                     statistical_duration, min_occurence, lookback_len=-1):

    periods = [period_1, period_2, period_3, period_4, period_5,
               period_6, period_7, period_8]

    if lookback_len < 0 and ridge_index > 0 and ridge_index < 7:
        lookback_len = periods[ridge_index] * 2
    dlen = statistical_duration + lookback_len

    tails = gen_cps4arranged_mas_by_random(periods, ridge_index, statistical_duration, min_occurence, sign=1)
    if tails is None:
        tails = gen_cps4arranged_mas(periods, ridge_index, min_occurence, sign=1)
    check_cps4arranged_mas(tails, periods, ridge_index, statistical_duration, min_occurence, sign=1)
    heads = np.cumprod(1 + np.random.normal(0, 0.01, size=dlen))[::-1] * tails[0]
    cps = np.concatenate([heads, tails])
    ret = [PlotInfo(title=f'MA {p}',
                    ptype=Ptype.MA,
                    data=get_ma(cps, p)[-dlen:]) for p in periods]
    return ret

def interval_evaluator(statistical_duration, **kwargs):
    return statistical_duration

stone_pp008 = RawMacro(code, name, description, parameters, macro,
                       sample_generator, interval_evaluator,
                       arg_checker, '2022082301', '2022081001')