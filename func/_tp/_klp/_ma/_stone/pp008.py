import numpy as np

from func._tp._ma import _stone as tp
from .common import get_ma, RawMacro, MacroParam, ParamType
from .common import (gen_cps4arranged_mas, gen_cps4arranged_mas_by_random, 
                    check_cps4arranged_mas)

code = 'stone_pp008'
name = '商智MA指標-PP008'
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
    if ridge_index < 0 or ridge_index > 7:
        ret['ridge_index'] = "值必須介於0~7"
    limit = 0
    if period_1 <= limit:
        ret['period_1'] = f"值必須大於{limit}"
    limit = max([limit, period_1])
    if period_2 <= limit:
        ret['period_2'] = f"值必須大於{limit}"
    limit = max([limit, period_2])
    if period_3 <= limit:
        ret['period_3'] = f"值必須大於{limit}"   
    limit = max([limit, period_3])
    if period_4 <= limit:
        ret['period_4'] = f"值必須大於{limit}"   
    limit = max([limit, period_4])
    if period_5 <= limit:
        ret['period_5'] = f"值必須大於{limit}"     
    limit = max([limit, period_5])
    if period_6 <= limit:
        ret['period_6'] = f"值必須大於{limit}"     
    limit = max([limit, period_6])
    if period_7 <= limit:
        ret['period_7'] = f"值必須大於{limit}"         
    limit = max([limit, period_7])
    if period_8 <= limit:
        ret['period_8'] = f"值必須大於{limit}"  
    limit = 1
    if ridge_index <= 0 or ridge_index >= 7:
        if min_occurence < limit:
            ret['min_occurence'] = f"值必須大於等於{limit}"
    else:
        ulimit = periods[ridge_index+1] - periods[ridge_index]
        if min_occurence < limit or min_occurence > ulimit:
            ret['min_occurence'] = f"值必須界於{limit}與{ulimit}之間" 
    limit = max([limit, min_occurence])
    if statistical_duration < limit:
        ret['statistical_duration'] = f"值必須大於等於{limit}"
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
    ret = {f'MA {p}': get_ma(cps, p)[-dlen:].tolist() for p in periods}
    return ret

def interval_evaluator(statistical_duration, **kwargs):
    return statistical_duration

stone_pp008 = RawMacro(code, name, description, parameters, macro, 
                       sample_generator, interval_evaluator, 
                       arg_checker)