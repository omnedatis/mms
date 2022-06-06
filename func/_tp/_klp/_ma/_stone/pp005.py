import numpy as np

from func._tp._ma import _stone as tp
from .common import get_ma, RawMacro, MacroParam, ParamType

code = 'stone_pp005'
name = '商智MA指標-PP005(盤整指標)'
macro = tp.stone_pp005
description = macro.__doc__
parameters = [
    MacroParam('base_period', '基準均線天期(b)', '基準均線天期(b)', 
               ParamType.INT, 1),
    MacroParam('short_term_period', '短期均線天期(s)', '短期均線天期(s)', 
               ParamType.INT, 5),
    MacroParam('long_term_period_1', '長期均線天期(t1)', '長期均線天期(t1)', 
               ParamType.INT, 10),
    MacroParam('long_term_period_2', '長期均線天期(t2)', '長期均線天期(t2)', 
               ParamType.INT, 20),
    MacroParam('long_term_period_3', '長期均線天期(t3)', '長期均線天期(t3)', 
               ParamType.INT, 40),
    MacroParam('long_term_period_4', '長期均線天期(t4)', '長期均線天期(t4)',
               ParamType.INT, 60),
    MacroParam('long_term_period_5', '長期均線天期(t5)', '長期均線天期(t5)', 
               ParamType.INT, 120),
    MacroParam('long_statistical_duration', '統計長天期時使用的取樣天數(n1)', 
               '統計長天期時使用的取樣天數(n1)', ParamType.INT, 40),  
    MacroParam('min_occurence', '判斷發生事件是否大於或等於幾次的次數(o)', 
               '判斷發生事件是否大於或等於幾次的次數(o)', ParamType.INT, 20),
    MacroParam('short_statistical_duration', '統計短天期時使用的取樣天數(n2)', 
               '統計短天期時使用的取樣天數(n2)', ParamType.INT, 5),
    MacroParam('threshold_of_difference_rate', '差異率臨界值(dr)', 
               '差異率臨界值(dr)', ParamType.FLOAT, 0.02)]

def arg_checker(base_period, short_term_period, long_term_period_1, 
                long_term_period_2, long_term_period_3, long_term_period_4, 
                long_term_period_5, long_statistical_duration, min_occurence,
                short_statistical_duration, threshold_of_difference_rate):
    ret = {}
    if threshold_of_difference_rate <= 0:
        ret['threshold_of_difference_rate'] = "值必須大於0"
    limit = 0
    if base_period <= limit:
        ret['base_period'] = f"值必須大於{limit}"
    limit = max([limit, base_period])
    if short_term_period <= limit:
        ret['short_term_period'] = f"值必須大於{limit}"
    limit = max([limit, short_term_period])
    if long_term_period_1 <= limit:
        ret['long_term_period_1'] = f"值必須大於{limit}"   
    limit = max([limit, long_term_period_1])
    if long_term_period_2 <= limit:
        ret['long_term_period_2'] = f"值必須大於{limit}"   
    limit = max([limit, long_term_period_2])
    if long_term_period_3 <= limit:
        ret['long_term_period_3'] = f"值必須大於{limit}"     
    limit = max([limit, long_term_period_3])
    if long_term_period_4 <= limit:
        ret['long_term_period_4'] = f"值必須大於{limit}"     
    limit = max([limit, long_term_period_4])
    if long_term_period_5 <= limit:
        ret['long_term_period_5'] = f"值必須大於{limit}"         
    limit = 0
    if min_occurence <= limit:
        ret['min_occurence'] = f"值必須大於{limit}"
    limit = max([limit, min_occurence-1])
    if long_statistical_duration <= limit:
        ret['long_statistical_duration'] = f"值必須大於{limit}"     
    if short_statistical_duration < 1 or short_statistical_duration > long_statistical_duration :
        ret['short_statistical_duration'] = f"值必須界於1~{long_statistical_duration}"           
    return ret

def gen_cp_by_random(base_period, short_period, long_periods, 
                     short_duration, long_duration, min_occurence, threshold, 
                     sigma=0.01, batch_size=10000, max_times=10):
    def get_ma(cps, period):
        if period <= 1:
            return cps
        temp = np.cumsum(cps, axis=1)
        ret = np.concatenate([temp[:,:period] / np.arange(1, period+1),
                              (temp[:,period:] - temp[:,:-period]) / period], axis=1)
        return ret
    N = batch_size
    L = long_periods[-1] + long_duration - 1
    for i in range(max_times):
        cps = np.cumprod(1 + np.random.normal(0, sigma, (N, L)), axis=1) * 100
        base = get_ma(cps, base_period)[:, -long_duration:]
        short = get_ma(cps, short_period)[:, -short_duration:]
        cond_1 = (abs(short / base[:, -short_duration:] - 1) <= threshold).all(axis=1)
        cond_2 = (np.array([abs(get_ma(cps, p)[:, -long_duration:] / base - 1) 
                            for p in long_periods]).mean(axis=0) < threshold
                  ).sum(axis=1) >= min_occurence
        """
        cond_2 = (np.array([abs(get_ma(cps, p)[:, -long_duration:] / base - 1
                                ) < threshold for p in long_periods]).all(axis=0)
                  ).sum(axis=1) >= min_occurence
        """
        conditions = cond_1 & cond_2
        if conditions.any():
            return cps[conditions][0]
    return None
    
def gen_cp_by_rule(base_period, short_period, long_periods, 
                   short_duration, min_occurence, threshold, sigma=0.01):
    def fix_cps(ts, r):
        rs = ts / ts[0] - 1
        for i in range(int(np.ceil(abs(rs).max() / r))):
            if rs.max() > r:
                fixs = rs - r
                fixs[fixs<0] = 0
                rs = rs - 2 * fixs
            if rs.min() < -r:
                fixs = -r - rs
                fixs[fixs<0] = 0
                rs = rs + 2 * fixs
            if abs(rs).max() <= r:
                break
        return (1+rs) * ts[0]
    dlen = max([long_periods[-1] + min_occurence,
                short_period + short_duration])
    cps = 100 * np.cumprod(1 + np.random.normal(0, 0.01, size=dlen))[::-1]
    ret = fix_cps(cps, threshold)
    return ret
    
def check_cps(cps, base_period, short_period, long_periods, 
              short_duration, long_duration, min_occurence, threshold):
    base = get_ma(cps, base_period)[-long_duration:]
    short = get_ma(cps, short_period)[-short_duration:]
    cond_1 = (abs(short / base[-short_duration:] - 1) <= threshold).all()
    cond_2 = (np.array([abs(get_ma(cps, p)[-long_duration:] / base - 1) 
                        for p in long_periods]).mean(axis=0) < threshold
              ).sum() >= min_occurence
    assert cond_1 and cond_2
    
def sample_generator(base_period, short_term_period, long_term_period_1, 
                     long_term_period_2, long_term_period_3, long_term_period_4, 
                     long_term_period_5, long_statistical_duration, min_occurence,
                     short_statistical_duration, threshold_of_difference_rate, 
                     lookback_len=-1):
    
    short_period = short_term_period
    long_periods = [long_term_period_1, long_term_period_2, long_term_period_3, 
                    long_term_period_4, long_term_period_5]
    threshold = threshold_of_difference_rate
    short_duration = short_statistical_duration
    long_duration = long_statistical_duration
    
    if lookback_len < 0:
        lookback_len = 0#long_term_period_5 * 2
    dlen = long_duration + lookback_len
    
    tails = gen_cp_by_random(base_period, short_period, long_periods, 
                             short_duration, long_duration, min_occurence, threshold)
    if tails is None:
        tails = gen_cp_by_rule(base_period, short_period, long_periods, 
                               short_duration, min_occurence, threshold)
    check_cps(tails, base_period, short_period, long_periods, 
              short_duration, long_duration, min_occurence, threshold)
    heads = np.cumprod(1 + np.random.normal(0, 0.01, size=dlen))[::-1] * tails[0]
    cps = np.concatenate([heads, tails])
    periods = [base_period, short_period] + long_periods
    ret = {f'MA {p}': get_ma(cps, p)[-dlen:].tolist() for p in periods}
    return ret  

def interval_evaluator(long_statistical_duration, **kwargs):
    return long_statistical_duration

stone_pp005 = RawMacro(code, name, description, parameters, macro, 
                       sample_generator, interval_evaluator, 
                       arg_checker)
