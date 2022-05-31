# -*- coding: utf-8 -*-
from .._context import TechnicalIndicator, TimeUnit, get_cp, ts_all, ts_any, ts_average, ts_max


def is_positive_int(value):
    if isinstance(value, int) and value > 0:
        return ''
    return '輸入值必須為正整數'

def all_positive_int_check(kwargs):
    ret = {}
    for key, value in kwargs.items():
        if is_positive_int(value):
            ret[key] = is_positive_int(value)
    if 'min_occurence' in ret:
        if kwargs['min_occurence'] == 0:
            del ret['min_occurence']
    if 'occurence_threshold' in ret:
        if kwargs['min_occurence'] == 0:
            del ret['min_occurence']
    return ret

def valid_index_check(kwargs):
    ret = {}
    ret.update(all_positive_int_check(kwargs))
    if kwargs.get('ridge_index'):
        if kwargs['ridge_index'] > 8:
            ret['ridge_index'] = ret.get('ridge_index', '') + ' 輸入值不可超出變數數量'
    if kwargs.get('trough_index'):
        if kwargs['trough_index'] > 8:
            ret['trough_index'] = ret.get('trough_index', '') +' 輸入值不可超出變數數量'
    return ret

def occur_time_check(kwargs):
    ret = {}
    occur = kwargs.get('occurence_threshold') or kwargs.get('min_occurence')
    if occur > kwargs['statistical_duration']:
        ret['statistical_duration'] = ret.get('statistical_duration', '') + " 輸入值應大於發生天數"
    return ret

def index_and_occur_time_check(kwargs):
    ret = {}
    ret.update(valid_index_check(kwargs))
    ret.update(occur_time_check(kwargs))

    return ret

def stone_pp005_check(kwargs):
    ret = {}
    if is_positive_int(kwargs['base_period']):
        ret['base_period'] = is_positive_int(kwargs['base_period'])
    if is_positive_int(kwargs['short_term_period']):
        ret['short_term_period'] = is_positive_int(kwargs['short_term_period'])
    if is_positive_int(kwargs['long_term_period_1']):
        ret['long_term_period_1'] = is_positive_int(kwargs['long_term_period_1'])
    if is_positive_int(kwargs['long_term_period_2']):
        ret['long_term_period_2'] = is_positive_int(kwargs['long_term_period_2'])
    if is_positive_int(kwargs['long_term_period_3']):
        ret['long_term_period_3'] = is_positive_int(kwargs['long_term_period_3'])
    if is_positive_int(kwargs['long_term_period_4']):
        ret['long_term_period_4'] = is_positive_int(kwargs['long_term_period_4'])
    if is_positive_int(kwargs['long_term_period_5']):
        ret['long_term_period_5'] = is_positive_int(kwargs['long_term_period_5'])
    if is_positive_int(kwargs['long_statistical_duration']):
        ret['long_statistical_duration'] = is_positive_int(kwargs['long_statistical_duration'])
    if is_positive_int(kwargs['short_statistical_duration']):
        ret['short_statistical_duration'] = is_positive_int(kwargs['short_statistical_duration'])
    if kwargs['min_occurence'] > kwargs['short_statistical_duration']:
        ret['long_statistical_duration'] = ret.get('long_statistical_duration', '') + "輸入值應大於發生天數"
    if not isinstance(kwargs['threshold_of_difference_rate'], float) or kwargs['threshold_of_difference_rate'] < 0:
        ret['threshold_of_difference_rate'] = ret.get('threshold_of_difference_rate', '') + "輸入值應為正浮點數"
    return ret

def stone_pp000(market_id: str, **kwargs):
    """pp000.

    規則：
        MA (t) 大於 MA (b).

    Arguments
    ---------
    market_id : string
        目標市場ID
    target_period : int
        目標均線天期(t).
    base_period : int
        基準均線天期(o).
    
    """
    try:
        target_period = kwargs['target_period']
        base_period = kwargs['base_period']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp000'")
    mat = TechnicalIndicator.MA(market_id, target_period)
    mab = TechnicalIndicator.MA(market_id, base_period)
    ret = mat > mab
    ret.rename(f'{market_id}.stone_pp000({kwargs})')
    return ret
stone_pp000.check = all_positive_int_check

def stone_pp001(market_id: str, **kwargs):
    """pp001.

    規則：
        過去 n 日內是否發生 o 次 MA (t) 大於 MA (b).

    Arguments
    ---------
    market_id : string
        目標市場ID
    target_period : int
        目標均線天期(t).
    base_period : int
        基準均線天期(b).
    statistical_duration : int
        判斷過去幾天內的天數(n).
    min_occurence : int
        判斷發生事件是否大於或等於幾次的次數(o).

    """
    try:
        target_period = kwargs['target_period']
        base_period = kwargs['base_period']
        statistical_duration = kwargs['statistical_duration']
        min_occurence = kwargs['min_occurence']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp001'")
    mat = TechnicalIndicator.MA(market_id, target_period)
    mab = TechnicalIndicator.MA(market_id, base_period)
    ret = (mat > mab).sampling(statistical_duration).sum() >= min_occurence
    ret.rename(f'{market_id}.stone_pp001({kwargs})')
    return ret
stone_pp001.check = occur_time_check

def stone_pp002(market_id: str, **kwargs):
    """pp002.

    規則：
        過去 n 天內的MA (t) 都大於各個基準比較對象 MA (b1), MA(b2), …, MA(b7).

    Arguments
    ---------
    market_id : string
        目標市場ID
    target_period : int
        目標均線天期(t).
    base_period_1 : int
        基準均線天期(b1).
    base_period_2 : int
        基準均線天期(b2).
    base_period_3 : int
        基準均線天期(b3).
    base_period_4 : int
        基準均線天期(b4).
    base_period_5 : int
        基準均線天期(b5).
    base_period_6 : int
        基準均線天期(b6).
    base_period_7 : int
        基準均線天期(b7).
    statistical_duration : int
        判斷過去幾天內的天數(n).

    """
    try:
        target_period = kwargs['target_period']
        base_periods = [kwargs[f'base_period_{idx}'] for idx in range(1, 8)]
        statistical_duration = kwargs['statistical_duration']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp002'")
    mat = TechnicalIndicator.MA(market_id, target_period)
    mabs = [TechnicalIndicator.MA(market_id, each) for each in base_periods]
    ret = (mat > ts_max(*mabs)).sampling(statistical_duration).all()
    ret.rename(f'{market_id}.stone_pp002({kwargs})')
    return ret
stone_pp002.check = all_positive_int_check

def stone_pp003(market_id: str, **kwargs):
    """pp003.

    規則：
        MA(t1) >= MA(t2) >= ... >= MA(tk-1) >= MA(tk)，且
        MA(tk) <= MA(tk + 1) <= ... <= MA(t8)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_1 : int
        目標均線天期(t1).
    period_2 : int
        目標均線天期(t2).
    period_3 : int
        目標均線天期(t3).
    period_4 : int
        目標均線天期(t4).
    period_5 : int
        目標均線天期(t5).
    period_6 : int
        目標均線天期(t6).
    period_7 : int
        目標均線天期(t7).
    period_8 : int
        目標均線天期(t8).
    trough_index : int
        指定第幾個天期為波谷天期(k).

    """
    try:
        trough_index = kwargs['trough_index']
        down_periods = [kwargs[f'period_{idx+1}'] for idx in range(trough_index+1)]
        up_periods = [kwargs[f'period_{idx+1}'] for idx in range(trough_index, 8)]
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp003'")
    upmas = [TechnicalIndicator.MA(market_id, each) for each in up_periods]
    downmas = [TechnicalIndicator.MA(market_id, each) for each in down_periods]
    conds = ([upmas[idx] <= upmas[idx+1] for idx in range(len(upmas)-1)] +
             [downmas[idx] >= downmas[idx+1] for idx in range(len(downmas)-1)])
    ret = ts_all(*conds)
    ret.rename(f'{market_id}.stone_pp003({kwargs})')
    return ret
stone_pp003.check = valid_index_check

def stone_pp004(market_id: str, **kwargs):
    """pp004.

    規則：
        MA(t1) <= MA(t2) <= ... <= MA(tk-1) <= MA(tk)，且
        MA(tk) >= MA(tk + 1) >= ... >= MA(t8)

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_1 : int
        目標均線天期(t1).
    period_2 : int
        目標均線天期(t2).
    period_3 : int
        目標均線天期(t3).
    period_4 : int
        目標均線天期(t4).
    period_5 : int
        目標均線天期(t5).
    period_6 : int
        目標均線天期(t6).
    period_7 : int
        目標均線天期(t7).
    period_8 : int
        目標均線天期(t8).
    ridge_index : int
        指定第幾個天期為波峰天期(k).

    """
    try:
        ridge_index = kwargs['ridge_index']
        up_periods = [kwargs[f'period_{idx+1}'] for idx in range(ridge_index+1)]
        down_periods = [kwargs[f'period_{idx+1}'] for idx in range(ridge_index, 8)]
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp004'")
    upmas = [TechnicalIndicator.MA(market_id, each) for each in up_periods]
    downmas = [TechnicalIndicator.MA(market_id, each) for each in down_periods]
    conds = ([upmas[idx] <= upmas[idx+1] for idx in range(len(upmas)-1)] +
             [downmas[idx] >= downmas[idx+1] for idx in range(len(downmas)-1)])
    ret = ts_all(*conds)
    ret.rename(f'{market_id}.stone_pp004({kwargs})')
    return ret
stone_pp004.check = valid_index_check

def stone_pp005(market_id: str, **kwargs):
    """pp005.

    規則：
        -
        1. 在過去n1天內，
           Average(abs(MA(l1)-MA(b)),...abs(MA(l5)-MA(b))) / MA(b) < dr
           至少發生o次；
        2. 在過去n2天內從未發生過 abs(MA(s)-MA(b)) / MA(b) > dr

    Arguments
    ---------
    market_id : string
        目標市場ID
    base_period : int
        基準均線天期(b).
    short_term_period : int
        短期均線天期(s).
    long_term_period_1 : int
        長期均線天期(t1).
    long_term_period_2 : int
        長期均線天期(t2).
    long_term_period_3 : int
        長期均線天期(t3).
    long_term_period_4 : int
        長期均線天期(t4).
    long_term_period_5 : int
        長期均線天期(t5).
    long_statistical_duration : int
        統計長天期時使用的取樣天數(n1).
    min_occurence : int
        判斷發生事件是否大於或等於幾次的次數(o).
    short_statistical_duration : int
        統計短天期時使用的取樣天數(n2).
    threshold_of_difference_rate : float
        差異率臨界值(dr)

    """
    try:
        base_period = kwargs['base_period']
        short_period = kwargs['short_term_period']
        long_periods = [kwargs[f'long_term_period_{idx}'] for idx in range(1, 6)]
        long_duration = kwargs['long_statistical_duration']
        short_duration = kwargs['short_statistical_duration']
        min_occurence = kwargs['min_occurence']
        threshold = kwargs['threshold_of_difference_rate']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp004'")
    base_ma = TechnicalIndicator.MA(market_id, base_period)
    short_ma = TechnicalIndicator.MA(market_id, short_period)
    long_mas = [TechnicalIndicator.MA(market_id, each) for each in long_periods]
    cond1 = (ts_average(*[abs(each / base_ma - 1) for each in long_mas]) < threshold
             ).sampling(long_duration).sum() >= min_occurence
    cond2 = (abs(short_ma / base_ma - 1) <= threshold).sampling(short_duration).all()
    ret = cond1 & cond2
    ret.rename(f'{market_id}.stone_pp005({kwargs})')
    return ret
stone_pp005.check = stone_pp005_check

def stone_pp006(market_id: str, **kwargs):
    """pp006.

    規則：
        在過去n天內
        1. MA(t1) >= MA(t2) >= ... >= MA(tk-1) >= MA(tk)，且
        2. MA(tk) <= MA(tk + 1) <= ... <= MA(t8)
        至少 o 次

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_1 : int
        目標均線天期(t1).
    period_2 : int
        目標均線天期(t2).
    period_3 : int
        目標均線天期(t3).
    period_4 : int
        目標均線天期(t4).
    period_5 : int
        目標均線天期(t5).
    period_6 : int
        目標均線天期(t6).
    period_7 : int
        目標均線天期(t7).
    period_8 : int
        目標均線天期(t8).
    trough_index : int
        指定第幾個天期為波谷天期(k).
    statistical_duration : int
        判斷過去幾天內的天數(n).
    min_occurence : int
        判斷發生事件是否大於或等於幾次的次數(o).

    """
    try:
        trough_index = kwargs['trough_index']
        down_periods = [kwargs[f'period_{idx+1}'] for idx in range(trough_index+1)]
        up_periods = [kwargs[f'period_{idx+1}'] for idx in range(trough_index, 8)]
        statistical_duration = kwargs['statistical_duration']
        min_occurence = kwargs['min_occurence']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp006'")
    upmas = [TechnicalIndicator.MA(market_id, each) for each in up_periods]
    downmas = [TechnicalIndicator.MA(market_id, each) for each in down_periods]
    conds = ([upmas[idx] <= upmas[idx+1] for idx in range(len(upmas)-1)] +
             [downmas[idx] >= downmas[idx+1] for idx in range(len(downmas)-1)])
    ret = ts_all(*conds).sampling(statistical_duration).sum() >= min_occurence
    ret.rename(f'{market_id}.stone_pp006({kwargs})')
    return ret
stone_pp006.check = index_and_occur_time_check

def stone_pp007(market_id: str, **kwargs):
    """pp007.

    規則：
        在過去n天內
        1. MA(t1) >= MA(t2) >= ... >= MA(tk-1) >= MA(tk)，且
        2. MA(tk) <= MA(tk + 1) <= ... <= MA(t8)
        超過 o 次

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_1 : int
        目標均線天期(t1).
    period_2 : int
        目標均線天期(t2).
    period_3 : int
        目標均線天期(t3).
    period_4 : int
        目標均線天期(t4).
    period_5 : int
        目標均線天期(t5).
    period_6 : int
        目標均線天期(t6).
    period_7 : int
        目標均線天期(t7).
    period_8 : int
        目標均線天期(t8).
    trough_index : int
        指定第幾個天期為波谷天期(k).
    statistical_duration : int
        判斷過去幾天內的天數(n).
    occurence_threshold : int
        在指定的統計天數中發生次數的臨界值(o).

    """
    try:
        trough_index = kwargs['trough_index']
        down_periods = [kwargs[f'period_{idx+1}'] for idx in range(trough_index+1)]
        up_periods = [kwargs[f'period_{idx+1}'] for idx in range(trough_index, 8)]
        statistical_duration = kwargs['statistical_duration']
        occurence_threshold = kwargs['occurence_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp007'")
    upmas = [TechnicalIndicator.MA(market_id, each) for each in up_periods]
    downmas = [TechnicalIndicator.MA(market_id, each) for each in down_periods]
    conds = ([upmas[idx] <= upmas[idx+1] for idx in range(len(upmas)-1)] +
             [downmas[idx] >= downmas[idx+1] for idx in range(len(downmas)-1)])
    ret = ts_all(*conds).sampling(statistical_duration).sum() > occurence_threshold
    ret.rename(f'{market_id}.stone_pp007({kwargs})')
    return ret
stone_pp007.check = index_and_occur_time_check

def stone_pp008(market_id: str, **kwargs):
    """pp008.

    規則：
        在過去n天內
        1. MA(t1) <= MA(t2) <= ... <= MA(tk-1) <= MA(tk)，且
        2. MA(tk) >= MA(tk + 1) >= ... >= MA(t8)
        至少 o 次

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_1 : int
        目標均線天期(t1).
    period_2 : int
        目標均線天期(t2).
    period_3 : int
        目標均線天期(t3).
    period_4 : int
        目標均線天期(t4).
    period_5 : int
        目標均線天期(t5).
    period_6 : int
        目標均線天期(t6).
    period_7 : int
        目標均線天期(t7).
    period_8 : int
        目標均線天期(t8).
    ridge_index : int
        指定第幾個天期為波峰天期(k).
    statistical_duration : int
        判斷過去幾天內的天數(n).
    min_occurence : int
        判斷發生事件是否大於或等於幾次的次數(o).

    """
    try:
        ridge_index = kwargs['ridge_index']
        up_periods = [kwargs[f'period_{idx+1}'] for idx in range(ridge_index+1)]
        down_periods = [kwargs[f'period_{idx+1}'] for idx in range(ridge_index, 8)]
        statistical_duration = kwargs['statistical_duration']
        min_occurence = kwargs['min_occurence']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp008'")
    upmas = [TechnicalIndicator.MA(market_id, each) for each in up_periods]
    downmas = [TechnicalIndicator.MA(market_id, each) for each in down_periods]
    conds = ([upmas[idx] <= upmas[idx+1] for idx in range(len(upmas)-1)] +
             [downmas[idx] >= downmas[idx+1] for idx in range(len(downmas)-1)])
    ret = ts_all(*conds).sampling(statistical_duration).sum() >= min_occurence
    ret.rename(f'{market_id}.stone_pp008({kwargs})')
    return ret
stone_pp008.check = index_and_occur_time_check

def stone_pp009(market_id: str, **kwargs):
    """pp009.

    規則：
        在過去n天內
        1. MA(t1) <= MA(t2) <= ... <= MA(tk-1) <= MA(tk)，且
        2. MA(tk) >= MA(tk + 1) >= ... >= MA(t8)
        超過 o 次

    Arguments
    ---------
    market_id : string
        目標市場ID
    period_1 : int
        目標均線天期(t1).
    period_2 : int
        目標均線天期(t2).
    period_3 : int
        目標均線天期(t3).
    period_4 : int
        目標均線天期(t4).
    period_5 : int
        目標均線天期(t5).
    period_6 : int
        目標均線天期(t6).
    period_7 : int
        目標均線天期(t7).
    period_8 : int
        目標均線天期(t8).
    ridge_index : int
        指定第幾個天期為波峰天期(k).
    statistical_duration : int
        判斷過去幾天內的天數(n).
    occurence_threshold : int
        在指定的統計天數中發生次數的臨界值(o).

    """
    try:
        ridge_index = kwargs['ridge_index']
        up_periods = [kwargs[f'period_{idx+1}'] for idx in range(ridge_index+1)]
        down_periods = [kwargs[f'period_{idx+1}'] for idx in range(ridge_index, 8)]
        statistical_duration = kwargs['statistical_duration']
        occurence_threshold = kwargs['occurence_threshold']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'stone_pp009'")
    upmas = [TechnicalIndicator.MA(market_id, each) for each in up_periods]
    downmas = [TechnicalIndicator.MA(market_id, each) for each in down_periods]
    conds = ([upmas[idx] <= upmas[idx+1] for idx in range(len(upmas)-1)] +
             [downmas[idx] >= downmas[idx+1] for idx in range(len(downmas)-1)])
    ret = ts_all(*conds).sampling(statistical_duration).sum() > occurence_threshold
    ret.rename(f'{market_id}.stone_pp009({kwargs})')
    return ret
stone_pp009.check = index_and_occur_time_check
