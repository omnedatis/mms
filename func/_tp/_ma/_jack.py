# -*- coding: utf-8 -*-

from .._context import TechnicalIndicator as TI
from .._context import TimeUnit, get_cp, ts_any

def checker_wrapper(func):
    PARAM_CHECKER = {
        'jack_ma_order_up':three_increasing,
        'jack_ma_order_down':three_increasing,
        'jack_ma_order_thick':three_increasing,
        'jack_ma_through_price_down':night_increasing,
        'jack_ma_through_price_up':night_increasing,
        'jack_ma_through_ma_down_thrend':night_increasing,
        'jack_ma_through_ma_up_thrend':night_increasing,
    }
    def wrapper(kwargs):
        return PARAM_CHECKER[func.__name__](kwargs)
    func.check = wrapper
    return func

def three_increasing(kwargs):
    ret = {}
    for key, value in kwargs.items():
        if is_positive_int(value):
            ret[key] = is_positive_int(value)
    if kwargs['ma_short_period'] > kwargs['ma_mid_period']:
        ret['ma_short_period'] = ret.get('ma_short_period', '') +', 必須小於 中期均線天數'
    if kwargs['ma_mid_period'] > kwargs['ma_mid_period']:
        ret['ma_mid_period'] = ret.get('ma_short_period', '') + ', 必須小於 長期均線天數'
    return ret

def night_increasing(kwargs):
    ret = {}
    for key, value in kwargs.items():
        if is_positive_int(value):
            ret[key] = is_positive_int(value)
    if kwargs['period_1'] <= kwargs['period_2']:
        ret['period_1'] = ret.get('period_1', '') + ', 需大於MA均線天數(第二小)'
    if kwargs['period_2'] <= kwargs['period_3']:
        ret['period_2'] = ret.get('period_2', '') + ', 需大於MA均線天數(第三小)'
    if kwargs['period_3'] <= kwargs['period_4']:
        ret['period_3'] = ret.get('period_3', '') + ', 需大於MA均線天數(第四小)'
    if kwargs['period_4'] <= kwargs['period_5']:
        ret['period_4'] = ret.get('period_4', '') + ', 需大於MA均線天數(第五小)'
    if kwargs['period_5'] <= kwargs['period_6']:
        ret['period_5'] = ret.get('period_5', '') + ', 需大於MA均線天數(第六小)'
    if kwargs['period_6'] <= kwargs['period_7']:
        ret['period_6'] = ret.get('period_6', '') + ',需大於MA均線天數(第七小)'
    if kwargs['period_7'] <= kwargs['period_8']:
        ret['period_7'] = ret.get('period_7', '') + ', 需大於MA均線天數(第八小)'
    if kwargs['period_8'] <= kwargs['period_9']:
        ret['period_8'] = ret.get('period_8', '') + ', 需大於MA均線天數(第九小)'
    return ret

def is_positive_int(value):
    if isinstance(value, int) and value > 0:
        return ''
    return '輸入值必須為正整數'

@checker_wrapper
def jack_ma_order_up(market_id: str, **kwargs):
    """短期MA大於中期MA大於長期MA (3條).
    
    規則：
        MA(短期) > MA(中期) > MA(長期).
    
    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    ma_short_period : int
        短期均線天數.
    ma_mid_period : int
        中期均線天數.
    ma_long_period : int
        長期均線天數
    
    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        ma_short_period = kwargs['ma_short_period']
        ma_mid_period = kwargs['ma_mid_period']
        ma_long_period = kwargs['ma_long_period']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'jack_ma_order_up'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'jack_ma_order_up': {esp}") 
    ma_short = TI.MA(market_id, ma_short_period, period_type)
    ma_mid = TI.MA(market_id, ma_mid_period, period_type)
    ma_long = TI.MA(market_id, ma_long_period, period_type)
    ret = (ma_short > ma_mid) &  (ma_mid > ma_long)
    ret.rename(f'{market_id}.jack_ma_order_up({kwargs})')
    return ret

@checker_wrapper
def jack_ma_order_thick(market_id: str, **kwargs):
    """中期MA大於或小於短期、長期MA (3條).
    
    規則：
        MA(中期) > MA(短期) > MA(長期) 或 MA(中期) < MA(短期) < MA(長期).
    
    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    ma_short_period : int
        短期均線天數.
    ma_mid_period : int
        中期均線天數.
    ma_long_period : int
        長期均線天數
    
    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        ma_short_period = kwargs['ma_short_period']
        ma_mid_period = kwargs['ma_mid_period']
        ma_long_period = kwargs['ma_long_period']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'jack_ma_order_thick'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'jack_ma_order_thick': {esp}") 
    ma_short = TI.MA(market_id, ma_short_period, period_type)
    ma_mid = TI.MA(market_id, ma_mid_period, period_type)
    ma_long = TI.MA(market_id, ma_long_period, period_type)
    ret = (((ma_mid > ma_short) & (ma_short > ma_long)) | 
           ((ma_mid < ma_short) & (ma_short < ma_long)))
    ret.rename(f'{market_id}.jack_ma_order_thick({kwargs})')
    return ret    

@checker_wrapper     
def jack_ma_order_down(market_id: str, **kwargs):
    """短期MA小於中期MA小於長期MA (3條).
    
    規則：
        MA(短期) < MA(中期) < MA(長期).
    
    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    ma_short_period : int
        短期均線天數.
    ma_mid_period : int
        中期均線天數.
    ma_long_period : int
        長期均線天數
    
    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        ma_short_period = kwargs['ma_short_period']
        ma_mid_period = kwargs['ma_mid_period']
        ma_long_period = kwargs['ma_long_period']
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'jack_ma_order_down'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'jack_ma_order_down': {esp}")  
    ma_short = TI.MA(market_id, ma_short_period, period_type)
    ma_mid = TI.MA(market_id, ma_mid_period, period_type)
    ma_long = TI.MA(market_id, ma_long_period, period_type)
    ret = (ma_short < ma_mid) &  (ma_mid < ma_long)
    ret.rename(f'{market_id}.jack_ma_order_down({kwargs})')
    return ret   

@checker_wrapper
def jack_ma_through_price_down(market_id: str, **kwargs):
    """收盤價向下穿越任一MA (9條).
    
    規則：
        收盤價向下穿越任一 MA.
    
    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    period_1 : int
        MA均線天數(最小).
    period_2 : int
        MA均線天數(第二小).
    period_3 : int
        MA均線天數(第三小).
    period_4 : int
        MA均線天數(第四小).
    period_5 : int
        MA均線天數(第五小).
    period_6 : int
        MA均線天數(第六小).
    period_7 : int
        MA均線天數(第七小).
    period_8 : int
        MA均線天數(第八小).
    period_9 : int
        MA均線天數(第九小).
        
    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        periods = [kwargs[f'period_{idx}'] for idx in range(1, 10)]
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'jack_ma_through_price_down'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'jack_ma_through_price_down': {esp}")         
    cp = get_cp(market_id)
    conds = [cp < TI.MA(market_id, each, period_type) 
             for each in periods]
    conds = [each & ~each.shift(1, period_type) for each in conds]
    ret = ts_any(*conds)
    ret.rename(f'{market_id}.jack_ma_through_price_down({kwargs})')
    return ret  

@checker_wrapper
def jack_ma_through_price_up(market_id: str, **kwargs):
    """收盤價向上穿越任一MA (9條).
    
    規則：
        收盤價向上穿越任一 MA.
    
    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    period_1 : int
        MA均線天數(最小).
    period_2 : int
        MA均線天數(第二小).
    period_3 : int
        MA均線天數(第三小).
    period_4 : int
        MA均線天數(第四小).
    period_5 : int
        MA均線天數(第五小).
    period_6 : int
        MA均線天數(第六小).
    period_7 : int
        MA均線天數(第七小).
    period_8 : int
        MA均線天數(第八小).
    period_9 : int
        MA均線天數(第九小).
        
    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        periods = [kwargs[f'period_{idx}'] for idx in range(1, 10)]
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'jack_ma_through_price_up'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'jack_ma_through_price_up': {esp}")         
    cp = get_cp(market_id)
    conds = [cp > TI.MA(market_id, each, period_type) 
             for each in periods]
    conds = [each & ~each.shift(1, period_type) for each in conds]
    ret = ts_any(*conds)
    ret.rename(f'{market_id}.jack_ma_through_price_up({kwargs})')
    return ret  

@checker_wrapper
def jack_ma_through_ma_down_thrend(market_id: str, **kwargs):
    """任一短天期MA向下穿越長天期MA (9條).
    
    規則：
        任一短天期 MA 向下穿越任一長天期 MA.
    
    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    period_1 : int
        MA均線天數(最小).
    period_2 : int
        MA均線天數(第二小).
    period_3 : int
        MA均線天數(第三小).
    period_4 : int
        MA均線天數(第四小).
    period_5 : int
        MA均線天數(第五小).
    period_6 : int
        MA均線天數(第六小).
    period_7 : int
        MA均線天數(第七小).
    period_8 : int
        MA均線天數(第八小).
    period_9 : int
        MA均線天數(第九小).
        
    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        periods = [kwargs[f'period_{idx}'] for idx in range(1, 10)]
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'jack_ma_through_ma_down_thrend'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'jack_ma_through_ma_down_thrend': {esp}")       
    mas = [TI.MA(market_id, each, period_type) for each in periods]
    conds = [mas[i] < each for i in range(len(mas)) for each in mas[i+1:]]
    conds = [each & ~each.shift(1, period_type) for each in conds]
    ret = ts_any(*conds)
    ret.rename(f'{market_id}.jack_ma_through_ma_down_thrend({kwargs})')
    return ret  

@checker_wrapper
def jack_ma_through_ma_up_thrend(market_id: str, **kwargs):
    """任一短天期MA向上穿越長天期MA (9條).
    
    規則：
        任一短天期 MA 向上穿越任一長天期 MA.
    
    Arguments
    ---------
    market_id : string
        目標市場ID
    period_type : string
        MA 取樣週期，有效值為
        - 'D' : 日
        - 'W' : 週
        - 'M' : 月
    period_1 : int
        MA均線天數(最小).
    period_2 : int
        MA均線天數(第二小).
    period_3 : int
        MA均線天數(第三小).
    period_4 : int
        MA均線天數(第四小).
    period_5 : int
        MA均線天數(第五小).
    period_6 : int
        MA均線天數(第六小).
    period_7 : int
        MA均線天數(第七小).
    period_8 : int
        MA均線天數(第八小).
    period_9 : int
        MA均線天數(第九小).
        
    """
    try:
        period_type = TimeUnit.get(kwargs['period_type'])
        periods = [kwargs[f'period_{idx}'] for idx in range(1, 10)]
    except KeyError as esp:
        raise RuntimeError(f"miss argument '{esp.args[0]}' when calling "
                           "'jack_ma_through_ma_up_thrend'")
    except ValueError as esp:
        raise RuntimeError("invalid argument error when calling"
                           f"'jack_ma_through_ma_up_thrend': {esp}")        
    mas = [TI.MA(market_id, each, period_type) for each in periods]
    conds = [mas[i] > each for i in range(len(mas)) for each in mas[i+1:]]
    conds = [each & ~each.shift(1, period_type) for each in conds]
    ret = ts_any(*conds)
    ret.rename(f'{market_id}.jack_ma_through_ma_up_thrend({kwargs})')
    return ret 
 