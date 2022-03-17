# -*- coding: utf-8 -*-

from .._context import TechnicalIndicator as TI
from .._context import TimeUnit, get_cp, ts_any

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
 