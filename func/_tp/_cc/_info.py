# -*- coding: utf-8 -*-

from enum import Enum
from typing import Any, List, NamedTuple

class _Dtype(NamedTuple):
    NAME: str
    TYPE: type
    
class Dtype(_Dtype, Enum):
    STRING = _Dtype('string', str)
    INTEGER = _Dtype('integer', int)
    FLOAT = _Dtype('float', float)
    
    @classmethod
    def from_name(cls, tar: str):
        for each in cls:
            if each.NAME == tar:
                return each
        raise ValueError()
        
class ParameterInfo(NamedTuple):
    id_: str
    name: str
    dtype: str
    ctype: str
    default: Any
    
    def to_dict(self):
        ret = {'id': self.id_,
               'name': self.name,
               'dtype': self.dtype,
               'ctype': self.ctype,
               'def_value': self.default}
        return ret
    
    @classmethod
    def from_dict(cls, info):
        id_ = info['id']
        name = info['name']
        dtype = info['dtype']
        ctype = info['ctype']
        default = info['def_value']
        return cls(id_, name, dtype, ctype, default)
    
class PatternInfo(NamedTuple):
    id_: str
    func: str
    name: str
    description: str
    paras: List[ParameterInfo]
    
    def to_dict(self):
        ret = {'id': self.id_,
               'func': self.func,
               'name': self.name,
               'description': self.description,
               'params': [each.to_dict() for each in self.paras]}
        return ret
    
    @classmethod
    def from_dict(cls, info):
        id_ = info['id']
        func = info['func']
        name = info['name']
        description = info['description']
        paras = [ParameterInfo.from_dict(each) for each in info['params']]
        return cls(id_, func, name, description, paras)
        
        