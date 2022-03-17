# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

@author: Jeff
"""
import pickle
import os
from typing import Any, List, Union
import pandas as pd
import numpy as np

def mkdir(path: str):
    """make dir recursively.
    
    Parameters
    ----------
    path: str
        Path of directory to be made.
        
    """
    if not os.path.exists(path):
        os.makedirs(path)

def pickle_dump(data: Any, file: str):
    """dump object to file by pickle.
    
    Parameters
    ----------
    data: object
        Object to be dump.
    file: str
        Filename where the data is dump to.    
        
    """
    mkdir(os.path.dirname(file))
    with open(file, 'wb') as f:
        pickle.dump(data, f)
        
def pickle_load(file: str) -> Any:
    """load object from file by pickle.
    
    Parameters
    ----------
    file: str
        Filename where the data is loaded from.
        
    Returns
    -------
    object
        The object loaded from given file.
        
    """    
    with open(file, 'rb') as f:
        ret = pickle.load(f)
    return ret
        
def esp2str(esp:Exception) -> str:
    """Trans exception to string."""
    return f'{type(esp).__name__}: {str(esp)}'

def fast_concat(data: List[Union[pd.Series, pd.DataFrame]]) -> pd.DataFrame:
    """Concat Pandas's Series and DataFrames with common index fastly."""
    index = data[0].index
    columns = []
    values = []
    for each in data:
        if isinstance(each, pd.DataFrame):
            columns += list(each.columns)
            values += list(each.values.T)
        else:
            columns.append(each.name)
            values.append(each.values)
    return pd.DataFrame(np.array(values).T, columns=columns, index=index)
