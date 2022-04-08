# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:34:18 2021
@author: Allen
"""
import logging
from logging import handlers
class Logger:
    def __init__(self, fp, level=logging.INFO, when='D', backup_count=7,
                 fmt='%(asctime)s - %(levelname)s : %(message)s'):
        self.logger = logging.getLogger(fp)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(level)
        # 檔案輸出
        th = handlers.TimedRotatingFileHandler(filename=fp, when=when, 
                                               backupCount=backup_count,
                                               encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(th)