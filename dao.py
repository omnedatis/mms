import datetime
import json
import os
import numpy as np
import pandas as pd
import pickle
import traceback
from sqlalchemy import create_engine
from threading import Lock
from typing import Any, Dict, List, NamedTuple, Optional, Union

from model import (ModelInfo, PatternInfo,
                pickle_dump, pickle_load,
                get_filed_name_of_future_return, set_model_execution_start)
from const import (LOCAL_DB, DATA_LOC, EXCEPT_DATA_LOC,
                   MarketDistField, ModelExecution, PredictResultField,
                   PatternResultField, MarketPeriodField, MarketScoreField, 
                   MarketInfoField, DSStockInfoField,
                   MarketStatField, ModelExecution, ScoreMetaField, BatchType)
import logging
from utils import Cache

class MimosaDB:
    """
    用來取得資料庫資料的物件，需在同資料夾下設置 config

    """
    config = {}
    CREATE_BY='SYS_BATCH'
    MODIFY_BY='SYS_BATCH'
    READ_ONLY=False

    def __init__(self, db_name='mimosa', mode='dev', read_only=False):
        """
        根據傳入參數取得指定的資料庫設定
        """
        def init_config():
            """
            load the configuration
            """
            current_path = os.path.split(os.path.realpath(__file__))[0]
            config = json.load(open('%s/config.json'%(current_path)))[db_name][mode]
            return config
        self.config = init_config()
        self._local_db_lock = Lock()
        self.pattern_cache = Cache(size=400000)
        self.future_reture_cache = Cache(size=100000)
        self.READ_ONLY = read_only

    def _engine(self):
      """
      透過資料庫設定取得資料庫連線引擎物件
      """
      engine_conf = self.config['engine']
      ip_addr = self.config['IP address']
      port = self.config['Port']
      user = self.config['User']
      password = self.config['Password']
      db_name = self.config['Database name']
      charset = self.config['charset']
      engine = create_engine('%s://%s:%s@%s:%s/%s?charset=%s'%
                              (engine_conf, user, password, ip_addr,
                              port, db_name, charset))
      return engine

    def _clean_market_data(self):
        """ 清除當前本地端市場歷史資料

        Returns
        -------
        None.

        """
        if os.path.exists(f'{DATA_LOC}/markets'):
            files = os.listdir(f'{DATA_LOC}/markets')
            for file in files:
                os.remove(f'{DATA_LOC}/markets/{file}')
            os.rmdir(f'{DATA_LOC}/markets')

    def _clone_market_data(self, batch_type:BatchType=BatchType.SERVICE_BATCH):
        """ 複製當前資料庫中的市場歷史資料至本地端

        Parameters
        ----------
        batch_type: BatchType
            指定批次的執行狀態為初始化還是服務呼叫

        Returns
        -------
        None.

        """
        table_name = ''
        if batch_type == BatchType.INIT_BATCH:
            table_name = 'FCST_MKT_PRICE_HISTORY'
        elif batch_type == BatchType.SERVICE_BATCH:
            table_name = 'FCST_MKT_PRICE_HISTORY_SWAP'
        else:
            raise Exception(f'_clone_market_data: Unknown batch type: {batch_type}')
        if not os.path.exists(f'{DATA_LOC}/markets'):
            logging.info('Clone market data from db')
            os.makedirs(f'{DATA_LOC}/markets', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    MARKET_CODE, PRICE_DATE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE
                FROM
                    {table_name}
            """
            data = pd.read_sql_query(sql, engine)
            result = pd.DataFrame(
                data[['MARKET_CODE', 'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE']].values,
                index=data['PRICE_DATE'].values.astype('datetime64[D]'),
                columns=['MARKET_CODE', 'OP', 'HP', 'LP', 'CP']
                )
            market_groups = result.groupby('MARKET_CODE')
            for market_id, market_group in market_groups:
                with open(f'{DATA_LOC}/markets/{market_id}.pkl', 'wb') as fp:
                    mkt = market_group.sort_index()[['OP', 'HP', 'LP', 'CP']]
                    mkt = pd.DataFrame(mkt.values.astype(float), index=mkt.index, columns=mkt.columns)
                    pickle.dump(mkt, fp)
            logging.info('Clone market data from db finished')

    def _clean_mkt_info(self):
        """ 清除當前本地端市場資訊

        Returns
        -------
        None.

        """
        if os.path.exists(f'{DATA_LOC}/market_info.pkl'):
            os.remove(f'{DATA_LOC}/market_info.pkl')

    def _clone_mkt_info(self, batch_type: BatchType=BatchType.INIT_BATCH):
        """複製當前資料庫中的市場資訊至本地端

        Parameters
        ----------
        batch_type: BatchType
            指定批次的執行狀態為初始化還是服務呼叫

        Returns
        -------
        None.
        """
        table_name = ''
        if batch_type == BatchType.INIT_BATCH:
            table_name = 'FCST_MKT'
        elif batch_type == BatchType.SERVICE_BATCH:
            table_name = 'FCST_MKT_SWAP'
        else:
            raise Exception(f'_clone_mkt_info: Unknown batch type: {batch_type}')
        if not os.path.exists(f'{DATA_LOC}/market_info.pkl'):
            logging.info('Clone market info from db')
            os.makedirs(f'{DATA_LOC}', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    *
                FROM
                    {table_name}
            """
            data = pd.read_sql_query(sql, engine)
            with open(f'{DATA_LOC}/market_info.pkl', 'wb') as fp:
                pickle.dump(data, fp)
            logging.info('Clone market info from db finished')

    def _clean_dsstock_info(self):
        """ 清除當前本地端 TEJ 股票類別資訊

        Returns
        -------
        None.

        """
        if os.path.exists(f'{DATA_LOC}/ds_s_stock_info.pkl'):
            os.remove(f'{DATA_LOC}/ds_s_stock_info.pkl')

    def _clone_dsstock_info(self, batch_type: BatchType=BatchType.INIT_BATCH):
        """複製當前資料庫中的 TEJ 股票類別資訊至本地端

        Parameters
        ----------
        batch_type: BatchType
            指定批次的執行狀態為初始化還是服務呼叫

        Returns
        -------
        None.
        """
        table_name = ''
        if batch_type == BatchType.INIT_BATCH:
            table_name = 'DS_S_STOCK'
        elif batch_type == BatchType.SERVICE_BATCH:
            table_name = 'DS_S_STOCK_SWAP'
        else:
            raise Exception(f'_clone_dsstock_info: Unknown batch type: {batch_type}')
        if not os.path.exists(f'{DATA_LOC}/ds_s_stock_info.pkl'):
            logging.info('Clone ds_s_stock info from db')
            os.makedirs(f'{DATA_LOC}', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    *
                FROM
                    {table_name}
            """
            data = pd.read_sql_query(sql, engine)
            with open(f'{DATA_LOC}/ds_s_stock_info.pkl', 'wb') as fp:
                pickle.dump(data, fp)
            logging.info('Clone ds_s_stock info from db finished')

    def _clean_patterns(self):
        """ 清除當前本地端現象資訊

        Returns
        -------
        None.

        """
        if os.path.exists(f'{DATA_LOC}/patterns.pkl'):
            os.remove(f'{DATA_LOC}/patterns.pkl')

    def _clone_patterns(self):
        """複製當前資料庫中的現象資料至本地端

        Returns
        -------
        None.
        """
        if not os.path.exists(f'{DATA_LOC}/patterns.pkl'):
            logging.info('Clone pattern from db')
            os.makedirs(f'{DATA_LOC}', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    ptn.PATTERN_ID, mcr.FUNC_CODE, para.PARAM_CODE, para.PARAM_VALUE, mp.PARAM_TYPE
                FROM
                    FCST_PAT AS ptn
                LEFT JOIN
                    (
                        SELECT
                            PATTERN_ID, MACRO_ID, PARAM_CODE, PARAM_VALUE
                        FROM
                            FCST_PAT_PARAM
                    ) AS para
                ON ptn.PATTERN_ID=para.PATTERN_ID
                LEFT JOIN
                    (
                        SELECT
                            MACRO_ID, FUNC_CODE
                        FROM
                            FCST_MACRO
                    ) AS mcr
                ON ptn.MACRO_ID=mcr.MACRO_ID
                LEFT JOIN
                    (
                        SELECT
                            MACRO_ID, PARAM_CODE, PARAM_TYPE
                        FROM
                            FCST_MACRO_PARAM
                    ) AS mp
                ON ptn.MACRO_ID=mp.MACRO_ID AND para.PARAM_CODE=mp.PARAM_CODE
                ORDER BY ptn.PATTERN_ID, mcr.FUNC_CODE ASC;
            """
            data = pd.read_sql_query(sql, engine)
            result = {}
            for i in range(len(data)):
                record = data.iloc[i]
                pid = record['PATTERN_ID']
                if pid in result:
                    continue
                func = record['FUNC_CODE']
                params = {}
                ptn_record = data[data['PATTERN_ID'].values==pid]
                for j in range(len(ptn_record)):
                    param_record = ptn_record.iloc[j]
                    param_type = param_record['PARAM_TYPE']
                    param_val = param_record['PARAM_VALUE']
                    if param_type == "int":
                        param_val = int(param_record['PARAM_VALUE'])
                    elif param_type == "float":
                        param_val = float(param_record['PARAM_VALUE'])
                    elif param_type == "string":
                        param_val = str(param_record['PARAM_VALUE'])
                    params[param_record['PARAM_CODE']] = param_val
                result[pid] = PatternInfo(pid, func, params)
            result = [result[pid] for pid in result]
            with open(f'{DATA_LOC}/patterns.pkl', 'wb') as fp:
                pickle.dump(result, fp)
            logging.info('Clone patterns from db finished')

    def _clean_models(self):
        """ 清除當前本地端市場清單

        Returns
        -------
        None.

        """
        if os.path.exists(f'{DATA_LOC}/models.pkl'):
            os.remove(f'{DATA_LOC}/models.pkl')

    def _clone_models(self):
        """複製當前資料庫中的觀點資訊至本地端

        Returns
        -------
        None.
        """
        if not os.path.exists(f'{DATA_LOC}/models.pkl'):
            logging.info('Clone models from db')
            os.makedirs(f'{DATA_LOC}', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    model.MODEL_ID
                FROM
                    FCST_MODEL AS model
                INNER JOIN
                    (
                        SELECT
                            MODEL_ID, STATUS_CODE, END_DT
                        FROM
                            FCST_MODEL_EXECUTION
                        WHERE
                            STATUS_CODE='{ModelExecution.ADD_PREDICT_FINISHED}' AND
                            END_DT IS NOT NULL
                    ) AS me
                ON model.MODEL_ID=me.MODEL_ID
            """
            data = pd.read_sql_query(sql, engine)['MODEL_ID'].values.tolist()
            with open(f'{DATA_LOC}/models.pkl', 'wb') as fp:
                pickle.dump(data, fp)
            logging.info('Clone models from db finished')

    def _clean_model_training_infos(self):
        """ 清除當前本地端觀點訓練資訊

        Returns
        -------
        None.

        """
        if os.path.exists(f'{DATA_LOC}/model_training_infos.pkl'):
            os.remove(f'{DATA_LOC}/model_training_infos.pkl')

    def _clone_model_training_infos(self):
        """複製當前資料庫中的觀點訓練資訊至本地端

        Returns
        -------
        None.
        """
        if not os.path.exists(f'{DATA_LOC}/model_training_infos.pkl'):
            logging.info('Clone model training infos from db')
            os.makedirs(f'{DATA_LOC}', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    *
                FROM
                    FCST_MODEL
            """
            model_info = pd.read_sql_query(sql, engine)
            with open(f'{DATA_LOC}/model_training_infos.pkl', 'wb') as fp:
                pickle.dump(model_info, fp)
            logging.info('Clone model training infos from db finished')

    def _clean_model_markets(self):
        """ 清除當前本地端觀點目標市場

        Returns
        -------
        None.

        """
        if os.path.exists(f'{DATA_LOC}/model_markets.pkl'):
            os.remove(f'{DATA_LOC}/model_markets.pkl')

    def _clone_model_markets(self):
        """複製當前資料庫中的觀點目標市場至本地端

        Returns
        -------
        None.
        """
        if not os.path.exists(f'{DATA_LOC}/model_markets.pkl'):
            logging.info('Clone model markets from db')
            os.makedirs(f'{DATA_LOC}', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    *
                FROM
                    FCST_MODEL_MKT_MAP
            """
            data = pd.read_sql_query(sql, engine)
            with open(f'{DATA_LOC}/model_markets.pkl', 'wb') as fp:
                pickle.dump(data, fp)
            logging.info('Clone model markets from db finished')

    def _clean_model_patterns(self):
        """ 清除當前本地端觀點使用現象

        Returns
        -------
        None.

        """
        if os.path.exists(f'{DATA_LOC}/model_patterns.pkl'):
            os.remove(f'{DATA_LOC}/model_patterns.pkl')

    def _clone_model_patterns(self):
        """複製當前資料庫中的觀點使用現象至本地端

        Returns
        -------
        None.
        """
        if not os.path.exists(f'{DATA_LOC}/model_patterns.pkl'):
            logging.info('Clone model patterns from db')
            os.makedirs(f'{DATA_LOC}', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    *
                FROM
                    FCST_MODEL_PAT_MAP
            """
            data = pd.read_sql_query(sql, engine)
            with open(f'{DATA_LOC}/model_patterns.pkl', 'wb') as fp:
                pickle.dump(data, fp)
            logging.info('Clone model patterns from db finished')

    def _clean_score_meta_info(self):
        """ 清除當前本地端分數標準差倍率資訊

        Returns
        -------
        None.

        """
        if os.path.exists(f'{DATA_LOC}/score_meta_info.pkl'):
            os.remove(f'{DATA_LOC}/score_meta_info.pkl')

    def _clone_score_meta_info(self):
        """複製當前資料庫中的分數標準差倍率資訊至本地端

        Returns
        -------
        None.
        """
        if not os.path.exists(f'{DATA_LOC}/score_meta_info.pkl'):
            logging.info('Clone score meta info from db')
            os.makedirs(f'{DATA_LOC}', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    *
                FROM
                    FCST_SCORE
            """
            data = pd.read_sql_query(sql, engine)
            with open(f'{DATA_LOC}/score_meta_info.pkl', 'wb') as fp:
                pickle.dump(data, fp)
            logging.info('Clone score meta info from db finished')

    def clean_db_cache(self):
        """ 清除本地資料庫快取

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self._clean_score_meta_info()
        self._clean_mkt_info()
        self._clean_dsstock_info()
        self._clean_market_data()
        self._clean_patterns()
        self._clean_models()
        self._clean_model_training_infos()
        self._clean_model_markets()
        self._clean_model_patterns()

    def clone_db_cache(self, batch_type:BatchType=BatchType.SERVICE_BATCH):
        """ 從資料庫載入快取

        Parameters
        ----------
        batch_type: BatchType
            指定批次的執行狀態為初始化還是服務呼叫

        Returns
        -------
        None.
        """
        self._clone_score_meta_info()
        self._clone_mkt_info(batch_type)
        self._clone_dsstock_info(batch_type)
        self._clone_market_data(batch_type)
        self._clone_patterns()
        self._clone_models()
        self._clone_model_training_infos()
        self._clone_model_markets()
        self._clone_model_patterns()

    def get_markets(self, market_type: str=None, category_code: str=None):
        """get market IDs from DB.
        Parameters
        ----------
        market_type:
            市場類型( 目前為 BLB 或 TEJ )
        category_code:
            市場分類( 電子股, 水泥股...等 )

        Returns
        -------
        list of str

        """
        self._clone_mkt_info()
        self._clone_dsstock_info()
        with open(f'{DATA_LOC}/market_info.pkl', 'rb') as fp:
            data = pickle.load(fp)
        if market_type is None:
            return data[MarketInfoField.MARKET_CODE.value].values.tolist()
        
        data = data[
            data[MarketInfoField.MARKET_SOURCE_TYPE.value].values == market_type]
        if category_code is None:
            return data[MarketInfoField.MARKET_CODE.value].values.tolist()
        
        with open(f'{DATA_LOC}/ds_s_stock_info.pkl', 'rb') as fp:
            cate_data = pickle.load(fp)
        cate_data = cate_data[
            cate_data[DSStockInfoField.TSE_INDUSTRY_CODE.value].values == category_code]
        data = data[MarketInfoField.MARKET_CODE.value].values.astype(str).tolist()
        cate_data = cate_data[DSStockInfoField.STOCK_CODE.value].values.astype(str).tolist()
        data = [x for x in data if x in cate_data]
        return data

    def get_patterns(self):
        """get patterns from DB.

        Returns
        -------
        list of PatternInfo

        """
        self._clone_patterns()
        with open(f'{DATA_LOC}/patterns.pkl', 'rb') as fp:
            data = pickle.load(fp)
        return data

    def _get_future_return_file(self, market_id: str) -> str:
        return f'{LOCAL_DB}/freturns/{market_id}.pkl'

    def save_future_return(self, market_id:str, data:pd.DataFrame):
        """Save future return results to DB.

        Parameters
        ----------
        market_id: str
            Id of market.
        data: Pandas's DataFrame
            A panel of float which columns TF(d) for each predicting period as
            d.

        """
        self.future_reture_cache[market_id] = data

    def dump_future_returns(self):
        """Dump future returns to DB. """
        logging.info('Dump future returns to db')
        for market_id in self.get_markets():
            self._local_db_lock.acquire()
            pickle_dump(self.future_reture_cache[market_id], self._get_future_return_file(market_id))
            self._local_db_lock.release()
        logging.info('Dump future returns finished')

    def get_future_return(self, market_id:str, period: int, begin_date:datetime.date):
        """Get future return results from begining date to the latest of given market from DB.

        Parameters
        ----------
        market_id: str
            ID of market.
        period: int
            Predicting period.
        begin_date: datetime.date, optional
            The begin date of the designated data range. If it is not set, get all
            history results.

        Returns
        -------
        Pandas's Series
            A timeseries of float.

        """
        field = get_filed_name_of_future_return(period)
        if market_id not in self.future_reture_cache:
            self._local_db_lock.acquire()
            self.future_reture_cache[market_id] = pickle_load(self._get_future_return_file(market_id))
            self._local_db_lock.release()
        ret = self.future_reture_cache[market_id][field]
        if begin_date:
            ret = ret[ret.index.values.astype('datetime64[D]') >= begin_date]
        return ret

    def _get_pattern_file(self, market_id: str) -> str:
        return f'{LOCAL_DB}/patterns/{market_id}.pkl'

    def save_pattern_results(self, market_id:str, data:pd.DataFrame):
        """Save pattern results to DB.

        Parameters
        ----------
        market_id: str
            Id of market.
        data: Pandas's DataFrame
            A panel of boolean which columns are the pattern IDs.

        """
        self.pattern_cache[market_id] = data

    def dump_pattern_results(self):
        """Dump pattern results to DB. """
        logging.info('Dump pattern results to db')
        for market_id in self.get_markets():
            self._local_db_lock.acquire()
            pickle_dump(self.pattern_cache[market_id], self._get_pattern_file(market_id))
            self._local_db_lock.release()
        logging.info('Dump pattern results finished')

    def get_pattern_results(self, market_id:str, patterns: List[str], begin_date:datetime.date):
        """Get pattern results from begining date to the latest of given market from DB.

        Parameters
        ----------
        market_id: str
            ID of market.
        patterns: list of str
            ID of patterns.
        begin_date: datetime.date, optional
            The begin date of the designated data range. If it is not set, get all
            history results.

        Returns
        -------
        Pandas's DataFrame
            A panel of boolean with all id of patterns as columns.

        """
        if market_id not in self.pattern_cache:
            self._local_db_lock.acquire()
            self.pattern_cache[market_id] = pickle_load(self._get_pattern_file(market_id))
            self._local_db_lock.release()
        ret = self.pattern_cache[market_id][patterns]
        if begin_date:
            ret = ret[ret.index.values.astype('datetime64[D]') >= begin_date]
        return ret

    def get_latest_dates(self, model_id:str):
        """get dates of the latest predict results of markets for given model.

        Paratmeters
        -----------
        model_id: str
            ID of model.

        Returns
        -------
        dict from str to datetime.date
            a dict mapping from marketID to the corresponding latest predicting
            date.

        """
        engine = self._engine()
        sql = f"""
            SELECT
                MARKET_CODE, MAX(DATA_DATE) AS DATA_DATE
            FROM
                FCST_MODEL_MKT_VALUE
            WHERE
                MODEL_ID='{model_id}'
            GROUP BY
                MARKET_CODE
        """
        data = pd.read_sql_query(sql, engine)
        result = {}
        for i in range(len(data)):
            market_id = data.iloc[i]['MARKET_CODE']
            result[market_id] = datetime.datetime.strptime(
                str(data.iloc[i]['DATA_DATE']), '%Y-%m-%d').date()
        return result

    def get_earliest_dates(self, model_id:str):
        """get dates of the earliest predict results of markets for given model.

        Paratmeters
        -----------
        model_id: str
            ID of model.

        Returns
        -------
        dict from str to datetime.date
            a dict mapping from marketID to the corresponding earliest predicting
            date.

        """
        engine = self._engine()
        sql = f"""
            SELECT
                MARKET_CODE, MIN(DATA_DATE) AS DATA_DATE
            FROM
                FCST_MODEL_MKT_VALUE_HISTORY
            WHERE
                MODEL_ID='{model_id}'
            GROUP BY
                MARKET_CODE
        """
        data = pd.read_sql_query(sql, engine)
        result = {}
        for i in range(len(data)):
            market_id = data.iloc[i]['MARKET_CODE']
            result[market_id] = datetime.datetime.strptime(
                str(data.iloc[i]['DATA_DATE']), '%Y-%m-%d').date()
        return result

    def get_models(self):
        """Get models from DB which complete ADD_PREDICT.

        Returns
        -------
        list of str
            IDs of models in DB

        """
        self._clean_models()
        self._clone_models()
        with open(f'{DATA_LOC}/models.pkl', 'rb') as fp:
            data = pickle.load(fp)
        return data

    def get_model_info(self, model_id:str):
        """Get model info from DB.

        Parameters
        ----------
        model_id: str

        Returns
        -------
        ModelInfo

        """
        self._clean_model_training_infos()
        self._clean_model_markets()
        self._clean_model_patterns()
        self._clone_model_training_infos()
        self._clone_model_markets()
        self._clone_model_patterns()

        # 取得觀點的訓練資訊
        with open(f'{DATA_LOC}/model_training_infos.pkl', 'rb') as fp:
            model_info = pickle.load(fp)
        m_cond = model_info['MODEL_ID'].values == model_id
        if len(model_info[m_cond]) == 0:
            # 若發生取不到資料的情況
            raise Exception(f"get_model_info: model not found: {model_id}")
        train_begin = model_info[m_cond].iloc[0]['TRAIN_START_DT']
        train_gap = model_info[m_cond].iloc[0]['RETRAIN_CYCLE']

        # 取得觀點的所有標的市場
        with open(f'{DATA_LOC}/model_markets.pkl', 'rb') as fp:
            markets = pickle.load(fp)
        m_cond = markets['MODEL_ID'].values == model_id
        markets = markets[m_cond]['MARKET_CODE'].values.tolist()

        # 取得觀點所有使用的現象 ID
        with open(f'{DATA_LOC}/model_patterns.pkl', 'rb') as fp:
            patterns = pickle.load(fp)
        m_cond = patterns['MODEL_ID'].values == model_id
        if len(patterns[m_cond]) == 0:
            # 若觀點下沒有任何現象, 則回傳例外
            raise Exception(f"get_model_info: 0 model pattern exception: {model_id}")
        patterns = patterns[m_cond]['PATTERN_ID'].values.tolist()

        result = ModelInfo(model_id, patterns, markets, train_begin, train_gap)
        return result

    def del_model_data(self, model_id: str):
        """Delete model information and predicting results from DB.

        Parameters
        ----------
        model_id: str
            ID of model to delete.

        """
        engine = self._engine()
        if not self.READ_ONLY:
            # DEL FCST_MODEL_MKT_VALUE
            sql = f"""
                DELETE FROM FCST_MODEL_MKT_VALUE
                WHERE
                    MODEL_ID='{model_id}'
            """
            engine.execute(sql)

            # DEL FCST_MODEL_MKT_VALUE_HISTORY
            sql = f"""
                DELETE FROM FCST_MODEL_MKT_VALUE_HISTORY
                WHERE
                    MODEL_ID='{model_id}'
            """
            engine.execute(sql)

    def save_model_latest_results(self, model_id:str, data:pd.DataFrame,
                                  exec_type:ModelExecution):
        """Save modle latest predicting results to DB.

        Parameters
        ----------
        model_id: str
            ID of model.
        data: Pandas's DataFrame
            A panel of floating with columns, 'lb_p', 'ub_p', 'pr_p' for each
            predicting period as p.
        exec_type: ModelExecution
            Allow enum: ADD_PREDICT, ADD_BACKTEST, BATCH_PREDICT

        """
        try:
            # 發生錯誤時需要看資料用
            except_catch = {
                'input': {
                    'model_id': model_id,
                    'data': data.copy(),
                    'exec_type': exec_type
                },
                'process': {}
            }
            logging.info('start saving model latest results')
            table_name = 'FCST_MODEL_MKT_VALUE_SWAP'
            if exec_type == ModelExecution.ADD_PREDICT:
                table_name = 'FCST_MODEL_MKT_VALUE'
            elif exec_type == ModelExecution.BATCH_PREDICT:
                table_name = 'FCST_MODEL_MKT_VALUE_SWAP'
            else:
                logging.error(f'Unknown execution type: {exec_type}')
                return -1
            engine = self._engine()

            # 製作儲存結構
            now = datetime.datetime.now()
            py = data[PredictResultField.PREDICT_VALUE.value].values * 100
            upper_bound = data[PredictResultField.UPPER_BOUND.value].values * 100
            lower_bound = data[PredictResultField.LOWER_BOUND.value].values * 100

            data = data[[
                PredictResultField.MODEL_ID.value,
                PredictResultField.MARKET_ID.value,
                PredictResultField.DATE.value,
                PredictResultField.PERIOD.value
                ]]
            data[PredictResultField.PREDICT_VALUE.value] = py
            data[PredictResultField.UPPER_BOUND.value] = upper_bound
            data[PredictResultField.LOWER_BOUND.value] = lower_bound
            create_dt = now
            data['CREATE_BY'] = self.CREATE_BY
            data['CREATE_DT'] = create_dt

            # 移除傳入預測結果中較舊的預測結果
            group_data = data.groupby([
                PredictResultField.MODEL_ID.value,
                PredictResultField.MARKET_ID.value,
                PredictResultField.PERIOD.value])
            latest_data = []
            for group_i, group in group_data:
                max_date = np.max(group[PredictResultField.DATE.value].values)
                latest_data.append(
                    group[group[PredictResultField.DATE.value].values == max_date])
            latest_data = pd.concat(latest_data, axis=0)

            # 新增最新預測結果
            # 合併現有的資料預測結果與當前的預測結果
            db_data = None
            sql = f"""
                SELECT
                    CREATE_BY, CREATE_DT, MODEL_ID, MARKET_CODE,
                    DATE_PERIOD, DATA_DATE, DATA_VALUE, UPPER_BOUND, LOWER_BOUND
                FROM
                    FCST_MODEL_MKT_VALUE
                WHERE
                    MODEL_ID='{model_id}'
            """
            db_data = pd.read_sql_query(sql, engine)
            except_catch['process']['db_data'] = db_data.copy()
            db_data[PredictResultField.DATE.value] = db_data[PredictResultField.DATE.value].astype('datetime64[D]')
            union_data = pd.concat([db_data, latest_data], axis=0)

            # 移除完全重複的預測結果
            union_data[PredictResultField.MODEL_ID.value] = union_data[PredictResultField.MODEL_ID.value].astype(str)
            union_data[PredictResultField.MARKET_ID.value] = union_data[PredictResultField.MARKET_ID.value].astype(str)
            union_data[PredictResultField.PERIOD.value] = union_data[PredictResultField.PERIOD.value].astype(np.int64)
            union_data[PredictResultField.DATE.value] = union_data[PredictResultField.DATE.value].astype('datetime64[D]')
            union_data = union_data.drop_duplicates(subset=[
                PredictResultField.MODEL_ID.value,
                PredictResultField.MARKET_ID.value,
                PredictResultField.PERIOD.value,
                PredictResultField.DATE.value
                ])

            # 移除較舊的預測結果
            group_data = union_data.groupby([
                PredictResultField.MODEL_ID.value,
                PredictResultField.MARKET_ID.value,
                PredictResultField.PERIOD.value])
            latest_data = []
            for group_i, group in group_data:
                max_date = np.max(group[PredictResultField.DATE.value].values)
                latest_data.append(
                    group[group[PredictResultField.DATE.value].values == max_date])
            latest_data = pd.concat(latest_data, axis=0)

            # 開始儲存
            if not self.READ_ONLY:
                sql = f"""
                    DELETE FROM {table_name}
                    WHERE
                        MODEL_ID='{model_id}'
                """
                engine.execute(sql)

            if not self.READ_ONLY:
                latest_data.to_sql(
                    table_name,
                    engine,
                    if_exists='append',
                    chunksize=10000,
                    method='multi',
                    index=False)
            logging.info('Saving model latest results finished')
        except Exception as e:
            logging.error('Save model latest results failed, save except data')
            os.makedirs(f'{EXCEPT_DATA_LOC}', exist_ok=True)
            EXCPT_DATA_PATH = f'{EXCEPT_DATA_LOC}/save_model_latest_results.pkl'
            except_catches = []
            if os.path.exists(EXCPT_DATA_PATH):
                except_catches = pickle_load(EXCPT_DATA_PATH)
            except_catches.append(except_catch)
            pickle_dump(except_catches, EXCPT_DATA_PATH)
            logging.error(traceback.format_exc())
            raise e
    
    def update_model_accuracy(self):
        """ 呼叫 SP 計算當前各模型準確率

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        if not self.READ_ONLY:
            logging.info('start update FCST model accuracy')
            engine=  self._engine()
            sql = f'CALL SP_UPDATE_FCST_MODEL_MKT_HIT_SUM_SWAP()'
            with engine.begin() as db_conn:
                db_conn.execute(sql)
            logging.info('Update FCST model accuracy finished')

    def checkout_fcst_data(self):
        """ 切換系統中的呈現資料位置

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        if not self.READ_ONLY:
            logging.info('start checkout FCST data')
            engine = self._engine()
            sql = f"CALL SP_SWAP_FCST()"
            with engine.begin() as db_conn:
                db_conn.execute(sql)
            logging.info('Checkout FCST data finished')

    def save_model_results(self, model_id:str, data:pd.DataFrame,
                           exec_type: ModelExecution=ModelExecution.ADD_BACKTEST):
        """Save modle predicting results to DB.

        Parameters
        ----------
        model_id: str
            ID of model.
        data: Pandas's DataFrame
            A panel of floating with columns, 'lb_p', 'ub_p', 'pr_p' for each
            predicting period as p.
        exec_type: ModelExecution
            Allow enum: ADD_PREDICT, ADD_BACKTEST, BATCH_PREDICT

        """
        if (exec_type == ModelExecution.ADD_PREDICT or
            exec_type == ModelExecution.BATCH_PREDICT):
            self.save_model_latest_results(model_id, data.copy(), exec_type)
        logging.info('start saving model results')
        # 製作儲存結構
        now = datetime.datetime.now()
        engine = self._engine()
        py = data[PredictResultField.PREDICT_VALUE.value].values * 100
        upper_bound = data[PredictResultField.UPPER_BOUND.value].values * 100
        lower_bound = data[PredictResultField.LOWER_BOUND.value].values * 100

        data = data[[
            PredictResultField.MODEL_ID.value,
            PredictResultField.MARKET_ID.value,
            PredictResultField.DATE.value,
            PredictResultField.PERIOD.value
            ]]
        data[PredictResultField.PREDICT_VALUE.value] = py
        data[PredictResultField.UPPER_BOUND.value] = upper_bound
        data[PredictResultField.LOWER_BOUND.value] = lower_bound
        create_dt = now
        data['CREATE_BY'] = self.CREATE_BY
        data['CREATE_DT'] = create_dt

        # 新增歷史預測結果
        if not self.READ_ONLY:
            try:
                data.to_sql(
                    'FCST_MODEL_MKT_VALUE_HISTORY_SWAP',
                    engine,
                    if_exists='append',
                    chunksize=10000,
                    method='multi',
                    index=False)
            except Exception as e:
                logging.info('save_model_results: Saving model history results failed, maybe PK duplicated, skipped it.')
                logging.debug(traceback.format_exc())
            logging.info('Saving model results finished')

    def get_market_data(self, market_id:str, begin_date:Optional[datetime.date]=None):
        """Get market data from the designated date to the latest from DB.

        Parameters
        ----------
        market_id: str
            ID of market.
        begin_date: datetime.date, optional
            The begin date of the designated data range. If it is not set, get all
            historical data.

        Returns
        -------
        Pandas's DataFrame
            A panel of floating with columns, 'OP', 'LP', 'HP', and 'CP'.

        """
        self._clone_market_data()
        if not os.path.exists(f'{DATA_LOC}/markets/{market_id}.pkl'):
            return pd.DataFrame(columns=['OP', 'LP', 'HP', 'CP'])
        with open(f'{DATA_LOC}/markets/{market_id}.pkl', 'rb') as fp:
            result = pickle.load(fp)
        if begin_date is not None:
            result = result[result.index.values >= begin_date]
        return result

    def set_model_execution_start(self, model_id:str, exection:str) -> str:
        """ create model exec

        Parameters
        ----------
        model_id: str
            ID of model
        exection: str
            status code set to this model

        Returns
        -------
        exec_id: str
            ID of model execution status

        """
        exec_id = 'READ_ONLY_MODE'
        if not self.READ_ONLY:
            # 檢查是否為合法的 exection
            if exection not in ModelExecution:
                raise Exception(f"Unknown excetion code {exection}")

            engine = self._engine()
            # 資料庫中對於一個觀點僅會存在一筆 AP 或 AB
            # 若是要儲存 AP 或是 AB, 那麼資料庫中就不應該已存在 AP 或 AB
            # 因此要先清除原先的執行紀錄
            if exection in [
                ModelExecution.ADD_PREDICT.value,
                ModelExecution.ADD_BACKTEST.value]:
                sql = f"""
                    DELETE FROM FCST_MODEL_EXECUTION
                    WHERE MODEL_ID='{model_id}' AND STATUS_CODE='{exection}'
                """
                engine.execute(sql)

            # 取得 EXEC_ID
            logging.info('Call SP_GET_SERIAL_NO')
            sql = f"CALL SP_GET_SERIAL_NO('EXEC_ID', @EXEC_ID)"
            with engine.begin() as db_conn:
                db_conn.execute(sql)
                results = db_conn.execute('SELECT @EXEC_ID').fetchone()
                exec_id = results[0]
            logging.info(f"Get SERIAL_NO: {exec_id}")

            # 建立 status
            COLUMNS = [
                'CREATE_BY', 'CREATE_DT', 'EXEC_ID',
                'MODEL_ID', 'STATUS_CODE', 'START_DT', 'END_DT'
                ]
            now = datetime.datetime.now()
            create_by = self.CREATE_BY
            create_dt = now
            start_dt = now
            end_dt = None
            data = [[
                create_by, create_dt, exec_id,
                model_id, exection, start_dt, end_dt]]
            data = pd.DataFrame(data, columns=COLUMNS)
            data.to_sql(
                'FCST_MODEL_EXECUTION',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        return exec_id

    def set_model_execution_complete(self, exec_id:str):
        """Set model execution complete on DB.

        Parameters
        ----------
        exec_id: str
            ID of model_exec
        execution: str
            status code set to this model
        model_id: str
            ID of model

        """
        now = datetime.datetime.now()
        engine = self._engine()

        finished_status = {
            ModelExecution.ADD_BACKTEST.value: ModelExecution.ADD_BACKTEST_FINISHED.value,
            ModelExecution.ADD_PREDICT.value: ModelExecution.ADD_PREDICT_FINISHED.value,
            ModelExecution.BATCH_PREDICT.value: ModelExecution.BATCH_PREDICT_FINISHED.value
        }

        sql = f"""
            SELECT
                STATUS_CODE
            FROM
                FCST_MODEL_EXECUTION
            WHERE
                EXEC_ID='{exec_id}';
        """
        exec_data = pd.read_sql_query(sql, engine)['STATUS_CODE']
        if len(exec_data) == 0:
            raise Exception('call set_model_execution_complete before set_model_execution_start')

        status = finished_status[exec_data.values[0]]
        if not self.READ_ONLY:
            sql = f"""
            UPDATE
                FCST_MODEL_EXECUTION
            SET
                END_DT='{now}', MODIFY_DT='{now}', STATUS_CODE='{status}'
            WHERE
                EXEC_ID='{exec_id}';
            """
            engine.execute(sql)

    def get_recover_model_execution(self):
        """ 取得模型最新執行狀態
        取得新增模型預測與新增模型回測的最新更新狀態，這兩個狀態對於任一模型而言
        都必須且應該只會各有一筆
        當新增模型狀態沒有找到且新增模型完成狀態也沒有找到時，狀態為需要新增預測
        當新增模型狀態找到了但沒有結束時間時，狀態為需要新增預測
        當新增模型完成找到但新增回測且新增回測完成狀態沒有沒找到時，狀態為需要新增回測
        當新增回測狀態找到但沒有結束時間時，狀態為需要新增回測

        Parameters
        ----------
        None.

        Returns
        -------
        exec_info: list of tuple
            [(model_id, ModelExecution), ...]

        """
        engine = self._engine()
        sql = f"""
            SELECT
                model.MODEL_ID, me.STATUS_CODE, me.END_DT
            FROM
                FCST_MODEL AS model
            LEFT JOIN
                (
                    SELECT
                        MODEL_ID, STATUS_CODE, END_DT
                    FROM
                        FCST_MODEL_EXECUTION
                ) AS me
            ON model.MODEL_ID=me.MODEL_ID
        """
        data = pd.read_sql_query(sql, engine)
        group_data = data.groupby("MODEL_ID")
        results = []
        for model_id, model_state_info in group_data:
            model_add_predict_info = model_state_info[
                model_state_info['STATUS_CODE'].values==
                ModelExecution.ADD_PREDICT.value]
            model_add_predict_finished_info = model_state_info[
                model_state_info['STATUS_CODE'].values==
                ModelExecution.ADD_PREDICT_FINISHED.value]
            model_add_backtest_info = model_state_info[
                model_state_info['STATUS_CODE'].values==
                ModelExecution.ADD_BACKTEST.value]
            model_add_backtest_finished_info = model_state_info[
                model_state_info['STATUS_CODE'].values==
                ModelExecution.ADD_BACKTEST_FINISHED.value]
            # ADD_PREDICT 未建立就掛掉
            if (len(model_add_predict_info) == 0) and (len(model_add_predict_finished_info) == 0):
                results.append((model_id, ModelExecution.ADD_PREDICT))
            # ADD_PREDICT 未完成就掛掉
            elif (len(model_add_predict_finished_info) == 0):
                results.append((model_id, ModelExecution.ADD_PREDICT))
            # ADD_BACKTEST 未建立就掛掉
            elif (len(model_add_backtest_info) == 0) and (len(model_add_backtest_finished_info) == 0):
                results.append((model_id, ModelExecution.ADD_BACKTEST))
            # ADD_BACKTEST 未完成就掛掉
            elif (len(model_add_backtest_finished_info) == 0):
                results.append((model_id, ModelExecution.ADD_BACKTEST))
        return results

    def save_latest_pattern_results(self, data:pd.DataFrame):
        """Save latest pattern results to DB.

        Parameters
        ----------
        data: Pandas's DataFrame
            A table of pattern results with columns for market_id, pattern_id,
            price_date and value.
        block_size: int
            Save each data_block that size is block_size in data until finished

        See Also
        --------
        PatternResultField

        """
        engine = self._engine()
        now = datetime.datetime.now()
        create_dt = now
        data['CREATE_BY'] = self.CREATE_BY
        data['CREATE_DT'] = create_dt

        logging.info(f'Save pattern event')
        if not self.READ_ONLY:
            data.to_sql(
                'FCST_PAT_MKT_EVENT_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        logging.info(f'Save pattern event finished')

    def save_latest_pattern_distribution(self, data: pd.DataFrame):
        """ 儲存最新現象統計分布統計量
        儲存發生指定現象後, 指定市場, 指定天期下的報酬分布統計量
        這個方法在儲存均值和標準差時會將結果轉換為 % , 因此會做
        乘 100 的動作

        Parameters
        ----------
        data: `pd.DataFrame`
            要儲存的資料

        Returns
        -------
        None.

        """
        engine = self._engine()

        now = datetime.datetime.now()
        create_by = self.CREATE_BY
        create_dt = now
        data['CREATE_BY'] = create_by
        data['CREATE_DT'] = create_dt
        data_mean = data[MarketDistField.RETURN_MEAN.value].values * 100
        data_std = data[MarketDistField.RETURN_STD.value].values * 100

        data = data[[
            'CREATE_BY', 'CREATE_DT', MarketDistField.PATTERN_ID.value,
            MarketDistField.MARKET_ID.value, MarketDistField.DATE_PERIOD.value]]
        data[MarketDistField.RETURN_MEAN.value] = data_mean
        data[MarketDistField.RETURN_STD.value] = data_std

        logging.info(f'Save pattern distribution')
        if not self.READ_ONLY:
            # 新增最新資料
            data.to_sql(
                'FCST_PAT_MKT_DIST_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        logging.info(f'Save pattern distribution finished')

    def save_latest_pattern_occur(self, data: pd.DataFrame):
        """ 儲存最新現象發生後次數統計
        儲存發生指定現象後, 指定市場, 指定天期下的發生與未發生總數,
        上漲總數, 持平總數與下跌總數

        Parameters
        ----------
        data: `pd.DataFrame`
            要儲存的資料

        Returns
        -------
        None.

        """
        engine = self._engine()

        now = datetime.datetime.now()
        create_by = self.CREATE_BY
        create_dt = now
        data['CREATE_BY'] = create_by
        data['CREATE_DT'] = create_dt

        logging.info(f'Save pattern occur')
        if not self.READ_ONLY:
        # 新增最新資料
            data.to_sql(
                'FCST_PAT_MKT_OCCUR_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        logging.info(f'Save pattern occur finished')

    def save_mkt_score(self, data: pd.DataFrame):
        """ 儲存市場最新分數上下界
        儲存各市場各天期要將報酬轉換為分數時的上下界範圍資訊

        Parameters
        ----------
        data: `pd.DataFrame`
            要儲存的資料

        Returns
        -------
        None.

        """
        engine = self._engine()

        now = datetime.datetime.now()
        create_by = self.CREATE_BY
        create_dt = now
        data['CREATE_BY'] = create_by
        data['CREATE_DT'] = create_dt
        data_upper_bound = data[MarketScoreField.UPPER_BOUND.value].values * 100
        data_lower_bound = data[MarketScoreField.LOWER_BOUND.value].values * 100

        data = data[[
            'CREATE_BY', 'CREATE_DT', MarketScoreField.MARKET_SCORE.value,
            MarketScoreField.MARKET_ID.value, MarketScoreField.DATE_PERIOD.value]]
        data[MarketScoreField.UPPER_BOUND.value] = data_upper_bound
        data[MarketScoreField.LOWER_BOUND.value] = data_lower_bound

        logging.info('Save market score')
        if not self.READ_ONLY:
            # 新增最新資料
            data.to_sql(
                'FCST_MKT_SCORE_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        logging.info('Save market score finished')

    def save_mkt_period(self, data: pd.DataFrame):
        """ 儲存市場各天期歷史報酬
        儲存各市場各天期歷史報酬

        Parameters
        ----------
        data: `pd.DataFrame`
            要儲存的資料

        Returns
        -------
        None.

        """
        engine = self._engine()

        now = datetime.datetime.now()
        create_by = self.CREATE_BY
        create_dt = now
        data['CREATE_BY'] = create_by
        data['CREATE_DT'] = create_dt
        data_net_change_rate = data[MarketPeriodField.NET_CHANGE_RATE.value].values * 100

        data = data[[
            'CREATE_BY', 'CREATE_DT', MarketPeriodField.MARKET_ID.value, 
            MarketPeriodField.DATE_PERIOD.value, MarketPeriodField.PRICE_DATE.value, 
            MarketPeriodField.DATA_DATE.value, MarketPeriodField.NET_CHANGE.value]]
        data[MarketPeriodField.NET_CHANGE_RATE.value] = data_net_change_rate

        logging.info('Save market period')
        if not self.READ_ONLY:
            # 新增最新資料
            data.to_sql(
                'FCST_MKT_PERIOD_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        logging.info('Save market period finished')

    def save_mkt_dist(self, data: pd.DataFrame):
        """ 儲存市場統計結果
        儲存各市場最新統計資訊結果

        Parameters
        ----------
        data: `pd.DataFrame`
            要儲存的資料

        Returns
        -------
        None.

        """
        engine = self._engine()

        now = datetime.datetime.now()
        create_by = self.CREATE_BY
        create_dt = now
        data['CREATE_BY'] = create_by
        data['CREATE_DT'] = create_dt
        data_return_mean = data[MarketStatField.RETURN_MEAN.value].values * 100
        data_return_std = data[MarketStatField.RETURN_STD.value].values * 100

        data = data[[
            'CREATE_BY', 'CREATE_DT', MarketStatField.MARKET_ID.value, 
            MarketStatField.DATE_PERIOD.value, MarketStatField.RETURN_CNT.value]]
        data[MarketStatField.RETURN_MEAN.value] = data_return_mean
        data[MarketStatField.RETURN_STD.value] = data_return_std

        logging.info('Save market distribution')
        if not self.READ_ONLY:
            # 新增最新資料
            data.to_sql(
                'FCST_MKT_DIST_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        logging.info('Save market distribution finished')

    def get_score_meta_info(self):
        """ 取得分數標準差倍率資訊
        取得各個分數區間所使用的上下界標準差倍率

        Parameters
        ----------
        None.

        Returns
        -------
        result: tuple of list
            [(score, upper_bound, lower_bound), ...]

        """
        self._clone_score_meta_info()
        with open(f'{DATA_LOC}/score_meta_info.pkl', 'rb') as fp:
            data = pickle.load(fp)
        result = []
        for i in range(len(data)):
            record = data.iloc[i]
            score = record[ScoreMetaField.SCORE_VALUE.value]
            upper_bound = record[ScoreMetaField.UPPER_BOUND.value]
            lower_bound = record[ScoreMetaField.LOWER_BOUND.value]
            result.append((score, upper_bound, lower_bound))
        return result

    def update_latest_pattern_occur(self, data: pd.DataFrame):
        """ 儲存新增的最新現象發生後次數統計
        儲存發生指定現象後, 指定市場, 指定天期下的發生與未發生總數,
        上漲總數, 持平總數與下跌總數

        Parameters
        ----------
        data: `pd.DataFrame`
            要儲存的資料

        Returns
        -------
        None.

        """
        engine = self._engine()

        now = datetime.datetime.now()
        create_by = self.CREATE_BY
        create_dt = now
        data['CREATE_BY'] = create_by
        data['CREATE_DT'] = create_dt

        logging.info(f'Update pattern occur')
        if not self.READ_ONLY:
        # 新增最新資料
            data.to_sql(
                'FCST_PAT_MKT_OCCUR',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        logging.info(f'Update pattern occur finished')
    
    def update_latest_pattern_distribution(self, data: pd.DataFrame):
        """ 儲存新增的最新現象統計分布統計量
        儲存發生指定現象後, 指定市場, 指定天期下的報酬分布統計量
        這個方法在儲存均值和標準差時會將結果轉換為 % , 因此會做
        乘 100 的動作

        Parameters
        ----------
        data: `pd.DataFrame`
            要儲存的資料

        Returns
        -------
        None.

        """
        engine = self._engine()

        now = datetime.datetime.now()
        create_by = self.CREATE_BY
        create_dt = now
        data['CREATE_BY'] = create_by
        data['CREATE_DT'] = create_dt
        data_mean = data[MarketDistField.RETURN_MEAN.value].values * 100
        data_std = data[MarketDistField.RETURN_STD.value].values * 100

        data = data[[
            'CREATE_BY', 'CREATE_DT', MarketDistField.PATTERN_ID.value,
            MarketDistField.MARKET_ID.value, MarketDistField.DATE_PERIOD.value]]
        data[MarketDistField.RETURN_MEAN.value] = data_mean
        data[MarketDistField.RETURN_STD.value] = data_std

        logging.info(f'Update pattern distribution')
        if not self.READ_ONLY:
            # 新增最新資料
            data.to_sql(
                'FCST_PAT_MKT_DIST',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        logging.info(f'Update pattern distribution finished')

    def update_latest_pattern_results(self, data: pd.DataFrame):
        """儲存新增的最新現象當前發生資訊至資料表

        Parameters
        ----------
        data: Pandas's DataFrame
            A table of pattern results with columns for market_id, pattern_id,
            price_date and value.
        block_size: int
            Save each data_block that size is block_size in data until finished

        See Also
        --------
        PatternResultField

        """
        engine = self._engine()
        now = datetime.datetime.now()
        create_dt = now
        data['CREATE_BY'] = self.CREATE_BY
        data['CREATE_DT'] = create_dt

        logging.info(f'Update pattern event')
        if not self.READ_ONLY:
            data.to_sql(
                'FCST_PAT_MKT_EVENT',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        logging.info(f'Update pattern event end')

    def get_pattern_info(self, pattern_id: str) -> PatternInfo:
        """取得指定現象 ID 的現象資訊

        Parameters
        ----------
        pattern_id: str
            要取得的現象 ID

        Returns
        -------
        result: model.PatternInfo
            現象計算所需的參數資訊
        """
        logging.info(f'Get pattern info {pattern_id} from db')
        engine = self._engine()
        sql = f"""
            SELECT
                ptn.PATTERN_ID, mcr.FUNC_CODE, para.PARAM_CODE, para.PARAM_VALUE, mp.PARAM_TYPE
            FROM
                FCST_PAT AS ptn
            LEFT JOIN
                (
                    SELECT
                        PATTERN_ID, MACRO_ID, PARAM_CODE, PARAM_VALUE
                    FROM
                        FCST_PAT_PARAM
                ) AS para
            ON ptn.PATTERN_ID=para.PATTERN_ID
            LEFT JOIN
                (
                    SELECT
                        MACRO_ID, FUNC_CODE
                    FROM
                        FCST_MACRO
                ) AS mcr
            ON ptn.MACRO_ID=mcr.MACRO_ID
            LEFT JOIN
                (
                    SELECT
                        MACRO_ID, PARAM_CODE, PARAM_TYPE
                    FROM
                        FCST_MACRO_PARAM
                ) AS mp
            ON ptn.MACRO_ID=mp.MACRO_ID AND para.PARAM_CODE=mp.PARAM_CODE
            ORDER BY ptn.PATTERN_ID, mcr.FUNC_CODE ASC;
        """
        data = pd.read_sql_query(sql, engine)
        result = {}
        for i in range(len(data)):
            record = data.iloc[i]
            pid = record['PATTERN_ID']
            if pid in result:
                continue
            func = record['FUNC_CODE']
            params = {}
            ptn_record = data[data['PATTERN_ID'].values==pid]
            for j in range(len(ptn_record)):
                param_record = ptn_record.iloc[j]
                param_type = param_record['PARAM_TYPE']
                param_val = param_record['PARAM_VALUE']
                if param_type == "int":
                    param_val = int(param_record['PARAM_VALUE'])
                elif param_type == "float":
                    param_val = float(param_record['PARAM_VALUE'])
                elif param_type == "string":
                    param_val = str(param_record['PARAM_VALUE'])
                params[param_record['PARAM_CODE']] = param_val
            result[pid] = PatternInfo(pid, func, params)
        if pattern_id not in result:
            raise Exception(f'get_pattern_info: pattern info not found in db: {pattern_id}')
        result = result[pattern_id]
        logging.info(f'Get pattern info {pattern_id} from db finished')
        return result
