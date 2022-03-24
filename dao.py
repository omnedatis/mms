import datetime
import json
import logging
import os
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from threading import Lock
from typing import Any, Dict, List, NamedTuple, Optional, Union

from model import (ModelInfo, PatternInfo, 
                pickle_dump, pickle_load, 
                get_filed_name_of_future_return, set_model_execution_start)
from const import (LOCAL_DB, DATA_LOC,
                   MarketDistField, ModelExecution, PredictResultField, 
                   PatternResultField, ModelExecution)
# 設定 logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(lineno)s - %(levelname)s - %(message)s')

    
class MimosaDB:
    """
    用來取得資料庫資料的物件，需在同資料夾下設置 config
    """
    config = {}
    CREATE_BY='SYS_BATCH'
    MODIFY_BY='SYS_BATCH'
    
    def __init__(self, db_name='mimosa', mode='dev'):
        """
        根據傳入參數取得指定的資料庫設定
        """
        def init_config():
            """
            load the configuration
            """
            current_path = os.path.split(os.path.realpath(__file__))[0]
            config = json.load(open('%s/config'%(current_path)))[db_name][mode]
            return config
        self.config = init_config()
        self._local_db_lock = Lock()
    
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
            os.remove(f'{DATA_LOC}/markets')

    def _clone_market_data(self):
        """ 複製當前資料庫中的市場歷史資料至本地端
        
        Returns
        -------
        None.
        
        """
        if not os.path.exists(f'{DATA_LOC}/markets'):
            logging.info('Clone market data from db')
            os.makedirs(f'{DATA_LOC}/markets', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    MARKET_CODE, PRICE_DATE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE
                FROM
                    FCST_MKT_PRICE_HISTORY
                ORDER BY
                    PRICE_DATE ASC
            """
            data = pd.read_sql_query(sql, engine)
            result = pd.DataFrame(
                data[['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE']].values.astype(float),
                index=data['PRICE_DATE'].values.astype('datetime64[D]'),
                columns=['OP', 'HP', 'LP', 'CP']
                )
            market_groups = result.groupby('MARKET_CODE')
            for market_id, market_group in market_groups:
                with open(f'{DATA_LOC}/markets/{market_id}.pkl', 'wb') as fp:
                    pickle.dump(market_group.sort_index(), fp)
            logging.info('Clone market data from db finished')

    def _clean_markets(self):
        """ 清除當前本地端市場清單
        
        Returns
        -------
        None.
        
        """
        if os.path.exists(f'{DATA_LOC}/market_list.pkl'):
            os.remove(f'{DATA_LOC}/market_list.pkl')
    
    def _clone_markets(self):
        """複製當前資料庫中的市場清單至本地端
        
        Returns
        -------
        None.
        """
        if not os.path.exists(f'{DATA_LOC}/market_list.pkl'):
            logging.info('Clone market list from db')
            os.makedirs(f'{DATA_LOC}', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    MARKET_CODE
                FROM
                    FCST_MKT
            """
            data = pd.read_sql_query(sql, engine)['MARKET_CODE'].values.tolist()
            with open(f'{DATA_LOC}/market_list.pkl', 'wb') as fp:
                pickle.dump(data, fp)
            logging.info('Clone market list from db finished')

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
                pickle.dump(data, fp)
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

    def clean_db_cache(self):
        """ 清除本地資料庫快取
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """
        self._clean_market_data()
        self._clean_markets()
        self._clean_patterns()
        self._clean_models()
        self._clean_model_training_infos()
        self._clean_model_markets()
        self._clean_model_patterns()

    def clone_db_cache(self):
        """ 從資料庫載入快取
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """
        self._clone_market_data()
        self._clone_markets()
        self._clone_patterns()
        self._clone_models()
        self._clone_model_training_infos()
        self._clone_model_markets()
        self._clone_model_patterns()

    def get_markets(self):
        """get market IDs from DB.
        
        Returns
        -------
        list of str
        
        """
        self._clone_markets()
        with open(f'{DATA_LOC}/market_list.pkl', 'rb') as fp:
            data = pickle.load(fp)
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
        self._local_db_lock.acquire()
        pickle_dump(data, self._get_future_return_file(market_id))
        self._local_db_lock.release()

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
        self._local_db_lock.acquire()
        field = get_filed_name_of_future_return(period)
        ret = pickle_load(self._get_future_return_file(market_id))[field]
        self._local_db_lock.release()
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
        self._local_db_lock.acquire()
        pickle_dump(data, self._get_pattern_file(market_id))
        self._local_db_lock.release()

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
        self._local_db_lock.acquire()
        ret = pickle_load(self._get_pattern_file(market_id))[patterns]
        self._local_db_lock.release()
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

    def get_models(self):
        """Get models from DB which complete ADD_PREDICT.
        
        Returns
        -------
        list of str
            IDs of models in DB
        
        """
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
        self._clone_model_training_infos()
        self._clone_model_markets()
        self._clone_model_patterns()
        # TODO 若發生取不到資料的情況
        # 取得觀點的訓練資訊
        with open(f'{DATA_LOC}/model_infos.pkl', 'rb') as fp:
            model_info = pickle.load(fp)
        m_cond = model_info['MODEL_ID'].values == model_id
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
        # DEL FCST_MODEL_MKT_VALUE
        sql = f"""
            DELETE FROM FCST_MODEL_MKT_VALUE
            WHERE
                MODEL_ID='{model_id}'
        """
        engine.execute(sql)
        
        
        # DEL FCST_MODEL_USER_MAP
        sql = f"""
            DELETE FROM FCST_MODEL_USER_MAP
            WHERE
                MODEL_ID='{model_id}'
        """
        # engine.execute(sql)

        # DEL FCST_MODEL_PAT_MAP
        sql = f"""
            DELETE FROM FCST_MODEL_PAT_MAP
            WHERE
                MODEL_ID='{model_id}'
        """
        # engine.execute(sql)

        # DEL FCST_MODEL_MKT_MAP
        sql = f"""
            DELETE FROM FCST_MODEL_MKT_MAP
            WHERE
                MODEL_ID='{model_id}'
        """
        # engine.execute(sql)

        # DEL FCST_WATCHLIST_MODEL_MAP
        sql = f"""
            DELETE FROM FCST_WATCHLIST_MODEL_MAP
            WHERE
                MODEL_ID='{model_id}'
        """
        # engine.execute(sql)

        # DEL FCST_MODEL_EXEC
        sql = f"""
            DELETE FROM FCST_MODEL_EXEC
            WHERE
                MODEL_ID='{model_id}'
        """
        # engine.execute(sql)

        # DEL FCST_MODEL
        sql = f"""
            DELETE FROM FCST_MODEL
            WHERE
                MODEL_ID='{model_id}'
        """
        # engine.execute(sql)

    def save_model_results(self, model_id:str, data:pd.DataFrame):
        """Save modle predicting results to DB.
        
        Parameters
        ----------
        model_id: str
            ID of model.
        data: Pandas's DataFrame
            A panel of floating with columns, 'lb_p', 'ub_p', 'pr_p' for each 
            predicting period as p.
        
        """
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

        data.to_sql(
            'FCST_MODEL_MKT_VALUE', 
            engine, 
            if_exists='append', 
            chunksize=1000,
            index=False)

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
        engine = self._engine()
        # 取得 EXEC_ID
        sql = f"CALL SP_GET_SERIAL_NO('EXEC_ID', @EXEC_ID)"
        with engine.begin() as db_conn:
            db_conn.execute(sql)
            results = db_conn.execute('SELECT @EXEC_ID').fetchone()
            exec_id = results[0]
        logging.info(exec_id)

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
            chunksize=1000,
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
            elif pd.isnull(model_add_predict_info.iloc[0]['END_DT']):
                results.append((model_id, ModelExecution.ADD_PREDICT))
            # ADD_BACKTEST 未建立就掛掉
            elif (len(model_add_backtest_info) == 0) and (len(model_add_backtest_finished_info) == 0):
                results.append((model_id, ModelExecution.ADD_BACKTEST))
            # ADD_BACKTEST 未完成就掛掉
            elif pd.isnull(model_add_backtest_info.iloc[0]['END_DT']):
                results.append((model_id, ModelExecution.ADD_BACKTEST))
        return results

    def save_latest_pattern_results(self, data:pd.DataFrame, block_size: int=100000):
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
        logging.info('delete deprecated records')
        engine = self._engine()
        cur_records = data[[
            PatternResultField.PATTERN_ID.value, 
            PatternResultField.MARKET_ID.value
            ]]
        sql = f"""
            SELECT
                {PatternResultField.PATTERN_ID.value}, {PatternResultField.MARKET_ID.value}
            FROM
                FCST_PAT_MKT_EVENT
        """
        prev_records = pd.read_sql_query(sql, engine)
        deprecated_records = pd.concat([
            prev_records, cur_records, cur_records
            ]).drop_duplicates(keep=False)
        if len(deprecated_records) > 0:
            # 若有需要刪除的資料
            deprecated_pks = [
                (
                    deprecated_records.iloc[i][PatternResultField.PATTERN_ID.value],
                    deprecated_records.iloc[i][PatternResultField.MARKET_ID.value]
                ) for i in range(len(deprecated_records))]
            pk_sets = f"('{deprecated_pks[0][0]}', '{deprecated_pks[0][1]}')"
            for i in range(1, len(deprecated_pks)):
                pat_id, market_id = deprecated_pks[i]
                pk_sets += f", ('{pat_id}', '{market_id}')"
            
            sql = f"""
                DELETE FROM
                    FCST_PAT_MKT_EVENT
                WHERE
                    ({PatternResultField.PATTERN_ID.value}, {PatternResultField.MARKET_ID.value})
                IN
                    ({pk_sets})
            """
            engine.execute(sql)
        logging.info('delete deprecated records finished')
        
        logging.info(f'saving start')
        for idx in range(0, len(data), block_size):
            self._save_latest_pattern_results(data[idx: idx+block_size])
            logging.info(f'save {idx} records')

    def _save_latest_pattern_results(self, data:pd.DataFrame):
        """Save latest pattern results to DB.
        
        Parameters
        ----------
        data: Pandas's DataFrame
            A table of pattern results with columns for market_id, pattern_id, 
            price_date and value.
            
        See Also
        --------
        PatternResultField
        
        """
        engine = self._engine()
        
        now = datetime.datetime.now()
        create_by = self.CREATE_BY
        create_dt = now
        modify_by = self.MODIFY_BY

        sql = f"""
        INSERT INTO 
            FCST_PAT_MKT_EVENT (
                CREATE_BY, CREATE_DT, MODIFY_BY, MODIFY_DT, 
                PATTERN_ID, MARKET_CODE, DATA_DATE, OCCUR_YN
            ) 
        VALUES """
        if len(data) > 0:
            ptn_event = data.iloc[0]
            sql += f"""
            (
                '{create_by}', '{str(create_dt)}', null, null,
                '{ptn_event[PatternResultField.PATTERN_ID.value]}', 
                '{ptn_event[PatternResultField.MARKET_ID.value]}', 
                '{str(ptn_event[PatternResultField.DATE.value])}', 
                '{'Y' if ptn_event[PatternResultField.VALUE.value] > 0 else 'N'}'
            )
            """
        for i in range(1, len(data)):
            ptn_event = data.iloc[i]
            sql += f"""
            , (
                '{create_by}', '{str(create_dt)}', null, null,
                '{ptn_event[PatternResultField.PATTERN_ID.value]}', 
                '{ptn_event[PatternResultField.MARKET_ID.value]}', 
                '{str(ptn_event[PatternResultField.DATE.value])}', 
                '{'Y' if ptn_event[PatternResultField.VALUE.value] > 0 else 'N'}'
            )
            """
        sql += f"""
        ON DUPLICATE KEY UPDATE DATA_DATE = VALUES(DATA_DATE),
                                MODIFY_BY = '{modify_by}',
                                MODIFY_DT = CURRENT_TIMESTAMP(),
                                OCCUR_YN = VALUES(OCCUR_YN)
        ;
        """
        engine.execute(sql)

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
        data_mean = data[MarketDistField.RETURN_MEAN].values * 100
        data_std = data[MarketDistField.RETURN_STD].values * 100
        
        data = data[[
            'CREATE_BY', 'CREATE_DT', MarketDistField.PATTERN_ID, 
            MarketDistField.MARKET_ID, MarketDistField.DATE_PERIOD]]
        data[MarketDistField.RETURN_MEAN] = data_mean
        data[MarketDistField.RETURN_STD] = data_std
        
        # 刪除現有資料庫
        sql = "DELETE FROM FCST_PAT_MKT_DIST"
        engine.execute(sql)

        # 新增最新資料
        data.to_sql(
            'FCST_PAT_MKT_DIST', 
            engine, 
            if_exists='append', 
            chunksize=1000,
            index=False)
    
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
        
        # 刪除現有資料庫
        sql = "DELETE FROM FCST_PAT_MKT_OCCUR"
        engine.execute(sql)

        # 新增最新資料
        data.to_sql(
            'FCST_PAT_MKT_OCCUR', 
            engine, 
            if_exists='append', 
            chunksize=1000,
            index=False)

