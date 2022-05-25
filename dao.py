import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import traceback
import threading as mt
from sqlalchemy import create_engine
from threading import Lock
from typing import Any, Dict, List, NamedTuple, Optional, Union
from model import (ModelInfo, PatternInfo, CatchableTread,
                pickle_dump, pickle_load,
                get_filed_name_of_future_return)
from const import (LOCAL_DB, DATA_LOC, EXCEPT_DATA_LOC, ExecMode, BatchType, MarketOccurField, ModelMarketHitSumField, PatternResultField,
                   TableName, MarketDistField, ModelExecution, PredictResultField,
                   MarketPeriodField, MarketScoreField, MarketHistoryPriceField,
                   MarketInfoField, DSStockInfoField, PatternInfoField,
                   MarketStatField, ModelExecution, ScoreMetaField,
                   PatternParamField, MacroInfoField, MacroParamField,
                   ModelInfoField, ModelMarketMapField, ModelPatternMapField,
                   ModelExecutionField, StoredProcedule)
from utils import dict_equals

class MimosaDB:
    """
    用來取得資料庫資料的物件，需在同資料夾下設置 config

    """
    config = {}
    CREATE_BY='SYS_BATCH'
    MODIFY_BY='SYS_BATCH'
    READ_ONLY=False
    WRITE_LOCAL=False

    def __init__(
        self, db_name='mimosa', mode: str=ExecMode.DEV.value,
        read_only: bool=False, write_local: bool=False
    ):
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
        self.READ_ONLY = read_only
        self.WRITE_LOCAL = write_local
        if self.WRITE_LOCAL:
            if not os.path.exists(f'{DATA_LOC}/local_out'):
                os.makedirs(f'{DATA_LOC}/local_out', exist_ok=True)

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

    def _convert_exec_to_status(self, exec_info: pd.DataFrame) -> Dict[str, Dict[str, datetime.date]]:
        """將 EXECUTION RECORD 轉換為以完成狀態字典表示, 若 exec_info 長度為 0, 
        則會回傳空字典 {}

        Parameters
        ----------
        exec_info: pd.DataFrame
            執行狀態直表
        
        Returns
        -------
        result: Dict[str, Dict[str, datetime.datetime]]
            完成狀態字典, 格式為 [model_id] - [status_code] - end_dt
        """
        FINISHED_STATUS = [
            ModelExecution.ADD_BACKTEST_FINISHED.value, 
            ModelExecution.ADD_PREDICT_FINISHED.value, 
            ModelExecution.BATCH_PREDICT_FINISHED.value
            ]
        exec_info = exec_info.copy()
        status_records = exec_info.groupby(by=[
            ModelExecutionField.MODEL_ID.value,
            ModelExecutionField.STATUS_CODE.value
            ])
        result = {}
        for (model_id, status_code), records in status_records:
            if model_id not in result:
                result[model_id] = {}
            if status_code not in FINISHED_STATUS:
                    continue
            max_date = np.max(
                records[ModelExecutionField.END_DT.value].values.astype('datetime64[ms]'))
            # datetime64[ms] 轉型為 datetime.datetime
            result[model_id][status_code] = max_date.tolist()
        return result

    def _sync_model_status(self):
        """與資料庫同步本地端模型預測結果執行狀態, 
        沒有紀錄 -> 不會寫檔, 
        有部分記錄 -> 沒有完成狀態的不會寫 key

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        logging.info("Sync local model execution status")
        # 取得所有觀點資訊
        execution_info = pickle_load(f'{DATA_LOC}/{TableName.MODEL_EXECUTION.value}.pkl')
        model_status_info = self._convert_exec_to_status(execution_info)
        for model_id in model_status_info:
            fp = f'{LOCAL_DB}/views/{model_id}'
            need_update = False
            # 取得觀點執行狀態
            status_fp = f"{fp}/status.pkl"
            # 判斷是否需要更新歷史預測結果
            if not os.path.exists(status_fp):
                need_update = True
            else:
                model_status = pickle_load(status_fp)
                if not dict_equals(model_status, model_status_info[model_id]):
                    need_update = True
            if need_update:
                result = model_status_info[model_id]
                os.makedirs(fp, exist_ok=True)
                pickle_dump(result, f"{fp}/status.pkl")
        logging.info("Sync local model execution status finished")
    
    def _sync_model_results(self, controller=None):
        """取得所有模型歷史預測結果資料, 若檔案已存在且 clean_first 為 False
        , 則將會沿用舊資料, 不會進行下載

        Parameters
        ----------
        controller: ThreadController
            用於強制中斷 Thread
        clean_first: bool
            是否要在執行複製前先清空本地端資料

        Returns
        -------
        None.
        """
        if not os.path.exists(f"{DATA_LOC}/model_results"):
            logging.info('Clone model predict result from db')
            os.makedirs(f'{DATA_LOC}/model_results', exist_ok=True)
        engine = self._engine()
        # 這邊犧牲效能, 換取較小的記憶體消耗量
        # 取得所有觀點ID
        model_info = pickle_load(f'{DATA_LOC}/model_training_infos.pkl')
        model_ids = model_info[ModelInfoField.MODEL_ID.value].values.tolist()

        # 取得最新執行狀態
        sql = f"""
            SELECT
                *
            FROM
                {TableName.MODEL_EXECUTION.value}
        """
        model_exec_info = pd.read_sql_query(sql, engine)
        
        model_status_records = self._convert_exec_to_status(model_exec_info)

        for model_id_i, model_id in enumerate(model_ids):
            if controller is not None and not controller.isactive:
                return
            fp = f'{LOCAL_DB}/views/{model_id}'
            need_update = False
            # 取得觀點執行狀態
            status_fp = f'{LOCAL_DB}/views/{model_id}/status.pkl'
            # 判斷是否需要更新歷史預測結果
            if not os.path.exists(status_fp):
                need_update = True
            else:
                model_status = pickle_load(status_fp)
                if not dict_equals(model_status, model_status_records[model_id]):
                    need_update = True
            if need_update:
                sql = f"""
                    SELECT
                        *
                    FROM
                        {TableName.PREDICT_RESULT_HISTORY.value}
                    WHERE
                        {PredictResultField.MODEL_ID.value}='{model_id}'
                """
                data = pd.read_sql_query(sql, engine)
                data[PredictResultField.DATE.value
                    ] = data[PredictResultField.DATE.value
                            ].values.astype('datetime64[D]')
                os.makedirs(fp, exist_ok=True)
                pickle_dump(data, f'{fp}/history_values.pkl')
                logging.info(f'Clone model result[{model_id_i+1}/{len(model_ids)}]: {model_id} finished')
        logging.info('Clone model predict result from db finished')

    def _sync_local_model_results(self, controller=None):
        """與資料庫同步本地端預測結果

        Parameters
        ----------
        controller: ThreadController
            用於強制中斷 Thread

        Returns
        -------
        None.
        """
        self._sync_model_results(controller=controller)
        self._sync_model_status()

    def clone_model_results(self, controller):
        """非同步取得所有模型歷史預測結果資料

        Parameters
        ----------
        controller: ThreadController
            用於強制中斷 Thread

        Returns
        -------
        t: CatchableTread
            下載資料的執行緒
        """
        t = CatchableTread(target=self._sync_local_model_results, args=(controller, ))
        t.start()
        return t

    def get_model_results(self, model_id: str) -> Dict[str, pd.DataFrame]:
        """取得指定模型的歷史預測結果資料

        Parameters
        ----------
        model_id: str
            觀點 ID

        Returns
        -------
        data: Dict[str, pd.DataFrame]
            指定模型下的各市場預測結果, 欄位資料有
             - PredictResultField.PERIOD.value,
             - PredictResultField.DATE.value,
             - PredictResultField.UPPER_BOUND.value,
             - PredictResultField.LOWER_BOUND.value
        """
        fp = f'{LOCAL_DB}/views/{model_id}/history_values.pkl'
        self._sync_model_results()

        data = pickle_load(fp)
        result = {
            market_id: group 
            for market_id, group in
            data.groupby(PredictResultField.MARKET_ID.value)
        }
        return result

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
            table_name = TableName.MKT_HISTORY_PRICE.value
        elif batch_type == BatchType.SERVICE_BATCH:
            table_name = f'{TableName.MKT_HISTORY_PRICE.value}_SWAP'
        else:
            raise Exception(f'_clone_market_data: Unknown batch type: {batch_type}')
        if not os.path.exists(f'{DATA_LOC}/markets'):
            logging.info('Clone market data from db')
            os.makedirs(f'{DATA_LOC}/markets', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    {MarketHistoryPriceField.MARKET_CODE.value},
                    {MarketHistoryPriceField.PRICE_DATE.value},
                    {MarketHistoryPriceField.OPEN_PRICE.value},
                    {MarketHistoryPriceField.HIGH_PRICE.value},
                    {MarketHistoryPriceField.LOW_PRICE.value},
                    {MarketHistoryPriceField.CLOSE_PRICE.value}
                FROM
                    {table_name}
            """
            data = pd.read_sql_query(sql, engine)
            result = pd.DataFrame(
                data[[
                    MarketHistoryPriceField.MARKET_CODE.value,
                    MarketHistoryPriceField.OPEN_PRICE.value,
                    MarketHistoryPriceField.HIGH_PRICE.value,
                    MarketHistoryPriceField.LOW_PRICE.value,
                    MarketHistoryPriceField.CLOSE_PRICE.value]].values,
                index=data[MarketHistoryPriceField.PRICE_DATE.value].values.astype('datetime64[D]'),
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
            table_name = TableName.MKT_INFO.value
        elif batch_type == BatchType.SERVICE_BATCH:
            table_name = f'{TableName.MKT_INFO.value}_SWAP'
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

    def _clean_model_execution(self):
        """清除當前本地端的觀點執行紀錄

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        if os.path.exists(f'{DATA_LOC}/{TableName.MODEL_EXECUTION.value}.pkl'):
            os.remove(f'{DATA_LOC}/{TableName.MODEL_EXECUTION.value}.pkl')
    
    def _clone_model_execution(self):
        if not os.path.exists(f'{DATA_LOC}/{TableName.MODEL_EXECUTION.value}.pkl'):
            logging.info(f'Clone {TableName.MODEL_EXECUTION.value} info from db')
            os.makedirs(f'{DATA_LOC}', exist_ok=True)
            engine = self._engine()
            sql = f"""
                SELECT
                    *
                FROM
                    {TableName.MODEL_EXECUTION.value}
            """
            data = pd.read_sql_query(sql, engine)
            pickle_dump(data, f'{DATA_LOC}/{TableName.MODEL_EXECUTION.value}.pkl')
            logging.info(f'Clone {TableName.MODEL_EXECUTION.value} info from db finished')

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
            table_name = TableName.DS_S_STOCK.value
        elif batch_type == BatchType.SERVICE_BATCH:
            table_name = f'{TableName.DS_S_STOCK.value}_SWAP'
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
                    ptn.{PatternInfoField.PATTERN_ID.value},
                    mcr.{MacroInfoField.FUNC_CODE.value},
                    para.{PatternParamField.PARAM_CODE.value},
                    para.{PatternParamField.PARAM_VALUE.value},
                    mp.{MacroParamField.PARAM_TYPE.value}
                FROM
                    {TableName.PAT_INFO.value} AS ptn
                LEFT JOIN
                    (
                        SELECT
                            {PatternParamField.PATTERN_ID.value},
                            {PatternParamField.MACRO_ID.value},
                            {PatternParamField.PARAM_CODE.value},
                            {PatternParamField.PARAM_VALUE.value}
                        FROM
                            {TableName.PAT_PARAM.value}
                    ) AS para
                ON ptn.{PatternInfoField.PATTERN_ID.value}=para.{PatternParamField.PATTERN_ID.value}
                LEFT JOIN
                    (
                        SELECT
                            {MacroInfoField.MACRO_ID.value},
                            {MacroInfoField.FUNC_CODE.value}
                        FROM
                            {TableName.MACRO_INFO.value}
                    ) AS mcr
                ON ptn.{PatternInfoField.MACRO_ID.value}=mcr.{MacroInfoField.MACRO_ID.value}
                LEFT JOIN
                    (
                        SELECT
                            {MacroParamField.MACRO_ID.value},
                            {MacroParamField.PARAM_CODE.value},
                            {MacroParamField.PARAM_TYPE.value}
                        FROM
                            {TableName.MACRO_PARAM.value}
                    ) AS mp
                ON ptn.{PatternInfoField.MACRO_ID.value}=mp.{MacroParamField.MACRO_ID.value} AND
                para.{PatternParamField.PARAM_CODE.value}=mp.{MacroParamField.PARAM_CODE.value}
                ORDER BY
                    ptn.{PatternInfoField.PATTERN_ID.value},
                    mcr.{MacroInfoField.FUNC_CODE.value} ASC;
            """
            data = pd.read_sql_query(sql, engine)
            result = {}
            for i in range(len(data)):
                record = data.iloc[i]
                pid = record[PatternInfoField.PATTERN_ID.value]
                if pid in result:
                    continue
                func = record[MacroInfoField.FUNC_CODE.value]
                params = {}
                ptn_record = data[data[PatternInfoField.PATTERN_ID.value].values==pid]
                for j in range(len(ptn_record)):
                    param_record = ptn_record.iloc[j]
                    param_type = param_record[MacroParamField.PARAM_TYPE.value]
                    param_val = param_record[PatternParamField.PARAM_VALUE.value]
                    if param_type == "int":
                        param_val = int(param_record[PatternParamField.PARAM_VALUE.value])
                    elif param_type == "float":
                        param_val = float(param_record[PatternParamField.PARAM_VALUE.value])
                    elif param_type == "string":
                        param_val = str(param_record[PatternParamField.PARAM_VALUE.value])
                    params[param_record[PatternParamField.PARAM_CODE.value]] = param_val
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
                    model.{ModelInfoField.MODEL_ID.value}
                FROM
                    {TableName.MODEL_INFO.value} AS model
                INNER JOIN
                    (
                        SELECT
                            {ModelExecutionField.MODEL_ID.value},
                            {ModelExecutionField.STATUS_CODE.value},
                            {ModelExecutionField.END_DT.value}
                        FROM
                            {TableName.MODEL_EXECUTION.value}
                        WHERE
                            {ModelExecutionField.STATUS_CODE.value}='{ModelExecution.ADD_PREDICT_FINISHED}' AND
                            {ModelExecutionField.END_DT.value} IS NOT NULL
                    ) AS me
                ON model.{ModelInfoField.MODEL_ID.value}=me.{ModelExecutionField.MODEL_ID.value}
            """
            data = pd.read_sql_query(sql, engine)[ModelInfoField.MODEL_ID.value].values.tolist()
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
                    {TableName.MODEL_INFO.value}
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
                    {TableName.MODEL_MKT_MAP.value}
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
                    {TableName.MODEL_PAT_MAP.value}
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
                    {TableName.SCORE_META.value}
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
        self._clean_model_execution()

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
        self._clone_model_execution()

    def get_macro_param_type(self, function_code: str) -> Dict[str, str]:
        """ 取得 Macro 中的 function 參數型態資訊

        Parameters
        ----------
        function_code: str
            function 代碼
        
        Returns
        -------
        result: Dict[str, str]
            function 參數對應的型態字典

        """
        engine = self._engine()
        sql = f"""
                SELECT
                    mcr.{MacroInfoField.MACRO_ID.value},
                    mcr.{MacroInfoField.FUNC_CODE.value},
                    mp.{MacroParamField.PARAM_CODE.value},
                    mp.{MacroParamField.PARAM_TYPE.value}
                FROM
                    {TableName.MACRO_INFO.value} AS mcr
                LEFT JOIN
                    (
                        SELECT
                            {MacroParamField.MACRO_ID.value},
                            {MacroParamField.PARAM_CODE.value},
                            {MacroParamField.PARAM_TYPE.value}
                        FROM
                            {TableName.MACRO_PARAM.value}
                    ) AS mp
                ON mcr.{MacroInfoField.MACRO_ID.value}=mp.{MacroParamField.MACRO_ID.value}
                WHERE
                    mcr.{MacroInfoField.FUNC_CODE.value} = '{function_code}';
            """
        data = pd.read_sql_query(sql, engine)
        result = {}
        for i in range(len(data)):
            param = data.iloc[i][MacroParamField.PARAM_CODE.value]
            val = data.iloc[i][MacroParamField.PARAM_TYPE.value]
            result[param] = val
        return result

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
        data = data[[MarketInfoField.MARKET_CODE.value,
                     MarketInfoField.MARKET_SOURCE_CODE.value]].values.astype(str).tolist()
        cate_data = cate_data[DSStockInfoField.STOCK_CODE.value].values.astype(str).tolist()
        data = [x[0] for x in data if x[1] in cate_data]
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
        result = {}
        data = self.get_model_results(model_id)
        for market_id in data:
            datas = (
                data[market_id][PredictResultField.DATE.value].values.astype('datetime64[D]'))
            latest_date = np.max(datas).tolist()
            result[market_id] = latest_date
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
        result = {}
        data = self.get_model_results(model_id)
        for market_id in data:
            datas = (
                data[market_id][PredictResultField.DATE.value].values.astype('datetime64[D]'))
            earliest_date = np.min(datas).tolist()
            result[market_id] = earliest_date
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
        m_cond = model_info[ModelInfoField.MODEL_ID.value].values == model_id
        if len(model_info[m_cond]) == 0:
            # 若發生取不到資料的情況
            raise Exception(f"get_model_info: model not found: {model_id}")
        train_begin = model_info[m_cond].iloc[0][ModelInfoField.TRAIN_START_DT.value]
        train_gap = model_info[m_cond].iloc[0][ModelInfoField.RETRAIN_CYCLE.value]

        # 取得觀點的所有標的市場
        with open(f'{DATA_LOC}/model_markets.pkl', 'rb') as fp:
            markets = pickle.load(fp)
        m_cond = markets[ModelMarketMapField.MODEL_ID.value].values == model_id
        markets = markets[m_cond][ModelMarketMapField.MARKET_CODE.value].values.tolist()

        # 取得觀點所有使用的現象 ID
        with open(f'{DATA_LOC}/model_patterns.pkl', 'rb') as fp:
            patterns = pickle.load(fp)
        m_cond = patterns[ModelPatternMapField.MODEL_ID.value].values == model_id
        if len(patterns[m_cond]) == 0:
            # 若觀點下沒有任何現象, 則回傳例外
            raise Exception(f"get_model_info: 0 model pattern exception: {model_id}")
        patterns = patterns[m_cond][ModelPatternMapField.PATTERN_ID.value].values.tolist()

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
                DELETE FROM {TableName.PREDICT_RESULT.value}
                WHERE
                    {PredictResultField.MODEL_ID.value}='{model_id}'
            """
            engine.execute(sql)

            # DEL FCST_MODEL_MKT_VALUE_HISTORY
            sql = f"""
                DELETE FROM {TableName.PREDICT_RESULT_HISTORY.value}
                WHERE
                    {PredictResultField.MODEL_ID.value}='{model_id}'
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
            latest_data = data
            logging.info('start saving model latest results')
            table_name = f'{TableName.PREDICT_RESULT.value}_SWAP'
            if exec_type == ModelExecution.ADD_PREDICT:
                table_name = TableName.PREDICT_RESULT.value
            elif exec_type == ModelExecution.BATCH_PREDICT:
                table_name = f'{TableName.PREDICT_RESULT.value}_SWAP'
            else:
                logging.error(f'Unknown execution type: {exec_type}')
                return -1

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

            db_data = pd.DataFrame(columns=latest_data.columns)
            if self.WRITE_LOCAL and self.READ_ONLY:
                if os.path.exists(f'{DATA_LOC}/local_out/{table_name}.pkl'):
                    db_data = pickle_load(f'{DATA_LOC}/local_out/{table_name}.pkl')
            elif not self.READ_ONLY:
                engine = self._engine()
                # 新增最新預測結果
                # 合併現有的資料預測結果與當前的預測結果
                db_data = None
                sql = f"""
                    SELECT
                        CREATE_BY, CREATE_DT,
                        {PredictResultField.MODEL_ID.value},
                        {PredictResultField.MARKET_ID.value},
                        {PredictResultField.PERIOD.value},
                        {PredictResultField.DATE.value},
                        {PredictResultField.PREDICT_VALUE.value},
                        {PredictResultField.UPPER_BOUND.value},
                        {PredictResultField.LOWER_BOUND.value}
                    FROM
                        {TableName.PREDICT_RESULT.value}
                    WHERE
                        {PredictResultField.MODEL_ID.value}='{model_id}'
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
            if self.WRITE_LOCAL and self.READ_ONLY:
                if os.path.exists(f'{DATA_LOC}/local_out/{table_name}.pkl'):
                    db_data = pickle_load(f'{DATA_LOC}/local_out/{table_name}.pkl')
                    db_data = db_data[db_data['MODEL_ID'].values != model_id]
                    pickle_dump(db_data, f'{DATA_LOC}/local_out/{table_name}.pkl')
            elif not self.READ_ONLY:
                sql = f"""
                    DELETE FROM {table_name}
                    WHERE
                        {PredictResultField.MODEL_ID.value}='{model_id}'
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

            if self.WRITE_LOCAL:
                pickle_dump(latest_data, f'{DATA_LOC}/local_out/{table_name}.pkl')
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
            sql = f'CALL {StoredProcedule.UPDATE_FCST_MODEL_MKT_HIT_SUM_SWAP.value}()'
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
            sql = f"CALL {StoredProcedule.SWAP_FCST.value}()"
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
        if (exec_type == ModelExecution.ADD_PREDICT or
            exec_type == ModelExecution.ADD_BACKTEST):
            table_name = f'{TableName.PREDICT_RESULT_HISTORY.value}'
        elif exec_type == ModelExecution.BATCH_PREDICT:
            table_name = f'{TableName.PREDICT_RESULT_HISTORY.value}_SWAP'
        else:
            logging.error(f'Unknown execution type: {exec_type}')
            return -1
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
                    table_name,
                    engine,
                    if_exists='append',
                    chunksize=10000,
                    method='multi',
                    index=False)
                # 與本地端資料同步更新
                history_fp = f'{LOCAL_DB}/views/{model_id}/history_values.pkl'
                history_data = pd.DataFrame(columns=data.columns)
                if os.path.exists(history_fp):
                    history_data = pickle_load(history_fp)
                history_data = pd.concat([history_data, data], axis=0)
                history_data = history_data.drop_duplicates(subset=[
                    PredictResultField.MODEL_ID.value,
                    PredictResultField.MARKET_ID.value,
                    PredictResultField.PERIOD.value,
                    PredictResultField.DATE.value
                ])
                pickle_dump(history_data, history_fp)
            except Exception as e:
                logging.info('save_model_results: Saving model history results failed, maybe PK duplicated, skipped it.')
                logging.debug(traceback.format_exc())
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_MODEL_MKT_VALUE_HISTORY_SWAP.pkl')
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
                    DELETE FROM {TableName.MODEL_EXECUTION.value}
                    WHERE {ModelExecutionField.MODEL_ID.value}='{model_id}' AND
                    {ModelExecutionField.STATUS_CODE.value}='{exection}'
                """
                engine.execute(sql)

            # 取得 EXEC_ID
            logging.info(f'Call {StoredProcedule.GET_SERIAL_NO.value}')
            sql = f"CALL {StoredProcedule.GET_SERIAL_NO.value}('EXEC_ID', null)"
            with engine.begin() as db_conn:
                results = db_conn.execute(sql).fetchone()
                exec_id = results[0]
            logging.info(f"Get SERIAL_NO: {exec_id}")

            # 建立 status
            COLUMNS = [
                'CREATE_BY', 'CREATE_DT',
                ModelExecutionField.EXEC_ID.value,
                ModelExecutionField.MODEL_ID.value,
                ModelExecutionField.STATUS_CODE.value,
                ModelExecutionField.START_DT.value,
                ModelExecutionField.END_DT.value
                ]
            now = datetime.datetime.now()
            now = np.datetime64(now).astype('datetime64[s]').tolist()
            create_by = self.CREATE_BY
            create_dt = now
            start_dt = now
            end_dt = None
            data = [[
                create_by, create_dt, exec_id,
                model_id, exection, start_dt, end_dt]]
            data = pd.DataFrame(data, columns=COLUMNS)
            data.to_sql(
                TableName.MODEL_EXECUTION.value,
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
        now = np.datetime64(now).astype('datetime64[s]').tolist()
        engine = self._engine()

        finished_status = {
            ModelExecution.ADD_BACKTEST.value: ModelExecution.ADD_BACKTEST_FINISHED.value,
            ModelExecution.ADD_PREDICT.value: ModelExecution.ADD_PREDICT_FINISHED.value,
            ModelExecution.BATCH_PREDICT.value: ModelExecution.BATCH_PREDICT_FINISHED.value
        }

        sql = f"""
            SELECT
                {ModelExecutionField.MODEL_ID.value},
                {ModelExecutionField.STATUS_CODE.value}
            FROM
                {TableName.MODEL_EXECUTION.value}
            WHERE
                {ModelExecutionField.EXEC_ID.value}='{exec_id}';
        """
        exec_data = pd.read_sql_query(sql, engine).iloc[0]
        if len(exec_data) == 0:
            raise Exception('call set_model_execution_complete before set_model_execution_start')

        status = finished_status[exec_data[ModelExecutionField.STATUS_CODE.value]]
        model_id = exec_data[ModelExecutionField.MODEL_ID.value]
        if not self.READ_ONLY:
            sql = f"""
            UPDATE
                {TableName.MODEL_EXECUTION.value}
            SET
                {ModelExecutionField.END_DT.value}='{now}',
                MODIFY_DT='{now}',
                MODIFY_BY='{self.MODIFY_BY}',
                {ModelExecutionField.STATUS_CODE.value}='{status}'
            WHERE
                {ModelExecutionField.EXEC_ID.value}='{exec_id}';
            """
            engine.execute(sql)

            # 同步更新本地端紀錄
            status_fp = f'{LOCAL_DB}/views/{model_id}/status.pkl'
            local_status = {}
            if os.path.exists(status_fp):
                local_status = pickle_load(status_fp)
            local_status[status] = now
            pickle_dump(local_status, status_fp)

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
        # TODO
        engine = self._engine()
        sql = f"""
            SELECT
                model.{ModelInfoField.MODEL_ID.value},
                me.{ModelExecutionField.STATUS_CODE.value},
                me.{ModelExecutionField.END_DT.value}
            FROM
                {TableName.MODEL_INFO.value} AS model
            LEFT JOIN
                (
                    SELECT
                        {ModelExecutionField.MODEL_ID.value},
                        {ModelExecutionField.STATUS_CODE.value},
                        {ModelExecutionField.END_DT.value}
                    FROM
                        {TableName.MODEL_EXECUTION.value}
                ) AS me
            ON model.{ModelInfoField.MODEL_ID.value}=me.{ModelExecutionField.MODEL_ID.value}
        """
        data = pd.read_sql_query(sql, engine)
        group_data = data.groupby(ModelInfoField.MODEL_ID.value)
        results = []
        for model_id, model_state_info in group_data:
            model_add_predict_info = model_state_info[
                model_state_info[ModelExecutionField.STATUS_CODE.value].values==
                ModelExecution.ADD_PREDICT.value]
            model_add_predict_finished_info = model_state_info[
                model_state_info[ModelExecutionField.STATUS_CODE.value].values==
                ModelExecution.ADD_PREDICT_FINISHED.value]
            model_add_backtest_info = model_state_info[
                model_state_info[ModelExecutionField.STATUS_CODE.value].values==
                ModelExecution.ADD_BACKTEST.value]
            model_add_backtest_finished_info = model_state_info[
                model_state_info[ModelExecutionField.STATUS_CODE.value].values==
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
            sql_template = f"""
                INSERT INTO
                    {TableName.PATTERN_RESULT.value}_SWAP
                (
                    CREATE_BY, CREATE_DT,
                    {PatternResultField.PATTERN_ID.value},
                    {PatternResultField.MARKET_ID.value},
                    {PatternResultField.DATE.value},
                    {PatternResultField.VALUE.value}
                )
                VALUES
            """
            vals = [
                str(
                    (
                        str(x[0]),
                        datetime.datetime.strftime(x[1], '%Y-%m-%d %H:%M:%S'),
                        str(x[2]),
                        str(x[3]),
                        str(x[4])[:10],
                        str(x[5])
                    )
                ) for x in data[[
                    'CREATE_BY', 'CREATE_DT',
                    PatternResultField.PATTERN_ID.value,
                    PatternResultField.MARKET_ID.value,
                    PatternResultField.DATE.value,
                    PatternResultField.VALUE.value
                ]].values
            ]
            batch_size = 10000
            batch_idxs = [i for i in range(0, len(vals)+batch_size, batch_size)]
            logging.info(f'Pattern event len: {len(vals)}')
            for idx_i, idx in enumerate(batch_idxs):
                if idx_i == 0:
                    continue
                sql = sql_template + ','.join(vals[batch_idxs[idx_i-1]:idx])
                logging.info(f'Save pattern event: #{batch_idxs[idx_i-1]} ~ #{idx}')
                engine.execute(sql)
                logging.info(f'Save pattern event: #{batch_idxs[idx_i-1]} ~ #{idx} finished')
            logging.info(f'Pattern event len: {len(vals)} finished')
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_PAT_MKT_EVENT_SWAP.pkl')
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
                f'{TableName.PAT_MKT_DIST.value}_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_PAT_MKT_DIST_SWAP.pkl')
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
                f'{TableName.PAT_MKT_OCCUR.value}_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_PAT_MKT_OCCUR_SWAP.pkl')
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
                f'{TableName.MKT_SCORE.value}_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_MKT_SCORE_SWAP.pkl')
        logging.info('Save market score finished')

    def save_latest_mkt_period(self, data: pd.DataFrame):
        """ 儲存市場各天期最新歷史報酬
        儲存各市場各天期最新歷史報酬

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
        data[MarketPeriodField.PRICE_DATE.value] = data[MarketPeriodField.PRICE_DATE.value].astype('datetime64[D]')

        logging.info('Save market period')
        if not self.READ_ONLY:
            # 新增最新資料
            data.to_sql(
                f'{TableName.MKT_PERIOD.value}_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_MKT_PERIOD_SWAP.pkl')
        logging.info('Save market period finished')

    def save_mkt_period(self, data: pd.DataFrame):
        """ 儲存市場各天期歷史報酬與最新報酬
        儲存各市場各天期歷史報酬與最新報酬

        Parameters
        ----------
        data: `pd.DataFrame`
            要儲存的資料

        Returns
        -------
        None.

        """
        engine = self._engine()

        # 儲存最新市場各天期報酬
        self.save_latest_mkt_period(data.copy(deep=True))

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

        logging.info('Save market period history')
        if not self.READ_ONLY:
            # 新增最新資料
            data.to_sql(
                f'{TableName.MKT_PERIOD_HISTORY.value}_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_MKT_PERIOD_HISTORY_SWAP.pkl')
        logging.info('Save market period history finished')

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
                f'{TableName.MKT_DIST.value}_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_MKT_DIST_SWAP.pkl')
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

    def update_latest_pattern_occur(self, pattern_id: str, data: pd.DataFrame):
        """ 儲存新增的最新現象發生後次數統計
        儲存發生指定現象後, 指定市場, 指定天期下的發生與未發生總數,
        上漲總數, 持平總數與下跌總數

        Parameters
        ----------
        pattern_id: `str`
            要更新的現象 ID
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

        if self.WRITE_LOCAL and self.READ_ONLY:
            if os.path.exists(f'{DATA_LOC}/local_out/FCST_PAT_MKT_OCCUR.pkl'):
                db_data = pickle_load(f'{DATA_LOC}/local_out/FCST_PAT_MKT_OCCUR.pkl')
                db_data = db_data[db_data['PATTERN_ID'].values != pattern_id]
                pickle_dump(db_data, f'{DATA_LOC}/local_out/FCST_PAT_MKT_OCCUR.pkl')
        elif not self.READ_ONLY:
            sql = f"""
                DELETE FROM {TableName.PAT_MKT_OCCUR.value}
                WHERE
                    {MarketOccurField.PATTERN_ID.value}='{pattern_id}'
            """
            engine.execute(sql)

        logging.info(f'Update pattern occur')
        if not self.READ_ONLY:
        # 新增最新資料
            data.to_sql(
                f'{TableName.PAT_MKT_OCCUR.value}',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_PAT_MKT_OCCUR.pkl')
        logging.info(f'Update pattern occur finished')

    def update_latest_pattern_distribution(self, pattern_id: str, data: pd.DataFrame):
        """ 儲存新增的最新現象統計分布統計量
        儲存發生指定現象後, 指定市場, 指定天期下的報酬分布統計量
        這個方法在儲存均值和標準差時會將結果轉換為 % , 因此會做
        乘 100 的動作

        Parameters
        ----------
        pattern_id: `str`
            要更新的現象 ID
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

        if self.WRITE_LOCAL and self.READ_ONLY:
            if os.path.exists(f'{DATA_LOC}/local_out/FCST_PAT_MKT_DIST.pkl'):
                db_data = pickle_load(f'{DATA_LOC}/local_out/FCST_PAT_MKT_DIST.pkl')
                db_data = db_data[db_data['PATTERN_ID'].values != pattern_id]
                pickle_dump(db_data, f'{DATA_LOC}/local_out/FCST_PAT_MKT_DIST.pkl')
        elif not self.READ_ONLY:
            sql = f"""
                DELETE FROM {TableName.PAT_MKT_DIST.value}
                WHERE
                    {MarketDistField.PATTERN_ID.value}='{pattern_id}'
            """
            engine.execute(sql)

        logging.info(f'Update pattern distribution')
        if not self.READ_ONLY:
            # 新增最新資料
            data.to_sql(
                f'{TableName.PAT_MKT_DIST.value}',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_PAT_MKT_DIST.pkl')
        logging.info(f'Update pattern distribution finished')

    def update_latest_pattern_results(self, pattern_id: str, data: pd.DataFrame):
        """儲存新增的最新現象當前發生資訊至資料表

        Parameters
        ----------
        pattern_id: `str`
            要更新的現象 ID
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

        if self.WRITE_LOCAL and self.READ_ONLY:
            if os.path.exists(f'{DATA_LOC}/local_out/FCST_PAT_MKT_EVENT.pkl'):
                db_data = pickle_load(f'{DATA_LOC}/local_out/FCST_PAT_MKT_EVENT.pkl')
                db_data = db_data[db_data['PATTERN_ID'].values != pattern_id]
                pickle_dump(db_data, f'{DATA_LOC}/local_out/FCST_PAT_MKT_EVENT.pkl')
        elif not self.READ_ONLY:
            sql = f"""
                DELETE FROM {TableName.PATTERN_RESULT.value}
                WHERE
                    {PatternResultField.PATTERN_ID.value}='{pattern_id}'
            """
            engine.execute(sql)

        logging.info(f'Update pattern event')
        if not self.READ_ONLY:
            data.to_sql(
                f'{TableName.PATTERN_RESULT.value}',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            pickle_dump(data, f'{DATA_LOC}/local_out/FCST_PAT_MKT_EVENT.pkl')
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
                ptn.{PatternInfoField.PATTERN_ID.value},
                mcr.{MacroInfoField.FUNC_CODE.value},
                para.{PatternParamField.PARAM_CODE.value},
                para.{PatternParamField.PARAM_VALUE.value},
                mp.{MacroParamField.PARAM_TYPE.value}
            FROM
                {TableName.PAT_INFO.value} AS ptn
            LEFT JOIN
                (
                    SELECT
                        {PatternParamField.PATTERN_ID.value},
                        {PatternParamField.MACRO_ID.value},
                        {PatternParamField.PARAM_CODE.value},
                        {PatternParamField.PARAM_VALUE.value}
                    FROM
                        {TableName.PAT_PARAM.value}
                ) AS para
            ON ptn.{PatternInfoField.PATTERN_ID.value}=para.{PatternParamField.PATTERN_ID.value}
            LEFT JOIN
                (
                    SELECT
                        {MacroInfoField.MACRO_ID.value},
                        {MacroInfoField.FUNC_CODE.value}
                    FROM
                        {TableName.MACRO_INFO.value}
                ) AS mcr
            ON ptn.{PatternInfoField.MACRO_ID.value}=mcr.{MacroInfoField.MACRO_ID.value}
            LEFT JOIN
                (
                    SELECT
                        {MacroParamField.MACRO_ID.value},
                        {MacroParamField.PARAM_CODE.value},
                        {MacroParamField.PARAM_TYPE.value}
                    FROM
                        {TableName.MACRO_PARAM.value}
                ) AS mp
            ON ptn.{PatternInfoField.MACRO_ID.value}=mp.{MacroParamField.MACRO_ID.value} AND
            para.{PatternParamField.PARAM_CODE.value}=mp.{MacroParamField.PARAM_CODE.value}
            ORDER BY
                ptn.{PatternInfoField.PATTERN_ID.value},
                mcr.{MacroInfoField.FUNC_CODE.value} ASC;
        """
        data = pd.read_sql_query(sql, engine)
        result = {}
        for i in range(len(data)):
            record = data.iloc[i]
            pid = record[PatternInfoField.PATTERN_ID.value]
            if pid in result:
                continue
            func = record[MacroInfoField.FUNC_CODE.value]
            params = {}
            ptn_record = data[data[PatternInfoField.PATTERN_ID.value].values==pid]
            for j in range(len(ptn_record)):
                param_record = ptn_record.iloc[j]
                param_type = param_record[MacroParamField.PARAM_TYPE.value]
                param_val = param_record[PatternParamField.PARAM_VALUE.value]
                if param_type == "int":
                    param_val = int(param_record[PatternParamField.PARAM_VALUE.value])
                elif param_type == "float":
                    param_val = float(param_record[PatternParamField.PARAM_VALUE.value])
                elif param_type == "string":
                    param_val = str(param_record[PatternParamField.PARAM_VALUE.value])
                params[param_record[PatternParamField.PARAM_CODE.value]] = param_val
            result[pid] = PatternInfo(pid, func, params)
        if pattern_id not in result:
            raise Exception(f'get_pattern_info: pattern info not found in db: {pattern_id}')
        result = result[pattern_id]
        logging.info(f'Get pattern info {pattern_id} from db finished')
        return result

    def truncate_swap_tables(self):
        """ 清除資料庫中執行批次時需要為空的 SWAP 表

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        if not self.READ_ONLY:
            logging.info('Start truncate swap tables')
            engine = self._engine()
            sql = f"CALL {StoredProcedule.TRUNCATE_AI_SWAP.value}()"
            with engine.begin() as db_conn:
                db_conn.execute(sql)
            logging.info('Start truncate swap tables finished')

    def save_model_hit_sum(self, data:pd.DataFrame):
        """儲存模型預測結果準確率

        Parameters
        ----------
        data: pd.DataFrame
            模型預測結果準確率

        Returns
        -------
        None.
        """
        engine = self._engine()
        now = datetime.datetime.now()
        create_dt = now
        data['CREATE_BY'] = self.CREATE_BY
        data['CREATE_DT'] = create_dt
        model_id = data.iloc[0][ModelMarketHitSumField.MODEL_ID.value]

        if self.WRITE_LOCAL and self.READ_ONLY:
            if os.path.exists(f'{DATA_LOC}/local_out/{TableName.MODEL_MKT_HIT_SUM.value}_SWAP.pkl'):
                db_data = pickle_load(f'{DATA_LOC}/local_out/{TableName.MODEL_MKT_HIT_SUM.value}_SWAP.pkl')
                db_data = db_data[db_data[ModelMarketHitSumField.MODEL_ID.value].values != model_id]
                pickle_dump(db_data, f'{DATA_LOC}/local_out/{TableName.MODEL_MKT_HIT_SUM.value}_SWAP.pkl')

        logging.info(f'Insert into model hit sum swap')
        if not self.READ_ONLY:
            data.to_sql(
                f'{TableName.MODEL_MKT_HIT_SUM.value}_SWAP',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            db_data = pickle_load(f'{DATA_LOC}/local_out/{TableName.MODEL_MKT_HIT_SUM.value}_SWAP.pkl')
            db_data = pd.concat([db_data, data], axis=0)
            pickle_dump(db_data, f'{DATA_LOC}/local_out/{TableName.MODEL_MKT_HIT_SUM.value}_SWAP.pkl')
        logging.info(f'Insert into model hit sum swap finished')

    def update_model_hit_sum(self, data:pd.DataFrame):
        """儲存模型預測結果準確率至當前準確率表

        Parameters
        ----------
        data: pd.DataFrame
            模型預測結果準確率

        Returns
        -------
        None.
        """
        engine = self._engine()
        now = datetime.datetime.now()
        create_dt = now
        data['CREATE_BY'] = self.CREATE_BY
        data['CREATE_DT'] = create_dt
        model_id = data.iloc[0][ModelMarketHitSumField.MODEL_ID.value]

        if self.WRITE_LOCAL and self.READ_ONLY:
            if os.path.exists(f'{DATA_LOC}/local_out/{TableName.MODEL_MKT_HIT_SUM.value}.pkl'):
                db_data = pickle_load(f'{DATA_LOC}/local_out/{TableName.MODEL_MKT_HIT_SUM.value}.pkl')
                db_data = db_data[db_data[ModelMarketHitSumField.MODEL_ID.value].values != model_id]
                pickle_dump(db_data, f'{DATA_LOC}/local_out/{TableName.MODEL_MKT_HIT_SUM.value}.pkl')

        logging.info(f'Update model hit sum')
        if not self.READ_ONLY:
            data.to_sql(
                f'{TableName.MODEL_MKT_HIT_SUM.value}',
                engine,
                if_exists='append',
                chunksize=10000,
                method='multi',
                index=False)
        if self.WRITE_LOCAL:
            db_data = pickle_load(f'{DATA_LOC}/local_out/{TableName.MODEL_MKT_HIT_SUM.value}.pkl')
            db_data = pd.concat([db_data, data], axis=0)
            pickle_dump(db_data, f'{DATA_LOC}/local_out/{TableName.MODEL_MKT_HIT_SUM.value}.pkl')
        logging.info(f'Update model hit sum finished')
