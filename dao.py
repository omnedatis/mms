import datetime
from enum import Enum
import json
import logging
import os
import pickle
import shutil
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pymysql import IntegrityError

from sqlalchemy import create_engine, exc
from _core._macro import MacroManager, MacroParaEnumManager, MacroTags
from const import (DATA_LOC, BatchType, DBModelStatus, DBPatternStatus,
                   DSStockInfoField, ExecMode, MacroInfoField,
                   MacroParamEnumField, MacroParamField, MacroTagField,
                   MacroTagMapField, MacroVersionInfoField,
                   MarketHistoryPriceField, MarketInfoField,
                   MarketPeriodField, MarketScoreField, ModelExecution,
                   ModelExecutionField, ModelInfoField, ModelMarketHitSumField,
                   ModelMarketMapField, ModelPatternMapField, PatternExecution,
                   PatternExecutionField, PatternInfoField, PatternParamField,
                   PatternResultField, PredictResultField, ScoreMetaField,
                   StoredProcedule, TableName, SerialNoType, DataType, CacheName,
                   ModelKernelField)
from _core import Pattern as PatternInfo
from _core import View as ModelInfo
from utils import CatchableTread, ThreadController, pickle_dump, pickle_load, dict_equals


class MimosaDBCacheManager:
    IS_SWAP_TYPE = {
        BatchType.INIT_BATCH: False,
        BatchType.SERVICE_BATCH: True
    }
    BASE_TABLES = [
        TableName.MKT_HISTORY_PRICE.value, TableName.MKT_INFO.value,
        TableName.DS_S_STOCK.value,
        TableName.PAT_INFO.value, TableName.PAT_PARAM.value,
        TableName.MACRO_INFO.value, TableName.MACRO_PARAM.value,
        TableName.MACRO_PARAM_ENUM.value, TableName.MACRO_VERSION.value,
        TableName.MODEL_INFO.value, TableName.MODEL_EXECUTION.value,
        TableName.MODEL_MKT_MAP.value, TableName.MODEL_PAT_MAP.value,
        TableName.SCORE_META.value, TableName.MODEL_KERNEL.value
    ]
    SWAP_TABLES = [
        TableName.MKT_HISTORY_PRICE.value,
        TableName.MKT_INFO.value,
        TableName.DS_S_STOCK.value
    ]
    TABLE_SCHEMA_INFO = {
        TableName.MKT_HISTORY_PRICE.value: [
            MarketHistoryPriceField.MARKET_CODE.value,
            MarketHistoryPriceField.PRICE_DATE.value,
            MarketHistoryPriceField.OPEN_PRICE.value,
            MarketHistoryPriceField.HIGH_PRICE.value,
            MarketHistoryPriceField.LOW_PRICE.value,
            MarketHistoryPriceField.CLOSE_PRICE.value
        ],
        f'{TableName.MKT_HISTORY_PRICE.value}_SWAP': [
            MarketHistoryPriceField.MARKET_CODE.value,
            MarketHistoryPriceField.PRICE_DATE.value,
            MarketHistoryPriceField.OPEN_PRICE.value,
            MarketHistoryPriceField.HIGH_PRICE.value,
            MarketHistoryPriceField.LOW_PRICE.value,
            MarketHistoryPriceField.CLOSE_PRICE.value
        ],
        TableName.MKT_INFO.value: [
            MarketInfoField.MARKET_CODE.value,
            MarketInfoField.MARKET_NAME.value,
            MarketInfoField.MARKET_SOURCE_CODE.value,
            MarketInfoField.MARKET_SOURCE_TYPE.value
        ],
        f'{TableName.MKT_INFO.value}_SWAP': [
            MarketInfoField.MARKET_CODE.value,
            MarketInfoField.MARKET_NAME.value,
            MarketInfoField.MARKET_SOURCE_CODE.value,
            MarketInfoField.MARKET_SOURCE_TYPE.value
        ],
        TableName.DS_S_STOCK.value: [
            DSStockInfoField.STOCK_CODE.value,
            DSStockInfoField.ISIN_CODE.value,
            DSStockInfoField.COMPANY_CODE.value,
            DSStockInfoField.EXCHANGE_TYPE.value,
            DSStockInfoField.INDUSTRY_CODE.value,
            DSStockInfoField.TSE_INDUSTRY_CODE.value,
            DSStockInfoField.CUR_CODE.value,
            DSStockInfoField.TSE_IPO_DATE.value,
            DSStockInfoField.OTC_IPO_DATE.value,
            DSStockInfoField.REG_IPO_DATE.value,
            DSStockInfoField.DELISTING_DATE.value
        ],
        f'{TableName.DS_S_STOCK.value}_SWAP': [
            DSStockInfoField.STOCK_CODE.value,
            DSStockInfoField.ISIN_CODE.value,
            DSStockInfoField.COMPANY_CODE.value,
            DSStockInfoField.EXCHANGE_TYPE.value,
            DSStockInfoField.INDUSTRY_CODE.value,
            DSStockInfoField.TSE_INDUSTRY_CODE.value,
            DSStockInfoField.CUR_CODE.value,
            DSStockInfoField.TSE_IPO_DATE.value,
            DSStockInfoField.OTC_IPO_DATE.value,
            DSStockInfoField.REG_IPO_DATE.value,
            DSStockInfoField.DELISTING_DATE.value
        ],
        TableName.PAT_INFO.value: [
            PatternInfoField.PATTERN_ID.value,
            PatternInfoField.MACRO_ID.value,
            PatternInfoField.PATTERN_STATUS.value
        ],
        TableName.PAT_PARAM.value: [
            PatternParamField.PATTERN_ID.value,
            PatternParamField.MACRO_ID.value,
            PatternParamField.PARAM_CODE.value,
            PatternParamField.PARAM_VALUE.value
        ],
        TableName.MACRO_INFO.value: [
            MacroInfoField.MACRO_ID.value,
            MacroInfoField.MACRO_NAME.value,
            MacroInfoField.MACRO_DESC.value,
            MacroInfoField.FUNC_CODE.value
        ],
        TableName.MACRO_PARAM.value: [
            MacroParamField.MACRO_ID.value,
            MacroParamField.PARAM_CODE.value,
            MacroParamField.PARAM_NAME.value,
            MacroParamField.PARAM_DESC.value,
            MacroParamField.PARAM_TYPE.value,
            MacroParamField.PARAM_DEFAULT.value
        ],
        TableName.MACRO_PARAM_ENUM.value:[
            MacroParamEnumField.ENUM_CODE.value,
            MacroParamEnumField.ENUM_VALUE_CODE.value,
            MacroParamEnumField.ENUM_VALUE_NAME.value,
            'CREATE_BY',
            'CREATE_DT'
        ],
        TableName.MACRO_VERSION.value: [
            MacroVersionInfoField.MACRO_ID.value,
            MacroVersionInfoField.CODE_VERSION.value,
            MacroVersionInfoField.INFO_VERSION.value
        ],
        TableName.MODEL_INFO.value: [
            ModelInfoField.MODEL_ID.value,
            ModelInfoField.MODEL_STATUS.value,
            ModelInfoField.RETRAIN_CYCLE.value,
            ModelInfoField.TRAIN_START_DT.value
        ],
        TableName.MODEL_EXECUTION.value: [
            ModelExecutionField.MODEL_ID.value,
            ModelExecutionField.STATUS_CODE.value,
            ModelExecutionField.END_DT.value
        ],
        TableName.MODEL_MKT_MAP.value: [
            ModelMarketMapField.MARKET_CODE.value,
            ModelMarketMapField.MODEL_ID.value
        ],
        TableName.MODEL_PAT_MAP.value: [
            ModelPatternMapField.MODEL_ID.value,
            ModelPatternMapField.PATTERN_ID.value
        ],
        TableName.SCORE_META.value: [
            ScoreMetaField.SCORE_CODE.value,
            ScoreMetaField.SCORE_VALUE.value,
            ScoreMetaField.LOWER_BOUND.value,
            ScoreMetaField.UPPER_BOUND.value
        ],
        TableName.MODEL_KERNEL.value: [
            ModelKernelField.MODEL_ID.value,
            ModelKernelField.VALID_START_DT.value,
            ModelKernelField.VALID_END_DT.value
        ]
    }
    model_execution_stamp_queue = {}

    def _init_config(self, db_name: str, mode: str) -> Dict[str, str]:
        """
        load the configuration
        """
        current_path = os.path.split(os.path.realpath(__file__))[0]
        config = json.load(open('%s/config.json' %
                           (current_path)))[db_name][mode]
        return config

    def __init__(self,
                 mode: str = ExecMode.DEV.value,
                 batch_type: BatchType = BatchType.SERVICE_BATCH):
        self.config = self._init_config('mimosa', mode)
        use_swap = self.IS_SWAP_TYPE[batch_type]
        self.BASE_TABLES = [
            f'{table}_SWAP' if use_swap and table in self.SWAP_TABLES
            else table for table in self.BASE_TABLES
        ]
        os.makedirs(f'{DATA_LOC}', exist_ok=True)

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
        engine = create_engine('%s://%s:%s@%s:%s/%s?charset=%s' %
                               (engine_conf, user, password, ip_addr,
                                port, db_name, charset))
        return engine

    def _convert_exec_to_status(self, exec_info: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """將 EXECUTION RECORD 轉換為以完成狀態字典表示, 若 exec_info 長度為 0,
        則會回傳空字典 {}

        Parameters
        ----------
        exec_info: pd.DataFrame
            執行狀態直表

        Returns
        -------
        result: Dict[str, Dict[str, str]]
            完成狀態字典, 格式為 [model_id] - [status_code] - end_dt
        """
        FINISHED_STATUS = [
            ModelExecution.ADD_BACKTEST_FINISHED.value,
            ModelExecution.ADD_PREDICT_FINISHED.value,
            ModelExecution.BATCH_PREDICT_FINISHED.value,
            ModelExecution.TRAIN_MODEL_FINISHED.value
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
                records[ModelExecutionField.END_DT.value].values.astype('datetime64[s]'))
            # datetime64[s] 轉型為 str
            result[model_id][status_code] = str(max_date.tolist())
        return result

    def _clone_table(self, table_name: str, cols: List[str]):
        filename = table_name if table_name[-5:
                                            ] != '_SWAP' else table_name[:-5]
        logging.info(f'Clone {table_name} from db')
        engine = self._engine()
        sql = f"""
            SELECT
                {','.join(cols)}
            FROM
                {table_name}
        """
        data = pd.read_sql_query(sql, engine)
        os.makedirs(DATA_LOC, exist_ok=True)
        with open(f'{DATA_LOC}/{filename}.pkl', 'wb') as fp:
            pickle.dump(data, fp)
        logging.info(f'Clone {table_name} from db finished')

    def _clone_base_tables(self):
        for table in self.BASE_TABLES:
            self._clone_table(table, self.TABLE_SCHEMA_INFO[table])

    def get_data(self, pkl_name: str) -> Union[pd.DataFrame, List[PatternInfo], List[str]]:
        """根據指定的檔案名稱, 取得本地快取資料.
        這個方法只會取得本地檔案, 不會在檔案不存在時進行自動下載的動作,
        因此在檔案不存在時, 會拋出 RuntimeError

        Parameters
        ----------
        pkl_name: str
            本地快取的檔案名稱, 可通過 CacheName 進行取得

        Returns
        -------
        data: DataFrame | Dict[str, PatternInfo] | List[str]
            指定的檔案內容

        """
        path = f'{DATA_LOC}/{pkl_name}.pkl'
        if not os.path.exists(path):
            raise RuntimeError(f"table {pkl_name} not found, call clone first")
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        return data

    def get_market_data(self, market_id: str) -> pd.DataFrame:
        """根據傳入的 market ID, 取得對應的市場資料 DataFrame.
        市場資料格式必須包含四個欄位:
            OP, CP, HP, LP
        並且 Index 必須為 datetime.date

        Parameters
        ----------
        market_id: str
            要取得的市場 ID

        Returns
        -------
        result: pd.DataFrame
            對應傳入 ID 的市場資料

        """
        if not os.path.exists(f'{DATA_LOC}/markets/{market_id}.pkl'):
            return pd.DataFrame(columns=['OP', 'LP', 'HP', 'CP'])
        with open(f'{DATA_LOC}/markets/{market_id}.pkl', 'rb') as fp:
            result = pickle.load(fp)
        return result

    def _df_find(self, data: pd.DataFrame, cols: List[str], vals=List[Any], method: str = 'and') -> pd.DataFrame:
        data = data.copy()
        conds = []
        for i, col in enumerate(cols):
            conds.append(data[col].values == vals[i])
        conds = np.array(conds)
        if method == 'and':
            conds = np.all(conds, axis=0)
        elif method == 'or':
            conds = np.any(conds, axis=0)
        else:
            raise ValueError(f'Unknown method: {method}')
        data = data[conds]
        return data

    def _construct_pattern_info(self,
                                pattern_info: pd.DataFrame, pattern_param_info: pd.DataFrame,
                                macro_info: pd.DataFrame, macro_param_info: pd.DataFrame
                                ) -> List[PatternInfo]:
        results = []
        enum_types = set(MacroParaEnumManager.dump()[MacroParamEnumField.ENUM_CODE.value].values)
        for i in range(len(pattern_info)):
            pattern = pattern_info.iloc[i]
            pattern_id = pattern[PatternInfoField.PATTERN_ID.value]
            macro_id = pattern[PatternInfoField.MACRO_ID.value]
            pattern_params = self._df_find(
                pattern_param_info,
                cols=[PatternParamField.PATTERN_ID.value],
                vals=[pattern_id])
            macro = self._df_find(
                macro_info,
                cols=[MacroInfoField.MACRO_ID.value],
                vals=[macro_id])
            macro_params = self._df_find(
                macro_param_info,
                cols=[MacroParamField.MACRO_ID.value],
                vals=[macro_id])
            if len(pattern_params) == 0:
                raise RuntimeError(
                    f'loss {pattern_id} data in {TableName.PAT_PARAM.value}, check database')
            if len(macro) == 0:
                raise RuntimeError(
                    f'loss {macro_id} data in {TableName.MACRO_INFO.value}, check database')
            if len(macro_params) == 0:
                raise RuntimeError(
                    f'loss {macro_id} data in {TableName.MACRO_PARAM.value}, check database')

            func_code = macro.iloc[0][MacroInfoField.FUNC_CODE.value]
            params = {}
            for j in range(len(pattern_params)):
                pattern_param = pattern_params.iloc[j]
                param_code = pattern_param[PatternParamField.PARAM_CODE.value]
                param_val = pattern_param[PatternParamField.PARAM_VALUE.value]
                macro_param = self._df_find(
                    macro_params,
                    cols=[MacroParamField.PARAM_CODE.value],
                    vals=[param_code]
                )
                if len(macro_param) == 0:
                    raise RuntimeError(
                        f'loss {macro_id}, {param_code} data in {TableName.MACRO_PARAM.value}, check database')
                param_type = macro_param.iloc[0][MacroParamField.PARAM_TYPE.value]
                if param_type == "int":
                    param_val = int(param_val)
                elif param_type == "float":
                    param_val = float(param_val)
                elif param_type == 'string':
                    param_val = str(param_val)
                elif param_type in enum_types:
                    param_val = MacroParaEnumManager.get(param_type, param_val)
                else:
                    raise RuntimeError(
                        f'Unknown type {param_type} in {TableName.MACRO_PARAM.value}, check database')
                params[param_code] = param_val
            results.append(PatternInfo.make(pattern_id, func_code, params))
        return results

    def _build_each_market_data(self):
        data = self.get_data(CacheName.MKT_HISTORY_PRICE.value)
        if not os.path.exists(f'{DATA_LOC}/markets'):
            logging.info('Clone each market data from cache')
            os.makedirs(f'{DATA_LOC}/markets', exist_ok=True)
            result = pd.DataFrame(
                data[[
                    MarketHistoryPriceField.MARKET_CODE.value,
                    MarketHistoryPriceField.OPEN_PRICE.value,
                    MarketHistoryPriceField.HIGH_PRICE.value,
                    MarketHistoryPriceField.LOW_PRICE.value,
                    MarketHistoryPriceField.CLOSE_PRICE.value]].values,
                index=data[MarketHistoryPriceField.PRICE_DATE.value].values.astype(
                    'datetime64[D]'),
                columns=['MARKET_CODE', 'OP', 'HP', 'LP', 'CP']
            )
            market_groups = result.groupby('MARKET_CODE')
            for market_id, market_group in market_groups:
                mkt = market_group.sort_index()[['OP', 'HP', 'LP', 'CP']]
                mkt = pd.DataFrame(mkt.values.astype(
                    float), index=mkt.index, columns=mkt.columns)
                with open(f'{DATA_LOC}/markets/{market_id}.pkl', 'wb') as fp:
                    pickle.dump(mkt, fp)
            logging.info('Clone market data from cache finished')

    def _save_all_model_status(self, data: Dict[str, Dict[str, datetime.date]]):
        """與資料庫同步本地端模型預測結果執行狀態,
        沒有紀錄 -> 不會寫檔,
        有部分記錄 -> 沒有完成狀態的不會寫 key

        Parameters
        ----------
        data: Dict[str, Dict[str, date]]
            欲寫入本地端的各 Model 執行狀態與對應的執行時間

        Returns
        -------
        None.
        """
        logging.info("Save local model execution status")
        # 取得所有觀點資訊
        for model_id in data:
            self.set_model_status(model_id, data[model_id])
        logging.info("Save local model execution status finished")

    def _model_results_need_update(self, model_id, db_status: Dict[str, Dict[str, str]]) -> bool:
        need_update = False
        fp = f'{DATA_LOC}/views/{model_id}'
        status_fp = f'{fp}/status.pkl'
        # 判斷是否需要更新歷史預測結果
        if model_id not in db_status:
            return True
        if not os.path.exists(status_fp):
            need_update = True
        else:
            local_status = self.get_model_status(model_id)
            if not dict_equals(local_status, db_status[model_id]):
                need_update = True
        return need_update

    def _sync_model_results_and_status(self, controller:ThreadController):
        """取得所有模型歷史預測結果資料, 若檔案已存在且 clean_first 為 False
        , 則將會沿用舊資料, 不會進行下載

        Parameters
        ----------
        controller: ThreadController
            用於強制中斷 Thread

        Returns
        -------
        None.
        """
        logging.info('Cloning model predict result from db started')

        engine = self._engine()
        # 這邊犧牲效能, 換取較小的記憶體消耗量
        # 取得所有觀點ID
        model_info = self.get_data(CacheName.MODEL_INFO.value)
        model_ids = model_info[ModelInfoField.MODEL_ID.value].values.tolist()

        # 取得最新執行狀態
        sql = f"""
            SELECT
                *
            FROM
                {TableName.MODEL_EXECUTION.value}
        """
        model_exec_info = pd.read_sql_query(sql, engine)
        db_status = self._convert_exec_to_status(model_exec_info)

        for model_id_i, model_id in enumerate(model_ids):
            if not controller.isactive:
                logging.info('Cloning model predict result from db terminated')
                return
            fp = f'{DATA_LOC}/views/{model_id}'
            if not os.path.exists(fp):
                os.makedirs(f'{DATA_LOC}/views', exist_ok=True)
            if self._model_results_need_update(model_id, db_status):
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
                logging.info(
                    f'Cloning model result[{model_id_i+1}/{len(model_ids)}]: {model_id} finished')
        self._save_all_model_status(db_status)
        logging.info('Cloning model predict result from db finished')

    def sync_model_result_and_status(self, model_id: str):
        """取得所有模型歷史預測結果資料, 若檔案已存在且 clean_first 為 False
        , 則將會沿用舊資料, 不會進行下載

        Parameters
        ----------
        model_id: str
            要刪除的 Model ID

        Returns
        -------
        None.
        """
        engine = self._engine()
        # 這邊犧牲效能, 換取較小的記憶體消耗量
        # 取得所有觀點ID
        model_info = self.get_data(CacheName.MODEL_INFO.value)
        model_ids = model_info[ModelInfoField.MODEL_ID.value].values.tolist()

        # 取得最新執行狀態
        sql = f"""
            SELECT
                *
            FROM
                {TableName.MODEL_EXECUTION.value}
        """
        model_exec_info = pd.read_sql_query(sql, engine)
        db_status = self._convert_exec_to_status(model_exec_info)

        fp = f'{DATA_LOC}/views/{model_id}'
        if not os.path.exists(fp):
            os.makedirs(f'{DATA_LOC}/views', exist_ok=True)
        if model_id in model_ids:
            if self._model_results_need_update(model_id, db_status):
                logging.info(f'Clone model predict result from db: {model_id}')
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
                logging.info(f'Clone model predict result from db finished: {model_id}')
        logging.info(f'Sync model predict result status from db: {model_id}')
        self._save_all_model_status(db_status)
        logging.info(f'Sync model predict result status from db finished: {model_id}')

    def get_model_status(self, model_id: str) -> Dict[str, str]:
        """取得本地快取中指定 Model 的執行狀態與對應時間

        Parameters
        ----------
        model_id: str
            要取得的 Model ID

        Returns
        -------
        result: Dict[str, date]
            Model 的執行狀態與對應時間

        """
        fp = f'{DATA_LOC}/views/{model_id}'
        result = pickle_load(f"{fp}/status.pkl")
        return result

    def set_model_status(self, model_id: str, data: Dict[str, str]):
        """設定本地快取中指定 Model 的執行狀態與對應時間

        Parameters
        ----------
        model_id: str
            要取得的 Model ID
        data: Dict[str, date]
            要儲存的 Model 執行狀態與對應時間

        Returns
        -------
        None.
        """
        fp = f'{DATA_LOC}/views/{model_id}'
        if not os.path.exists(fp):
            os.makedirs(f'{fp}', exist_ok=True)
        pickle_dump(data, f"{fp}/status.pkl")

    def put_status_to_queue(self, model_id:str, status: Dict[str, str]):
        """將更新的狀態戰存至記憶體

        Parameters
        ----------
        model_id: str
            要暫存的 Model ID

        Returns
        -------
        None.
        """
        self.model_execution_stamp_queue[model_id] = status

    def get_status_from_queue(self, model_id:str) -> Dict[str, str]:
        """從快取暫存中取得 Model 狀態

        Parameters
        ----------
        model_id: str
            觀點 ID

        Returns
        -------
        result: Dict[str, str]
            觀點狀態結果
        """
        result = self.model_execution_stamp_queue[model_id]
        return result

    def clean_status_queue(self):
        """清空快取中 Model 的執行狀態

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self.model_execution_stamp_queue = {}

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
             - PredictResultField.MODEL_ID
             - PredictResultField.MARKET_ID
             - PredictResultField.DATE
             - PredictResultField.PERIOD
             - PredictResultField.PREDICT_VALUE
             - PredictResultField.UPPER_BOUND
             - PredictResultField.LOWER_BOUND
        """
        fp = f'{DATA_LOC}/views/{model_id}/history_values.pkl'
        if not os.path.exists(fp):
            return {}
        data = pickle_load(fp)
        result = {
            market_id: group
            for market_id, group in
            data.groupby(PredictResultField.MARKET_ID.value)
        }
        return result

    def set_model_results(self, model_id: str, data: pd.DataFrame):
        # 與本地端資料同步更新
        fp = f'{DATA_LOC}/views/{model_id}/history_values.pkl'
        history_data = pd.DataFrame(columns=data.columns)
        if os.path.exists(fp):
            history_data = pickle_load(fp)
        history_data = pd.concat([history_data, data], axis=0)
        history_data = history_data.drop_duplicates(subset=[
            PredictResultField.MODEL_ID.value,
            PredictResultField.MARKET_ID.value,
            PredictResultField.PERIOD.value,
            PredictResultField.DATE.value
        ])
        pickle_dump(history_data, fp)

    def del_model_result(self, model_id: str):
        fp = f'{DATA_LOC}/views/{model_id}'
        if os.path.exists(fp):
            shutil.rmtree(fp)

    def clone(self):
        self._clone_base_tables()
        self._build_each_market_data()

    def clean(self):
        for table in CacheName:
            if os.path.exists(f'{DATA_LOC}/{table.value}.pkl'):
                os.remove(f'{DATA_LOC}/{table.value}.pkl')
        if os.path.exists(f'{DATA_LOC}/markets'):
            shutil.rmtree(f'{DATA_LOC}/markets')

    def refresh(self, tables: Optional[List[str]] = None):
        """更新本地快取資料

        Parameters
        ----------
        tables: List[str] | None
            指定本地端要刷新的 Table 名稱, 若為空, 則刷新全部的基本資料表

        Returns
        -------
        None.
        """
        if tables is None:
            tables = self.BASE_TABLES
        for table in tables:
            self._clone_table(table, self.TABLE_SCHEMA_INFO[table])


class MimosaDB:
    """
    用來取得資料庫資料的物件，需在同資料夾下設置 config

    """
    config = {}
    MODE = ExecMode.DEV.value
    CREATE_BY = 'SYS_BATCH'
    MODIFY_BY = 'SYS_BATCH'
    READ_ONLY = False
    WRITE_LOCAL = False
    WRITE_LOCAL_FP = f'{DATA_LOC}/local_out'
    SAVE_BATCH_SIZE = 10000
    cache_manager = MimosaDBCacheManager()

    def _init_config(self, db_name: str, mode: str):
        """
        load the configuration
        """
        current_path = os.path.split(os.path.realpath(__file__))[0]
        config = json.load(open('%s/config.json' %
                           (current_path)))[db_name][mode]
        return config

    def __init__(
        self, mode: str = ExecMode.DEV.value,
        read_only: bool = False, write_local: bool = False
    ):
        """
        根據傳入參數取得指定的資料庫設定
        """
        self.MODE = mode
        self.config = self._init_config('mimosa', mode)
        self.cache_manager = MimosaDBCacheManager(mode)
        self.READ_ONLY = read_only
        self.WRITE_LOCAL = write_local
        if self.WRITE_LOCAL:
            if os.path.exists(self.WRITE_LOCAL_FP):
                shutil.rmtree(f'{self.WRITE_LOCAL_FP}')
            os.makedirs(self.WRITE_LOCAL_FP, exist_ok=True)

    def _do_if_not_read_only(func: Callable) -> Callable:
        def wrap(self, *args, **kwargs):
            if not self.READ_ONLY:
                return func(self, *args, **kwargs)
        return wrap

    def _use_write_local(func: Callable, lock=Lock()) -> Callable:
        """使用 write local 裝飾詞時, 修飾的方法需要有
        data 與 table_name 兩個參數
        """
        def wrap(self, *args, **kwargs):
            if self.WRITE_LOCAL:
                now = datetime.datetime.now()
                table_name = args[0] if 'table_name' not in kwargs else kwargs['table_name']
                data = args[1] if 'data' not in kwargs else kwargs['data']
                fp = f'{self.WRITE_LOCAL_FP}/{table_name}.pkl'
                record = {}
                lock.acquire()
                if os.path.exists(fp):
                    record = pickle_load(fp)
                record[str(now)] = data
                pickle_dump(record, fp)
                lock.release()
            return func(self, *args, **kwargs)
        return wrap

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
        engine = create_engine('%s://%s:%s@%s:%s/%s?charset=%s' %
                               (engine_conf, user, password, ip_addr,
                                port, db_name, charset))
        return engine

    def _extend_basic_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        now = datetime.datetime.now()
        create_dt = now
        data['CREATE_BY'] = self.CREATE_BY
        data['CREATE_DT'] = create_dt
        return data


# 取得/刪除本地快取
    def clean_db_cache(self):
        """ 清除本地資料庫快取

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.cache_manager.clean()

    def clone_db_cache(self, batch_type: BatchType = BatchType.SERVICE_BATCH):
        """ 從資料庫載入快取

        Parameters
        ----------
        batch_type: BatchType
            指定批次的執行狀態為初始化還是服務呼叫

        Returns
        -------
        None.
        """
        self.cache_manager = MimosaDBCacheManager(self.MODE, batch_type)
        self.cache_manager.clone()

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
        t = CatchableTread(
            target=self.cache_manager._sync_model_results_and_status,
            args=(controller, ))
        t.start()
        return t

# 使用本地快取得資料
    def get_markets(self, market_type: str = None, category_code: str = None) -> List[str]:
        """取得指定分類下的市場 ID 清單

        Parameters
        ----------
        market_type:
            市場類型( 目前為 BLB 或 TEJ )
        category_code:
            市場分類( 電子股, 水泥股...等 )

        Returns
        -------
        data: List[str]
            市場 ID 清單

        """
        data = self.cache_manager.get_data(CacheName.MKT_INFO.value)
        if market_type is None:
            return data[MarketInfoField.MARKET_CODE.value].values.tolist()

        data = data[
            data[MarketInfoField.MARKET_SOURCE_TYPE.value].values == market_type]
        if category_code is None:
            return data[MarketInfoField.MARKET_CODE.value].values.tolist()

        cate_data = self.cache_manager.get_data(CacheName.DS_S_STOCK.value)
        cate_data = cate_data[
            cate_data[DSStockInfoField.TSE_INDUSTRY_CODE.value].values == category_code]

        data = data[[MarketInfoField.MARKET_CODE.value,
                     MarketInfoField.MARKET_SOURCE_CODE.value]].values.astype(str).tolist()
        cate_data = cate_data[DSStockInfoField.STOCK_CODE.value].values.astype(
            str).tolist()
        # 取得所有在指定市場分類下的市場清單
        data = [x[0] for x in data if x[1] in cate_data]
        return data

    def get_market_info(self) -> pd.DataFrame:
        """取得 MIMOSA 市場基本資訊

        Parameters
        ----------
        None.

        Returns
        -------
        ret: pd.DataFrame
            Index: 市場 CODE
            Columns: MARKET_SOURCE_TYPE, MARKE_SOURCE_CODE
        """
        recv = self.cache_manager.get_data(CacheName.MKT_INFO.value)
        ret = recv.set_index(MarketInfoField.MARKET_CODE.value
                             )[[MarketInfoField.MARKET_SOURCE_TYPE.value,
                                MarketInfoField.MARKET_SOURCE_CODE.value]]
        return ret

    def get_category_info(self) -> pd.DataFrame:
        """取得 TEJ 中的市場資訊

        Parameters
        ----------
        None.

        Returns
        -------
        ret: pd.DataFrame
            Index: 市場 CODE
            Columns: STOCK_CODE, TSE_INDUSTRY_CODE
        """
        recv = self.cache_manager.get_data(CacheName.DS_S_STOCK.value)
        ret = recv.set_index(DSStockInfoField.STOCK_CODE.value
                             )[DSStockInfoField.TSE_INDUSTRY_CODE.value]
        return ret

    def get_market_data(self, market_id: str, begin_date: Optional[datetime.date] = None) -> pd.DataFrame:
        """根據傳入的市場 ID 與起始日期, 取得指定的市場資料

        Parameters
        ----------
        market_id: str
            市場 ID
        begin_date: datetime.date | None
            市場資料的起始日期, 回傳資料會取得該日期至最新的市場資料, 若為 None,
            就會回傳全部的市場資料

        Returns
        -------
        result: pd.DataFrame
            包含欄位 'OP', 'CP', 'HP', 'LP', 並且 Index 為對應的資料日期

        """
        result = self.cache_manager.get_market_data(market_id)
        if begin_date is not None:
            result = result[result.index.values >= begin_date]
        return result

    def get_patterns(self) -> List[PatternInfo]:
        """取得 Pattern 清單

        Returns
        -------
        list of PatternInfo

        """
        pattern_info = self.cache_manager.get_data(TableName.PAT_INFO.value)
        pattern_param_info = self.cache_manager.get_data(TableName.PAT_PARAM.value)
        macro_info = self.cache_manager.get_data(TableName.MACRO_INFO.value)
        macro_param_info = self.cache_manager.get_data(TableName.MACRO_PARAM.value)

        pattern_info = self.cache_manager._df_find(pattern_info,
                                     cols=[PatternInfoField.PATTERN_STATUS.value,
                                           PatternInfoField.PATTERN_STATUS.value],
                                     vals=[(DBPatternStatus.PRIVATE_AND_VALID.value),
                                           DBPatternStatus.PUBLIC_AND_VALID.value],
                                     method='or')
        patterns = self.cache_manager._construct_pattern_info(
            pattern_info, pattern_param_info, macro_info, macro_param_info)
        return patterns

    def get_models(self) -> List[str]:
        """取得所有執行狀態已經紀錄為 ADD_PREDICT_FINISHED 的 Model ID

        Parameters
        ----------
        None.

        Returns
        -------
        data: List[str]
            所有執行狀態已標記為 ADD_PREDICT_FINISHED 的 Model ID 清單

        """
        model_info = self.cache_manager.get_data(CacheName.MODEL_INFO.value)
        model_info = self.cache_manager._df_find(
            model_info,
            cols=[ModelInfoField.MODEL_STATUS.value,
                  ModelInfoField.MODEL_STATUS.value],
            vals=[DBModelStatus.PRIVATE_AND_VALID.value,
                  DBModelStatus.PUBLIC_AND_VALID.value],
            method='or'
        )
        exec_info = self.cache_manager.get_data(CacheName.MODEL_EXECUTION.value)
        apf_exec_info = self.cache_manager._df_find(
            exec_info,
            cols=[ModelExecutionField.STATUS_CODE.value],
            vals=[ModelExecution.TRAIN_MODEL_FINISHED.value])

        model_ids = model_info[ModelInfoField.MODEL_ID.value]
        result = []
        for model_id in model_ids:
            if model_id in apf_exec_info[ModelExecutionField.MODEL_ID.value].values:
                result.append(model_id)
        return result

    def get_score_meta_info(self) -> List[Tuple[int, float, float]]:
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
        data = self.cache_manager.get_data(CacheName.SCORE_META.value)
        result = []
        for i in range(len(data)):
            record = data.iloc[i]
            score = record[ScoreMetaField.SCORE_VALUE.value]
            upper_bound = record[ScoreMetaField.UPPER_BOUND.value]
            lower_bound = record[ScoreMetaField.LOWER_BOUND.value]
            result.append((score, upper_bound, lower_bound))
        return result

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
        self.cache_manager.sync_model_result_and_status(model_id)
        data = self.cache_manager.get_model_results(model_id)
        return data

    def get_latest_dates(self, model_id: str) -> datetime.date:
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

    def get_earliest_dates(self, model_id: str) -> Dict[str, datetime.date]:
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

    def get_removed_model(self) -> List[str]:
        """ 取得資料庫中狀態被標記為已移除的 Model ID

        Parameters
        ----------
        None.

        Returns
        -------
        model_ids: List[str]
            資料庫中全部已被標記已移除的 Model ID
        """
        model_info = self.cache_manager.get_data(CacheName.MODEL_INFO.value)
        model_info = self.cache_manager._df_find(
            model_info,
            cols=[ModelInfoField.MODEL_STATUS.value],
            vals=[DBModelStatus.REMOVED.value],
            method='or'
        )
        model_ids = model_info[ModelInfoField.MODEL_ID.value].values.tolist()
        return model_ids

    def get_view_kernel(self) -> Dict[str, Tuple[datetime.datetime, datetime.datetime]]:
        """取得觀點訓練結果使用資訊

        Returns
        -------
        result: Dict[str, Tuple[datetime.datetime, datetime.datetime]]
            使用觀點ID取得可用起始日與可用終止日, 結構為:觀點ID-(可用起始日, 可用終止日)
        """
        view_kernel = self.cache_manager.get_data(TableName.MODEL_KERNEL.value)
        result = {}
        for i in range(len(view_kernel)):
            data = view_kernel.iloc[i]
            model_id = data[ModelKernelField.MODEL_ID.value]
            start_dt = data[ModelKernelField.VALID_START_DT.value]
            end_dt = data[ModelKernelField.VALID_END_DT.value]
            result[model_id] = (start_dt, end_dt)
        return result

# 當下由資料庫取得資料
    def get_macro_param_type(self, function_code: str) -> Dict[str, str]:
        """ 從資料庫取得當下 Macro 中的 function 參數型態資訊

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

    def get_recover_model_execution(self) -> List[Tuple[str, ModelExecution]]:
        """ 取得模型最新執行狀態
        取得新增模型預測與新增模型回測的最新更新狀態，這兩個狀態對於任一模型而言
        都必須且應該只會各有一筆
         - 當新增模型狀態沒有找到且新增模型完成狀態也沒有找到時，狀態為需要新增預測
         - 當新增模型狀態找到了但沒有結束時間時，狀態為需要新增預測
         - 當新增模型完成找到但新增回測且新增回測完成狀態沒有沒找到時，狀態為需要新增回測
         - 當新增回測狀態找到但沒有結束時間時，狀態為需要新增回測

        Parameters
        ----------
        None.

        Returns
        -------
        exec_info: List[Tuple[str, ModelExecution]]
            每個 Model ID 與對應的執行狀態, 格式為
                [(model_id, ModelExecution), ...]

        """
        engine = self._engine()
        sql = f"""
            SELECT
                model.{ModelInfoField.MODEL_ID.value},
                model.{ModelInfoField.MODEL_STATUS.value},
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
            WHERE
                model.{ModelInfoField.MODEL_STATUS.value}='{DBModelStatus.PRIVATE_AND_VALID.value}' OR
                model.{ModelInfoField.MODEL_STATUS.value}='{DBModelStatus.PUBLIC_AND_VALID.value}'
        """
        data = pd.read_sql_query(sql, engine)
        group_data = data.groupby(ModelInfoField.MODEL_ID.value)
        results = []
        for model_id, model_state_info in group_data:
            model_add_predict_info = model_state_info[
                model_state_info[ModelExecutionField.STATUS_CODE.value].values ==
                ModelExecution.ADD_PREDICT.value]
            model_add_predict_finished_info = model_state_info[
                model_state_info[ModelExecutionField.STATUS_CODE.value].values ==
                ModelExecution.ADD_PREDICT_FINISHED.value]
            model_add_backtest_info = model_state_info[
                model_state_info[ModelExecutionField.STATUS_CODE.value].values ==
                ModelExecution.ADD_BACKTEST.value]
            model_add_backtest_finished_info = model_state_info[
                model_state_info[ModelExecutionField.STATUS_CODE.value].values ==
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

    def get_pattern_info(self, pattern_id: str) -> PatternInfo:
        """取得指定現象 ID 的現象資訊

        Parameters
        ----------
        pattern_id: str
            要取得的現象 ID

        Returns
        -------
        result: PatternInfo
            現象計算所需的參數資訊
        """
        self.cache_manager.refresh([
            TableName.PAT_INFO.value, TableName.PAT_PARAM.value
        ])
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
            WHERE
                ptn.{PatternInfoField.PATTERN_STATUS.value}={DBPatternStatus.PRIVATE_AND_VALID.value} OR
                ptn.{PatternInfoField.PATTERN_STATUS.value}={DBPatternStatus.PUBLIC_AND_VALID.value}
            ORDER BY
                ptn.{PatternInfoField.PATTERN_ID.value},
                mcr.{MacroInfoField.FUNC_CODE.value} ASC;
        """
        data = pd.read_sql_query(sql, engine)
        result = {}
        enum_types = set(MacroParaEnumManager.dump()[MacroParamEnumField.ENUM_CODE.value].values)
        for i in range(len(data)):
            record = data.iloc[i]
            pid = record[PatternInfoField.PATTERN_ID.value]
            if pid in result:
                continue
            func = record[MacroInfoField.FUNC_CODE.value]
            params = {}
            ptn_record = data[data[PatternInfoField.PATTERN_ID.value].values == pid]
            for j in range(len(ptn_record)):
                param_record = ptn_record.iloc[j]
                param_type = param_record[MacroParamField.PARAM_TYPE.value]
                param_val = param_record[PatternParamField.PARAM_VALUE.value]
                if param_type == "int":
                    param_val = int(
                        param_record[PatternParamField.PARAM_VALUE.value])
                elif param_type == "float":
                    param_val = float(
                        param_record[PatternParamField.PARAM_VALUE.value])
                elif param_type == "string":
                    param_val = str(
                        param_record[PatternParamField.PARAM_VALUE.value])
                elif param_type in enum_types:
                    param_val = MacroParaEnumManager.get(
                        param_type,
                        param_record[PatternParamField.PARAM_VALUE.value])
                else:
                    raise RuntimeError(
                        f'Unknown type {param_type} in {TableName.MACRO_PARAM.value}, check database')
                params[param_record[PatternParamField.PARAM_CODE.value]] = param_val
            result[pid] = PatternInfo.make(pid, func, params)
        if pattern_id not in result:
            raise Exception(
                f'get_pattern_info: pattern info not found in db: {pattern_id}')
        result = result[pattern_id]
        logging.info(f'Get pattern info {pattern_id} from db finished')
        return result

    def get_model_info(self, model_id: str) -> ModelInfo:
        """根據指定的 Model ID, 取得對應的 ModelInfo 物件,
        此動作會同步部分本地端快取資料

        Parameters
        ----------
        model_id: str
            觀點 ID

        Returns
        -------
        result: ModelInfo
            對應的 ModelInfo 物件

        """
        self.cache_manager.refresh([
            TableName.MODEL_INFO.value, TableName.MODEL_MKT_MAP.value,
            TableName.MODEL_PAT_MAP.value
        ])
        # 取得觀點的訓練資訊
        model_info = self.cache_manager.get_data(CacheName.MODEL_INFO.value)
        m_cond = model_info[ModelInfoField.MODEL_ID.value].values == model_id
        if len(model_info[m_cond]) == 0:
            # 若發生取不到資料的情況
            raise Exception(f"get_model_info: model not found: {model_id}")

        train_begin = model_info[m_cond].iloc[0][ModelInfoField.TRAIN_START_DT.value]
        # (train_begin == train_begin) 用於判斷是否為 nan 或 nat
        if train_begin is not None and (train_begin == train_begin):
            train_begin = train_begin.date()
        else:
            train_begin = None

        train_gap = model_info[m_cond].iloc[0][ModelInfoField.RETRAIN_CYCLE.value]
        # (train_gap == train_gap) 用於判斷是否為 nan 或 nat
        if train_gap is not None and (train_gap == train_gap):
            train_gap = int(train_gap)
        else:
            train_gap = None

        # 取得觀點的所有標的市場
        markets = self.cache_manager.get_data(CacheName.MODEL_MKT_MAP.value)
        m_cond = markets[ModelMarketMapField.MODEL_ID.value].values == model_id
        markets = markets[m_cond][ModelMarketMapField.MARKET_CODE.value].values.tolist(
        )

        # 取得觀點所有使用的現象 ID
        patterns = self.cache_manager.get_data(CacheName.MODEL_PAT_MAP.value)
        m_cond = patterns[ModelPatternMapField.MODEL_ID.value].values == model_id
        if len(patterns[m_cond]) == 0:
            # 若觀點下沒有任何現象, 則回傳例外
            raise Exception(
                f"get_model_info: 0 model pattern exception: {model_id}")
        patterns = patterns[m_cond][ModelPatternMapField.PATTERN_ID.value].values.tolist(
        )

        result = ModelInfo.make(
            model_id, patterns, markets, train_begin, train_gap)
        return result

    def get_macro_info(self) -> pd.DataFrame:
        """取得資料庫中的 Macro 資訊

        Parameters
        ----------
        None.

        Returns
        -------
        macro_info: pd.DataFrame
            Macro 資訊 DataFrame
        """
        self.cache_manager.refresh([
            TableName.MACRO_INFO.value, TableName.MACRO_PARAM.value,
            TableName.MACRO_VERSION.value
        ])
        macro_info = self.cache_manager.get_data(CacheName.MACRO_INFO.value)
        return macro_info

    def get_macro_param_info(self) -> pd.DataFrame:
        """取得資料庫中的 Macro 參數資訊

        Parameters
        ----------
        None.

        Returns
        -------
        macro_param_info: pd.DataFrame
            Macro 參數資訊的 DataFrame
        """
        self.cache_manager.refresh([TableName.MACRO_PARAM.value])
        macro_param_info = self.cache_manager.get_data(CacheName.MACRO_PARAM.value)
        return macro_param_info

    def get_macro_version_info(self) -> pd.DataFrame:
        """取得資料庫中的 Macro 版本資訊

        Parameters
        ----------
        None.

        Returns
        -------
        macro_version: pd.DataFrame
            Macro 版本資訊 DataFrame
        """
        self.cache_manager.refresh([TableName.MACRO_VERSION.value])
        macro_version = self.cache_manager.get_data(CacheName.MACRO_VERSION.value)
        return macro_version

    def get_macro_param_enum(self) -> pd.DataFrame:
        """取得當前資料庫中的 Macro 參數名目資料

        Parameters
        ----------
        None.

        Returns
        -------
        macro_param_enum: pd.DataFrame
            Macro 參數的名目資料資訊
        """
        self.cache_manager.refresh([TableName.MACRO_PARAM_ENUM.value])
        macro_param_enum = self.cache_manager.get_data(CacheName.MACRO_PARAM_ENUM.value)
        return macro_param_enum

# 寫入資料進資料庫
    @_use_write_local
    @_do_if_not_read_only
    def _save_df_to_db(self, table_name:str, data: pd.DataFrame):
        """將 Dataframe 寫入資料庫中

        Parameters
        ----------
        data: pd.DataFrame
            要儲存至資料庫中的資料
        table_name: str
            要儲存資料的資料表名稱

        Returns
        -------
        None.

        """
        engine = self._engine()
        try:
            data.to_sql(
            table_name,
            engine,
            if_exists='append',
            chunksize=self.SAVE_BATCH_SIZE,
            method='multi',
            index=False)
        except IntegrityError as e:
            logging.error(str(e))

    @_do_if_not_read_only
    def _execute_sql(self, sql: str):
        """執行傳入的 sql 語句

        Parameters
        ----------
        sql: str
            要執行的 SQL 語句

        Returns
        -------
        None.
        """
        engine = self._engine()
        try:
            engine.execute(sql)
        except IntegrityError as e:
            logging.error(str(e))

    def _insert(self, table_name: str, data: pd.DataFrame):
        """將傳入的資料寫入資料庫

        Parameters
        ----------
        table_name: str
            要存入的資料表名稱
        data: pd.DataFrame
            要存入的資料

        Returns
        -------
        None.
        """
        self._save_df_to_db(table_name, data)

    @_use_write_local
    def _insert_by_sql(self, table_name: str, data: pd.DataFrame,
                       col_info: List[Tuple[str, DataType]]):
        """根據傳入的資料資訊, 建構出對應的 sql 儲存語法並執行

        Parameters
        ----------
        table_name: str
            要儲存的目標資料表名稱
        data: pd.DataFrame
            要儲存的目標資料
        col_info: List[Tuple[str, DataType]]
            儲存資料表的各欄位名稱與對應資料型態

        Returns
        -------
        None.
        """
        logging.info(f'Save {table_name}')
        cols = [x[0] for x in col_info]
        sql_template = f"""
            INSERT INTO
                {table_name}
            (
                {','.join(cols)}
            )
            VALUES
        """
        for col, col_type in col_info:
            data[col] = data[col].astype(col_type.value)
        vals = [
            str(
                tuple(x)
            ) for x in data[cols].values
        ]
        batch_idxs = [i for i in range(
            0, len(vals)+self.SAVE_BATCH_SIZE, self.SAVE_BATCH_SIZE)]
        logging.info(f'{table_name} len: {len(vals)}')
        for idx_i, idx in enumerate(batch_idxs):
            if idx_i == 0:
                continue
            sql = sql_template + ','.join(vals[batch_idxs[idx_i-1]:idx])
            logging.info(f'Save {table_name}: #{batch_idxs[idx_i-1]} ~ #{idx}')
            self._execute_sql(sql)
            logging.info(
                f'Save {table_name}: #{batch_idxs[idx_i-1]} ~ #{idx} finished')
        logging.info(f'{table_name} len: {len(vals)} finished')

    def _insert_and_delete(self,
                           table_name: str, data: pd.DataFrame,
                           whereby: List[Tuple[str, Union[str, int, float]]]):
        """根據指定欄位刪除資料後, 再將傳入的資料寫入資料庫

        Parameters
        ----------
        table_name: str
            要存入的資料表名稱
        data: pd.DataFrame
            要存入的資料
        whereby: List[Tuple[str, Union[str, int, float]]])
            指定要刪除資料的欄位, 指定後會先刪除

        Returns
        -------
        None.
        """
        whereby = [
            f"{col}={val}" if (isinstance(val, int) or isinstance(val, float))
            else f"{col}='{val}'" for col, val in whereby]
        sql = f"""
            DELETE FROM {table_name}
            WHERE
                {' AND '.join(whereby)}
        """ if len(whereby) != 0 else f"""
            DELETE FROM {table_name}
        """
        self._execute_sql(sql)
        self._save_df_to_db(table_name, data)

    def _insert_and_ignore_exist(self, table_name: str, data: pd.DataFrame,
                                 pk_info: List[Tuple[str, DataType]],
                                 whereby: List[Tuple[str, Union[str, int, float]]]):
        """將傳入的資料寫入資料庫, 並且會避免寫入重複的資料

        Parameters
        ----------
        table_name: str
            要存入的資料表名稱
        data: pd.DataFrame
            要存入的資料
        pk_info: List[Tuple[str, str]]
            要避免重複的資料欄位與該欄位的型態
        whereby: List[Tuple[str, Union[str, int, float]]])
            指定要刪除資料的欄位, 指定後會先刪除

        Returns
        -------
        None.
        """
        engine = self._engine()
        pks = [x[0] for x in pk_info]
        whereby = [
            f"{col}={val}" if (isinstance(val, int) or isinstance(val, float))
            else f"{col}='{val}'" for col, val in whereby]
        # 取得當前資料庫中的資料
        db_data = None
        sql = f"""
            SELECT
                {','.join(pks)}
            FROM
                {table_name}
            WHERE
                {' AND '.join(whereby)}
        """
        db_data = pd.read_sql_query(sql, engine)

        # 移除完全重複的資料
        union_data = pd.concat([data, db_data, db_data], axis=0)
        for pk in pk_info:
            pk_col, pk_type = pk
            union_data[pk_col] = union_data[pk_col].astype(pk_type.value)
        union_data = union_data.drop_duplicates(subset=pks)

        self._save_df_to_db(table_name, union_data)

    def _insert_and_update_exist(self, table_name: str, data: pd.DataFrame,
                               pk_info: List[Tuple[str, DataType]],
                               cols: List[str],
                               whereby: List[Tuple[str, Union[str, int, float]]]):
        """將傳入的資料合併資料庫中的資料後一併寫入, 合併時當發生 pk 重複的狀況,
        會將重複的部分以只寫入傳入的資料進行處理

        Parameters
        ----------
        table_name: str
            要存入的資料表名稱
        data: pd.DataFrame
            要存入的資料
        pk_info: List[Tuple[str, str]]
            要避免重複的資料欄位與該欄位的型態
        cols: List[str]
            儲存時的欄位順序
        whereby: List[Tuple[str, Union[str, int, float]]])
            指定要刪除資料的欄位, 指定後會先刪除

        Returns
        -------
        None.
        """
        engine = self._engine()
        pks = [x[0] for x in pk_info]
        whereby = [
            f"{col}={val}" if (isinstance(val, int) or isinstance(val, float))
            else f"{col}='{val}'" for col, val in whereby]
        # 取得當前資料庫中的資料
        db_data = None
        sql = f"""
            SELECT
                *
            FROM
                {table_name}
            WHERE
                {' AND '.join(whereby)}
        """ if len(whereby) != 0 else f"""
            SELECT
                *
            FROM
                {table_name}
        """
        db_data = pd.read_sql_query(sql, engine)

        # 合併資料庫的資料
        union_data = pd.concat([data[cols], db_data[cols]], axis=0)
        for pk_col, pk_type in pk_info:
            union_data[pk_col] = union_data[pk_col].values.astype(
                pk_type.value)
        union_data = union_data.drop_duplicates(subset=pks, keep='first')

        sql = f"""
            DELETE FROM {table_name}
            WHERE
                {' AND '.join(whereby)}
        """
        self._execute_sql(sql)

        self._save_df_to_db(table_name, union_data)

    def save_model_latest_results(self, model_id: str, data: pd.DataFrame,
                                  exec_type: ModelExecution):
        """儲存最新的 Model 預測結果, 在儲存時會判定傳入的資料必定為最新的預測結果

        Parameters
        ----------
        model_id: str
            Model ID
        data: pd.DataFrame
            歷史預測結果資料的 DataFrame, 其中欄位須包含
             - PredictResultField.MODEL_ID
             - PredictResultField.MARKET_ID
             - PredictResultField.DATE
             - PredictResultField.PERIOD
             - PredictResultField.PREDICT_VALUE
             - PredictResultField.UPPER_BOUND
             - PredictResultField.LOWER_BOUND
        exec_type: ModelExecution
            模型正在執行的狀態, 允許的內容為
             - ModelExecution.ADD_PREDICT
             - ModelExecution.ADD_BACKTEST
             - ModelExecution.BATCH_PREDICT

        Returns
        -------
        None.

        """
        logging.info(f'start saving model latest results: {model_id}')
        table_name = f'{TableName.PREDICT_RESULT.value}_SWAP'
        if exec_type == ModelExecution.ADD_PREDICT:
            table_name = TableName.PREDICT_RESULT.value
        elif exec_type == ModelExecution.BATCH_PREDICT:
            table_name = f'{TableName.PREDICT_RESULT.value}_SWAP'
        else:
            logging.error(f'Unknown execution type: {exec_type}')
            return -1

        # 製作儲存結構
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

        # 取得歷史最新預測結果資料
        history_data = list(self.cache_manager.get_model_results(model_id).values())
        # 與最新預測結果合併
        sort_cols = [
            PredictResultField.MODEL_ID.value,
            PredictResultField.MARKET_ID.value,
            PredictResultField.DATE.value,
            PredictResultField.PERIOD.value,
            PredictResultField.PREDICT_VALUE.value,
            PredictResultField.UPPER_BOUND.value,
            PredictResultField.LOWER_BOUND.value
        ]
        history_data = [x[sort_cols] for x in history_data]
        data = pd.concat(history_data+[data[sort_cols]], axis=0)

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
        if len(latest_data) > 0:
            latest_data = pd.concat(latest_data, axis=0)
            latest_data = self._extend_basic_cols(latest_data)

            # 開始儲存
            self._insert_and_update_exist(table_name, latest_data,
                                    pk_info=[
                                        (PredictResultField.MODEL_ID.value,
                                         DataType.STRING),
                                        (PredictResultField.MARKET_ID.value,
                                         DataType.STRING),
                                        (PredictResultField.PERIOD.value,
                                         DataType.INT)
                                    ],
                                    cols=latest_data.columns.values.tolist(),
                                    whereby=[(PredictResultField.MODEL_ID.value, model_id)])
        logging.info(f'Saving model latest results finished: {model_id}')

    def save_model_results(self, model_id: str, data: pd.DataFrame,
                           exec_type: ModelExecution = ModelExecution.ADD_BACKTEST):
        """Save modle predicting results to DB.

        Parameters
        ----------
        model_id: str
            Model ID
        data: pd.DataFrame
            歷史預測結果資料的 DataFrame, 其中欄位須包含
             - PredictResultField.MODEL_ID
             - PredictResultField.MARKET_ID
             - PredictResultField.DATE
             - PredictResultField.PERIOD
             - PredictResultField.PREDICT_VALUE
             - PredictResultField.UPPER_BOUND
             - PredictResultField.LOWER_BOUND
        exec_type: ModelExecution
            模型正在執行的狀態, 允許的內容為
             - ModelExecution.ADD_PREDICT
             - ModelExecution.ADD_BACKTEST
             - ModelExecution.BATCH_PREDICT

        """
        if (exec_type == ModelExecution.ADD_PREDICT or
                exec_type == ModelExecution.BATCH_PREDICT):
            self.save_model_latest_results(model_id, data.copy(), exec_type)
        logging.info(f'Saving model history results: {model_id} started')
        if (exec_type == ModelExecution.ADD_PREDICT or
                exec_type == ModelExecution.ADD_BACKTEST):
            table_name = f'{TableName.PREDICT_RESULT_HISTORY.value}'
        elif exec_type == ModelExecution.BATCH_PREDICT:
            table_name = f'{TableName.PREDICT_RESULT_HISTORY.value}_SWAP'
        else:
            logging.error(f'Unknown execution type: {exec_type}')
            return -1

        # 製作儲存結構
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
        data = self._extend_basic_cols(data)

        # 新增歷史預測結果
        self._insert(table_name, data)
        # 與本地端資料同步更新
        self.cache_manager.set_model_results(model_id, data)
        logging.info(f'Model results history finished: {model_id} finished')

    def save_latest_pattern_results(self, data: pd.DataFrame):
        """儲存最新 Pattern 計算結果進資料庫

        Parameters
        ----------
        data: pd.DataFrame
            要儲存的資料 DataFrame, 欄位包含
             - PatternResultField.PATTERN_ID
             - PatternResultField.MARKET_ID
             - PatternResultField.DATE
             - PatternResultField.VALUE

        Return
        ------
        None.

        """
        table_name = f'{TableName.PATTERN_RESULT.value}_SWAP'
        data = self._extend_basic_cols(data)

        logging.info(f'Save pattern event')
        self._insert_by_sql(table_name,
                            data=data,
                            col_info=[
                                ('CREATE_BY', DataType.STRING),
                                ('CREATE_DT', DataType.DATETIME),
                                (PatternResultField.PATTERN_ID.value, DataType.STRING),
                                (PatternResultField.MARKET_ID.value, DataType.STRING),
                                (PatternResultField.DATE.value, DataType.DATE),
                                (PatternResultField.VALUE.value, DataType.STRING),
                            ])
        logging.info(f'Save pattern event finished')

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
        table_name = f'{TableName.MKT_SCORE.value}_SWAP'
        data = self._extend_basic_cols(data)
        data_upper_bound = data[MarketScoreField.UPPER_BOUND.value].values * 100
        data_lower_bound = data[MarketScoreField.LOWER_BOUND.value].values * 100

        data = data[[
            'CREATE_BY', 'CREATE_DT', MarketScoreField.MARKET_SCORE.value,
            MarketScoreField.MARKET_ID.value, MarketScoreField.DATE_PERIOD.value]]
        data[MarketScoreField.UPPER_BOUND.value] = data_upper_bound
        data[MarketScoreField.LOWER_BOUND.value] = data_lower_bound

        logging.info('Save market score')
        self._insert(table_name, data)
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
        table_name = f'{TableName.MKT_PERIOD.value}_SWAP'
        data = self._extend_basic_cols(data)

        data_net_change_rate = data[MarketPeriodField.NET_CHANGE_RATE.value].values * 100

        data = data[[
            'CREATE_BY', 'CREATE_DT', MarketPeriodField.MARKET_ID.value,
            MarketPeriodField.DATE_PERIOD.value, MarketPeriodField.PRICE_DATE.value,
            MarketPeriodField.DATA_DATE.value, MarketPeriodField.NET_CHANGE.value]]
        data[MarketPeriodField.NET_CHANGE_RATE.value] = data_net_change_rate
        data[MarketPeriodField.PRICE_DATE.value] = data[MarketPeriodField.PRICE_DATE.value].astype(
            DataType.DATE.value)

        logging.info('Save market period')
        self._insert(table_name, data)
        logging.info('Save market period finished')

    def save_model_hit_sum(self, data: pd.DataFrame):
        """儲存模型預測結果準確率

        Parameters
        ----------
        data: pd.DataFrame
            模型預測結果準確率

        Returns
        -------
        None.
        """
        table_name = f'{TableName.MODEL_MKT_HIT_SUM.value}_SWAP'
        data = self._extend_basic_cols(data)

        logging.info(f'Insert into model hit sum swap')
        self._insert(table_name, data)
        logging.info(f'Insert into model hit sum swap finished')

    def save_macro_info(self, data: pd.DataFrame):
        """新增 Macro 資訊

        Parameters
        ----------
        data: pd.DataFrame
            要新增的 Macro 資訊

        Returns
        -------
        None.
        """
        data = self._extend_basic_cols(data)
        self._insert(table_name=TableName.MACRO_INFO.value, data=data)

    def save_macro_param_info(self, data: pd.DataFrame):
        """新增 Macro 參數資訊

        Parameters
        ----------
        data: pd.DataFrame
            要新增的 Macro 參數資訊

        Returns
        -------
        None.
        """
        data = self._extend_basic_cols(data)
        self._insert(table_name=TableName.MACRO_PARAM.value, data=data)

    def save_macro_version_info(self, data: pd.DataFrame):
        """新增 Macro 資訊

        Parameters
        ----------
        data: pd.DataFrame
            要新增的 Macro 資訊

        Returns
        -------
        None.
        """
        data = self._extend_basic_cols(data)
        self._insert(table_name=TableName.MACRO_VERSION.value, data=data)

    def save_macro_param_enum(self, data: pd.DataFrame):
        """儲存 Macro 參數名目資料, 儲存時會將重複 PK 的資料用傳入資料進行取代

        Parameters
        ----------
        data: pd.DataFrame
            要儲存的名目資料

        Returns
        -------
        None.
        """
        def get_new_enum(old, new):
            now_time = datetime.datetime.now()
            old = {f'{a}.{b}': old[idx: idx+1]
                for idx, (a, b) in enumerate(
                    zip(old[MacroParamEnumField.ENUM_CODE.value].values,
                        old[MacroParamEnumField.ENUM_VALUE_CODE.value].values))}
            new = {f'{a}.{b}': new[idx: idx+1]
                for idx, (a, b) in enumerate(
                    zip(new[MacroParamEnumField.ENUM_CODE.value].values,
                        new[MacroParamEnumField.ENUM_VALUE_CODE.value].values))}
            ret = []
            for key in old:
                if key in new:
                    if (new[key][MacroParamEnumField.ENUM_VALUE_NAME.value].values ==
                        old[key][MacroParamEnumField.ENUM_VALUE_NAME.value].values):
                        ret.append(old[key])
                    else:
                        new[key]['CREATE_BY'] = old[key]['CREATE_BY']
                        new[key]['CREATE_DT'] = old[key]['CREATE_DT']
                        new[key]['MODIFY_BY'] = self.MODIFY_BY
                        new[key]['MODIFY_DT'] = now_time
                        ret.append(new[key])
                    del new[key]
            for key in new:
                new[key]['CREATE_BY'] = self.CREATE_BY
                new[key]['CREATE_DT'] = now_time
            ret += list(new.values())
            return pd.concat(ret, axis=0)
        self.cache_manager.refresh(tables=[TableName.MACRO_PARAM_ENUM.value])
        db_enum = self.cache_manager.get_data(CacheName.MACRO_PARAM_ENUM.value)
        new_data = get_new_enum(db_enum, data)

        self._delete(TableName.MACRO_PARAM_ENUM.value, [])
        self._insert(TableName.MACRO_PARAM_ENUM.value, new_data)

    def update_macro_tag(self):
        """更新 Macro 對應 Tag ID 資料表

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        engine = self._engine()
        # 刪除舊版 Tag 資訊
        sql = f"""
            SELECT
                {MacroTagMapField.TAG_ID.value}
            FROM
                {TableName.MACRO_TAG_MAP.value}
        """
        data = pd.read_sql_query(sql, engine).drop_duplicates(keep='first')
        deprecated_tag_ids = data[MacroTagMapField.TAG_ID.value].values
        for tag_id in deprecated_tag_ids:
            self._delete(TableName.MACRO_TAG.value,
                whereby=[
                    (MacroTagField.TAG_ID.value, tag_id)
                ])
        # 刪除舊版 TAG 對應表資訊
        self._delete(TableName.MACRO_TAG_MAP.value, [])

        # 新增新版 Macro TAG 資訊
        macro_tag, macro_tag_map = self._get_macro_tag_info()
        self._insert(TableName.MACRO_TAG.value, macro_tag)
        self._insert(TableName.MACRO_TAG_MAP.value, macro_tag_map)

    def save_view_kernel(self, view_id: str, start_dt: datetime.date):
        """儲存觀點訓練結果資訊

        Parameters
        ----------
        view_id: str
            觀點ID
        start_dt: datetime.date
            觀點訓練結果可用起始日
        
        Returns
        -------
        None.
        """
        data = pd.DataFrame([[view_id, start_dt]], 
            columns=[ModelKernelField.MODEL_ID.value, 
            ModelKernelField.VALID_START_DT.value])
        data = self._extend_basic_cols(data)
        self._insert(table_name=TableName.MODEL_KERNEL.value, data=data)

    @_do_if_not_read_only
    def set_model_execution_start(self, model_id: str, exection: str) -> str:
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
        table_name = TableName.MODEL_EXECUTION.value
        exec_id = 'READ_ONLY_MODE'
        # 檢查是否為合法的 exection
        if exection not in ModelExecution:
            raise Exception(f"Unknown excetion code {exection}")

        # 資料庫中對於一個觀點僅會存在一筆 AP 或 AB
        # 若是要儲存 AP 或是 AB, 那麼資料庫中就不應該已存在 AP 或 AB
        # 因此要先清除原先的執行紀錄
        if exection in [
                ModelExecution.ADD_PREDICT.value,
                ModelExecution.ADD_BACKTEST.value]:
            self._delete(table_name,
                            whereby=[
                                (ModelExecutionField.MODEL_ID.value, model_id),
                                (ModelExecutionField.STATUS_CODE.value, exection)
                            ])

        # 取得 EXEC_ID
        exec_id = self._get_serial_no(SerialNoType.EXECUTION)

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

        logging.info(f"[DB Create] Set model execution start: {model_id} -> {exection}")
        self._insert(table_name, data)
        logging.info(f"[DB Create] Set model execution start finished: {model_id} -> {exection}")

        # 同步本地端快取資料
        local_status = self.cache_manager._convert_exec_to_status(data)[
            model_id]
        self.cache_manager.set_model_status(model_id, local_status)
        return exec_id

    def set_pattern_execution_start(self, pid: str, exection: str) -> str:
        """ create pattern exec

        Parameters
        ----------
        pid: str
            ID of pattern
        exection: str
            status code set to this pattern

        Returns
        -------
        exec_id: str
            ID of pattern execution status

        """
        table_name = TableName.PATTERN_EXECUTION.value
        # 檢查是否為合法的 execution
        if exection not in PatternExecution:
            raise Exception(f"Unknown excetion code {exection}")

        # 資料庫中對於一個現象僅會存在一筆 ADD_PATTERN
        # 若是要儲存 ADD_PATTERN, 那麼資料庫中就不應該已存在 ADD_PATTERN
        # 因此要先清除原先的執行紀錄
        if exection in [
                PatternExecution.ADD_PATTERN.value]:
            self._delete(table_name,
                         whereby=[
                             (PatternExecutionField.PATTERN_ID.value, pid),
                             (PatternExecutionField.STATUS_CODE.value, exection)
                         ])

        # 取得 EXEC_ID
        exec_id = self._get_serial_no(SerialNoType.EXECUTION)

        # 建立 status
        COLUMNS = [
            'CREATE_BY', 'CREATE_DT',
            PatternExecutionField.EXEC_ID.value,
            PatternExecutionField.PATTERN_ID.value,
            PatternExecutionField.STATUS_CODE.value,
            PatternExecutionField.START_DT.value,
            PatternExecutionField.END_DT.value
        ]
        now = datetime.datetime.now()
        now = np.datetime64(now).astype('datetime64[s]').tolist()
        create_by = self.CREATE_BY
        create_dt = now
        start_dt = now
        end_dt = None
        data = [[
            create_by, create_dt, exec_id,
            pid, exection, start_dt, end_dt]]
        data = pd.DataFrame(data, columns=COLUMNS)
        self._insert(table_name, data)
        return exec_id

# 更新資料庫中資料
    def _update(self, table_name: str,
                set_value: List[Tuple[str, Union[str, int, float]]],
                whereby: List[Tuple[str, Union[str, int, float]]]):
        set_str = []
        for col, val in set_value:
            if isinstance(val, int) or isinstance(val, float):
                sql = f"{col}={val}"
            else:
                new_val = val.replace("'", "''")
                sql = f"{col}='{new_val}'"
            set_str.append(sql)
        whereby = [
            f"{col}={val}" if (isinstance(val, int) or isinstance(val, float))
            else f"{col}='{val}'" for col, val in whereby]
        sql = f"""
        UPDATE
            {table_name}
        SET
            {','.join(set_str)}
        WHERE
            {' AND '.join(whereby)};
        """
        self._execute_sql(sql)

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
        table_name = TableName.PATTERN_RESULT.value
        data = self._extend_basic_cols(data)

        logging.info(f'Update pattern event')
        self._insert_and_delete(table_name, data,
                                whereby=[
                                    (PatternResultField.PATTERN_ID.value, pattern_id)
                                ])
        logging.info(f'Update pattern event end')

    def update_model_hit_sum(self, data: pd.DataFrame):
        """儲存模型預測結果準確率至當前準確率表

        Parameters
        ----------
        data: pd.DataFrame
            模型預測結果準確率

        Returns
        -------
        None.
        """
        table_name = TableName.MODEL_MKT_HIT_SUM.value
        data = self._extend_basic_cols(data)
        group_data = data.groupby([
            ModelMarketHitSumField.MODEL_ID.value])

        logging.info(f'Update model hit sum')
        for model_id, model_data in group_data:
            self._insert_and_update_exist(
                table_name, model_data,
                pk_info=[
                    (ModelMarketHitSumField.MODEL_ID.value, DataType.STRING),
                    (ModelMarketHitSumField.MARKET_CODE.value, DataType.STRING),
                    (ModelMarketHitSumField.DATE_PERIOD.value, DataType.INT)
                ],
                cols=model_data.columns.values.tolist(),
                whereby=[
                    (ModelMarketHitSumField.MODEL_ID.value, model_id)
                ]
            )
        logging.info(f'Update model hit sum finished')

    def set_pattern_execution_complete(self, exec_id: str):
        """Set pattern execution complete on DB.

        Parameters
        ----------
        exec_id: str
            ID of pattern_exec

        Returns
        -------
        None.

        """
        table_name = TableName.PATTERN_EXECUTION.value
        now = datetime.datetime.now()
        now = np.datetime64(now).astype('datetime64[s]').tolist()
        engine = self._engine()

        finished_status = {
            PatternExecution.BATCH_SERVICE.value: PatternExecution.BATCH_SERVICE_FINISHED.value,
            PatternExecution.ADD_PATTERN.value: PatternExecution.ADD_PATTERN_FINISHED.value,
        }

        sql = f"""
            SELECT
                {PatternExecutionField.PATTERN_ID.value},
                {PatternExecutionField.STATUS_CODE.value}
            FROM
                {TableName.PATTERN_EXECUTION.value}
            WHERE
                {PatternExecutionField.EXEC_ID.value}='{exec_id}';
        """
        exec_data = pd.read_sql_query(sql, engine).iloc[0]
        if len(exec_data) == 0:
            raise Exception(
                'call set_pattern_execution_complete before set_pattern_execution_start')

        status = finished_status[exec_data[PatternExecutionField.STATUS_CODE.value]]
        self._update(table_name,
                     set_value=[
                         (PatternExecutionField.END_DT.value, str(now)),
                         ('MODIFY_DT', str(now)),
                         ('MODIFY_BY', self.MODIFY_BY),
                         (PatternExecutionField.STATUS_CODE.value, status)
                     ],
                     whereby=[
                         (PatternExecutionField.EXEC_ID.value, exec_id)
                     ])

    def update_macro_info(self, data: pd.DataFrame):
        """更新 Macro 資訊

        Parameters
        ----------
        data: pd.DataFrame
            要進行儲存的 Macro 資訊

        Returns
        -------
        None.
        """
        now = datetime.datetime.now()
        now = np.datetime64(now).astype('datetime64[s]').tolist()

        for i in range(len(data)):
            row = data.iloc[i]
            macro_id = row[MacroInfoField.MACRO_ID.value]
            macro_name = row[MacroInfoField.MACRO_NAME.value]
            macro_desc = row[MacroInfoField.MACRO_DESC.value]
            macro_func = row[MacroInfoField.FUNC_CODE.value]
            whereby = [
                (MacroInfoField.MACRO_ID.value, macro_id)
            ]
            self._update(
                table_name=TableName.MACRO_INFO.value,
                set_value=[
                    (MacroInfoField.MACRO_NAME.value, macro_name),
                    (MacroInfoField.MACRO_DESC.value, macro_desc),
                    (MacroInfoField.FUNC_CODE.value, macro_func),
                    ('MODIFY_DT', str(now)),
                    ('MODIFY_BY', self.MODIFY_BY)
                ],
                whereby=whereby
            )

    def _gen_update_macro_param_info(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """產生所有有異動的參數 DataFrame
        """
        _para_columns = [MacroParamField.MACRO_ID.value,
                         MacroParamField.PARAM_CODE.value,
                         MacroParamField.PARAM_NAME.value,
                         MacroParamField.PARAM_DESC.value,
                         MacroParamField.PARAM_DEFAULT.value,
                         MacroParamField.PARAM_TYPE.value]
        _db_columns = ['CREATE_BY', 'CREATE_DT', 'MODIFY_BY', 'MODIFY_DT']

        def get_macro_para_table():
            engine = self._engine()
            sql = ("SELECT "
                   f"{', '.join([s for s in _db_columns + _para_columns])} "
                   f"FROM {TableName.MACRO_PARAM.value}")
            ret = {mid: pinfo for mid, pinfo in
                   pd.read_sql_query(sql, engine
                                     ).groupby(MacroParamField.MACRO_ID.value)}
            return ret

        def update_pinfo(new, old, now):
            ret = []
            old = {pid: pinfo for pid, pinfo in
                   old.groupby(MacroParamField.PARAM_CODE.value)}
            for pid, pinfo in new.groupby(MacroParamField.PARAM_CODE.value):
                if pid in old:
                    cur = old[pid]
                    if (pinfo[_para_columns].values != old[pid][_para_columns].values).any():
                        for each in _para_columns:
                            cur[each] = pinfo[each].values
                        cur['MODIFY_DT'] = now
                        cur['MODIFY_BY'] = self.CREATE_BY
                    ret.append(cur)
                else:
                    pinfo['CREATE_DT'] = now
                    pinfo['CREATE_BY'] = self.CREATE_BY
                    pinfo['MODIFY_DT'] = None
                    pinfo['MODIFY_BY'] = None
                    ret.append(pinfo)
            return pd.concat(ret, axis=0)

        now = datetime.datetime.now()
        now = np.datetime64(now).astype('datetime64[s]').tolist()
        pinfo_from_db = get_macro_para_table()
        ret = {}
        for mid, pinfo in data.groupby(MacroParamField.MACRO_ID.value):
            if mid not in pinfo_from_db:
                pinfo['CREATE_DT'] = now
                pinfo['CREATE_BY'] = self.CREATE_BY
                pinfo['MODIFY_DT'] = None
                pinfo['MODIFY_BY'] = None
                ret[mid] = pinfo
                continue
            ret[mid] = update_pinfo(pinfo, pinfo_from_db[mid], now)
        return ret

    def update_macro_param_info(self, data: pd.DataFrame):
        """更新 Macro 參數資訊

        Parameters
        ----------
        data: pd.DataFrame
            要進行儲存的 Macro 參數資訊

        Returns
        -------
        None.
        """
        update_data = self._gen_update_macro_param_info(data)
        for macro_id in update_data:
            remove_where = [
                (MacroParamField.MACRO_ID.value, macro_id)
            ]
            self._delete(TableName.MACRO_PARAM.value, whereby=remove_where)
        update_data = pd.concat(list(update_data.values()), axis=0)
        self._insert(TableName.MACRO_PARAM.value, update_data)

    def update_macro_version_info(self, data: pd.DataFrame):
        """更新 Macro 版本資訊

        Parameters
        ----------
        data: pd.DataFrame
            要進行儲存的 Macro 版本資訊

        Returns
        -------
        None.
        """
        now = datetime.datetime.now()
        now = np.datetime64(now).astype('datetime64[s]').tolist()

        for i in range(len(data)):
            row = data.iloc[i]
            macro_id = row[MacroVersionInfoField.MACRO_ID.value]
            code_version = row[MacroVersionInfoField.CODE_VERSION.value]
            info_version = row[MacroVersionInfoField.INFO_VERSION.value]
            whereby = [
                (MacroInfoField.MACRO_ID.value, macro_id)
            ]
            self._update(
                table_name=TableName.MACRO_VERSION.value,
                set_value=[
                    (MacroVersionInfoField.CODE_VERSION.value, code_version),
                    (MacroVersionInfoField.INFO_VERSION.value, info_version),
                    ('MODIFY_DT', str(now)),
                    ('MODIFY_BY', self.MODIFY_BY)
                ],
                whereby=whereby
            )

    def update_view_kernel(self, model_id: str, start_dt: datetime.date, end_dt: datetime.date):
        """更新觀點訓練結果資訊

        Parameters
        ----------
        model_id: str
            觀點ID
        start_dt: datetime.date
            觀點訓練結果可用起始日
        end_dt: datetime.date
            觀點訓練結果可用終止日
        
        Returns
        -------
        None.
        """
        now = datetime.datetime.now()
        now = np.datetime64(now).astype('datetime64[s]').tolist()

        whereby = [
            (ModelKernelField.MODEL_ID.value, model_id),
            (ModelKernelField.VALID_START_DT.value, 
                np.datetime64(start_dt).astype('datetime64[D]').tolist())
        ]
        self._update(
            table_name=TableName.MODEL_KERNEL.value,
            set_value=[
                (ModelKernelField.VALID_END_DT.value, 
                 str(np.datetime64(end_dt).astype('datetime64[D]').tolist())),
                ('MODIFY_DT', str(now)),
                ('MODIFY_BY', self.MODIFY_BY)
            ],
            whereby=whereby
        )

    @_do_if_not_read_only
    def stamp_model_execution(self, exec_ids: List[str]):
        """將所有模型執行狀態儲存至資料庫, 並同時更新本地端資料庫狀態

        Parameters
        ----------
        exec_ids: List[str]
            模型執行狀態 ID 清單

        Returns
        -------
        None.
        """
        table_name = TableName.MODEL_EXECUTION.value
        now = datetime.datetime.now()
        now = np.datetime64(now).astype('datetime64[s]').tolist()
        finished_status = {
            ModelExecution.ADD_BACKTEST.value: ModelExecution.ADD_BACKTEST_FINISHED.value,
            ModelExecution.ADD_PREDICT.value: ModelExecution.ADD_PREDICT_FINISHED.value,
            ModelExecution.BATCH_PREDICT.value: ModelExecution.BATCH_PREDICT_FINISHED.value,
            ModelExecution.TRAIN_MODEL_FINISHED.value: ModelExecution.TRAIN_MODEL_FINISHED.value
        }
        engine = self._engine()
        sql = f"""
            SELECT
                {ModelExecutionField.EXEC_ID.value},
                {ModelExecutionField.MODEL_ID.value},
                {ModelExecutionField.STATUS_CODE.value}
            FROM
                {TableName.MODEL_EXECUTION.value};
        """
        exec_data = pd.read_sql_query(sql, engine)

        logging.info("[DB Update] Stamp model execution started")
        for exec_id in exec_ids:
            # 取得資料庫模型ID 與執行狀態
            eid_cond = exec_data[ModelExecutionField.EXEC_ID.value].values == exec_id
            if len(exec_data[eid_cond]) == 0:
                raise Exception(
                    'call stamp_model_execution before set_model_execution_start')
            e_data = exec_data[eid_cond].iloc[0]
            model_id = e_data[ModelExecutionField.MODEL_ID.value]
            status = finished_status[e_data[ModelExecutionField.STATUS_CODE.value]]

            # 取得本地端結束時間紀錄
            local_status = self.cache_manager.get_status_from_queue(model_id)
            end_dt = local_status[status]
            self._update(table_name,
                         set_value=[
                             (ModelExecutionField.END_DT.value, end_dt),
                             ('MODIFY_DT', str(now)),
                             ('MODIFY_BY', self.MODIFY_BY),
                             (ModelExecutionField.STATUS_CODE.value, status)
                         ],
                         whereby=[
                             (ModelExecutionField.EXEC_ID.value, exec_id)
                         ]
                         )
            self.cache_manager.set_model_status(model_id, local_status)
        self.cache_manager.clean_status_queue()
        logging.info("[DB Update] Stamp model execution finished")

# 刪除資料庫資料
    def _delete(self, table_name: str, whereby: List[Tuple[str, Union[str, int, float]]]):
        whereby = [
            f"{col}={val}" if (isinstance(val, int) or isinstance(val, float))
            else f"{col}='{val}'" for col, val in whereby]
        sql = f"""
            DELETE FROM {table_name}
            WHERE
                {' AND '.join(whereby)}
        """ if len(whereby) > 0 else f"""
            DELETE FROM {table_name}
        """
        self._execute_sql(sql)

    def del_model_execution(self, model_id: str):
        """移除資料庫中 Model 的所有 execution 狀態

        Parameters
        ----------
        model_id: str
            Model ID

        Returns
        -------
        None.
        """
        self._delete(TableName.MODEL_EXECUTION.value,
                     whereby=[
                         (ModelExecutionField.MODEL_ID.value, model_id)
                     ])

    def del_model_kernel(self, model_id: str):
        """移除資料庫中 Model 的所有 execution 狀態

        Parameters
        ----------
        model_id: str
            Model ID

        Returns
        -------
        None.
        """
        self._delete(TableName.MODEL_KERNEL.value,
                     whereby=[
                         (ModelKernelField.MODEL_ID.value, model_id)
                     ])

    def del_model_data(self, model_id: str):
        """移除資料庫中指定 Model 由 python 計算出的預測結果資訊

        Parameters
        ----------
        model_id: str
            Model ID

        Returns
        -------
        None.

        """
        self._delete(TableName.PREDICT_RESULT.value,
                     whereby=[
                         (PredictResultField.MODEL_ID.value, model_id)
                     ])
        self._delete(TableName.PREDICT_RESULT_HISTORY.value,
                     whereby=[
                         (PredictResultField.MODEL_ID.value, model_id)
                     ])
        self._delete(TableName.MODEL_MKT_HIT_SUM.value,
                     whereby=[
                         (ModelMarketHitSumField.MODEL_ID.value, model_id)
                     ])
        self.cache_manager.del_model_result(model_id)

    def del_pattern_data(self, pattern_id: str):
        """刪除 DB 中 Pattern 計算相關資訊，包含
        1. FCST_PAT_EXECUTION
        2. FCST_PAT_MKT_EVENT

        Parameters
        ----------
        pattern_id: str
            ID of pattern to delete.

        Returns
        -------
        None.

        """
        self._delete(TableName.PATTERN_RESULT.value,
                     whereby=[
                         (PatternResultField.PATTERN_ID.value, pattern_id)
                     ])
        self._delete(TableName.PATTERN_EXECUTION.value,
                     whereby=[
                         (PatternExecutionField.PATTERN_ID.value, pattern_id)
                     ])

    def del_macro_info(self, macro_id: str):
        """移除資料庫中指定 Macro 資訊

        Parameters
        ----------
        macro_id: str
            要刪除的 Macro ID

        Returns
        -------
        None.

        """
        self._delete(TableName.MACRO_INFO.value,
                     whereby=[
                         (MacroInfoField.MACRO_ID.value, macro_id)
                     ])

    def del_macro_param_info(self, macro_id: str):
        """移除資料庫中指定 Macro 參數資訊

        Parameters
        ----------
        macro_id: str
            要刪除參數資訊的 Macro ID

        Returns
        -------
        None.

        """
        self._delete(TableName.MACRO_PARAM.value,
                     whereby=[
                         (MacroParamField.MACRO_ID.value, macro_id)
                     ])

    def del_macro_version_info(self, macro_id: str):
        """移除資料庫中指定 Macro 版本資訊

        Parameters
        ----------
        macro_id: str
            要刪除版本資訊的 Macro ID

        Returns
        -------
        None.

        """
        self._delete(TableName.MACRO_VERSION.value,
                     whereby=[
                         (MacroVersionInfoField.MACRO_ID.value, macro_id)
                     ])

    def _get_macro_tags(self) -> Dict[MacroTags, str]:
        """取得 Macro Tag 與對應的 Tag ID

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        tags = []
        for macro in MacroManager:
            tags += macro.tags
        tags = list(set(tags))
        ids = self._get_serial_no(SerialNoType.TAG, len(tags))
        ret = {tag: id_  for tag, id_ in zip(tags, ids)}
        return ret

    def _get_macro_tag_info(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """取得 Macro Tag 資訊以及 Macro Tag 對應表資訊

        Parameters
        ----------
        None.

        Returns
        -------
        result: Tuple[pd.DataFrame, pd.DataFrame]
            結構為: (Macro Tag 資訊, Macro Tag 對應表資訊)
        """
        tags = self._get_macro_tags()
        info = pd.DataFrame([[i, t.value.name, t.value.desc]
                             for t, i in tags.items()],
                            columns=[MacroTagField.TAG_ID.value,
                                     MacroTagField.TAG_NAME.value,
                                     MacroTagField.TAG_DESC.value])
        info[MacroTagField.AUTHOR.value] = self.CREATE_BY

        engine = self._engine()
        sql = f"""
            SELECT
                {MacroInfoField.MACRO_ID.value},
                {MacroInfoField.FUNC_CODE.value}
            FROM
                {TableName.MACRO_INFO.value}
        """
        data = pd.read_sql_query(sql, engine)
        func_to_macro_id = {
            data.iloc[i][MacroInfoField.FUNC_CODE.value] : data.iloc[i][MacroInfoField.MACRO_ID.value]
            for i in range(len(data))
        }
        values = []
        for macro in MacroManager:
            values += [[func_to_macro_id[macro.code], tags[tag]] for tag in macro.tags]
        maps = pd.DataFrame(values, columns=[MacroTagMapField.MACRO_ID.value,
                                             MacroTagMapField.TAG_ID.value])

        info = self._extend_basic_cols(info)
        maps = self._extend_basic_cols(maps)
        return info, maps

# 呼叫 Stored Procedule
    @_do_if_not_read_only
    def update_model_accuracy(self):
        """ 呼叫 SP 計算當前各模型準確率

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        logging.info('start update FCST model accuracy')
        engine = self._engine()
        sql = f'CALL {StoredProcedule.UPDATE_FCST_MODEL_MKT_HIT_SUM_SWAP.value}()'
        with engine.begin() as db_conn:
            db_conn.execute(sql)
        logging.info('Update FCST model accuracy finished')

    @_do_if_not_read_only
    def checkout_fcst_data(self):
        """ 切換系統中的呈現資料位置

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        logging.info('start checkout FCST data')
        engine = self._engine()
        sql = f"CALL {StoredProcedule.SWAP_FCST.value}()"
        with engine.begin() as db_conn:
            db_conn.execute(sql)
        logging.info('Checkout FCST data finished')

    @_do_if_not_read_only
    def truncate_swap_tables(self):
        """ 清除資料庫中執行批次時需要為空的 SWAP 表

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        logging.info('Start truncate swap tables')
        engine = self._engine()
        sql = f"CALL {StoredProcedule.TRUNCATE_AI_SWAP.value}()"
        with engine.begin() as db_conn:
            db_conn.execute(sql)
        logging.info('Start truncate swap tables finished')

    def _get_serial_no(self, code: SerialNoType, size: Optional[int]=None) -> Union[str, List[str]]:
        """利用 SP 取得指定的 SerialNo 序列

        Parameters
        ----------
        code: SerialNo
        """
        engine = self._engine()
        # 取得 EXEC_ID
        logging.info(f'Call {StoredProcedule.GET_SERIAL_NO.value}')
        sp_arg = 'null' if size is None else str(size)
        sql = f"CALL {StoredProcedule.GET_SERIAL_NO.value}('{code.value}', {sp_arg})"
        with engine.begin() as db_conn:
            if size is None:
                result = db_conn.execute(sql).fetchone()[0]
            else:
                results = db_conn.execute(sql).fetchall()
                result = [r[0] for r in results]
        logging.info(f"Get SERIAL_NO: {result}")
        return result

    @_do_if_not_read_only
    def broadcast_invalid_macro(self, macro_id: str, effect_msg: str):
        """當 Macro 發生異動後, 將異動造成的影響擴散至整個DB

        Parameters
        ----------
        macro_id: str
            被異動的 Macro ID
        effect_msg: str
            異動訊息

        Returns
        -------
        None.
        """
        engine = self._engine()
        # 取得 EXEC_ID
        logging.info(f'Call {StoredProcedule.BROADCAST_INVALID_MACRO_EFFECT.value}')
        sql = f"CALL {StoredProcedule.BROADCAST_INVALID_MACRO_EFFECT.value}('{macro_id}', '{effect_msg}')"
        with engine.begin() as db_conn:
            db_conn.execute(sql)
        logging.info(f"Call {StoredProcedule.BROADCAST_INVALID_MACRO_EFFECT.value} finished")

# 更新本地 Model 狀態
    @_do_if_not_read_only
    def set_model_execution_complete(self, exec_id: str):
        """更新本地快取紀錄的 Model 執行狀態

        Parameters
        ----------
        exec_id: str
            ID of model_exec
        execution: str
            status code set to this model
        model_id: str
            ID of model

        Returns
        -------
        None.

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
        exec_data = pd.read_sql_query(sql, engine)
        if len(exec_data) == 0:
            raise Exception(
                f'call set_model_execution_complete before set_model_execution_start: {exec_id}')
        exec_data = exec_data.iloc[0]

        status = finished_status[exec_data[ModelExecutionField.STATUS_CODE.value]]
        model_id = exec_data[ModelExecutionField.MODEL_ID.value]

        logging.info(f"[Local Update] Set model execution complete: {model_id} -> {status} started")
        local_status = self.cache_manager.get_model_status(model_id)
        local_status[status] = str(now)
        self.cache_manager.put_status_to_queue(model_id, local_status)
        logging.info(f"[Local Update] Set model execution complete: {model_id} -> {status} finished")


    @_do_if_not_read_only
    def set_model_train_complete(self, model_id: str):
        """ create model train complete exec

        Parameters
        ----------
        model_id: str
            ID of model

        Returns
        -------
        None.

        """
        table_name = TableName.MODEL_EXECUTION.value
        exection = ModelExecution.TRAIN_MODEL_FINISHED.value
        exec_id = 'READ_ONLY_MODE'

        # 取得 EXEC_ID
        exec_id:str = self._get_serial_no(SerialNoType.EXECUTION)

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
        end_dt = now
        data = [[
            create_by, create_dt, exec_id,
            model_id, exection, start_dt, end_dt]]
        data = pd.DataFrame(data, columns=COLUMNS)

        logging.info(f"[DB Create] Set model train complete {model_id} -> {exection}")
        self._insert(table_name, data)
        logging.info(f"[DB Create] Set model train complete finished: {model_id} -> {exection}")

    def get_recover_models(self) -> List[str]:
        engine = self._engine()
        sql = f"""
            SELECT
                model.{ModelInfoField.MODEL_ID.value},
                model.{ModelInfoField.MODEL_STATUS.value},
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
            WHERE
                model.{ModelInfoField.MODEL_STATUS.value}='{DBModelStatus.PRIVATE_AND_VALID.value}' OR
                model.{ModelInfoField.MODEL_STATUS.value}='{DBModelStatus.PUBLIC_AND_VALID.value}'
        """
        data = pd.read_sql_query(sql, engine)
        group_data = data.groupby(ModelInfoField.MODEL_ID.value)
        results = []
        for model_id, model_state_info in group_data:
            model_add_predict_info = model_state_info[
                model_state_info[ModelExecutionField.STATUS_CODE.value].values ==
                ModelExecution.TRAIN_MODEL_FINISHED.value]
            # ADD_PREDICT 未建立就掛掉
            if len(model_add_predict_info) == 0:
                results.append(model_id)
        return results
