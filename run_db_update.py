import argparse
import logging
from logging import handlers
import os
import sys
import pandas as pd
from typing import List
import warnings
from const import LOG_LOC, ExecMode, MacroInfoField, MacroParamField, MacroVersionInfoField
from dao import MimosaDB
from _core._macro import MacroManager, MacroParaEnumManager

def generate_macro_ids(size: int, db: MimosaDB) -> List[str]:
    """產生 macro id 序列
    Parameters
    ----------
    size: int
        序列長度
    db: MimosaDB
        資料庫連接工具
    
    Returns
    -------
    result: List[str]
        Macro ID 序列
    """
    exist_ids = db.get_macro_info()[MacroInfoField.MACRO_ID.value].values
    prefix = 'MAC00000000'
    results = []
    series_no = 0
    while len(results) < size:
        new_id = f'{prefix}{str(series_no).zfill(9)}'
        if new_id not in exist_ids:
            results.append(series_no)
        series_no += 1
    return results

def get_macro_info_from_db(db: MimosaDB):
    """取得資料庫中 Macro 資訊

    Parameters
    ----------
    db: MimosaDB
        資料庫物件
    
    Returns
    -------
    result: Dict[str, MacroDict]
        Macro 資料, 
        MacroDict: {
            'FUNC_CODE': str,
            'MACRO_NAME': str,
            'MACRO_DESC': str,
            'MACRO_INFO_VERSION': str,
            'MACRO_CODE_VERSION': str,
            'PARAM': List[ParamDict]
        }
        ParamDict: {
            'PARAM_CODE': str,
            'PARAM_NAME': str,
            'PARAM_DESC': str,
            'PARAM_DEFAULT': str,
            'PARAM_TYPE': str
        }
    """
    macro_info = db.get_macro_info()
    macro_param_info = db.get_macro_param_info()
    macro_version_info = db.get_macro_version_info()
    result = {}
    for i in range(len(macro_info)):
        macro_id = macro_info.iloc[i][MacroInfoField.MACRO_ID.value]
        macro_name = macro_info.iloc[i][MacroInfoField.MACRO_NAME.value]
        macro_desc = macro_info.iloc[i][MacroInfoField.MACRO_DESC.value]
        macro_func = macro_info.iloc[i][MacroInfoField.FUNC_CODE.value]
        ver_info_cond = macro_version_info[MacroVersionInfoField.MACRO_ID.value].values == macro_id
        ver_info = macro_version_info[ver_info_cond].iloc[0]
        macro_info_version = ver_info[MacroVersionInfoField.INFO_VERSION.value]
        macro_code_version = ver_info[MacroVersionInfoField.CODE_VERSION.value]
        param_cond = macro_param_info[MacroParamField.MACRO_ID.value].values == macro_id
        param_info = macro_param_info[param_cond]
        params = []
        for j in range(len(param_info)):
            param = param_info.iloc[j]
            param_code = param[MacroParamField.PARAM_CODE.value]
            param_name = param[MacroParamField.PARAM_NAME.value]
            param_desc = param[MacroParamField.PARAM_DESC.value]
            param_default = param[MacroParamField.PARAM_DEFAULT.value]
            param_type = param[MacroParamField.PARAM_TYPE.value]
            param_dict = {
                'PARAM_CODE': param_code,
                'PARAM_NAME': param_name,
                'PARAM_DESC': param_desc,
                'PARAM_DEFAULT': param_default,
                'PARAM_TYPE': param_type
            }
            params.append(param_dict)
        macro_dict = {
            'FUNC_CODE': macro_func,
            'MACRO_NAME': macro_name,
            'MACRO_DESC': macro_desc,
            'MACRO_INFO_VERSION': macro_info_version,
            'MACRO_CODE_VERSION': macro_code_version,
            'PARAM': params
        }
        result[macro_id] = macro_dict
    return result

def check_macro(new, old):
    """

    Parameters
    ----------
    new: List[MacroDict]
        新的 Macro 資料
        MacroDict: {
            'FUNC_CODE': str,
            'MACRO_NAME': str,
            'MACRO_DESC': str,
            'MACRO_INFO_VERSION': str,
            'MACRO_CODE_VERSION': str,
            'PARAM': List[ParamDict]
        }
        ParamDict: {
            'PARAM_CODE': str,
            'PARAM_NAME': str,
            'PARAM_DESC': str,
            'PARAM_DEFAULT': str,
            'PARAM_TYPE': str
        }
    old: Dict[str, MacroDict]
        舊的 Macro 資料
        MacroDict: {
            'FUNC_CODE': str,
            'MACRO_NAME': str,
            'MACRO_DESC': str,
            'MACRO_INFO_VERSION': str,
            'MACRO_CODE_VERSION': str,
            'PARAM': List[ParamDict]
        }
        ParamDict: {
            'PARAM_CODE': str,
            'PARAM_NAME': str,
            'PARAM_DESC': str,
            'PARAM_DEFAULT': str,
            'PARAM_TYPE': str
        }
    
    Returns
    -------
    ret_i: List[MacroDict]
        要進行新增的 Macro 資料清單
    ret_m: Dict[str, str]
        所有有異動的 Macro 異動訊息
    ret_d: List[str]
        要進行刪除的 Macro ID 清單
    ret_u: Dict[str, MacroDict]
        要進行更新的 Macro 資料清單
    """
    # check delete
    ret_i = []
    ret_m = {}
    ret_d = []
    ret_u = {}

    new = {each['FUNC_CODE']: each for each in new}
    for mid in old:
        if old[mid]['FUNC_CODE'] not in new:
            ret_m[mid] = f"運算函式 {old[mid]['MACRO_NAME']}已被系統刪除"
            ret_d.append(mid)
    func_to_mid = {old[each]['FUNC_CODE']: each for each in old}
    for fid in new:
        if fid not in func_to_mid:
            ret_i.append(new[fid])
            continue
        cur_n = new[fid]
        mid = func_to_mid[fid]
        cur_o = old[mid]
        if cur_n['MACRO_CODE_VERSION'] == cur_o['MACRO_CODE_VERSION']:
            if cur_n['MACRO_INFO_VERSION'] != cur_o['MACRO_INFO_VERSION']:
                ret_u[mid] = cur_n
            continue
        ret_u[mid] = cur_n
        idx = 1
        msg = (f"系統已發布 運算函式 {cur_o['MACRO_NAME']} 的更新版本：\n"
               f"{idx}. 運算邏輯更新；\n")
        idx += 1
        if cur_n['MACRO_NAME'] != cur_o['MACRO_NAME']:
            msg += f"{idx}. 更新名稱為 `{cur_o['MACRO_NAME']}`；\n"
            idx += 1
        para_n = {each['PARAM_NAME']: each['PARAM_TYPE']
                  for each in cur_n['PARAM']}
        para_o = {each['PARAM_NAME']: each['PARAM_TYPE']
                  for each in cur_o['PARAM']}
        for each in para_n:
            if each not in para_o:
                msg += f"{idx}. 新增參數 `{each}`；\n"
                idx += 1
        for each in para_o:
            if each not in para_n:
                msg += f"{idx}. 移除參數 `{each}`；\n"
                idx += 1
        for each in para_n:
            if each in para_o:
                if para_n[each] != para_o[each]:
                    msg += f"{idx}. 變更參數 {each} 的型態為 `{para_n[each]}`；\n"
                    idx += 1
        ret_m[mid] = msg
    return ret_i, ret_u, ret_m, ret_d

def convert_macro_to_df(macro, macro_id: str):
    """將 Macro Dict 轉換為其中包含的各張 pd.DataFrame

    Parameters
    ----------
    macro: Dict
        Macro 資訊
        MacroDict: {
            'FUNC_CODE': str,
            'MACRO_NAME': str,
            'MACRO_DESC': str,
            'MACRO_INFO_VERSION': str,
            'MACRO_CODE_VERSION': str,
            'PARAM': List[ParamDict]
        }
        ParamDict: {
            'PARAM_CODE': str,
            'PARAM_NAME': str,
            'PARAM_DESC': str,
            'PARAM_DEFAULT': str,
            'PARAM_TYPE': str
        }

    Returns
    -------
    None.
    """
    macro_info = [
        macro_id,
        macro['FUNC_CODE'],
        macro['MACRO_NAME'],
        macro['MACRO_DESC'],
    ]
    macro_info = pd.DataFrame([macro_info], columns=[
        MacroInfoField.MACRO_ID.value,
        MacroInfoField.MACRO_NAME.value,
        MacroInfoField.MACRO_DESC.value
    ])
    macro_version = [
        macro_id,
        macro['MACRO_INFO_VERSION'],
        macro['MACRO_CODE_VERSION'],
    ]
    macro_version = pd.DataFrame([macro_version], columns=[
        MacroVersionInfoField.MACRO_ID.value,
        MacroVersionInfoField.INFO_VERSION.value,
        MacroVersionInfoField.CODE_VERSION.value
    ])
    params = []
    for param in macro['PARAM']:
        params.append([
            macro_id,
            param['PARAM_CODE'],
            param['PARAM_NAME'],
            param['PARAM_DESC'],
            param['PARAM_DEFAULT'],
            param['PARAM_TYPE'],
        ])
    params = pd.DataFrame(params, columns=[
        MacroParamField.MACRO_ID.value,
        MacroParamField.PARAM_CODE.value,
        MacroParamField.PARAM_NAME.value,
        MacroParamField.PARAM_DESC.value,
        MacroParamField.PARAM_DEFAULT.value,
        MacroParamField.PARAM_TYPE.value,
    ])
    return macro_info, macro_version, params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="db_update_batch")
    parser.add_argument("--mode", "-md", action='store', default='uat')
    args = parser.parse_args()
    mode = args.mode
    exec_mode = ExecMode.get(mode)
    if exec_mode is None:
        raise RuntimeError(f'invalid execution mode {mode}')
    if exec_mode == ExecMode.PROD.value or exec_mode == ExecMode.UAT.value:
        warnings.filterwarnings("ignore")
    if not os.path.exists(LOG_LOC):
        os.mkdir(LOG_LOC)
    err_hdlr = logging.StreamHandler(stream=sys.stderr)
    err_hdlr.setLevel(logging.ERROR)
    info_hdlr = logging.StreamHandler(stream=sys.stdout)
    info_hdlr.setLevel(logging.INFO)
    file_hdlr = handlers.TimedRotatingFileHandler(
        filename=f'{LOG_LOC}/update_db_data_batch.log', when='D', backupCount=7)
    fmt = '%(asctime)s.%(msecs)03d - %(levelname)s - %(threadName)s - %(filename)s - line %(lineno)d: %(message)s'
    level = {ExecMode.DEV.value: logging.DEBUG,
                ExecMode.UAT.value: logging.INFO,
                ExecMode.PROD.value: logging.ERROR}[exec_mode]
    file_hdlr.setLevel(level)
    logging.basicConfig(level=0, format=fmt, handlers=[
                        err_hdlr, info_hdlr, file_hdlr], datefmt='%Y-%m-%d %H:%M:%S')
    db = MimosaDB(mode=mode)

    macro_info = MacroManager.dump()
    old_macro = get_macro_info_from_db(db)
    adds, updates, msgs, dels = check_macro(macro_info, old_macro)
    
    # 新增 Macro
    infos = []
    versions = []
    params = []
    macro_ids = generate_macro_ids(len(adds), db)
    for i, macro in enumerate(adds):
        info, version, param = convert_macro_to_df(macro, macro_ids[i])
        infos.append(info)
        versions.append(version)
        params.append(params)
    infos = pd.concat(infos, axis=0)
    versions = pd.concat(versions, axis=0)
    params = pd.concat(params, axis=0)
    db.save_macro_info(infos)
    db.save_macro_version_info(versions)
    db.save_macro_param_info(params)

    # 更新 Macro
    infos = []
    versions = []
    params = []
    for macro_id in updates:
        macro = updates[macro_id]
        info, version, param = convert_macro_to_df(macro, macro_id)
        infos.append(info)
        versions.append(version)
        params.append(params)
    infos = pd.concat(infos, axis=0)
    versions = pd.concat(versions, axis=0)
    params = pd.concat(params, axis=0)
    db.update_macro_info(infos)
    db.update_macro_version_info(versions)
    db.update_macro_param_info(params)

    # 刪除 Macro
    for macro_id in dels:
        db.del_macro_info(macro_id)
        db.del_macro_param_info(macro_id)
        db.del_macro_version_info(macro_id)
    
    # 發布 Macro 異動資訊
    for macro_id, msg in msgs.items():
        # CALL SP
        pass

    # 更新 Macro Enum
    enum_info = MacroParaEnumManager.dump()
    db.save_macro_param_enum(enum_info)
