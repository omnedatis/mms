# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

@author: Jeff

WSGI python server
"""
import argparse
import os
import traceback
import threading as mt
import sys
from model import (
    get_patterns_occur_dates, set_db, batch, init_db, get_mix_pattern_occur, get_mix_pattern_mkt_dist_info,
    get_mix_pattern_rise_prob, get_mix_pattern_occur_cnt, get_market_price_dates,
    get_market_rise_prob, get_mkt_dist_info, add_pattern, add_model, remove_model,
    task_queue, verify_pattern, get_frame, get_plot, del_pattern_data,
    cast_macro_kwargs, del_model_execution, check_macro_info, CatchableTread,
    get_occurred_patterns, get_draft_date)
from const import (ExecMode, PORT, LOG_LOC, MODEL_QUEUE_LIMIT, PATTERN_QUEUE_LIMIT,
                   MarketPeriodField, HttpResponseCode, TaskLimitCode)
from func.common import Ptype
import datetime
from dao import MimosaDB
from flask import Flask, request
from flasgger import Swagger
from waitress import serve
from werkzeug.exceptions import MethodNotAllowed, NotFound, InternalServerError, BadRequest
import logging
from logging import handlers
import warnings

app = Flask(__name__)

Swagger(app)

parser = argparse.ArgumentParser(prog="Program start server")
parser.add_argument("--motionless", "-ml", action='store_true', default=False)
parser.add_argument("--batchless", "-bl", action='store_true', default=False)
parser.add_argument("--mode", "-md", action='store', default='dev')
args = parser.parse_args()


@app.route("/model/batch", methods=["GET"])
def api_batch():
    """ Run batch. """
    logging.info("Calling batch, pending")
    task_queue.do_prioritized_task(batch)
    return HttpResponseCode.ACCEPTED.format()


@app.route("/models/<string:modelId>", methods=["POST"])
def api_add_model(modelId):
    """
    新增模型
    ---
    tags:
      - 前台
    parameters:
      - name: modelId
        in: path
        type: string
        required: true
    responses:
      202:
        description: 請求已接收，等待執行
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: string
              nullable: true
    """
    logging.info(f"api_add_model  receiving: {modelId}")
    task_queue.push(add_model, args=(modelId,), task_limit=TaskLimitCode.MODEL)
    return HttpResponseCode.ACCEPTED.format()


@app.route("/models/<string:modelId>", methods=["DELETE"])
def api_remove_model(modelId):
    """
    移除模型
    ---
    tags:
      - 前台
    parameters:
      - name: modelId
        in: path
        type: string
        required: true
    responses:
      202:
        description: 請求已接收，等待執行
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: string
              nullable: true
    """
    logging.info(f"api_remove_model  receiving: {modelId}")
    del_model_execution(modelId)
    t = CatchableTread(target=remove_model, args=(modelId,))
    t.start()
    return HttpResponseCode.ACCEPTED.format()


@app.route("/models/<string:modelId>", methods=["PATCH"])
def api_edit_model(modelId):
    """
    編輯模型
    ---
    tags:
      - Studio
    parameters:
      - name: modelId
        in: path
        type: string
        required: true
    responses:
      202:
        decription: 請求已接收，等待執行
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: string
              nullable: true
    """
    logging.info(f"api_edit_model receiving: {modelId}")
    del_model_execution(modelId)
    task_queue.push(add_model, args=(modelId,), task_limit=TaskLimitCode.MODEL)
    return HttpResponseCode.ACCEPTED.format()


@app.route("/pattern/compound/occurdates", methods=["POST"])
def api_get_pattern_dates():
    """
    取得複合現象歷史發生日期
    ---
    tags:
      - Studio
    parameters:
      - name: request
        in: body
        type: object
        properties:
          patterns:
            type: array
            items:
              type: string
          marketCode:
            type: string
          startDate:
            type: string
            format: date
          endDate:
            type: string
            format: date
    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: array
              items:
                type: string
                format: date
    """
    try:
        logging.info(f"api_get_pattern_dates receiving: {request.json}")
        data = request.json
        patterns = data['patterns']
        market_id = data['marketCode']
        start_date = data.get('startDate')
        end_date = data.get('endDate')
    except Exception as esp:
        raise BadRequest
    ret = get_mix_pattern_occur(market_id, patterns, start_date, end_date)
    ret = [each.strftime('%Y-%m-%d') for each in ret]

    return HttpResponseCode.OK.format(ret)


@app.route("/pattern/occurdates", methods=["POST"])
def api_get_patterns_dates():
    """
    取得多個現象的指定市場, 指定期間歷史發生日期
    ---
    tags:
      - Studio
    parameters:
      - name: request
        in: body
        type: object
        properties:
          patterns:
            type: array
            items:
              type: string
          marketCode:
            type: string
          startDate:
            type: string
            format: date
          endDate:
            type: string
            format: date
    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: array
              items:
                type: string
                format: date
    """
    try:
        logging.info(f"api_get_patterns_dates receiving: {request.json}")
        data = request.json
        patterns = data['patterns']
        market_id = data['marketCode']
        start_date = data.get('startDate')
        end_date = data.get('endDate')
    except Exception as esp:
        raise BadRequest
    ret = get_patterns_occur_dates(market_id, patterns, start_date, end_date)
    return HttpResponseCode.OK.format(ret)


@app.route("/pattern/count", methods=["POST"])
def api_get_pattern_count():
    """
    取得複合現象上漲次數
    ---
    tags:
      - Studio
    parameters:
      - name: request
        in: body
        type: object
        properties:
          patternId:
            type: string
          markets:
            type: array
            items:
             type: string
          startDate:
            type: string
            format: date
          endDate:
            type: string
            format: date
    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: object
              properties:
                occurCnt:
                  type: integer
                nonOccurCnt:
                  type: integer
    """
    try:
        logging.info(
            f"api_get_pattern_count receiving: {request.json}")
        data = request.json

        pattern_id = data['patternId']
        markets =  data['markets']
        start_date = data.get('startDate') and datetime.datetime.strptime(data.get('startDate'), '%Y-%m-%d').date()
        end_date = data.get('endDate') and datetime.datetime.strptime(data.get('endDate'), '%Y-%m-%d').date()
    except Exception as esp:
        raise BadRequest

    occur, non_occur = get_mix_pattern_occur_cnt(
        pattern_id, markets, start_date, end_date)
    ret = {"occurCnt": int(occur), "nonOccurCnt": int(non_occur)}
    return HttpResponseCode.OK.format(ret)


@app.route("/pattern/occurredpatterns", methods=["POST"])
def api_get_occurred_patterns():
    """
    取得指定日期有發生的現象
    ---
    tags:
      - Studio
    parameters:
      - name: request
        in: body
        type: object
        properties:
          date:
            type: string
            format: date
          patterns:
            type: array
            items:
              type: string
    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
                type: array
                items:
                  type: string
    """
    try:
        logging.info(
            f"api_get_occurred_patterns receiving: {request.json}")
        data = request.json
        date = datetime.datetime.strptime(data['date'], '%Y-%m-%d')
        patterns = data['patterns']
    except Exception:
        raise BadRequest
    ret = get_occurred_patterns(date, patterns)
    return HttpResponseCode.OK.format(ret)


@app.route("/pattern/updownprob", methods=["POST"])
def api_get_pattern_updownprob():
    """
    取得複合現象上漲/下跌機率
    ---
    tags:
      - Studio
    parameters:
      - name: request
        in: body
        type: object
        properties:
          markets:
            type: array
            items:
              type: string
          patterns:
            type: array
            items:
              type: string
    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: array
              items:
                type: object
                properties:
                  statisticsCategory:
                    type: string
                  upAndDownCategories:
                    type: string
                  probabilityValue:
                      type: array
                      items:
                        type: object
                        properties:
                          days:
                            type: integer
                          probability:
                            type: number
    """
    try:
        logging.info(
            f"api_get_pattern_upprob receiving: {request.json}")
        data = request.json
        markets = data['markets']
        patterns = data['patterns']
    except Exception:
        raise BadRequest
    ret = get_mix_pattern_rise_prob(markets, patterns)
    return HttpResponseCode.OK.format(ret)


@app.route("/pattern/distribution", methods=["POST"])
def api_get_pattern_distribution():
    """
    取得複合現象分布資訊
    ---
    tags:
      - Studio
    parameters:
      - name: request
        in: body
        type: object
        properties:
          patterns:
            type: array
            items:
              type: string
          datePeriod:
            type: integer
          markets:
            type: array
            items:
              type: string
    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: array
              items:
                type: object
                properties:
                  type:
                    type: string
                  rangeUp:
                    type: float
                  rangeDown:
                    type: float
                  name:
                    type: number
                  value:
                    type: number
    """
    try:
        logging.info(
            f"api_get_pattern_distribution receiving: {request.json}")
        data = request.json
        patterns = data['patterns']
        date_period = data['datePeriod']
        markets = data['markets']
    except Exception as esp:
        raise BadRequest
    ret = get_mix_pattern_mkt_dist_info(
        patterns, date_period, markets)
    return HttpResponseCode.OK.format(ret)


@app.route("/patterns/<string:patternId>", methods=["POST"])
def api_add_pattern(patternId):
    """
    新增現象
    ---
    tags:
      - Studio
    parameters:
      - name: patternId
        in: path
        type: string
        required: true
    responses:
      202:
        description: 請求已接收，等待執行
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: string
              nullable: true
    """
    logging.info(f"api_add_pattern receiving: {patternId}")
    task_queue.push(add_pattern, args=(patternId,), task_limit=TaskLimitCode.PATTERN)
    return HttpResponseCode.ACCEPTED.format()

@app.route("/patterns/<string:patternId>", methods=["PATCH"])
def api_edit_pattern(patternId):
    """
    編輯現象
    ---
    tags:
      - Studio
    parameters:
      - name: patternId
        in: path
        type: string
        required: true
    responses:
      202:
        decription: 請求已接收，等待執行
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: string
              nullable: true
    """
    logging.info(f"api_edit_pattern receiving: {patternId}")
    del_pattern_data(patternId)
    task_queue.push(add_pattern, args=(patternId,), task_limit=TaskLimitCode.PATTERN)
    return HttpResponseCode.ACCEPTED.format()


@app.route("/markets/upprob", methods=["GET"])
def api_get_market_upprob():
    """
    取得指定市場集上漲機率
    ---
    tags:
      - 前台
    parameters:
      - name: datePeriod
        in: query
        type: integer
        required: true
      - name: marketType
        in: query
        type: string
      - name: categoryCode
        in: query
        type: string
    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: object
              properties:
                positiveWeight:
                  type: number
    """
    try:
        logging.info(f"api_get_market_upprob receiving: {request.args}")
        data = request.args
        date_period = data["datePeriod"]
        market_type = data.get("marketType") or None
        category_code = data.get("categoryCode") or None
    except Exception as esp:
        raise BadRequest
    ret = get_market_rise_prob(int(date_period), market_type, category_code)
    ret =  {"positiveWeight": float(ret)}
    return HttpResponseCode.OK.format(ret)


@app.route("/markets/distribution", methods=["POST"])
def api_get_market_distribution():
    """
    取得指定市場集分布資訊
    ---
    tags:
      - 前台
    parameters:
      - name: request
        in: body
        type: object
        properties:
          datePeriod:
            type: integer
          markets:
            type: array
            items:
              type: string
    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: array
              items:
                type: object
                properties:
                  type:
                    type: string
                  rangeUp:
                    type: float
                  rangeDown:
                    type: float
                  name:
                    type: number
                  value:
                    type: number
    """
    try:
        logging.info(f"api_get_market_distribution receiving: {request.args}")
        data = request.json
        date_period = data["datePeriod"]
        markets = data['markets']
    except Exception as esp:
        raise BadRequest
    ret = get_mkt_dist_info(int(date_period), markets)
    return HttpResponseCode.OK.format(ret)


@app.route('/markets/<string:marketId>/pricedate', methods=["GET"])
def api_get_market_price_date(marketId):
    """
    取得起始日起各天期資料日期和對應價格天期
    ---
    tags:
      - 前台
    parameters:
      - name: startDate
        in: query
        type: string
        format: date
      - name: marketId
        in: path
        type: string
        required: true
    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: array
              items:
                type: object
                properties:
                  dataDate:
                    type: string
                  netChangeRate:
                    type: number
                  priceDate:
                    type: string
                    format: date
                  datePeriod:
                    type: integer
    """
    try:
        logging.info(f"api_get_market_price_date receiving: {request.args}")
        data = request.args
        start_date = data.get("startDate") or None
        if start_date is not None:
            start_date = datetime.datetime.strptime(
                start_date, "%Y-%m-%d").date()
    except:
        raise BadRequest
    ret = get_market_price_dates(marketId, start_date)
    ret = [{"dataDate": each[MarketPeriodField.DATA_DATE.value].strftime('%Y-%m-%d'),
            "priceDate":each[MarketPeriodField.PRICE_DATE.value].strftime('%Y-%m-%d'),
            "datePeriod":each[MarketPeriodField.DATE_PERIOD.value], } for each in ret]
    return HttpResponseCode.OK.format(ret)

@app.route('/pattern/paramcheck', methods=["POST"])
def api_pattern_paramscheck():
    """
    檢查 Pattern 參數合法性
    ---
    tags:
      - Studio
    parameters:
      - name: request
        in: body
        type: object
        properties:
          funcCode:
            type: string
          paramCodes:
            type: array
            items:
              type: object
              properties:
                paramCode:
                  type: string
                paramValue:
                  type: string

    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: array
              items:
                type: object
                properties:
                  errorParam:
                    type: string
                  errorMessage:
                    type: string
    """
    logging.info(f"api_pattern_paramscheck receiving: {request.json}")
    try:
        data = request.json
        func_code = data['funcCode']
        params_codes = data['paramCodes']
        kwargs = {each["paramCode"]: each["paramValue"]
                  for each in params_codes}
        if check_macro_info(func_code):
            raise InternalServerError(f'found inconsistent data type on {func_code}')
        kwargs, msg = cast_macro_kwargs(func_code, kwargs)
        if msg:
          return HttpResponseCode.OK.format(msg)
    except KeyError:
        raise BadRequest
    except ValueError:
        raise InternalServerError
    ret = [{"errorParam": key, "errorMessage": value}
           for key, value in verify_pattern(func_code, kwargs).items()]
    return HttpResponseCode.OK.format(ret)


@app.route('/pattern/frame', methods=["POST"])
def api_get_pattern_frame():
    """
    取得 pattern 示意圖規則區間長度
    ---
    tags:
      - Studio
    parameters:
      - name: request
        in: body
        type: object
        properties:
          funcCode:
            type: string
          paramCodes:
            type: array
            items:
              type: object
              properties:
                paramCode:
                  type: string
                paramValue:
                  type: string

    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: object
              properties:
                patternInterval:
                  type: integer
    """
    logging.info(f"api_get_pattern_frame receiving: {request.json}")
    try:
        data = request.json
        func_code = data['funcCode']
        params_codes = data['paramCodes']
        kwargs = {each["paramCode"]: each["paramValue"]
                  for each in params_codes}
        if check_macro_info(func_code):
            raise InternalServerError(f'found inconsistent data type on {func_code}')
        kwargs, msg = cast_macro_kwargs(func_code, kwargs)
        if msg:
            return HttpResponseCode.BAD_REQUEST.format(msg)
    except KeyError:
        raise BadRequest
    except ValueError:
        raise InternalServerError
    ret = get_frame(func_code, kwargs)
    ret = {"patternInterval": ret}
    return HttpResponseCode.OK.format(ret)


@app.route('/pattern/plot', methods=["POST"])
def api_get_pattern_plot():
    """
    取得 pattern 示意圖
    ---
    tags:
      - Studio
    parameters:
      - name: request
        in: body
        type: object
        properties:
          funcCode:
            type: string
          paramCodes:
            type: array
            items:
              type: object
              properties:
                paramCode:
                  type: string
                paramValue:
                  type: string

    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: array
              items:
                type: object
                properties:
                  figType:
                    type: string
                  figName:
                    type: string
                  figData:
                    type: array
                    items:
                      type: object
                      properties:
                        group:
                          type: string
                        value:
                          type: number
                          format: float
                        index:
                          type: integer
    """
    logging.info(f"api_get_pattern_plot receiving: {request.json}")
    try:
        data = request.json
        func_code = data['funcCode']
        params_codes = data['paramCodes']
        kwargs = {each["paramCode"]: each["paramValue"]
                  for each in params_codes}
        if check_macro_info(func_code):
            raise InternalServerError(f'found inconsistent data type on {func_code}')
        kwargs, msg = cast_macro_kwargs(func_code, kwargs)
        if msg:
            return HttpResponseCode.BAD_REQUEST.format(msg)
    except KeyError:
        raise BadRequest
    except ValueError:
        raise InternalServerError
    ret = []
    plot_infos = get_plot(func_code, kwargs)
    for pinfo in plot_infos:
        if pinfo.ptype == Ptype.CANDLE:
            cc_data = []
            for index, each in enumerate(pinfo.data):
                group = {
                        0: Ptype.OP.value,
                        1: Ptype.HP.value,
                        2: Ptype.LP.value,
                        3: Ptype.CP.value
                    }[index]
                cc_data.extend([{
                        "group": group,
                        "index": idx,
                        "value": value
                    } for idx, value in enumerate(each.tolist())]
                )
            ret.append({
                    "figType": pinfo.ptype.value,
                    "figName": pinfo.title,
                    "figData": cc_data
                })
        else:
            ret.append({
                "figType": pinfo.ptype.value,
                "figName": pinfo.title,
                "figData": [{
                    "group": pinfo.ptype.value,
                    "index": idx,
                    "value": value
                } for idx, value in enumerate(pinfo.data.tolist())]
            })
    return HttpResponseCode.OK.format(ret)

@app.route('/pattern/draft/occurdates', methods=['POST'])
def api_get_pattern_draft_date():
    """
    取得 pattern 草稿發生日期
    ---
    tags:
      - Studio
    parameters:
      - name: request
        in: body
        type: object
        properties:
          funcCode:
            type: string
          paramCodes:
            type: array
            items:
              type: object
              properties:
                paramCode:
                  type: string
                paramValue:
                  type: string
          marketCode:
            type: string
          startDate:
            type: string
            format: date
          endDate:
            type: string
            format: date
    responses:
      200:
        description: 成功取得
        schema:
          type: object
          properties:
            status:
              type: integer
            message:
              type: string
            data:
              type: array
              items:
                type: string
                format: date
    """
    logging.info(f"api_pattern_get_frame receiving: {request.json}")
    try:
        data = request.json
        func_code = data['funcCode']
        params_codes = data['paramCodes']
        market_code = data['marketCode']
        kwargs = {each["paramCode"]: each["paramValue"]
                  for each in params_codes}
        start_date = data.get('startDate')
        end_date = data.get('endDate')
        if start_date is not None:
            start_date = datetime.datetime.strptime(
                start_date, "%Y-%m-%d").date()
        if end_date is not None:
            end_date = datetime.datetime.strptime(
                end_date, "%Y-%m-%d").date()
        if check_macro_info(func_code):
            raise InternalServerError(f'found inconsistent data type on {func_code}')
        print(kwargs)
        kwargs, msg = cast_macro_kwargs(func_code, kwargs)
        if msg:
            return HttpResponseCode.BAD_REQUEST.format(msg)
    except KeyError:
        raise BadRequest
    except ValueError:
        raise InternalServerError
    ret = get_draft_date(func_code, kwargs, market_code, start_date, end_date)
    ret = [datetime.datetime.strftime(i, '%Y-%m-%d') for i in ret]
    return HttpResponseCode.OK.format(ret)


@app.errorhandler(MethodNotAllowed)
def handle_not_allow_request(e):
    logging.error(traceback.format_exc())
    return HttpResponseCode.METHOD_NOT_ALLOWED.format()


@app.errorhandler(NotFound)
def handle_not_allow_request(e):
    logging.error(traceback.format_exc())
    return HttpResponseCode.NOT_FOUND.format()


@app.errorhandler(InternalServerError)
def handle_internal_server_error(e):
    logging.error(traceback.format_exc())
    return HttpResponseCode.INTERNAL_SERVER_ERROR.format()


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    logging.error(traceback.format_exc())
    return HttpResponseCode.BAD_REQUEST.format()


if __name__ == '__main__':
    mode = args.mode
    exec_mode = ExecMode.get(mode)
    if exec_mode is None:
        logging.error(f'invalid execution mode {mode}')
        raise RuntimeError(f'invalid execution mode {mode}')
    if exec_mode == ExecMode.PROD.value or exec_mode == ExecMode.UAT.value:
        warnings.filterwarnings("ignore")
    if not os.path.exists(LOG_LOC):
        os.mkdir(LOG_LOC)
    try:
        err_hdlr = logging.StreamHandler(stream=sys.stderr)
        err_hdlr.setLevel(logging.ERROR)
        info_hdlr = logging.StreamHandler(stream=sys.stdout)
        info_hdlr.setLevel(logging.INFO)
        file_hdlr = handlers.TimedRotatingFileHandler(
            filename=f'{LOG_LOC}/app.log', when='D', backupCount=7)
        fmt = '%(asctime)s.%(msecs)03d - %(levelname)s - %(threadName)s - %(filename)s - line %(lineno)d: %(message)s'
        level = {ExecMode.DEV.value: logging.DEBUG,
                 ExecMode.UAT.value: logging.INFO,
                 ExecMode.PROD.value: logging.ERROR}[exec_mode]
        file_hdlr.setLevel(level)
        logging.basicConfig(level=0, format=fmt, handlers=[
                            err_hdlr, info_hdlr, file_hdlr], datefmt='%Y-%m-%d %H:%M:%S')
        task_queue.start()
        set_db(MimosaDB(mode=exec_mode))

    except Exception:
        logging.error("setting up failed")
        logging.error(traceback.format_exc())
    if (not args.motionless) and (not args.batchless):
        init_db()
    if (not args.motionless):
        serve(app, port=PORT, threads=10, )
