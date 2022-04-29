# -*- coding: utf-8 -*-
"""
Created on Tue March 8 17:07:36 2021

@author: Jeff

WSGI python server
--------------------------------------------------------------

Server build using `init_db` to create local database when

detecting none. When building local database, pattern update

and model recovery are executed.


When server fails, restarting will automatically run batch,

while api call `api_batch` could terminate this batch run.

Batch api call could fail to execute if the previous controller

has not yet being released.
"""
import argparse
from concurrent.futures import thread
import os
import traceback
import threading as mt
import sys
from func._td._db import set_market_data_provider
from model import (set_db, batch, init_db, get_pattern_occur, get_mix_pattern_occur,
                   get_mix_pattern_mkt_dist_info, get_mix_pattern_rise_prob, get_mix_pattern_occur_cnt,
                   get_pattern_mkt_dist_info, get_pattern_rise_prob, get_pattern_occur_cnt, get_market_price_dates,
                   get_market_rise_prob, get_mkt_dist_info, set_exec_mode, add_pattern, add_model,
                   remove_model, MarketDataFromDb, model_queue, pattern_queue)
from const import ExecMode, PORT, LOG_LOC, MODEL_QUEUE_LIMIT, PATTERN_QUEUE_LIMIT, MarketPeriodField
import datetime
from dao import MimosaDB
from flask import Flask, request
from flasgger import Swagger
from waitress import serve
from werkzeug.exceptions import MethodNotAllowed, NotFound, InternalServerError
import logging
from logging import handlers
import json
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
    model_queue.cut_line(batch, size=MODEL_QUEUE_LIMIT)
    pattern_queue.cut_line(batch, size=PATTERN_QUEUE_LIMIT)
    return {"status": 202, "message": "accepted", "data": None}


@app.route("/model", methods=["POST"])
def api_add_model():
    """
    新增模型
    ---
    tags:
      - 前台
    parameters:
      - name: request
        in: body
        type: object
        properties:
          modelId:
            type: string
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
    try:
        logging.info(f"api_add_model  receiving: {request.json}")
        data = request.json
        model_id = data['modelId']
    except KeyError as esp:
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    model_queue.push(add_model, size=1, args=(model_id,))
    return {"status": 202, "message": "accepted", "data": None}


@app.route("/model", methods=["DELETE"])
def api_remove_model():
    """
    移除模型
    ---
    tags:
      - 前台
    parameters:
      - name: request
        in: body
        type: object
        properties:
          modelId:
            type: string
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
    try:
        logging.info(f"api_remove_model  receiving: {request.json}")
        data = request.json
        model_id = data['modelId']
    except KeyError as esp:
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    t = mt.Thread(target=remove_model, args=(model_id,))
    t.start()
    return {"status": 202, "message": "accepted", "data": None}


@app.route("/pattern/compound/dates", methods=["POST"])
def api_get_compound_pattern_dates():
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
          marketId:
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
                occurDates:
                  type: array
                  items:
                    type: string
                    format: date
    """
    try:
        logging.info(f"api_get_compound_pattern_dates receiving: {request.json}")
        data = request.json
        patterns = data['patterns']
        market_id = data['marketId']
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    try:
        ret = get_mix_pattern_occur(market_id, patterns)
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status": 200, "message": "OK", "data": {"occurDates": ret}}


@app.route("/pattern/dates", methods=["POST"])
def api_get_pattern_dates():
    """
    取得指定現象歷史發生日期
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
          marketId:
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
                occurDates:
                  type: array
                  items:
                    type: string
                    format: date
    """
    try:
        logging.info(f"api_get_pattern_dates receiving: {request.json}")
        data = request.json
        pattern_id = data['patternId']
        market_id = data['marketId']
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    try:
        ret = get_pattern_occur(market_id, pattern_id)
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status": 200, "message": "OK", "data": {"occurDates": ret}}


@app.route("/pattern/compound/count", methods=["POST"])
def api_get_compound_pattern_count():
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
          patterns:
            type: array
            items:
              type: string
          marketType:
            type: string
          categoryCode:
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
                occurCnt:
                  type: integer
                nonOccurCnt:
                  type: integer
    """
    try:
        logging.info(
            f"api_get_compound_pattern_count receiving: {request.json}")
        data = request.json
        patterns = data['patterns']
        market_type = data.get('marketType')
        category_code = data.get('categoryCode')
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    try:
        occur, non_occur = get_mix_pattern_occur_cnt(
            patterns, market_type, category_code)

    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status": 200, "message": "OK", "data": {"occurCnt": int(occur), "nonOccurCnt": int(non_occur)}}


@app.route("/pattern/count", methods=["GET"])
def api_get_pattern_count():
    """
    取得指定現象發生次數
    ---
    tags:
      - 前台
    parameters:
      - name: request
        in: body
        type: object
        properties:
          patterns:
            type: array
            items:
              type: string
          marketType:
            type: string
          categoryCode:
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
                occurCnt:
                  type: integer
                nonOccurCnt:
                  type: integer
    """
    try:
        logging.info(f"api_get_pattern_count receiving: {request.args}")
        data = request.args
        pattern_id = data['patternId']
        market_type = data.get('marketType') or None
        category_code = data.get('categoryCode') or None
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    try:
        occur, non_occur = get_pattern_occur_cnt(
            pattern_id, market_type, category_code)

    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status": 200, "message": "OK", "data": {"occurCnt": int(occur), "nonOccurCnt": int(non_occur)}}


@app.route("/pattern/compound/upprob", methods=["POST"])
def api_get_compound_pattern_upprob():
    """
    取得複合現象上漲機率
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
          marketType:
            type: string
          categoryCode:
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
        logging.info(
            f"api_get_compound_pattern_upprob receiving: {request.json}")
        data = request.json
        patterns = data['patterns']
        date_period = data['datePeriod']
        market_type = data.get('marketType')
        category_code = data.get('categoryCode')
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    try:
        ret = get_mix_pattern_rise_prob(
            patterns, date_period, market_type, category_code)
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status": 200, "message": "OK", "data": {"positiveWeight": float(ret)}}


@app.route("/pattern/upprob", methods=["GET"])
def api_get_pattern_upprob():
    """
    取得指定現象上漲機率
    ---
    tags:
      - 前台
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
          marketType:
            type: string
          categoryCode:
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
        logging.info(f"api_get_pattern_upprob receiving: {request.args}")
        data = request.args
        pattern_id = data['patternId']
        date_period = data['datePeriod']
        market_type = data.get('marketType') or None
        category_code = data.get('categoryCode') or None
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    try:
        ret = get_pattern_rise_prob(pattern_id, int(
            date_period), market_type, category_code)
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status": 200, "message": "OK", "data": {"positiveWeight": float(ret)}}


@app.route("/pattern/compound/distribution", methods=["POST"])
def api_get_compound_pattern_distribution():
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
          marketType:
            type: string
          categoryCode:
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
                  marketCode:
                    type: string
                  returnMean:
                    type: number
                  returnStd:
                    type: number
                  samples:
                    type: integer
    """
    try:
        logging.info(
            f"api_get_compound_pattern_distribution receiving: {request.json}")
        data = request.json
        patterns = data['patterns']
        date_period = data['datePeriod']
        market_type = data.get('marketType')
        category_code = data.get('categoryCode')
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    try:
        ret = get_mix_pattern_mkt_dist_info(
            patterns, date_period, market_type, category_code)
        ret = [{"marketCode": key, "returnMean": float(mean), "returnStd": float(
            std), "samples": int(counts)} for key, (mean, std, counts) in ret.items()]
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status": 200, "message": "OK", "data": ret}


@app.route("/pattern/distribution", methods=["GET"])
def api_get_pattern_distribution():
    """
    取得指定現象分布資訊
    ---
    tags:
      - 前台
    parameters:
      - name: patternId
        in: query
        type: string
        required: true
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
              type: array
              items:
                type: object
                properties:
                  marketCode:
                    type: string
                  returnMean:
                    type: number
                  returnStd:
                    type: number
                  samples:
                    type: integer
    """
    try:
        logging.info(f"api_get_pattern_distribution receiving: {request.args}")
        data = request.args
        pattern_id = data['patternId']
        date_period = data['datePeriod']
        market_type = data.get('marketType') or None
        category_code = data.get('categoryCode') or None
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    try:
        ret = get_pattern_mkt_dist_info(pattern_id, int(
            date_period), market_type, category_code)
        ret = [{"marketCode": key, "returnMean": float(mean), "returnStd": float(
            std), "samples": int(counts)} for key, (mean, std, counts) in ret.items()]
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status": 200, "message": "OK", "data": ret}


@app.route("/patterns/<string:patternId>", methods=["GET"])
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
    pattern_queue.push(add_pattern, size=1, args=(patternId,))
    return {"status": 202, "message": "accepted", "data": None}


@app.route("/markets/upprob", methods=["GET"])
def api_market_upprob():
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
        logging.info(f"api_market_upprob receiving: {request.args}")
        data = request.args
        date_period = data["datePeriod"]
        market_type = data.get("marketType") or None
        category_code = data.get("categoryCode") or None
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    ret = get_market_rise_prob(int(date_period), market_type, category_code)
    return {"status": 200, "message": "OK", "data": {"positiveWeight": float(ret)}}


@app.route("/markets/distribution", methods=["GET"])
def api_market_distribution():
    """
    取得指定市場集分布資訊
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
              type: array
              items:
                type: object
                properties:
                  marketCode:
                    type: string
                  returnMean:
                    type: number
                  returnStd:
                    type: number
                  samples:
                    type: integer
    """
    try:
        logging.info(f"api_market_distribution receiving: {request.args}")
        data = request.args
        date_period = data["datePeriod"]
        market_type = data.get("marketType") or None
        category_code = data.get("categoryCode") or None
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    ret = get_mkt_dist_info(int(date_period), market_type, category_code)
    ret = [{"marketCode": key, "returnMean": float(mean), "returnStd": float(
        std), "samples": int(counts)} for key, (mean, std, counts) in ret.items()]
    return {"status": 200, "message": "OK", "data": ret}


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
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    except:
        logging.error(traceback.format_exc())
        return {"status": 400,
                "message": "Invalid request argument",
                "data": None}
    ret = get_market_price_dates(marketId, start_date)
    ret = [{"dataDate": each[MarketPeriodField.DATA_DATE.value].strftime('%Y-%m-%d'),
            "priceDate":each[MarketPeriodField.PRICE_DATE.value].strftime('%Y-%m-%d'), 
            "datePeriod":each[MarketPeriodField.DATE_PERIOD.value],} for each in ret]
    return {"status": 200, "message": "OK", "data": ret}


@app.errorhandler(MethodNotAllowed)
def handle_not_allow_request(e):
    return {"status": 405, "message": "Method not allowed", "data": None}


@app.errorhandler(NotFound)
def handle_not_allow_request(e):
    return {"status": 404, "message": "Not Found", "data": None}


@app.errorhandler(InternalServerError)
def handle_internal_server_error(e):
    logging.error(traceback.format_exc())
    return {"status": 500, "message": "internal server error", "data": None}


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
        fmt = '%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s - line %(lineno)d: %(message)s'
        level = {ExecMode.DEV.value: logging.DEBUG,
                 ExecMode.UAT.value: logging.INFO,
                 ExecMode.PROD.value: logging.ERROR}[exec_mode]
        file_hdlr.setLevel(level)
        logging.basicConfig(level=0, format=fmt, handlers=[err_hdlr, info_hdlr, file_hdlr])
        model_queue.start()
        pattern_queue.start()
        set_exec_mode(exec_mode)
        set_db(MimosaDB(mode=exec_mode))
        set_market_data_provider(MarketDataFromDb())

    except Exception as esp:
        logging.error(f"setting up failed")
        logging.error(traceback.format_exc())
    if (not args.motionless) and (not args.batchless):
        t = mt.Thread(target=init_db)
        t.start()
    if (not args.motionless):
        serve(app, port=PORT, threads=10)
