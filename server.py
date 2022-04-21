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
import time
from func._td._db import set_market_data_provider
from model import (set_db, batch, init_db, get_pattern_occur, get_mix_pattern_occur,
 get_mix_pattern_mkt_dist_info, get_mix_pattern_rise_prob, get_mix_pattern_occur_cnt,
 get_pattern_mkt_dist_info, get_pattern_rise_prob, get_pattern_occur_cnt,
 add_pattern, add_model, remove_model, MarketDataFromDb, model_queue, pattern_queue)
from const import ExecMode, PORT, LOG_LOC, MODEL_QUEUE_LIMIT, PATTERN_QUEUE_LIMIT
from dao import MimosaDB
from flask import Flask, request
from waitress import serve
from werkzeug.exceptions import MethodNotAllowed, NotFound, InternalServerError
import logging
from logging import handlers
import json
import warnings
app =  Flask(__name__)

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
    return {"status":202, "message":"accepted", "data":None}

@app.route("/model", methods=["POST"])
def api_add_model():
    """ Create new model. """
    try:
        logging.info(f"api_add_model  receiving: {request.data}")
        data = json.loads(request.data)
        model_id = data['modelId']
    except KeyError as esp:
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    model_queue.push(add_model, size=1, args=(model_id,))
    return {"status":202, "message":"accepted", "data":data}

@app.route("/model", methods=["DELETE"])
def api_remove_model():
    """ Remove model. """
    try:
        logging.info(f"api_remove_model  receiving: {request.data}")
        data = json.loads(request.data)
        model_id = data['modelId']
    except KeyError as esp:
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    t = mt.Thread(target=remove_model, args=(model_id,))
    t.start()
    return {"status":202, "message":"accepted", "data":None}

@app.route("/pattern/compound/dates", methods=["POST"])
def api_get_compound_pattern():
    try:
        logging.info(f"api_get_compound_pattern receiving: {request.data}")
        data = json.loads(request.data)
        patterns = data['patterns']
        market_id = data['marketId']
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    try:
        ret = get_mix_pattern_occur(market_id, patterns)
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status":200, "message":"OK", "data":{"occurDates":ret}}

@app.route("/pattern/dates", methods=["POST"])
def api_get_pattern():
    try:
        logging.info(f"api_get_pattern receiving: {request.data}")
        data = json.loads(request.data)
        pattern_id = data['patternId']
        market_id = data['marketId']
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    try:
        ret = get_pattern_occur(market_id, pattern_id)
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status":200, "message":"OK", "data":{"occurDates":ret}}

@app.route("/pattern/compound/count", methods=["POST"])
def api_get_compound_pattern_count():
    try:
        logging.info(f"api_get_compound_pattern_count receiving: {request.data}")
        data = json.loads(request.data)
        patterns = data['patterns']
        market_type = data.get('marketType')
        category_code = data.get('categoryCode')
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    try:
        occur, non_occur = get_mix_pattern_occur_cnt(patterns, market_type, category_code)

    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status":200, "message":"OK", "data":{"occurCnt":int(occur), "nonOccurCnt":int(non_occur)}}

@app.route("/pattern/count", methods=["POST"])
def api_get_pattern_count():
    try:
        logging.info(f"api_get_pattern_count receiving: {request.data}")
        data = json.loads(request.data)
        pattern_id = data['patternId']
        market_type = data.get('marketType')
        category_code = data.get('categoryCode')
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    try:
        occur, non_occur = get_pattern_occur_cnt(pattern_id, market_type, category_code)

    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status":200, "message":"OK", "data":{"occurCnt":int(occur), "nonOccurCnt":int(non_occur)}}

@app.route("/pattern/compound/upprob", methods=["POST"])
def api_get_compound_pattern_upprob():
    try:
        logging.info(f"api_get_compound_pattern_upprob receiving: {request.data}")
        data =  json.loads(request.data)
        patterns = data['patterns']
        date_period = data['datePeriod']
        market_type = data.get('marketType')
        category_code = data.get('categoryCode')
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    try:
        ret = get_mix_pattern_rise_prob(patterns, date_period, market_type, category_code)
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status":200, "message":"OK", "data":{"positiveWeight":float(ret)}}

@app.route("/pattern/upprob", methods=["POST"])
def api_get_pattern_upprob():
    try:
        logging.info(f"api_get_pattern_upprob receiving: {request.data}")
        data =  json.loads(request.data)
        pattern_id = data['patternId']
        date_period = data['datePeriod']
        market_type = data.get('marketType')
        category_code = data.get('categoryCode')
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    try:
        ret = get_pattern_rise_prob(pattern_id, date_period, market_type, category_code)
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status":200, "message":"OK", "data":{"positiveWeight":float(ret)}}

@app.route("/pattern/compound/distribution", methods=["POST"])
def api_get_compound_pattern_distribution():
    try:
        logging.info(f"api_get_compound_pattern_distribution receiving: {request.data}")
        data =  json.loads(request.data)
        patterns = data['patterns']
        date_period = data['datePeriod']
        market_type = data.get('marketType')
        category_code = data.get('categoryCode')
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    try:
        ret = get_mix_pattern_mkt_dist_info(patterns, date_period, market_type, category_code)
        ret = [{"marketCode":key, "returnMean":float(mean), "returnStd":float(std), "samples":int(counts)} for key, (mean, std, counts) in ret.items()]
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status":200, "message":"OK", "data":ret}

@app.route("/pattern/distribution", methods=["POST"])
def api_get_pattern_distribution():
    try:
        logging.info(f"api_get_pattern_distribution receiving: {request.data}")
        data =  json.loads(request.data)
        pattern_id = data['patternId']
        date_period = data['datePeriod']
        market_type = data.get('marketType')
        category_code = data.get('categoryCode')
    except Exception as esp:
        logging.error(traceback.format_exc())
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    try:
        ret = get_pattern_mkt_dist_info(pattern_id, date_period, market_type, category_code)
        ret = [{"marketCode":key, "returnMean":float(mean), "returnStd":float(std), "samples":int(counts)} for key, (mean, std, counts) in ret.items()]
    except Exception as esp:
        logging.error(traceback.format_exc())
        raise Exception
    return {"status":200, "message":"OK", "data":ret}

@app.route("/patterns/<string:pattern_id>", methods=["GET"])
def api_add_pattern(pattern_id):
    pattern_queue.push(add_pattern, size=1, args=(pattern_id,))
    return {"status":202, "message":"accepted", "data":None}

@app.route("/test", methods=["GET"])
def api_test_queue():
    def test_queue():
        for each in range(10):
            time.sleep(1)
            logging.info(each)
    model_queue.push(test_queue, size=1)
    return {"status":200, "message":"OK", "data":None}


@app.errorhandler(MethodNotAllowed)
def handle_not_allow_request(e):
    return {"status":405, "message":"Method not allowed", "data":None}

@app.errorhandler(NotFound)
def handle_not_allow_request(e):
    return {"status":404, "message":"Not Found", "data":None}

@app.errorhandler(InternalServerError)
def handle_internal_server_error(e):
    logging.error(traceback.format_exc())
    return {"status":500, "message":"internal server error", "data":None}

if __name__ == '__main__':
    mode = args.mode
    if ExecMode.get(mode) is None:
        logging.error(f'invalid execution mode {mode}')
        raise RuntimeError(f'invalid execution mode {mode}')
    if ExecMode.get(mode) == ExecMode.PROD.value or ExecMode.get(mode) == ExecMode.UAT.value:
        warnings.filterwarnings("ignore")
    if not os.path.exists(LOG_LOC):
        os.mkdir(LOG_LOC)
    try:
        stream_hdlr = logging.StreamHandler()
        sys_hdlr = handlers.SysLogHandler()
        file_hdlr = handlers.TimedRotatingFileHandler(filename=f'{LOG_LOC}/.log', when='D', backupCount=7)
        fmt = '%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s - line %(lineno)d: %(message)s'
        level = {ExecMode.DEV.value:logging.DEBUG,
                 ExecMode.UAT.value:logging.INFO,
                 ExecMode.PROD.value:logging.ERROR}[ExecMode.get(mode)]
        logging.basicConfig(level=level, format=fmt, handlers=[stream_hdlr, file_hdlr, sys_hdlr])
        model_queue.start()
        pattern_queue.start()
        set_db(MimosaDB(mode=ExecMode.get(mode)))
        set_market_data_provider(MarketDataFromDb())

    except Exception as esp:
        logging.error(f"setting up failed")
        logging.error(traceback.format_exc())     
    if (not args.motionless) and (not args.batchless):
        t = mt.Thread(target=init_db)
        t.start()
    if (not args.motionless):
        serve(app, port=PORT, threads=10)
    