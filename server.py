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
import os
import traceback
import threading as mt
import datetime
import time
from func._td._db import set_market_data_provider
from model import (set_db, batch, init_db,
 add_model, remove_model, MarketDataFromDb, MT_MANAGER)
from const import BATCH_EXE_CODE, ExecMode, PORT, LOG_LOC
from dao import MimosaDB
from flask import Flask, request
from waitress import serve
from werkzeug.exceptions import MethodNotAllowed, NotFound
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
    MT_MANAGER.acquire(BATCH_EXE_CODE).switch_off()
    MT_MANAGER.release(BATCH_EXE_CODE)
    while MT_MANAGER.exists(BATCH_EXE_CODE):
        time.sleep(1)
    excute_id = datetime.datetime.now()

    t = mt.Thread(target=batch, args=(excute_id,))
    t.start()

    return {"status":202, "message":"accepted", "data":None}

@app.route("/model", methods=["POST"])
def api_add_model():
    """ Create new model. """
    excute_id = datetime.datetime.now()
    data = json.loads(request.data)
    logging.info(f"api_add_model {excute_id} start, receiving: {data}")
    try:
        model_id = data['modelId']
    except KeyError as esp:
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    def _add_model(model_id):
        add_model(model_id)
        logging.info(f"api_add_model {excute_id} complete")
        return
    t = mt.Thread(target=_add_model, args=(model_id,))
    t.start()
    return {"status":202, "message":"accepted", "data":data}

@app.route("/model", methods=["DELETE"])
def api_remove_model():
    """ Remove model. """
    excute_id = datetime.datetime.now()
    data = json.loads(request.data)
    logging.info(f"api_remove_model {excute_id} receiving: {data}")
    try:
        model_id = data['modelId']
    except KeyError as esp:
        return {"status":400,
                "message":"Invalid request argument",
                "data":None}
    def _remove_model(model_id):
        remove_model(model_id)
        logging.info(f"api_remove_model {excute_id} complete")
        return
    t = mt.Thread(target=_remove_model, args=(model_id,))
    t.start()
    return {"status":202, "message":"accepted", "data":data}

@app.errorhandler(MethodNotAllowed)
def handle_not_allow_request(e):
    return {"status":405, "message":"Method not allowed", "data":None}

@app.errorhandler(NotFound)
def handle_not_allow_request(e):
    return {"status":404, "message":"Not Found", "data":None}

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
        file_hdlr = handlers.TimedRotatingFileHandler(filename=f'{LOG_LOC}/.log', when='D', backupCount=7)
        level = {ExecMode.DEV.value:logging.DEBUG, ExecMode.UAT.value:logging.INFO, ExecMode.PROD.value:logging.ERROR}[ExecMode.get(mode)]
        logging.basicConfig(level=level, format='%(asctime)s - %(threadName)s: %(filename)s - line %(lineno)d: %(message)s', handlers=[stream_hdlr, file_hdlr])
        set_db(MimosaDB(mode=mode))
        set_market_data_provider(MarketDataFromDb())
    except Exception as esp:
        logging.error(f"setting up failed")
        logging.error(traceback.format_exc())     
    if (not args.motionless) and (not args.batchless):
        t = mt.Thread(target=init_db)
        t.start()
    if (not args.motionless):
        serve(app, port=PORT)
    