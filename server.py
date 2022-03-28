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
import threading as mt
import logging
import datetime
import time
from func._td._db import set_market_data_provider
from model import (set_db, batch, init_db,
 add_model, remove_model, MarketDataFromDb, MT_MANAGER)
from const import BATCH_EXE_CODE
from dao import MimosaDB
from flask import Flask, request
from waitress import serve
from werkzeug.exceptions import MethodNotAllowed, NotFound
import json

logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)   
app =  Flask(__name__)

parser = argparse.ArgumentParser(prog="Program start server")
parser.add_argument("--init", action='store_true')
parser.add_argument("--motionless", "-ml", action='store_true', default=False)
parser.add_argument("--batchless", "-bl", action='store_true', default=False)
args = parser.parse_args()

@app.route("/model/batch", methods=["GET"])
def api_batch():
    """ Run batch. """
    MT_MANAGER.acquire(BATCH_EXE_CODE).switch_off()
    MT_MANAGER.release(BATCH_EXE_CODE)
    time.sleep(10)
    excute_id = datetime.datetime.now()

    t = mt.Thread(target=batch, arg=(excute_id, logger))
    t.start()
    
    return {"status":202, "message":"accepted", "data":None}

@app.route("/model", methods=["POST"])
def api_add_model():
    """ Create new model. """
    excute_id = datetime.datetime.now()
    data = json.loads(request.data)
    logger.info(f"api_add_model {excute_id} start, receiving: {data}")
    try:
        model_id = data['modelId']
    except KeyError as esp:
        return {"status":400, 
                "message":"Invalid request argument", 
                "data":None}
    def _add_model(model_id):
        add_model(model_id)
        logger.info(f"api_add_model {excute_id} complete")
        return
    t = mt.Thread(target=_add_model, args=(model_id,))
    t.start()
    return {"status":202, "message":"accepted", "data":data}

@app.route("/model", methods=["DELETE"])
def api_remove_model():
    """ Remove model. """
    excute_id = datetime.datetime.now()
    data = json.loads(request.data)
    logger.info(f"api_remove_model {excute_id} receiving: {data}")
    try:
        model_id = data['modelId']
    except KeyError as esp:
        return {"status":400, 
                "message":"Invalid request argument", 
                "data":None}
    def _remove_model(model_id):
        remove_model(model_id)
        logger.info(f"api_remove_model {excute_id} complete")
        return
    t = mt.Thread(target=_remove_model, args=(model_id,))
    t.start()
    return {"status":202, "message":"accepted", "data":data}

@app.errorhandler(MethodNotAllowed)
def handle_not_allow_request(e):
    return {"status":405, "message":"Method not allowed", "data":None}

@app.errorhandler(NotFound)
def handle_not_allow_request(e):
    return {"status":404, "message":"Not Found", "data":json.loads(request.data)}

if __name__ == '__main__':
    set_db(MimosaDB())
    set_market_data_provider(MarketDataFromDb())
    if not os.path.exists('./_local_db') or args.init:
        init_db()
    if (not args.motionless) and (not args.batchless):
        excute_id = datetime.datetime.now()
        t = mt.Thread(target=batch, args=(excute_id, logger))
        t.start()
    if not args.motionless:
        serve(app, port=5000)
    