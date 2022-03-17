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

logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)   
app =  Flask(__name__)



@app.route("/model/batch", methods=["GET"])
def api_batch():
    """ Run batch. """
    MT_MANAGER.acquire(BATCH_EXE_CODE).switch_off()
    MT_MANAGER.release(BATCH_EXE_CODE)
    time.sleep(10)
    excute_id = datetime.datetime.now()
    logger.info(f"api_batch {excute_id} start")

    t = mt.Thread(target=batch, arg=(excute_id, logger))
    t.start()
    
    return {"status":"ok"}

@app.route("/model/", methods=["POST"])
def api_add_model():
    """ Create new model. """
    excute_id = datetime.datetime.now()
    logger.info(f"api_add_model {excute_id} start, receiving: {request.form}")
    model_id = request.form.get('modelId')
    def _add_model(model_id):
        add_model(model_id)
        logger.info(f"api_add_model {excute_id} complete")
        return
    t = mt.Thread(target=_add_model, args=(model_id,))
    t.start()
    return {"status":"ok", "model_id":model_id}

@app.route("/model/removal", methods=["POST"])
def api_remove_model():
    """ Remove model. """
    excute_id = datetime.datetime.now()
    logger.info(f"api_remove_model {excute_id} receiving: {request.form}")
    model_id = request.form.get('modelId')
    def _remove_model(model_id):
        remove_model(model_id)
        logger.info(f"api_remove_model {excute_id} complete")
        return
    t = mt.Thread(target=_remove_model, args=(model_id,))
    t.start()
    return {"status":"ok", "model_id":model_id}

if __name__ == '__main__':
    set_db(MimosaDB())
    set_market_data_provider(MarketDataFromDb())
    if not os.path.exists('./_local_db'):
        init_db()
    excute_id = datetime.datetime.now()
    t = mt.Thread(target=batch, args=(excute_id, logger))
    t.start()
    serve(app, port=5000)
    