import sys
import os
import argparse
import logging
import warnings
import io
import json
import subprocess

print('0', __file__)

# import cv2
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import math
import warnings
import numpy as np

warnings.filterwarnings("ignore",category=FutureWarning)

# sys.path.append(os.path.join(os.path.dirname(__file__), '/opt/ml/code/package'))

import pickle
# from io import StringIO
from timeit import default_timer as timer
from collections import Counter

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # from autogluon import ImageClassification as task

import flask

# import sagemaker
# from sagemaker import get_execution_role, local, Model, utils, fw_utils, s3
# import boto3
import tarfile

print('1', __file__)

# import pandas as pd
import model

prefix = '/opt/ml/'
# model_path = os.path.join(prefix, 'model')
model_path = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
# ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
# ctx = mx.cpu()

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

# The flask app for serving predictions
app = flask.Flask(__name__)

print('2', __file__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

print('3', __file__)

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    f_csv = "0000.csv"
    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
    elif flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        print(data)
        print('-')
        data = data.split('`')
        data = '\n'.join(data)
        print(data)
        with open(f_csv, "w") as fp:
            fp.write(data)
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

    # print('Invoked with {} records'.format(data.keys()))

    # Do the prediction
    outputs = model.predict(f_csv)
    # print(outputs)

    # out = io.StringIO()
    # out.write(output_dict_str)
    # result = out.getvalue()
    result = '\n'.join([','.join(out) for out in outputs])
    print(result)

    # return flask.Response(response=result, status=200, mimetype='application/json')
    return flask.Response(response=result, status=200, mimetype='text/csv')
