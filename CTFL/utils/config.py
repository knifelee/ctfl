import numpy as np
import torch
import datetime
from utils.args import args

DATA_DIR = 'data/'

SPLIT = 0.1

SEED = args.seed

BATCH_SIZE = 128

StartTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
StartDate = datetime.date.today().strftime('%Y-%m-%d')

NUM_PARTS = 8

MACRO_THRES = 8  # more than X records to share a test credit

TRACE_THRES = 0.8
