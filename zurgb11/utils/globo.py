import os

## docker run -e DOCKER_RUN=1 (...)
DOCKER_RUN = os.environ.get('DOCKER_RUN')

if DOCKER_RUN:
    BASE_VIGIA_DIR = "/home/vigia"
    BASE_XDV_DIR = "/home/anomaly"
else:
    BASE_VIGIA_DIR = "/raid/DATASETS/.zuble/vigia"
    BASE_XDV_DIR = "/raid/DATASETS/anomaly"

print(BASE_VIGIA_DIR , BASE_XDV_DIR)


## with time stats wrong
SERVER_TRAIN_PATH = BASE_XDV_DIR+'/XD_Violence/training/'
SERVER_TEST_PATH = BASE_XDV_DIR+'/XD_Violence/testing'

## with time stats right
SERVER_TRAIN_COPY_PATH = BASE_XDV_DIR+'/XD_Violence/training_copy'
SERVER_TEST_COPY_PATH =  BASE_XDV_DIR+'/XD_Violence/testing_copy'

## alter cut 
SERVER_TRAIN_COPY_ALTER_PATH1 = BASE_XDV_DIR+'/XD_Violence/training_copy_alter'
SERVER_TRAIN_COPY_ALTER_PATH2 = SERVER_TRAIN_COPY_ALTER_PATH1 + '/CUT'



MODEL_PATH = BASE_VIGIA_DIR+'/zurgb11/model/model/'
CKPT_PATH = BASE_VIGIA_DIR+'/zurgb11/model/ckpt/'
HIST_PATH = BASE_VIGIA_DIR+'/zurgb11/model/hist/'
RSLT_PATH = BASE_VIGIA_DIR+'/zurgb11/model/rslt/'
WEIGHTS_PATH = BASE_VIGIA_DIR+'/zurgb11/model/weights/'
LOGS_PATH = BASE_VIGIA_DIR+'/zurgb11/model/logs/'