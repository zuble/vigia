''' PATH VARS '''
BASE_VIGIA_DIR = "/raid/DATASETS/.zuble/vigia"

## with time stats wrong
SERVER_TRAIN_PATH = '/raid/DATASETS/anomaly/XD_Violence/training/'
SERVER_TEST_PATH = '/raid/DATASETS/anomaly/XD_Violence/testing'

## with time stats right
SERVER_TRAIN_COPY_PATH = '/raid/DATASETS/anomaly/XD_Violence/training_copy'
SERVER_TEST_COPY_PATH =  '/raid/DATASETS/anomaly/XD_Violence/testing_copy'

## alter cut 
SERVER_TRAIN_COPY_ALTER_PATH1 = "/raid/DATASETS/anomaly/XD_Violence/training_copy_alter"
SERVER_TRAIN_COPY_ALTER_PATH2 = SERVER_TRAIN_COPY_ALTER_PATH1 + '/CUT'



MODEL_PATH = BASE_VIGIA_DIR+'/zuwav11/model/model/'
CKPT_PATH = BASE_VIGIA_DIR+'/zuwav11/model/ckpt/'
HIST_PATH = BASE_VIGIA_DIR+'/zuwav11/model/hist/'
RSLT_PATH = BASE_VIGIA_DIR+'/zuwav11/model/rslt/'
WEIGHTS_PATH = BASE_VIGIA_DIR+'/zuwav11/model/weights/'