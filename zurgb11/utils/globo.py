import os , json

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'


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


ZU_PATH = BASE_VIGIA_DIR+'/zurgb11'
MODEL_PATH = BASE_VIGIA_DIR+'/zurgb11/model/model/'
CKPT_PATH = BASE_VIGIA_DIR+'/zurgb11/model/ckpt/'
HIST_PATH = BASE_VIGIA_DIR+'/zurgb11/model/hist/'
RSLT_PATH = BASE_VIGIA_DIR+'/zurgb11/model/rslt/'
WEIGHTS_PATH = BASE_VIGIA_DIR+'/zurgb11/model/weights/'
LOGS_PATH = BASE_VIGIA_DIR+'/zurgb11/model/logs/'



''' CONFIGS '''
in_shapes = {
    "original":     (120, 160, 3),
    "resnet50":     (224, 224, 3),
    "mobilenetv2":  (224, 224, 3),  #(224, 224, 3)
    "xception":     (150, 150, 3)     #(299, 299, 3)
}
BACKBONE = 'mobilenetv2'  

CFG_RGB_TRAIN  = {
    "frame_step":4, # fstep 2 -> 12fps | fstep 3 -> 8fps | fstep 4 -> 6fps
    
    "backbone" : BACKBONE,
    "in_shape": in_shapes[BACKBONE],
    "in_height":in_shapes[BACKBONE][0],
    "in_width":in_shapes[BACKBONE][1],
    
    "batch_size":1,
    "augment":True,
    "shuffle":False,
    
    "ativa" : 'relu',
    "optima" : 'sgd',
    "lr" : 0.0002 ,
    "batch_type" : 0,   # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : 4000, # in relation to original video, NN takes (frame_max / fstep) frames at max
    "ckpt_start" : f"{0:0>8}",  #used in train_model: if 00000000 start from scratch, else start from ckpt with config stated
    
    "epochs" : 30 , # 30
    
    "debug" : True , 
    "dummy" : 0     ## if 0 use clbks and saves
}


CFG_WEIGHTS4TEST = {
    "backbone" : BACKBONE,
    "in_shape": in_shapes[BACKBONE],
    "in_height":in_shapes[BACKBONE][0],
    "in_width":in_shapes[BACKBONE][1],
    "ativa" : 'relu',
    "optima" : 'sgd',
}

CFG_RGB_TEST = {
    "batch_type" : 2, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : 1000,
    
    "debug" : False,
    "watch" : False,
    "dummy" : 10
}



################
YAMMET_PATH = '/raid/DATASETS/.zuble/vigia/zuwav00/audioset-yamet'
CFG_YAMMET = {
    'graph_filename' : YAMMET_PATH + '/audioset-yamnet-1.pb', 
    
    'metadata_file' : YAMMET_PATH + '/audioset-yamnet-1.json',
    'labels' :  json.load(open(YAMMET_PATH + '/audioset-yamnet-1.json', "r"))["classes"],
    
    'anom_labels_i' : [317,318,319,390,394,420,421,422,423,424,425,426,427,428,429,430],
    
    'audio_fs_input' : 16000,
    'full_or_max' : 'max'
}