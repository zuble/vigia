import os , argparse , time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"


## DATASETS
ucf101_root = '/raid/DATASETS/anomaly/UCF101'
ucf101 = {
    'train' : ucf101_root+'/train' ,
    'test' : ucf101_root+'/test' ,
}

minisports1m_root = '/media/jtstudents/T77/miniSports1M/videos_256p_dense_cache'
minisports1m = {
    'train' : minisports1m_root+'/train' ,
    'test' : minisports1m_root+'/test' ,
}

mitv2_root = '/'
mitv2={}

kinetics400_root = '/media/jtstudents/T77/kinetics400'
kinetics400 = {
    'train' : kinetics400_root+'/videos_train' ,
    'test' : kinetics400_root+'/videos_val' ,    
}


####################################################################################################
parser = argparse.ArgumentParser(description='zuFE')
parser.add_argument('--dummy', type=int , default=512)

parser.add_argument('--ds',  default = kinetics400, choices=[ucf101 , minisports1m , mitv2 , kinetics400])
parser.add_argument('--backbone', type=str , default='mobilenetv2' , choices=['xception','mobilenetv2'])

parser.add_argument('--epochs', type=int , default=1 )

ARGS = parser.parse_args(args=[])
####################################################################################################


ds = [name for name, value in locals().items() if value is ARGS.ds][0]
in_shapes = {
    "mobilenetv2":  (224, 224, 3),  #(224, 224, 3)
    "xception":     (299, 299, 3)   #(299, 299, 3)
}
out_nclasses = {
    'ucf101' : 101,
    'minisports1m' : 0,
    'mitv2' : 0 , 
    'kinetics400' : 400 
}
CFG_TRAIN = { 
    "backbone" : ARGS.backbone,
    "in_shape": in_shapes[ARGS.backbone],
    "in_height":in_shapes[ARGS.backbone][0],
    "in_width": in_shapes[ARGS.backbone][1],
    "out_nclasses" : out_nclasses[ds],

    "optima": 'adam',
    "lr" : 1e-5,
    
    "batch_size": 8,
    "frame_max": 16,
    "frame_step": 1,
    "transform": False
}  

CFG_TEST = {
    "backbone" : ARGS.backbone,
    "in_shape": in_shapes[ARGS.backbone],
    "in_height":in_shapes[ARGS.backbone][0],
    "in_width": in_shapes[ARGS.backbone][1],
    "out_nclasses" : out_nclasses[ds],
    
    "batch_size": 1,
    "frame_max": 16, 
    "frame_step": 1,
    "transform": False
}  


## INIT MODEL NAME AND FOLDERING
MODEL_NAME = "{:.4f}_{}-{}_{}_{}-{}".format(time.time(),ds,CFG_TRAIN["frame_max"],CFG_TRAIN["backbone"],CFG_TRAIN["optima"],CFG_TRAIN["lr"]);print(MODEL_NAME)
BASE_MODEL_PATH = os.path.join('model/',ds,MODEL_NAME)
MODEL_PATH = BASE_MODEL_PATH+'/'+MODEL_NAME+'.h5'
WEIGHTS_PATH = os.path.join(BASE_MODEL_PATH,'weights')
LOG_PATH = os.path.join(BASE_MODEL_PATH,'log')

def init():
    print("\nINIT MODEL FOLDER @",BASE_MODEL_PATH)
    if not os.path.exists(BASE_MODEL_PATH):
        os.makedirs(BASE_MODEL_PATH)
        os.makedirs(WEIGHTS_PATH)
        os.makedirs(LOG_PATH)
    else: raise Exception(f"{BASE_MODEL_PATH} eristes")
    


##################################################################
## dataloader.py experiment
#
# configuration of the data layer
#
DATA_FRAME_RATE = 1         # the rate at which to sample the input
DATA_TEMP_DURATION = 16     # the temporal duration or number of frames of the input
DATA_NUM_INPUT_CHANNELS = 3 # the number of channels in the input
# the minimum and maximum scale for image resize operation
DATA_TRAIN_JITTER_SCALES = [182, 228]
# the spatial resolution of the input
DATA_TRAIN_CROP_SIZE = 112
DATA_TEST_CROP_SIZE = 160
# The mean value of the video raw pixels across the R G B channels.
DATA_MEAN = [0.45, 0.45, 0.45]
# The standard deviation of the video raw pixels across the R G B channels.
DATA_STD = [0.225, 0.225, 0.225]

#
# configuration for training
#
# number of examples in the training set
TRAIN_DATASET_SIZE = 240436
# batch size
TRAIN_BATCH_SIZE = 1
# number of training epochs
TRAIN_EPOCHS = 1
# loss function
TRAIN_OPTIMIZER = "SGD"
# momentum for optimizer
TRAIN_MOMENTUM = 0.9
# base learning rate
TRAIN_BASE_LR = 0.1
# number of training epochs to warm up
TRAIN_WARMUP_EPOCHS = 1
# initial learning rate during warmup
TRAIN_WARMUP_LR = 0.01

#
# configuration for inference
#
# number of spatial crops
TEST_NUM_SPATIAL_CROPS = 3
# number of temporal views
TEST_NUM_TEMPORAL_VIEWS = 1
# batch size
TEST_BATCH_SIZE = 1