import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

''' UCFCRIME '''

## video paths lists
UCFCRIME_VPATHS_LISTS = {
    "train_normal" : '../.lists_ucfcrime/train_normal.list' ,
    "train_abnormal" : '../.lists_ucfcrime/train_abnormal.list' ,
    "test" : '../.lists_ucfcrime/test.list'
}

UCFCRIME_FEATC3D_BASE_DIR = '/raid/DATASETS/anomaly/UCF_Crimes/features/C3DSPORTS1M/'
UCFCRIME_FEATC3D = {
    "train_normal" : UCFCRIME_FEATC3D_BASE_DIR + 'train_normal' , 
    "train_abnormal" : UCFCRIME_FEATC3D_BASE_DIR + 'train_abnormal' , 
    "test" : UCFCRIME_FEATC3D_BASE_DIR + 'test'    
}

C3DSPORTS1M_WEIGHTS = 'weights/sports1M_weights_tf.h5'