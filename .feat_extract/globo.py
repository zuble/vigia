import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

''' UCFCRIME '''
UCFCRIME_ROOT = "/raid/DATASETS/anomaly/UCF_Crimes"
UCFCRIME_FROOT = UCFCRIME_ROOT + '/features'


## video paths lists
UCFCRIME_VPATHS_LISTS = {
    "train_normal" : '.lists_ucfcrime/train_normal.list' ,
    "train_abnormal" : '.lists_ucfcrime/train_abnormal.list' ,
    "test" : '.lists_ucfcrime/test.list'
}

## C3D features
## 0000 : slides of 16 / no frame_step / no normalize / 4096 features / 30mins max 
## 0001 : slides of 16 / no frame_step / no normalize / 4096 features

C3D_VERSION = '0001'
UCFCRIME_C3D_FPATHS = {
    "train_normal" :    UCFCRIME_FROOT+'/C3DSPORTS1M/'+C3D_VERSION+'/train_normal',
    "train_abnormal":   UCFCRIME_FROOT+'/C3DSPORTS1M/'+C3D_VERSION+'/train_abnormal',
    "test" :            UCFCRIME_FROOT+'/C3DSPORTS1M/'+C3D_VERSION+'/test'
}
C3DSPORTS1M = '.models/c3d_sports1m/sports1M_weights_tf.h5'


## I3D features
## 0000 : slides of 16 / no frame_step / no normalize / 1024 features (rgb_imagenet_and_kinetics)
## 0001 : 

I3D_VERSION = '0000'
UCFCRIME_I3D_FPATHS = {
    "train_normal" :    UCFCRIME_FROOT+'/I3D/'+I3D_VERSION+'/train_normal',
    "train_abnormal":   UCFCRIME_FROOT+'/I3D/'+I3D_VERSION+'/train_abnormal',
    "test" :            UCFCRIME_FROOT+'/I3D/'+I3D_VERSION+'/test'
}



VSWIN = '.models/vswin/swin_tiny_patch244_window877_kinetics400_1k_1'