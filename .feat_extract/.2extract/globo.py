import os , argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

''' UCFCRIME '''
UCFCRIME_FROOT = "/raid/DATASETS/anomaly/UCF_Crimes/features"


## video paths lists
UCFCRIME_VPATHS_LISTS = {
    "train_normal" : '.lists/ucfcrime/train_normal.list' ,
    "train_abnormal" : '.lists/ucfcrime/train_abnormal.list' ,
    "test" : '.lists/ucfcrime/test.list'
}
VPATHS = UCFCRIME_VPATHS_LISTS


## features folders

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


FEATURES = 'c3d'    # c3d , i3d
if FEATURES == 'i3d':   FPATHS = UCFCRIME_I3D_FPATHS
elif FEATURES == 'c3d': FPATHS = UCFCRIME_C3D_FPATHS


DRY_RUNA = True ## if True doesnt save any npy file and just iterates over 10 videos