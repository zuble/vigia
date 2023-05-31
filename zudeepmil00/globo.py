import os , argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"


UCFCRIME_ROOT = "/raid/DATASETS/anomaly/UCF_Crimes"
UCFCRIME_FROOT = UCFCRIME_ROOT + '/features'


## C3D features
## 0000 : slides of 16 / no frame_step / no normalize / 4096 features / 30mins max 
## 0001 : slides of 16 / no frame_step / no normalize / 4096 features / (generator)
C3D_VERSION = '0001'
UCFCRIME_C3D_FPATHS = {
    "train_normal" :    UCFCRIME_FROOT+'/C3DSPORTS1M/'+C3D_VERSION+'/train_normal',
    "train_abnormal":   UCFCRIME_FROOT+'/C3DSPORTS1M/'+C3D_VERSION+'/train_abnormal',
    "test" :            UCFCRIME_FROOT+'/C3DSPORTS1M/'+C3D_VERSION+'/test'
}
UCFCRIME_C3D_LISTS = {
    "train_normal" :    'list/c3d/'+C3D_VERSION+'/ucf_c3d_'+C3D_VERSION+'_train_normal.list',
    "train_abnormal" :  'list/c3d/'+C3D_VERSION+'/ucf_c3d_'+C3D_VERSION+'_train_abnormal.list',
    "test" :            'list/c3d/'+C3D_VERSION+'/ucf_c3d_'+C3D_VERSION+'_test.list', 
}


## I3D features
## 0000 : slides of 16 / no frame_step / no normalize / 1024 features (rgb_imagenet_and_kinetics)
## 0001 : 
I3D_VERSION = '0000'
UCFCRIME_I3D_FPATHS = {
    "train_normal" :    UCFCRIME_FROOT+'/I3D/'+I3D_VERSION+'/train_normal',
    "train_abnormal":   UCFCRIME_FROOT+'/I3D/'+I3D_VERSION+'/train_abnormal',
    "test" :            UCFCRIME_FROOT+'/I3D/'+I3D_VERSION+'/test'
}
UCFCRIME_I3D_LISTS = {
    "train_normal" :    'list/i3d/'+I3D_VERSION+'/ucf_i3d_'+I3D_VERSION+'_train_normal.list',
    "train_abnormal" :  'list/i3d/'+I3D_VERSION+'/ucf_i3d_'+I3D_VERSION+'_train_abnormal.list',
    "test" :            'list/i3d/'+I3D_VERSION+'/ucf_i3d_'+I3D_VERSION+'_test.list', 
}


UCFCRIME_I3DDEEPMIL_FPATHS = {
    "train_normal" :    UCFCRIME_FROOT + '/I3DDEEPMIL/train_normal' , 
    "train_abnormal" :  UCFCRIME_FROOT + '/I3DDEEPMIL/train_abnormal' ,
    "test" :            UCFCRIME_FROOT + '/I3DDEEPMIL/test'
}
UCFCRIME_I3DDEEPMIL_LISTS = {
    "train_normal" :    "list/deepmil/ucf_i3d_train_normal.list",
    "train_abnormal" :  "list/deepmil/ucf_i3d_train_abnormal.list",
    "test" :            "list/deepmil/ucf_i3d_test.list", 
}


UCFCRIME_I3DRTFM_FPATHS = {
    "train_normal" :    UCFCRIME_FROOT + '/I3DRTFM_10CROP/train_normal' , 
    "train_abnormal" :  UCFCRIME_FROOT + '/I3DRTFM_10CROP/train_abnormal' ,
    "test" :            UCFCRIME_FROOT + '/I3DRTFM_10CROP/test'
}
UCFCRIME_I3DRTFM_LISTS = {
    "train_normal" :    "list/rtfm/ucf_i3d_train_normal.list",
    "train_abnormal" :  "list/rtfm/ucf_i3d_train_abnormal.list" ,
    "test" :            "list/rtfm/ucf_i3d_test.list", 
}


FEATURE_LISTS ={
    'i3ddeepmil' : UCFCRIME_I3DDEEPMIL_LISTS,
    'i3drtfm' : UCFCRIME_I3DRTFM_LISTS,
    'c3d' : UCFCRIME_C3D_LISTS
}

## deepmil
UCFCRIME_GT = 'gt/gt-ucf.npy'

## each video totalframes % 16 == 0
UCFCRIME_GT16 ='gt/gt-ucf_16f.npy' ## [i]=(fn,gt)
UCFCRIME_GT16_ALL = 'gt/gt-ucf_16f_all.npy' ## gt1,gt2..

## train folders
BASE_MODEL_PATH = '.model/'
CKPT_PATH = BASE_MODEL_PATH + 'ckpt'


## option.py
parser = argparse.ArgumentParser(description='WSAD')
parser.add_argument('--dummy', default=0 )
parser.add_argument('--debug', default=True )

parser.add_argument('--features', default='c3d', choices=['i3ddeepmil', 'i3drtfm' , 'c3d'])
parser.add_argument('--lossfx', default='milbert', choices=['deepmil', 'milbert' , 'espana'])
parser.add_argument('--classifier', default='MLP', choices=['MLP'])

parser.add_argument('--epochs', type=int, default=100, help='maximum iteration to train (default: 100)')
parser.add_argument('--batch_size', type=int, default=16, help='number of instances in a batch of data (default: 16)')

parser.add_argument('--gpus', default=1, type=int, choices=[0 , 1 , 2 , 3], help='gpus')

ARGS = parser.parse_args(args=[])


if ARGS.features == 'i3ddeepmil':
    VERSION = ''
    NCROPS = 10
    NSEGMENTS = 32 
    NFEATURES = 1024 
    OPTIMA = 'Adam'
    LR = 0.0001 #0.00005
    
elif ARGS.features == 'i3drtfm':
    VERSION = ''
    NCROPS = 10
    NSEGMENTS = 32 
    NFEATURES = 2048 
    OPTIMA = 'Adagrad'
    LR = 0.0001

elif ARGS.features == 'c3d':
    VERSION = C3D_VERSION
    NCROPS = 0
    NSEGMENTS = 32 
    NFEATURES = 4096
    OPTIMA = 'Adam'
    LR = 0.0001