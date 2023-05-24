import os , argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

BASE_UCFCRIME_DIR = "/raid/DATASETS/anomaly/UCF_Crimes"

UCFCRIME_VPATHS = {
    "train_normal" : BASE_UCFCRIME_DIR + "/videos/normal" ,
    "train_abnormal" : BASE_UCFCRIME_DIR + "videos/abnormal" , 
    "test" : BASE_UCFCRIME_DIR + "videos/test"
}

UCFCRIME_C3DRAW_FPATHS = {
    "train_normal" :    BASE_UCFCRIME_DIR+"/c3d-sports1m_raw_features/normal",
    "train_abnormal":   BASE_UCFCRIME_DIR+"/c3d-sports1m_raw_features/abnormal",
    "test" :            BASE_UCFCRIME_DIR+"/c3d-sports1m_raw_features/test",
}
UCFCRIME_C3DRAW_LISTS = {
    "train_normal" :    "list/ucf_c3d_raw_train_normal.list",
    "train_abnormal" :  "list/ucf_c3d_raw_train_abnormal.list",
    "test" :            "list/ucf_c3d_raw_test.list", 
}

UCFCRIME_I3DDEEPMIL_FPATHS = {
    "train_normal" :    BASE_UCFCRIME_DIR + '/features/I3DDEEPMIL/train_normal' , 
    "train_abnormal" :  BASE_UCFCRIME_DIR + '/features/I3DDEEPMIL/train_abnormal' ,
    "test" :            BASE_UCFCRIME_DIR + '/features/I3DDEEPMIL/test'
}
UCFCRIME_I3DDEEPMIL_LISTS = {
    "train_normal" :    "list/deepmil/ucf_i3d_train_normal.list",
    "train_abnormal" :  "list/deepmil/ucf_i3d_train_abnormal.list",
    "test" :            "list/deepmil/ucf_i3d_test.list", 
}

UCFCRIME_I3DRTFM_FPATHS = {
    "train_normal" :    BASE_UCFCRIME_DIR + '/features/I3DRTFM_10CROP/train_normal' , 
    "train_abnormal" :  BASE_UCFCRIME_DIR + '/features/I3DRTFM_10CROP/train_abnormal' ,
    "test" :            BASE_UCFCRIME_DIR + '/features/I3DRTFM_10CROP/test'
}
UCFCRIME_I3DRTFM_LISTS = {
    "train_normal" :    "list/rtfm/ucf_i3d_train_normal.list",
    "train_abnormal" :  "list/rtfm/ucf_i3d_train_abnormal.list" ,
    "test" :            "list/rtfm/ucf_i3d_test.list", 
}

## deepmil
UCFCRIME_GT = 'list/gt-ucf.npy'

BASE_MODEL_PATH = '.model/'
CKPT_PATH = BASE_MODEL_PATH + 'ckpt'


## option.py
parser = argparse.ArgumentParser(description='WSAD')
parser.add_argument('--dummy', default=0 )
parser.add_argument('--debug', default=True )

parser.add_argument('--features', default='i3ddeepmil', choices=['i3ddeepmil', 'i3drtfm'])
parser.add_argument('--lossfx', default='milbert', choices=['deepmil', 'milbert' , 'espana'])

parser.add_argument('--epochs', type=int, default=100, help='maximum iteration to train (default: 100)')
parser.add_argument('--batch_size', type=int, default=16, help='number of instances in a batch of data (default: 16)')

parser.add_argument('--gpus', default=1, type=int, choices=[0 , 1 , 2 , 3], help='gpus')

ARGS = parser.parse_args(args=[])


if ARGS.features == 'i3ddeepmil':
    NCROPS = 10
    NSEGMENTS = 32 
    NFEATURES = 1024 
    OPTIMA = 'Adam'
    LR = 0.0001 #0.00005
    
elif ARGS.features == 'i3drtfm':
    NCROPS = 10
    NSEGMENTS = 32 
    NFEATURES = 2048 
    OPTIMA = 'Adagrad'
    LR = 0.0001