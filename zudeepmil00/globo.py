import os , argparse , time , csv
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="2"


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

## https://github.com/Roc-Ng/DeepMIL/issues/2
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

## https://github.com/tianyu0207/RTFM
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
    'i3d' : UCFCRIME_I3D_LISTS,
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
parser.add_argument('--dummy', type=int , default=0 , help='number of files in dataset, 0:full')
parser.add_argument('--debug', type=bool, default=True )

parser.add_argument('--druna', type=bool, default=False , help='if False creates folders/save ckpts && final weights')

parser.add_argument('--features',   type=str , default='c3d',       choices=['i3ddeepmil','i3drtfm','i3d','c3d'])
parser.add_argument('--l2norm',     type=int , default=1 ,          choices=[0,1,2] , help='0:none , 1:raw feats, 2:raw && segmnt feats')
parser.add_argument('--lossfx',     type=str , default='milbert',   choices=['deepmil','milbert','espana'])
parser.add_argument('--classifier', type=str , default='MLP',       choices=['MLP'])

parser.add_argument('--epochs',     type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16,    help='pairs of normal/abnormal features (if =16 , model in is (32,NSEGMENTS,NFEATURES))')

ARGS = parser.parse_args(args=[])

    
#################################

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
    
elif ARGS.features == 'i3d':
    VERSION = I3D_VERSION
    NCROPS = 0
    NSEGMENTS = 32 
    NFEATURES = 1024 
    OPTIMA = 'Adagrad'
    LR = 0.0001
    
elif ARGS.features == 'c3d':
    VERSION = C3D_VERSION
    NCROPS = 0
    NSEGMENTS = 32 
    NFEATURES = 4096
    OPTIMA = 'Adam'
    LR = 0.0001
    

MODEL_NAME = "{:.4f}_{}_{}-{}_{}_{}-{}".format(time.time(),ARGS.classifier,ARGS.features+VERSION,ARGS.l2norm,ARGS.lossfx,OPTIMA,LR);print(MODEL_NAME)
BASE_MODEL_PATH = os.path.join('.model',MODEL_NAME)
MODEL_PATH = BASE_MODEL_PATH+'/'+MODEL_NAME+'.h5'


if not ARGS.druna:
    if not os.path.exists(BASE_MODEL_PATH):
        os.makedirs(BASE_MODEL_PATH);print("\nINIT MODEL FOLDER @",BASE_MODEL_PATH)
    else: raise Exception(f"{BASE_MODEL_PATH} eristes")
    
    WEIGHTS_PATH = os.path.join(BASE_MODEL_PATH,'weights'); os.makedirs(WEIGHTS_PATH)
    LOG_PATH = os.path.join(BASE_MODEL_PATH,'log'); os.makedirs(LOG_PATH)
    
    HIST_CSV = LOG_PATH + '/hist.csv'
    with open(HIST_CSV, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'avg_loss', 'auc' , 'pr_auc' , 'ap' , 'fpr' , 'tpr' , 'th_targ'])

def histlog(row):
    if not ARGS.druna:
        with open(HIST_CSV, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(row)