from yacs.config import CfgNode as CN

_C = CN()

## the root path to your data
_C.DATA_DIR = ''
## text file describing the dataset, each line per video sample. There are three items in each line: (1) video path; (2) video length and (3) video label
#_C.DATA_LIST = ''
## number of gpus to use. Use -1 for CPU
_C.GPUID = []
## wheter to log cfg after merge with default
_C.LOG_CFG_INFO = True
## directory of saved results
_C.EXPERIMENTDIR = ''
## based on the .yml file name a folder will be created in the experiment dir
_C.EXPERIMENTID = ''
_C.EXPERIMENTPATH = ''
## 
_C.DEBUG = True

# model
_C.MODEL = CN()
## type of model to use. see vision_model for options
_C.MODEL.NAME = 'i3d_resnet50_v1_kinetics400'
## data type for the network. default is float32
_C.MODEL.DTYPE = 'float32'
## number of classes
_C.MODEL.NUM_CLASSES = 400
## mode in which to train the model. options are symbolic, imperative, hybrid
_C.MODEL.MODE = 'hybrid'
## enable using pretrained model from GluonCV
_C.MODEL.USE_PRETRAINED = True
## hashtag for pretrained models
_C.MODEL.HASHTAG = '568a722e'
## path of parameters to load from
_C.MODEL.RESUME_PARAMS = ''

# transform
_C.TRANSFORM = CN()
## size of the input image size. default is 224
_C.TRANSFORM.INPUT_SIZE = 224
## whether to use ten crop evaluation
_C.TRANSFORM.TEN_CROP = False
## whether to use three crop evaluation
_C.TRANSFORM.THREE_CROP = False

# data
_C.DATA = CN()
## Number of segments to evenly divide the video into clips. A useful technique to obtain global video-level information
_C.DATA.NUM_SEGMENTS = -1
## Number of crops for each image. default is 1. Common choices are three crops and ten crops during evaluation
_C.DATA.NUM_CROP = 1
## The length of input video clip. For example, new_length=16 means we will extract a video clip of consecutive 16 frames
_C.DATA.NEW_LENGTH = 32 # FRAME_MAX
## Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames. new_step=2 means we will extract a video clip of every other frame
_C.DATA.NEW_STEP = 1 # FRAME_STEP
## Scale the width of loaded image to "new_width" for later multiscale cropping and resizing
_C.DATA.NEW_WIDTH = 340
## Scale the height of loaded image to "new_height" for later multiscale cropping and resizing
_C.DATA.NEW_HEIGHT = 256
## if set to True, decode video frames with decord , else opencv
_C.DATA.USE_DECORD = True
## if set to True, use data loader designed for SlowFast network.'
_C.DATA.SLOWFAST = False
## the temporal stride for sparse sampling of video frames for slow branch in SlowFast network
_C.DATA.SLOW_TEMPORAL_STRIDE = 16
## the temporal stride for sparse sampling of video frames for fast branch in SlowFast network
_C.DATA.FAST_TEMPORAL_STRIDE = 2
## different types of data augmentation pipelines. Supports v1, v2, v3 and v4.
_C.DATA.DATA_AUG = 'v1'


# ---------------------------------------------------------------------------- #


def get_cfg(args):
    """
    Get the experiment config.
    """
    cfg = _C.clone()
    cfg.merge_from_file(args.experiment)
    return cfg