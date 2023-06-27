#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Dataset paths
# ---------------------------------------------------------------------------- #
_C.DS = CN()

ucf101_root = '/raid/DATASETS/anomaly/UCF101'
_C.DS.ucf101 = CN() 
_C.DS.ucf101.vpaths = [ ucf101_root+'/train' , ucf101_root+'/test' ]

minisports1m_root = '/media/jtstudents/T77/miniSports1M/videos_256p_dense_cache'
_C.DS.minisports1m = CN()
_C.DS.minisports1m.vpaths = [ minisports1m_root+'/train' , minisports1m_root+'/test' ]
    
mitv2_root = '/'
_C.DS.mitv2=[]

_C.DS.kinetics400 = CN()
kinetics400_root = '/media/jtstudents/T77/kinetics400'
_C.DS.kinetics400.vpaths = [ kinetics400_root+'/videos_train' , kinetics400_root+'/videos_val' ]
_C.DS.kinetics400.lpaths = [kinetics400_root+'/.zu/kinetics400_train_list_videos.txt' ,
                          kinetics400_root+'/.zu/kinetics400_val_list_videos.txt' ]

'''
# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False
# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200
# Weight decay value that applies on BN parameters
_C.BN.WEIGHT_DECAY = 0.0
# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"
# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1
# Parameter for NaiveSyncBatchNorm, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized. `NUM_SYNC_DEVICES` cannot be larger than number of
# devices per machine; if global sync is desired, set `GLOBAL_SYNC`.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.NUM_SYNC_DEVICES = 1
# Parameter for NaiveSyncBatchNorm. Setting `GLOBAL_SYNC` to True synchronizes
# stats across all devices, across all machines; in this case, `NUM_SYNC_DEVICES`
# must be set to None.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.GLOBAL_SYNC = False
'''



# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True
# Kill training if loss explodes over this ratio from the previous 5 measurements.
# Only enforced if > 0.0
_C.TRAIN.KILL_LOSS_EXPLOSION_FACTOR = 0.0
# Dataset.
_C.TRAIN.DATASET = "kinetics400"
# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64
# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10
# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10
# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True
# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""
# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"
# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False
# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False
# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)
# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = False
# if True, inflate some params from imagenet model.
_C.TRAIN.CHECKPOINT_IN_INIT = False


# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
'''
_C.AUG = CN()

# Whether to enable randaug.
_C.AUG.ENABLE = False
# Number of repeated augmentations to used during training.
# If this is greater than 1, then the actual batch size is
# TRAIN.BATCH_SIZE * AUG.NUM_SAMPLE.
_C.AUG.NUM_SAMPLE = 1
# Not used if using randaug.
_C.AUG.COLOR_JITTER = 0.4
# RandAug parameters.
_C.AUG.AA_TYPE = "rand-m9-mstd0.5-inc1"
# Interpolation method.
_C.AUG.INTERPOLATION = "bicubic"
# Probability of random erasing.
_C.AUG.RE_PROB = 0.25
# Random erasing mode.
_C.AUG.RE_MODE = "pixel"
# Random erase count.
_C.AUG.RE_COUNT = 1
# Do not random erase first (clean) augmentation split.
_C.AUG.RE_SPLIT = False
# Whether to generate input mask during image processing.
_C.AUG.GEN_MASK_LOADER = False
# If True, masking mode is "tube". Default is "cube".
_C.AUG.MASK_TUBE = False
# If True, masking mode is "frame". Default is "cube".
_C.AUG.MASK_FRAMES = False
# The size of generated masks.
_C.AUG.MASK_WINDOW_SIZE = [8, 7, 7]
# The ratio of masked tokens out of all tokens. Also applies to MViT supervised training
_C.AUG.MASK_RATIO = 0.0
# The maximum number of a masked block. None means no maximum limit. (Used only in image MaskFeat.)
_C.AUG.MAX_MASK_PATCHES_PER_BLOCK = None
'''


# ---------------------------------------------------------------------------- #
# MipUp options.
# ---------------------------------------------------------------------------- #
'''
_C.MIXUP = CN()

# Whether to use mixup.
_C.MIXUP.ENABLE = False
# Mixup alpha
_C.MIXUP.ALPHA = 0.8
# Cutmix alpha.
_C.MIXUP.CUTMIX_ALPHA = 1.0
# Probability of performing mixup or cutmix when either/both is enabled.
_C.MIXUP.PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled.
_C.MIXUP.SWITCH_PROB = 0.5
# Label smoothing.
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1
'''

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True
# Dataset for testing.
_C.TEST.DATASET = "kinetics400"
# Ds path
_C.TEST.DATASET_PATH = ""
# Total mini-batch size
_C.TEST.BATCH_SIZE = 32
# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""
# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_TEMPORAL_VIEWS = 10 #NUM_ENSEMBLE_VIEWS
# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3
# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"
# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""
_C.TEST.NUM_TEMPORAL_VIEWS = 1
_C.TEST.NUM_TEMPORAL_CLIPS = []


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CN()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""
# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "
# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""
# The number of frames of the input clip.
_C.DATA.FRAME_MAX = 0
# The video sampling rate of the input clip.
_C.DATA.FRAME_STEP = 0
'''
# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.DATA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]
# Eigenvectors for PCA jittering.
_C.DATA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]
# If a imdb have been dumpped to a local file with the following format:
# `{"im_path": im_path, "class": cont_id}`
# then we can skip the construction of imdb and load it from the local file.
_C.DATA.PATH_TO_PRELOAD_IMDB = ""
'''
# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.
_C.DATA.NINPUT_CHANNELS = 3
# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]
# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]
# The relative scale range of Inception-style area based random resizing augmentation.
# If this is provided, DATA.TRAIN_JITTER_SCALES above is ignored.
_C.DATA.TRAIN_JITTER_SCALES_RELATIVE = []
# The relative aspect ratio range of Inception-style area based random resizing
# augmentation.
_C.DATA.TRAIN_JITTER_ASPECT_RELATIVE = []
# If True, perform stride length uniform temporal sampling.
_C.DATA.USE_OFFSET_SAMPLING = False
# Whether to apply motion shift for augmentation.
_C.DATA.TRAIN_JITTER_MOTION_SHIFT = False
# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224
# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256
# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30
# JITTER TARGET_FPS by +- this number randomly
_C.DATA.TRAIN_JITTER_FPS = 0.0
# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "torchvision"
# Decoding resize to short size (set to native size for best speed)
_C.DATA.DECODING_SHORT_SIZE = 256

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False
# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True
# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False
# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"
# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False
# how many samples (=clips) to decode from a single video
_C.DATA.TRAIN_CROP_NUM_TEMPORAL = 1
# how many spatial samples to crop from a single clip
_C.DATA.TRAIN_CROP_NUM_SPATIAL = 1
# color random percentage for grayscale conversion
_C.DATA.COLOR_RND_GRAYSCALE = 0.0
# loader can read .csv file in chunks of this chunk size
_C.DATA.LOADER_CHUNK_SIZE = 0
# if LOADER_CHUNK_SIZE > 0, define overall length of .csv file
_C.DATA.LOADER_CHUNK_OVERALL_SIZE = 0
# for chunked reading, dataloader can skip rows in (large)
# training csv file
_C.DATA.SKIP_ROWS = 0
# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "
# augmentation probability to convert raw decoded video to
# grayscale temporal difference
_C.DATA.TIME_DIFF_PROB = 0.0
'''
# Apply SSL-based SimCLR / MoCo v1/v2 color augmentations,
#  with params below
_C.DATA.SSL_COLOR_JITTER = False
# color jitter percentage for brightness, contrast, saturation
_C.DATA.SSL_COLOR_BRI_CON_SAT = [0.4, 0.4, 0.4]
# color jitter percentage for hue
_C.DATA.SSL_COLOR_HUE = 0.1
# SimCLR / MoCo v2 augmentations on/off
_C.DATA.SSL_MOCOV2_AUG = False
# SimCLR / MoCo v2 blur augmentation minimum gaussian sigma
_C.DATA.SSL_BLUR_SIGMA_MIN = [0.0, 0.1]
# SimCLR / MoCo v2 blur augmentation maximum gaussian sigma
_C.DATA.SSL_BLUR_SIGMA_MAX = [0.0, 2.0]

# If combine train/val split as training for in21k
_C.DATA.IN22K_TRAINVAL = False
# If not None, use IN1k as val split when training in21k
_C.DATA.IN22k_VAL_IN1K = ""
# Large resolution models may use different crop ratios
_C.DATA.IN_VAL_CROP_RATIO = 0.875 # 224/256 = 0.875
'''
# don't use real video for kinetics.py
_C.DATA.DUMMY_LOAD = False

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
'''
_C.DATA_LOADER = CN()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8
# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True
# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False
'''

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
'''
_C.RESNET = CN()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"
# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1
# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64
# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True
# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False
#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False
#  If true, initialize the final conv layer of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_CONV = False
# Number of weight layers.
_C.RESNET.DEPTH = 50
# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]
# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]
# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]
'''

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# Model architecture.
_C.MODEL.ARCH = "slowfast"
# Model name
_C.MODEL.MODEL_NAME = "SlowFast"
# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400
# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"
# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = [
    "2d",
    "c2d",
    "i3d",
    "slow",
    "x3d",
    "mvit",
    "maskmvit",
]
# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]
# Dropout rate before final projection in the backbone.
#_C.MODEL.DROPOUT_RATE = 0.5 ## X3D.DROPOUT_RATE 
# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0
# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01
# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"
# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False
# If True, detach the final fc layer from the network, by doing so, only the
# final fc layer will be trained.
_C.MODEL.DETACH_FINAL_FC = False
# If True, frozen batch norm stats during training.
_C.MODEL.FROZEN_BN = False
# If True, AllReduce gradients are compressed to fp16
_C.MODEL.FP16_ALLREDUCE = False

# ---------------------------------------------------------------------------- #
# X3D  options
# See https://arxiv.org/abs/2004.04730 for details about X3D Networks.
# ---------------------------------------------------------------------------- #
_C.X3D = CN()

# Width expansion factor.
_C.X3D.WIDTH_FACTOR = 1.0
# Depth expansion factor.
_C.X3D.DEPTH_FACTOR = 1.0
# Bottleneck expansion factor for the 3x3x3 conv.
_C.X3D.BOTTLENECK_WIDTH_FACTOR = 1.0  #
# Dimensions of the last linear layer before classificaiton.
_C.X3D.DIM_C5 = 2048
# Dimensions of the first 3x3 conv layer.
_C.X3D.DIM_C1 = 12
# the size of the temporal filter in the conv1 layer
_C.X3D.C1_TEMP_FILTER = 5
# Whether to scale the width of Res2, default is false.
_C.X3D.SCALE_RES2 = False
# Whether to use a BatchNorm (BN) layer before the classifier, default is false.
_C.X3D.BN_LIN5 = False
# Whether to use channelwise (=depthwise) convolution in the center (3x3x3)
# convolution operation of the residual blocks. 
_C.X3D.CHANNELWISE_3x3x3 = True

# dropout rate for the dropout layer before the final fully-connected layer
# Dropout rate before final projection in the backbone.
_C.X3D.DROPOUT_RATE = 0.5

# l2 regularizer passed to the layer as regularizer
_C.X3D.L2_WEIGHT_DECAY = 0.00005 ## pyslowfast default is 1e-4 (==SOLVER.WEIGHT_DECAY)

## x3d-tf
_C.X3D.BN = CN()
#_C.X3D.BN.L2_WEIGHT_DECAY (pyslowfast use 0 as default)
# the momentum parameter for all batch norm layers
_C.X3D.BN.MOMENTUM = 0.9
# the epsilon parameter for all batch norm layers
_C.X3D.BN.EPS = 1e-5

# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CN()

# Corresponds to frame step of the input in slowpath
_C.SLOWFAST.TAU = 16
# Corresponds to the channel reduction ratio, between the Slow and Fast pathways.
_C.SLOWFAST.BETA = 1/8
# frame rate reduction ratio, alpha between Slow and Fast pathways.
# frame_step fast: tau/alpha
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2
# ['T_conv','T_sample','TtoC_sum','TtoC_concat']
_C.SLOWFAST.FUSION_METHOD = 'T_conv'

_C.SLOWFAST.BN = CN()
# the momentum parameter for all batch norm layers
_C.SLOWFAST.BN.MOMENTUM = 0.9
# the epsilon parameter for all batch norm layers
_C.SLOWFAST.BN.EPS = 1e-5

# adds a inital layer responsible to transform the video input
# 0 : accepts only as input the fast_input, and gathers th eslow input from it
# 1 : accepts a  
_C.SLOWFAST.DATALAYER = 0


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# Learning rate policy (cosine or steps_with_relative_lrs).
_C.SOLVER.LR_POLICY = "cosine" 
# Base learning rate.
_C.SOLVER.BASE_LR = 0.1
# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1
# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01
# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0
# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False
# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1
# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []
# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []


# Momentum.
_C.SOLVER.MOMENTUM = 0.9
# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0
# Nesterov momentum.
_C.SOLVER.NESTEROV = True
# L2 regularization for non BN parameters
##_C.SOLVER.WEIGHT_DECAY = 1e-4 ## aplied in X3D.BN



# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

## Exponential decay factor.
#_C.SOLVER.GAMMA = 0.1
## Base learning rate is linearly scaled with NUM_SHARDS.
#_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False
## If True, perform no weight decay on parameter with one dimension (bias term, etc).
#_C.SOLVER.ZERO_WD_1D_PARAM = False
## Clip gradient at this value before optimizer update
#_C.SOLVER.CLIP_GRAD_VAL = None
## Clip gradient at this norm before optimizer update
#_C.SOLVER.CLIP_GRAD_L2NORM = None
## LARS optimizer
#_C.SOLVER.LARS_ON = False
## The layer-wise decay of learning rate. Set to 1. to disable.
#_C.SOLVER.LAYER_DECAY = 1.0
## Adam's .DATA.
#_C.SOLVER.BETAS = (0.9, 0.999)


# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CN()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False
# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5**0.5]

_C.MULTIGRID.LONG_CYCLE = False
# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5**0.5),
    (0.5, 0.5**0.5),
    (0.5, 1),
    (1, 1),
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0


# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CN()

# Number of epochs for data loading benchmark.
_C.BENCHMARK.NUM_EPOCHS = 5
# Log period in iters for data loading benchmark.
_C.BENCHMARK.LOG_PERIOD = 100
# If True, shuffle dataloader for epoch during benchmark.
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Sets the tf.debugging.experimental.enable_dump_debug_info
_C.DEBUG = False

# Path to the base dir holding all exprimentations dumps
_C.EXPERIMENTDIR = ""
# Name for the folder holding all the experiment log/ckpts/etc (default will be .yml fn)
_C.EXPERIMENTID = ""
# Full path to the dir of actual experiment (determined base on the 2 above values
# after in setup)
_C.EXPERIMENTPATH = ""
# Full path to the sub folder ckpts of EXPERIMENPATH
_C.EXPERIMENTCKPTPATH = ""

# GPU's to use at training 
# If + than 1 ; eg. 1 , 2 ; a strategy will be applied to all model related fx
_C.GPUID = 1
# Sets the system environment 'TF_FORCE_GPU_ALLOW_GROWTH'
_C.GPUSLIM = True

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1
# Log period in iters.
_C.LOG_PERIOD = 10
# If True, log the model info.
_C.LOG_MODEL_INFO = True
# if True, log the experiment cfg after merge
_C.LOG_CFG_INFO = True


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()