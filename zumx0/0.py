from os import path as osp , makedirs as makedirs
import glob, sys, time, argparse, gc

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from gluoncv.data import VideoClsCustom

import utils.setup as setup
from utils.vid import gerador_clips
import utils.list as listils

parser = argparse.ArgumentParser(description='fext')
parser.add_argument('--experiment', 
                    type=str, 
                    default='cfg/xdviol0.yml', 
                    help='relative path to experiment .yml')
args = parser.parse_args(args=[])


if __name__ == "__main__":
    
    cfg = setup.init(args)
    from  utils.log import get_logger
    logger = get_logger(__name__)
    
    ## ??
    gc.set_threshold(100, 5, 5)

    ## set mx ctx 
    if cfg.GPUID[0] == -1: context = mx.cpu()
    elif len(cfg.GPUID) == 1:
        context = mx.gpu()
    else: 
        context = []
        for gpu in cfg.GPUID:context.append(mx.gpu(gpu))
    #logger.info("context",str(cfg.GPUID[:]))
    
    
    ## get data preprocess/transofrm
    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
    if cfg.TRANSFORM.TEN_CROP:
        transform_test = transforms.Compose([
            video.VideoTenCrop(cfg.TRANSFORM.INPUT_SIZE),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        cfg.DATA.NUM_CROP = 10
    elif cfg.TRANSFORM.THREE_CROP:
        transform_test = transforms.Compose([
            video.VideoThreeCrop(cfg.TRANSFORM.INPUT_SIZE),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        cfg.DATA.NUM_CROP = 3
    else:
        transform_test = video.VideoGroupValTransform(size=cfg.TRANSFORM.INPUT_SIZE, mean=image_norm_mean, std=image_norm_std)
        cfg.DATA.NUM_CROP = 1
    
    
    ## get model
    if cfg.MODEL.USE_PRETRAINED and len(cfg.MODEL.HASHTAG) > 0:
        cfg.MODEL.USE_PRETRAINED = cfg.MODEL.HASHTAG
    classes = cfg.MODEL.NUM_CLASSES
    model_name = cfg.MODEL.NAME
    net = get_model(name=model_name, nclass=classes, pretrained=cfg.MODEL.USE_PRETRAINED,
                    feat_ext=True, num_segments=cfg.DATA.NUM_SEGMENTS, num_crop=cfg.DATA.NUM_CROP)
    net.cast(cfg.MODEL.DTYPE)
    #net.collect_params().reset_ctx(context)
    net.collect_params().initialize(force_reinit=True, ctx=context)
    #print(net.collect_params())

    if cfg.MODEL.MODE == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    if cfg.MODEL.RESUME_PARAMS != '' and not cfg.MODEL.USE_PRETRAINED:
        net.load_parameters(cfg.MODEL.RESUME_PARAMS, ctx=context)
        logger.info('Pre-trained model %s is successfully loaded.' % (cfg.MODEL.RESUME_PARAMS))
    else: logger.info('Pre-trained model is successfully loaded from the model zoo.')

    logger.info("Successfully built model {}".format(model_name))


    ## real-shhh
    sub_dirs = glob.glob(cfg.DATA_DIR+"/*_copy")
    for sub_dir in sub_dirs:
        
        ## create a folder in the exp path with same name as the video folder
        out_folder = osp.join(cfg.EXPERIMENTPATH , sub_dir.split("/")[-1])
        if not osp.exists(out_folder):
            makedirs(out_folder)
        logger.info(f"saving features in2 {out_folder}")
        
        ## get the video paths
        vpaths = listils.get(sub_dir)
        
        start_time = time.time()
        for vid, vline in enumerate(vpaths):
            vpath = vline.split()[0]
            vname = osp.splitext(osp.basename(vpath))[0]
            logger.info(f'{vid} {vpath}')
            
            ## checks if video is already processed  
            out_npy = osp.join(out_folder, vname + ".npy")
            if osp.exists(out_npy):
                #logger.info(str(out_npy),"already created")
                print(f'')
                continue
            
            ## creates the gerador
            clips = gerador_clips(vpath , cfg , transform_test , True)
            nclips = next(clips)
            
            #feats = np.ones((nclips,2048),np.float32) #f'np.{cfg.MODEL.DTYPE}'
            feats = []
            for j,clip in enumerate(clips):
                video_input = nd.array(clip).as_in_context(context)
                video_feat = net(video_input.astype(cfg.MODEL.DTYPE, copy=False))
                feats.append(video_feat.asnumpy())
                #feats[j] = video_feat.asnumpy()
                
                if cfg.DEBUG:
                    print('clip', clip.shape)
                    print('video_input', video_input.shape)
                    print('video_feat', video_feat.shape,video_feat.dtype,"\n")
            
            feats = np.concatenate(feats)
            #print(f'\t{feats.dtype} , np ? {isinstance(feats, np.ndarray)}')
            logger.info(f'{feats.shape} saved in2 {out_npy}')
            np.save(out_npy, feats)

            if vid == 1 : break
            
        end_time = time.time()
        logger.info('Total feature extraction time is %4.2f minutes' % ((end_time - start_time) / 60))
