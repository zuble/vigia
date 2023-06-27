import globo

import cv2 , time , os
import numpy as np , tensorflow as tf 
from sklearn.preprocessing import normalize

import i3d, c3d
from utils import gerador_clips 

'''
    Iterates over all UCF Crime video folders (train_normal , train_abnormal , test)
    Divide each video into snippets of 16 frames 
    Process each snippet trough c3dsports1m model
    #l2norm the features
    Saves a .npy file per video
'''

    
## gets preprocess fx and model
if globo.FEATURES == 'i3d':
    squeeze = True
    prep_input = i3d.preprocess_input    
    feat_extractor = i3d.Inception_Inflated3d(
        include_top=True,
        weights='rgb_imagenet_and_kinetics',
        input_shape=(16, 224, 224, 3),
        feat_extractor=True )
    #print(feature_extractor.summary())
    
elif globo.FEATURES == 'c3d':
    squeeze = False
    prep_input = c3d.preprocess_input         ## returns float64
    feat_extractor = c3d.c3d_feature_extractor() ## returns float32


@tf.function
def predict(x): return feat_extractor(x, training=False)

def l2norm(x): return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)  

## iterates over train,valdt,test
for type, dir in globo.VPATHS.items():

    vpaths = list(open(dir))
    fpath = globo.FPATHS[type]

    print(f'\n\n****************\n\nProcessing {type} @ {dir}\n\n{len(vpaths)} vids\nsaving into {fpath}')
    
    for i, vpath in enumerate(vpaths):
        t = time.time()
        
        vpath = vpath.strip("\n");fn = os.path.splitext(os.path.basename(vpath))[0]
        print(f'\n{i} {fn}.mp4')
        
        ## checks if video is already processed  
        out_npy = os.path.join(fpath, fn + ".npy")
        if not globo.DRY_RUNA and os.path.exists(out_npy):
            print(out_npy,"already created")
            continue
             
        features = []
        for clip in gerador_clips(vpath):
        
            prep_clip = prep_input(clip)
            print(f'\tPREP{np.shape(prep_clip)} , {prep_clip.dtype} , np ? {isinstance(prep_clip, np.ndarray)}')
            
            feature = predict(prep_clip) 
            print(f'\tFEAT {np.shape(feature)}')
            
            if squeeze: ## i3d
                feature = np.squeeze(feature, axis=(1, 2, 3))
                print(f'\tFEAT {np.shape(feature)} , {feature.dtype} , np ? {isinstance(feature, np.ndarray)}')
            
            features.append(feature)
        
        features = np.concatenate(features) ## == np.array
        print(f'\tFEATED ( {np.shape(features)[0]} timesteps , {np.shape(features)[1]} features ) {features.dtype} , np ? {isinstance(features, np.ndarray)} , tf ? {isinstance(features, tf.Tensor)}')
        
        #features2 = normalize(features, axis=1)
        #features = l2norm(features)
        #print("\tL2NORM same ?",np.allclose(features,features2))
              
        if not globo.DRY_RUNA: np.save(out_npy, features)
        print(f'\n\tsaved @ {out_npy}')
        
        tt = time.time()
        print(f'\t~ {int(tt-t)} seconds')
        
        if globo.DRY_RUNA and i == 10: break




## too much t handle
'''
for type, dir in globo.UCFCRIME_VPATHS_LISTS.items():

    vpaths = list(open(dir))
    
    print(f'\n****************\n\nProcessing {type} @ {dir}\n\n{len(vpaths)} vids')
    
    features_list , fns_list = [] , []
    for i, vpath in enumerate(vpaths):
        
        vpath = vpath.strip('\n');fn = os.path.basename(vpath)
        print(f'\n{i} {fn}\n')

        clips, frames = get_video_clips(vpath) ## divide in2 snippets of 16
        
        prep_clips = [c3d.preprocess_input(np.array(clip)) for clip in clips]
        prep_clips = np.vstack(prep_clips)
        print(f'\n\tPREP {np.shape(prep_clips)}\n\t( {np.shape(prep_clips)[0]} clips , {np.shape(prep_clips)[1]} frames , {np.shape(prep_clips)[2:]} resolution )',prep_clips.dtype)
        
    
        features = feature_extractor.predict(prep_clips)
        #features = normalize(features, axis=1)
        print(f'\n\tFEAT {np.shape(features)}\n\t( {np.shape(features)[0]} timesteps , {np.shape(features)[1]} features )',features.dtype , isinstance(features, np.ndarray))
        
        features_list.append(features)
        fns_list.append(fn)
        
    
    features_list = np.array(features_list, dtype=object)
    fns_list = np.array(fns_list, dtype=object)
    print(f'\n\tALL FEAT {np.shape(features_list)} , {np.shape(fns_list)}')


    out_npz = os.path.join(globo.UCFCRIME_FEATC3D_BASE_DIR+VERSION, type + ".npz")
    np.savez_compressed(out_npz, fns=fns_list , features=features_list)
    print(f'\n\tsaved {type} features @ {out_npz}')
'''