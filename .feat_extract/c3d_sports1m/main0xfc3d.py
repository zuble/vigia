import globo

import os , numpy as np
from sklearn.preprocessing import normalize

import c3d
from utils import get_video_clips


'''
    Iterates over all UCF Crime video folders (train_0 , train_1 , test)
    Divide in each video into snippets of 16 frames 
    Process each snippet trough c3dsports1m model
    l2norm the features
    Saves a .npy file per video
'''


feature_extractor = c3d.c3d_feature_extractor()

for type, dir in globo.UCFCRIME_VPATHS_LISTS.items():

    vpaths = list(open(dir))
    
    print(f'\n****************\n\nProcessing {type} @ {dir}\n\n{len(vpaths)} vids')
    
    for i, vpath in enumerate(vpaths):
        
        vpath = vpath.strip('\n')
        fn = os.path.splitext(os.path.basename(vpath))[0]
        print(f'\n{i} {fn}.mp4\n')

        out_npy = os.path.join(globo.UCFCRIME_FEATC3D[type], fn + ".npy")
        if os.path.exists(out_npy):
            print(out_npy,"already created")
            continue
            
        ## divide in2 snippets of 16
        clips, frames = get_video_clips(vpath)
        
        ## preprocess 4 c3d
        prep_clips = [c3d.preprocess_input(np.array(clip)) for clip in clips]
        prep_clips = np.vstack(prep_clips)
        print(f'\n\tPREP {np.shape(prep_clips)}\n\t( {np.shape(prep_clips)[0]} clips , {np.shape(prep_clips)[1]} frames , {np.shape(prep_clips)[2:]} resolution )')
        
        ## get them features
        features = feature_extractor.predict(prep_clips)
        #features = normalize(features, axis=1)
        print(f'\n\tFEAT {np.shape(features)}\n\t( {np.shape(features)[0]} timesteps , {np.shape(features)[1]} features )')
        
        #np.save(out_npy, features)
        print(f'\n\tsaved @ {out_npy}')
        break