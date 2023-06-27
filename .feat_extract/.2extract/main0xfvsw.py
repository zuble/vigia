import globo , os
import numpy as np
import tensorflow as tf
from utils import *
from sklearn.preprocessing import normalize


def normalize_input(batch_clips):
    mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
    std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
    return (batch_clips - mean) / std


model = tf.saved_model.load(globo.VSWIN)


for type, dir in globo.UCFCRIME_VPATHS_LISTS.items():

    vpaths = list(open(dir))
    
    print(f'\n****************\n\nProcessing {type} @ {dir}\n\n{len(vpaths)} vids')
    
    features_list = []
    for i, vpath in enumerate(vpaths):
        
        vpath = vpath.strip('\n')
        fn = os.path.splitext(os.path.basename(vpath))[0]
        print(f'\n{i} {fn}.mp4\n')

        ## divide in2 snippets of 32
        clips, frames = get_video_clips(vpath , clip_length = 32 , resolution = 224)
        clips = clips.astype(np.float32)
        
        clips_prep_tf = tf.image.per_image_standardization(clips)
        print("\tclips_prep_tf",np.shape(clips_prep_tf) , clips_prep_tf.dtype)
        
        clips_prep_tf = np.transpose(clips_prep_tf, (0, 4, 1, 2, 3))
        print("\tclips_prep_tf",np.shape(clips_prep_tf) , clips_prep_tf.dtype)
        
        features = model(clips_prep_tf).numpy()
        print("\tfeatures",np.shape(features) , features.dtype)
        
        #features = normalize(features, axis=1)
        features_list.append(features)
        break

    print("\n\nEND END ",np.shape(features_list))    
    # Save all the features for the current type as a single .npz file
    out_npz = os.path.join(globo.UCFCRIME_FEATC3D_BASE_DIR, type + ".npz")
    #np.savez(out_npz, *features_list)
    print(f'\n\tsaved {type} features @ {out_npz}')