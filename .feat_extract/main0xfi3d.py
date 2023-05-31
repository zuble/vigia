import globo

import cv2 , time , os
import numpy as np , tensorflow as tf 
from sklearn.preprocessing import normalize

import i3d_horse
from utils import gerador_clips 



feature_extractor = i3d_horse.Inception_Inflated3d(
    include_top=True,
    weights='rgb_imagenet_and_kinetics',
    input_shape=(16, 224, 224, 3),
    feat_extractor=True )

#feature_extractor = i3d_horse.Inception_Inflated3d(
#    include_top=False,
#    weights='rgb_imagenet_and_kinetics',
#    input_shape=(16, 224, 224, 3) )

#print(rgb_feat_model.summary())


def preprocess_input(clip):
    ## https://github.com/OanaIgnat/I3D_Keras/blob/master/src/preprocess.py
    SMALLEST_DIM = 256
    IMAGE_CROP_SIZE = 224

    def resize(img):
        ''' 
            frame resized preserving aspect ratio so that the smallest dimension is 256 pixels,
            with bilinear interpolation
        '''
        # print('Original Dimensions : ', img.shape)
        original_width = int(img.shape[1])
        original_height = int(img.shape[0])
        aspect_ratio = original_width / original_height

        if original_height < original_width:
            new_height = SMALLEST_DIM
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = SMALLEST_DIM
            new_height = int(original_width / aspect_ratio)

        dim = (new_width, new_height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        # print('Resized Dimensions : ', resized.shape)
        return resized

    def crop_center(img, new_size):
        y, x, c = img.shape
        (cropx, cropy) = new_size
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]

    def rescale_pixel_values(img):
        #print('Data Type: %s' % img.dtype)
        #print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
        img = img.astype('float32') 
        #img /= 255.0 ## normalize to the range 0:1
        img = (img / 255.0) * 2 - 1 ## normalize to the range -1:1
        #print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
        return img

    ## COMPARE WITH THE ATUAL FX
    ## https://github.com/piergiaj/pytorch-i3d/blob/master/charades_dataset.py
    def load_rgb_frames(image_dir, vid, start, num):
        frames = []
        for i in range(start, start+num):
            img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
            w,h,c = img.shape
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            img = (img/255.)*2 - 1
            frames.append(img)
        return np.asarray(frames, dtype=np.float32)


    prep_clip = []
    for frame in clip:
        resized = resize(frame)
        frame_cropped = crop_center(resized, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE))
        frame = rescale_pixel_values(frame_cropped)
        #print(f'\t{np.shape(frame)} , numpy ? {isinstance(frame, np.ndarray)}')
        prep_clip.append(frame)
    return np.expand_dims(prep_clip, axis=0)



@tf.function
def predict(x): return feature_extractor(x, training=False)

def l2norm(x): return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)  


for type, dir in globo.UCFCRIME_VPATHS_LISTS.items():

    vpaths = list(open(dir))
    fpath = globo.UCFCRIME_I3D_FPATHS[type]

    print(f'\n\n****************\n\nProcessing {type} @ {dir}\n\n{len(vpaths)} vids\nsaving into {fpath}')
    
    for i, vpath in enumerate(vpaths):
        t = time.time()
        
        vpath = vpath.strip("\n");fn = os.path.splitext(os.path.basename(vpath))[0]
        print(f'\n{i} {fn}.mp4')
            
        out_npy = os.path.join(fpath, fn + ".npy")
        if os.path.exists(out_npy):
            print(out_npy,"already created")
            continue
             
        features = []
        for clip in gerador_clips(vpath):
        
            prep_clip = preprocess_input(clip)
            #print(f'\tPREP{np.shape(prep_clip)} , {prep_clip.dtype} , np ? {isinstance(prep_clip, np.ndarray)}')
            feature = predict(prep_clip) 
            #print(f'\tFEAT {np.shape(feature)}')
            feature = np.squeeze(feature, axis=(1, 2, 3))
            #print(f'\tFEAT {np.shape(feature)} , {feature.dtype} , np ? {isinstance(feature, np.ndarray)}')
            features.append(feature)

        features = np.concatenate(features) ## == np.array
        print(f'\tFEATED ( {np.shape(features)[0]} timesteps , {np.shape(features)[1]} features ) {features.dtype} , np ? {isinstance(features, np.ndarray)} , tf ? {isinstance(features, tf.Tensor)}')
        
        #features2 = normalize(features, axis=1)
        #features = l2norm(features)
        #print("\tL2NORM same ?",np.allclose(features,features2))
              
        np.save(out_npy, features)
        print(f'\n\tsaved @ {out_npy}')
        
        tt = time.time()
        print(f'\t~ {int(tt-t)} seconds')


