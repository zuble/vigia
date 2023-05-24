import time , os , cv2

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]='3'
os.environ['TF_ENABLE_CUDNN_AUTOTUNE'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet50, MobileNetV2, Xception
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_prep
from tensorflow.keras.applications.xception import preprocess_input as xception_prep

from utils import xdv


@tf.function
def vas_predict(model,x):
    return model(x, training=False)    

def get_n_frames(video_path, n):
    frames = []
    frames_orig = []
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [i * total_frames // n for i in range(n)]

    for i in range(total_frames):
        ret, frame = cap.read()
        if i in frame_indices:
            #frames_orig.append(frame)
            
            frame = cv2.resize(frame, ( input_shape[0] , input_shape[1]) )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = np.array(frame)/255.0
            frames.append(frame)
            
    cap.release()
    print(np.array(frames).shape)
    p = prep_fx(np.array(frames))
    print(p.shape)
    
    #frame_array = np.array(frames).astype(np.float32)
    return np.expand_dims(p, 0)

def test_model(n):
    test_fn, *_ = xdv.test_files()
    filtered_test_fn = [fn for fn in test_fn if 'label_A' not in fn]
    for i,fn in enumerate(test_fn):
        print(fn)
        if i == 2:break
        inp = get_n_frames(fn, n)
        t = time.time()
        #output = vas_predict(model,inp)[0][0].numpy()
        output = model(inp)
        print(f"Output shape: {output.shape}")
        tt = time.time()
        print(str(tt-t))

    
backbones = ["resnet50", "mobilenetv2", "xception"]
backbone = "mobilenetv2"
input_shapes = {
    "resnet50": (224, 224, 3),
    "mobilenetv2": (224, 224, 3),
    "xception": (150, 150, 3),
}
prep_fx = mobilenetv2_prep
input_shape = input_shapes[backbone]
image_input = keras.Input(shape=(None, *input_shape ))

if backbone == 'resnet50':
    backbone = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape = input_shape)
    backbone_out = keras.layers.TimeDistributed(backbone)(image_input)
        
elif backbone == 'mobilenetv2':
    backbone = tf.keras.applications.MobileNetV2(include_top=False, 
                                                 weights='imagenet', 
                                                 input_shape = input_shape , 
                                                 #pooling = 'max' ,
                                                 #alpha = 0.5 , 
                                                 #depth_multiplier=2
                                                )
    backbone_out = keras.layers.TimeDistributed(backbone)(image_input)
    
elif backbone == 'xception':
    #input_shape = (299, 299, 3)
    #image_input = keras.Input(shape=(None, *input_shape))
    backbone = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape = input_shape)
    backbone_out = keras.layers.TimeDistributed(backbone)(image_input)
    
for layer in backbone.layers: 
    layer.trainable = False    
    
model = keras.Model(inputs=[image_input], outputs=[backbone_out])

    
test_model(10)
