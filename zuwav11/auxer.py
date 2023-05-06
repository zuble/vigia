'''import cv2

video = cv2.VideoCapture('/raid/DATASETS/anomaly/XD_Violence/testing_copy/v=38GQ9L2meyE__#1_label_B6-0-0.mp4')
while video.isOpened() :
    sucess, frame = video.read()
    if sucess:
        cv2.imshow("w",frame)
        key = cv2.waitKey(1)
        
        if key == ord('q'): break  # quit
        if key == ord(' '):  # pause
            while True:
                key = cv2.waitKey(1)
                if key == ord(' '):break
    else: break
video.release()
cv2.destroyAllWindows()
'''

import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



'''
import os, time, random, logging , datetime , cv2 , csv , subprocess , json
import numpy as np

import tensorflow as tf
print("tf",tf.version.VERSION)
from tensorflow import keras
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from utils import globo , xdv , tfh5 , sinet


##  GPU CONFIGURATION

#os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
#tfh5.set_tf_loglevel(logging.ERROR)
#tfh5.tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
#tfh5.set_memory_growth()
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


CFG_SINET = {
    
    'sinet_version': 'fsd-sinet-vgg42-tlpf_aps-1',
    
    'graph_filename' : os.path.join(globo.FSDSINET_PATH,"fsd-sinet-vgg42-tlpf_aps-1.pb"),
    'metadata_file'  : os.path.join(globo.FSDSINET_PATH,"fsd-sinet-vgg42-tlpf_aps-1.json"),
    
    'audio_fs_input':22050,
    'batchSize' : 64,
    'lastPatchMode': 'repeat',
    'patchHopSize' : 50,
    
    
    'anom_labels' : ["Alarm","Boom","Crowd","Dog","Drill","Explosion","Fire","Gunshot and gunfire","Hammer","Screaming","Screech",\
                    "Shatter","Shout","Siren","Slam","Squeak","Yell"],
    'anom_labels_i' : [4,18,51,59,65,72,78,92,94,145,146,147,148,152,154,161,198],
    
    'anom_labels2' : ["Boom","Explosion","Fire","Gunshot and gunfire","Screaming",\
                    "Shout","Siren","Yell"],
    'anom_labels_i2' : [18,72,78,92,147,148,152,198],
    
    'full_or_max' : 'max', #chose output to (timesteps,labels_total) ot (1,labels_total)
    'labels_total' : 200
    
}
CFG_WAV= {
    
    "full_or_max" : CFG_SINET["full_or_max"],
    "sinet_aas_len" : CFG_SINET["labels_total"], 
    
    "shuffle" : False,
    
    "ativa" : 'relu',
    "optima" : 'sgd',
    "batch_type" : 0,   # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : 8000,
    "ckpt_start" : f"{0:0>8}",  #used in train_model: if 00000000 start from scratch, else start from ckpt with CFG_WAV stated
    
    "epochs" : 1,
    "batch_size" : 1
    
}

data = np.load(os.path.join(globo.AAS_PATH,f"{CFG_SINET['sinet_version']}--fl_train.npz"), allow_pickle=True)["data"]


def rescale_sigmoid(data):
    
    def sigmoid(x):return 1 / (1 + np.exp(-x))
    
    num_features = data.shape[1]
    scaled_data = np.zeros_like(data)

    for feature_idx in range(num_features):
        feature_data = data[:, feature_idx]

        # Center the data around the mean
        centered_data = feature_data - np.mean(feature_data)

        # Apply sigmoid function
        scaled_feature_data = sigmoid(centered_data)
        scaled_data[:, feature_idx] = scaled_feature_data

    return scaled_data

def min_max_scaling(data):
    num_features = data.shape[1]
    scaled_data = np.zeros_like(data)

    for feature_idx in range(num_features):
        feature_data = data[:, feature_idx]

        min_val = np.min(feature_data)
        max_val = np.max(feature_data)

        # Avoid division by zero
        feature_range = max_val - min_val
        if feature_range == 0:
            feature_range = 1
        
        scaled_feature_data = (feature_data - min_val) / feature_range
        scaled_data[:, feature_idx] = scaled_feature_data
        
        #print(f"{feature_idx} , max{max_val} , min{min_val} , fr{feature_range}")
    
    return scaled_data


metadata = json.load(open(CFG_SINET['metadata_file'], "r"))
labels = metadata["classes"]

for k in range(len(data)):
    #k = 214
    label = data[k]['label']

    p_es_arr = data[k]['p_es_array']
    #p_es_arr_norm = min_max_scaling(p_es_arr)
    #p_es_arr_sigm = rescale_sigmoid(p_es_arr)

    print("*************************\n")
    print(label,"\n",np.shape(p_es_arr))
    # np.shape(p_es_arr_norm))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    timesteps = np.shape(p_es_arr)[0]
    print(timesteps)
    # Preparing the coordinates for the 3D scatter plot
    x = np.repeat(np.arange(timesteps), len(labels))
    y = np.tile(np.arange(len(labels)), timesteps)
    z = p_es_arr.ravel()

    # Plotting the 3D scatter plot
    ax.scatter(x, y, z)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Classes')
    ax.set_zlabel('Values')
    plt.title(f'3D Visualization for Position {k}')
    plt.show()
   
   
    
    #for label_i in range(len(labels)):
    #    #print('\n\n',p_es_arr[:, label_i])
    #    print("\n\n",labels[label_i])
    #    label_array = p_es_arr[:, label_i]
    #    
    #    maxx = np.amax(label_array,axis=0)
    #    avgg = np.mean(label_array)
    #    print("@ MAX", maxx)
    #    print("@ AVG", avgg)
    #    #mmmaxx = np.amax(p_es_arr_norm[:, label_i])
    #    #mmavgg = np.mean(p_es_arr_norm[:, label_i])
    #    
    #    #print("\nORIGNAL",label_array.shape,"\n",label_array)
    #    #print("\nMinMaxNORM\n",p_es_arr_norm[:,label_i])
    #    #print("\nSIGMOID\n",p_es_arr_sigm[:,label_i])
    #    
    #    #print("@ mmMAX",mmmaxx)
    #    #print("@ AVG", avgg,"@ mmAVG",mmavgg)
    


    
    #for i in range(len(CFG_SINET['anom_labels_i2'])):
    #    label_i = CFG_SINET['anom_labels_i2'][i]
    #    print("\n\n********\n",label_i, CFG_SINET['anom_labels2'][i], label)
    #
    #    label_array = p_es_arr[:, label_i]
    #    
    #    maxx = np.amax(label_array,axis=0)
    #    avgg = np.mean(label_array)
    #    
    #    print("@ MAX", maxx)
    #    print("@ AVG", avgg)
    #    
    #    #mmmaxx = np.amax(p_es_arr_norm[:, label_i])
    #    #mmavgg = np.mean(p_es_arr_norm[:, label_i])
    #    
    #    #print("\nORIGNAL",label_array.shape,"\n",label_array)
    #    #print("\nMinMaxNORM\n",p_es_arr_norm[:,label_i])
    #    #print("\nSIGMOID\n",p_es_arr_sigm[:,label_i])
    #    
    #    #print("@ mmMAX",mmmaxx)
    #    #print("@ AVG", avgg,"@ mmAVG",mmavgg)
    #
    #if k == 6: break

'''