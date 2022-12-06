# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:18:10 2022

@author: zuble
"""

import cv2
import numpy as np
#import mtcnn
import keras
from keras import models, layers
import tensorflow as tf
import os, time, random, logging
from tqdm.keras import TqdmCallback
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


base_vigia_dir = "/media/jtstudents/HDD/.zuble/vigia"


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def input_video_data(file_name):
    print("\n\nINPUT_VIDEO_DATA\n")
    #file_name = 'C:\\Bosch\\Anomaly\\training\\videos\\13_007.avi'
    #file_name = '/raid/DATASETS/anomaly/UCF_Crimes/Videos/Training_Normal_Videos_Anomaly/Normal_Videos308_x264.mp4'
    video = cv2.VideoCapture(file_name)
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    #mtcnn_detector = mtcnn.mtcnn.MTCNN()
    #print(file_name + '  ' + str(total_frame))
    divid_no = 1
    
    if total_frame > frame_max:
        total_frame_int = int(total_frame)
        if total_frame_int % frame_max == 0:
            divid_no = int(total_frame / frame_max)
        else:
            divid_no = int(total_frame / frame_max) + 1
        
    batch_no = 0
    batch_frames = []
    batch_frames_flip = []
    counter = 0
    if 'Normal' in file_name:
        print("\n\nNORMAL\n\n")
        if divid_no != 1:
            slice_no = int(random.random()*divid_no)
            passby = 0
            if slice_no != divid_no - 1:
                while video.isOpened and passby < frame_max * slice_no:
                    passby += 1
                    success, image = video.read()
                    if success == False:
                        break
            else:
                while video.isOpened and passby < total_frame - frame_max:
                    passby += 1
                    success, image = video.read()
                    if success == False:
                        break

    while video.isOpened:               
        success, image = video.read()
        if success == False:
            break
            
        #ratio = image.shape[0] / image.shape[1]
        #print(str(image.shape[0])+ ' ' + str(image.shape[1]))
        #image = cv2.resize(image, (800, int(800*ratio)))
        #print(image.shape)
        #faces = face_detector.detectMultiScale(image,1.1,8)
        '''
        faces = mtcnn_detector.detect_faces(image)
        
        for face in faces:
            (x,y,w,h) = face['box']
            #print(face)
            cv2.rectangle(image,(x,y), (x+w,y+h), (255,255,0), 2)
            cv2.putText(image, str(face['confidence'])[:4], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        '''
        image = cv2.resize(image, (target_width, target_height))
        image_flip = cv2.flip(image, 1)
        
        image_array = np.array(image)/255.0
        image_array_flip = np.array(image_flip)/255.0
        
        batch_frames.append(image_array)
        batch_frames_flip.append(image_array_flip)
        
        counter += 1
        if counter > frame_max:
            break
            
    video.release()
    batch_frames = np.array(batch_frames)
    #print(batch_frames.shape)
        
    return np.expand_dims(batch_frames,0), np.expand_dims(batch_frames_flip, 0), total_frame
    
    '''
    cv2.imshow('show', image)
    keyInput = cv2.waitKey(1)
    if keyInput == 27:
        break
    '''

#cv2.destroyWindow('show')
#from keras.utils import to_categorical
def generate_input():
    #has_visited = [0 for i in range(len(train_fn))]
    
    print("\n\nGENERATE_INPUT\n")
    loop_no = 0
    while 1:
        index = update_index[loop_no]
        loop_no += 1
        if loop_no == len(train_fn):
            loop_no = 0
            
        #index = 0
        batch_frames, batch_frames_flip, total_frames = input_video_data(train_fn[index])
        print("\ntrain_fn[",index,"]=",train_fn[index],"\ntotal_frames=",total_frames,"\nbatch_frames.shape=",batch_frames.shape)
        #if batch_frames.ndim != 5:
        #   break
        
        # GENERATORS    
        #       a kind of iterators that can only iterate over once
        #       NO store of values in memory
        # YIELD 
        #       like a return, except the function will return a generator
        
        '''
        if 'Abuse' in train_fn[index]:
            yield batch_frames,  np.array([np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0])])
        elif 'Arrest' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0])])
        elif 'Arson' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0])])
        elif 'Assault' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0])]) 
        elif 'Burglary' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0])]) 
        elif 'Explosion' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0])])        
        elif 'Fighting' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0])]) 
        elif 'RoadAccidents' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0])]) 
        elif 'Robbery' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0])]) 
        elif 'Shooting' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0])]) 
        elif 'Shoplifting' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0])]) 
        elif 'Stealing' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0])])
        elif 'Normal' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0])]) 
        elif 'Vandalism' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1])]) 
        '''
        
        #batch_frames
        if 'label_A' in train_fn[index]:
            yield batch_frames, np.array([0])   #normal
        else:
            yield batch_frames, np.array([1])   #abnormal

        #batch_frames_flip
        if 'label_A' in train_fn[index]:
            yield batch_frames_flip, np.array([0])  #normal
        else:
            yield batch_frames_flip, np.array([1])  #abnormal
    
    print("loop_no=",loop_no)

def input_test_video_data(file_name, batch_no=0):
    #file_name = 'C:\\Bosch\\Anomaly\\training\\videos\\13_007.avi'
    #file_name = '/raid/DATASETS/anomaly/UCF_Crimes/Videos/Training_Normal_Videos_Anomaly/Normal_Videos308_x264.mp4'
    video = cv2.VideoCapture(file_name)
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    #mtcnn_detector = mtcnn.mtcnn.MTCNN()
    divid_no = 1

    if total_frame > frame_max:
        total_frame_int = int(total_frame)
        if total_frame_int % frame_max == 0:
            divid_no = int(total_frame / frame_max)
        else:
            divid_no = int(total_frame / frame_max) + 1

    passby = 0
    #updates the start frame to 0,4000,8000... excluding the last batch
    if batch_no != divid_no - 1:
        while video.isOpened and passby < frame_max * batch_no:
            passby += 1
            success, image = video.read()
            if success == False:
                break
    #updates the last batch starting frame 
    else:
        if batch_type==2:
            #print("2")
            while video.isOpened and passby < frame_max * batch_no:
                passby += 1
                success, image = video.read()
                if success == False:
                    break
        if batch_type==1:
            #print("1")
            while video.isOpened and passby < total_frame - frame_max:
                passby += 1
                success, image = video.read()
                if success == False:
                    break
            
    batch_frames = []
    counter = 0
    
    while video.isOpened:               
        success, image = video.read()
        if success == False:
            break
        
        image = cv2.resize(image, (target_width, target_height))
        image_array = np.array(image)/255.0 #normalize
        batch_frames.append(image_array)
        cv2.imshow('frame', image)
        
        counter += 1
        if counter > frame_max:
            break
            
    video.release()
    batch_frames = np.array(batch_frames)
    
    print("\t-batch",batch_no,"[",passby,", ... ] ",batch_frames.shape)    
    
    return np.expand_dims(batch_frames,0), divid_no, total_frame, fps


def test_files():
    """
    GENERATE LIST of TEST FILES
    """
    test_fn, test_normal_fn, test_abnormal_fn = [],[],[]
    y_test_labels, y_test_norm, y_test_abnor = [],[],[]
    #server_testname_folder = '/raid/DATASETS/anomaly/XD_Violence/testing'
    server_testname_folder = '/media/jtstudents/HDD/.zuble/xdviol/test'
    for root, dirs, files in os.walk(server_testname_folder):
        for file in files:
            if file.find('.mp4') != -1:
                if 'label_A' in file:
                    test_normal_fn.append(os.path.join(root, file))
                    y_test_norm.append(0)
                else:
                    test_abnormal_fn.append(os.path.join(root, file))
                    y_test_abnor.append(1)
                    
    test_fn = test_normal_fn + test_abnormal_fn
    y_test_labels = y_test_norm + y_test_abnor
    
    print("\ntest_fn",np.shape(test_fn),"\ntest_normal_fn",np.shape(test_normal_fn),"\ntest_abnormal_fn",np.shape(test_abnormal_fn))
    print("\ny_test_labels",np.shape(y_test_labels),"\ny_test_norm",np.shape(y_test_norm),"\ny_test_abnor",np.shape(y_test_abnor))
    
    return test_fn, y_test_labels

def train_files():
    """
    GENERATING LIST of TRAIN FILES
    """
    train_fn = []
    #server_video_loc = '/home/zhen/Documents/Remote/raid/DATASETS/anomaly/UCF_Crimes/Videos'
    #server_video_loc = '/raid/DATASETS/anomaly/XD_Violence/training/'
    server_video_loc = '/media/jtstudents/HDD/.zuble/xdviol/train'
    for root, dirs, files in os.walk(server_video_loc):
        for file in files:
            if file.find('.mp4') != -1:
                train_fn.append(os.path.join(root, file))
                
    print("\ntrain_fn=",np.shape(train_fn))
    
    return train_fn


def all_operations(args):
    x = args[0]
    #tf.print(x.shape)
    x = tf.reshape(x, [1, -1,x.shape[1]*x.shape[2]*x.shape[3]])
    return x
@tf.function
def loss_category(y_true, y_pred):    
    #tf.print(y_pred, y_true, 'Prediction')
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return cce


def find_weights(): 
    weights_fn = []
    weights_path = []
    weights_base_path = base_vigia_dir+"/zhen++/parameters_saved"

    for file in os.listdir(weights_base_path):
        fname, fext = os.path.splitext(file)
        if fext == ".h5" and file.find('_2_4_8_xdviolence_model_weights') != -1 :
            print(file)
            weights_path.append(os.path.join(weights_base_path, file))
            weights_fn.append(file)

    return weights_fn, weights_path

def form_model():
    print("\nFORM_MODEL\n")
    image_input = keras.Input(shape=(None, target_height, target_width, 3))
    #Freeze the batch normalization
    
    c3d_layer1 = keras.layers.Conv3D(4,(2,3,3), activation='relu')(image_input)
    c3d_pooling1 = keras.layers.MaxPooling3D((1,2,2))(c3d_layer1)
    c3d_layer2 = keras.layers.Conv3D(8,(4,3,3), activation='relu')(c3d_pooling1)
    c3d_pooling2 = keras.layers.MaxPooling3D((2,2,2))(c3d_layer2)
    c3d_layer3 = keras.layers.Conv3D(16,(8,3,3), activation='relu')(c3d_pooling2)
    c3d_pooling3 = keras.layers.MaxPooling3D((4,2,2))(c3d_layer3)
    #c3d_layer4 = keras.layers.Conv3D(32,(2,3,3), activation='relu')(c3d_pooling3)
    #c3d_pooling4 = keras.layers.MaxPooling3D((2,2,2))(c3d_layer4)
    
    feature_conv_4 = keras.layers.Lambda(all_operations)(c3d_pooling3)
    
    lstm1 = keras.layers.LSTM(1024,input_shape=(1200,feature_conv_4.shape[2]), return_sequences=True)(feature_conv_4)
    lstm2 = keras.layers.LSTM(512, return_sequences=True)(lstm1)
    global_feature = keras.layers.GlobalMaxPooling1D()(lstm1)
    
    #ADD THE AUDIO FEATURE HERE 
    
    dense_1 = keras.layers.Dense(128, activation='relu')(global_feature)
    #dense_2 = keras.layers.Dense(13, activation='sigmoid')(dense_1)
    
    soft_max = keras.layers.Dense(1, activation='sigmoid')(dense_1)
    
    
    model = models.Model(inputs=[image_input], outputs=[soft_max])
    model.summary()
    
    
    #class_weights = [10,10,10,10,10,10,10,10,10,10,10,10,0.1,10]
    optimizer_adam = keras.optimizers.SGD(learning_rate = 0.0002)
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    
    model.compile(optimizer=optimizer_adam, 
                    loss= 'binary_crossentropy', 
                    #loss_weights = class_weights,
                    #metrics=['accuracy']
                    metrics=METRICS)
    return model

def train_model():
    '''
    MODEL TRAIN/VALIDATION 
    (silent mode - verbose = 0)
    '''

    #https://keras.io/api/callbacks/model_checkpoint/
    ckpt_path = base_vigia_dir+'/zhen_++/'+time_str+'_2_4_8_xdviolence_anomaly_{epoch:08d}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(ckpt_path, save_freq='epoch')

    #para_file_name = '.262731_2_4_8_xdviolence_anomaly_00000010.h5'
    #model.load_weights(para_file_name)

    print("\n\nMODEL.FIT")
    history = model.fit(generate_input(), 
                        steps_per_epoch=len(train_fn)*2, 
                        epochs=30, 
                        verbose=1, 
                        callbacks=[checkpoint, TqdmCallback(verbose=2)])

    model.save(time_str + '_2_4_8_xdviolence')
    model.save_weights(time_str + '_2_4_8_xdviolence_model_weights.h5')  

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = time_str + '_xdviolence_history.csv'
    with open(hist_csv_file, mode = 'w') as f:
        hist_df.to_csv(f)
        
    return model

def test_model(model):
    print("\nTEST MODEL\n")
    f = open(base_vigia_dir+'/zhen++/parameters_results/'+model_weight_fn+'_'+str(batch_type)+'.txt', 'w')
    content_str = ''
    total_frames_test = 0
    predict_total= [] 
    
    start_test = time.time()
    for i in range(len(test_fn)):
        if test_fn[i] != '':
            file_path = test_fn[i]
            
            #the frist 4000 frames from actual test video                
            frame1, divid_no, total_frames, fps = input_test_video_data(file_path)
            video_time = total_frames/fps
            total_frames_test += total_frames
            
            start_predict1 = time.time()
            predict_result = model.predict(frame1)[0][0]
            end_predict1 = time.time()
            time_predict = end_predict1-start_predict1
            
            high_score_patch = 0
            print("\t ",predict_result,"%") 
            
            
            #when frame1 (input video) has > 4000 frames
            patch_num = 1
            while patch_num < divid_no:
                frame1, divid_no, total_frames, fps = input_test_video_data(file_path, patch_num)
                
                start_predict2 = time.time()
                predict_new = model.predict(frame1)[0][0]
                end_predict2 = time.time()
                time_predict += end_predict2 - start_predict2
                
                if predict_new > predict_result:
                    predict_result = predict_new
                    high_score_patch = patch_num
                    
                print("\t ",predict_result,"%")  
                patch_num += 1
                
            predict_total.append(predict_result)
            
            if 'label_A' in test_fn[i]:
                print('\nNORM',str(i),':',f'{total_frames:.0f}',"@",f'{fps:.0f}',"fps =",f'{video_time:.2f}',"sec\n\t",
                        test_fn[i][test_fn[i].rindex('/')+1:],
                        "\n\t "+str(predict_result),"% @batch",high_score_patch,"in",str(time_predict),"seconds\n",
                        "----------------------------------------------------\n")
            else:
                print('\nABNORM',str(i),':',f'{total_frames:.0f}',"@",f'{fps:.0f}',"fps =",f'{video_time:.2f}',"sec\n\t",
                        test_fn[i][test_fn[i].rindex('/')+1:],
                        "\n\t"+str(predict_result),"% @batch",high_score_patch,"in",str(time_predict),"seconds\n",
                        "----------------------------------------------------\n")
                
            content_str += test_fn[i][test_fn[i].rindex('/')+1:] + '\t' + str(predict_result)  + '\n'
    
    end_test = time.time()
    time_test = end_test - start_test

    f.write(content_str)
    f.close()
    print("\nDONE\n\ttotal of",str(total_frames_test),"frames processed in",time_test," seconds",
            "\n\t"+str(total_frames_test / time_test),"frames per second",
            "\n\n********************************************************",
            "\n\n********************************************************")                  

    return predict_total


def plot_cm(y_test_pred,p=0.5):
    '''
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#download_the_kaggle_credit_card_fraud_data_set
    '''
    y_test_pred_array = np.array(y_test_pred)
    cm = confusion_matrix(y_test_labels, y_test_pred_array > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(base_vigia_dir+'/zhen++/parameters_results/'+model_weight_fn+'_CM'+str(batch_type)+'.png')              



'''
    GPU CONFIGURATION
    https://www.tensorflow.org/guide/gpu
'''
set_tf_loglevel(logging.WARNING)

#Enabling device placement logging causes any Tensor allocations or operations to be printed.
tf.debugging.set_log_device_placement(False) 

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
gpus = tf.config.list_physical_devices('GPU')


#https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth

#if gpus:
#    print("\nAvaiable GPU's",gpus)
#    try:
#        # Currently, memory growth needs to be the same across GPUs
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#        
#        logical_gpus = tf.config.list_logical_devices('GPU')
#        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#    except RuntimeError as e:
#        # Memory growth must be set before GPUs have been initialized
#        print(e)



target_height = 120
target_width = 160
frame_max = 4000

test_fn, y_test_labels = test_files()
train_fn = train_files()

update_index = range(0, len(train_fn))
time_str = str(time.time())
print("\ntime_str=",time_str,"\n")        


#model = form_model()
#model = train_model()
#test_model(model) 

model = form_model()
weights_fn, weights_path = find_weights()


# =1 last batch has 4000 frames // =2 last batch has no repetead frames
batch_type = 2
print("\n\n\t\tBATCH TYPE",batch_type)
    
for i in range(len(weights_fn)):
    print("\n\nLoading weights from",weights_fn[i],"\n")
    model.load_weights(weights_path[i])
    model_weight_fn,model_weight_ext = os.path.splitext(str(weights_fn[i]))
    
    y_test_pred = test_model(model)
    plot_cm(y_test_pred)


'''
for i in range(8,11):
    if i < 10:
        para_file_name = '1626947956.798592_2_4_8_xdviolence_anomaly_0000000' + str(i) + '.h5'
    else:
        para_file_name = '1626947956.798592_2_4_8_xdviolence_anomaly_000000' + str(i) + '.h5'
        
model.load_weights(para_file_name)
'''
