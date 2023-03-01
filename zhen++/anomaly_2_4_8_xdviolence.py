# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:04:10 2020

@author: Zhen
"""

import cv2
import numpy as np
#import mtcnn
import keras
from keras import models, layers
import tensorflow as tf
import os, time, random
import time
import random
import sklearn
from sklearn.metrics import confusion_matrix

#frame_no = 8


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
        
        if 'label_A' in train_fn[index]:
            yield batch_frames, np.array([0])   #normal
        else:
            yield batch_frames, np.array([1])   #abnormal

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
    #mtcnn_detector = mtcnn.mtcnn.MTCNN()
    #print(file_name + '  ' + str(total_frame))
    divid_no = 1

    if total_frame > frame_max:
        total_frame_int = int(total_frame)
        if total_frame_int % frame_max == 0:
            divid_no = int(total_frame / frame_max)
        else:
            divid_no = int(total_frame / frame_max) + 1

    passby = 0
    if batch_no != divid_no - 1:
        while video.isOpened and passby < frame_max * batch_no:
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
            
    batch_frames = []
    
    counter = 0


    while video.isOpened:               
        success, image = video.read()
        if success == False:
            break
            
        image = cv2.resize(image, (target_width, target_height))
        
        image_array = np.array(image)/255.0
        
        batch_frames.append(image_array)
        
        counter += 1
        if counter > frame_max:
            break
            
    video.release()
    batch_frames = np.array(batch_frames)
    #print(batch_frames.shape)
        
    return np.expand_dims(batch_frames,0), divid_no, total_frame


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


def form_model():
    print("\nFORM_MODEL")
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
    
    dense_1 = keras.layers.Dense(128, activation='relu')(global_feature)
    #dense_2 = keras.layers.Dense(13, activation='sigmoid')(dense_1)
    
    soft_max = keras.layers.Dense(1, activation='sigmoid')(dense_1)
    
    
    model = models.Model(inputs=[image_input], outputs=[soft_max])
    model.summary()
    
    
    #class_weights = [10,10,10,10,10,10,10,10,10,10,10,10,0.1,10]
    optimizer_adam = keras.optimizers.SGD(learning_rate = 0.0002)
    model.compile(optimizer=optimizer_adam, 
                    loss= 'binary_crossentropy', 
                    #loss_weights = class_weights,
                    metrics=['accuracy'])
    return model

def crime_test(model):
    #f = open('/home/zhen/anomaly/test/'+ para_file_name + '.txt', 'w')
    f = open('/media/jtstudents/HDD/.zuble/zhen_helperino' + time_str + '.txt', 'w')
    content_str = ''
    start_time = time.time()
    total_frames_test = 0
    for i in range(len(test_fn)):
        if test_fn[i] != '':
            file_path = test_fn[i]
                            
            frame1, divid_no, total_frames = input_test_video_data(file_path)
            total_frames_test += total_frames
            #print(file_path)
            predict_result = model.predict(frame1)[0][0]
            #pred_index = np.argmax(predict_result)
            
            patch_num = 1
            while patch_num < divid_no:                           
                frame1, divid_no, total_frames = input_test_video_data(file_path, patch_num)
                predict_new = model.predict(frame1)[0][0]
                if predict_new > predict_result:
                    predict_result = predict_new
                patch_num += 1
            if 'label_A' in test_fn[i]:
                print(test_fn[i][test_fn[i].rindex('/')+1:] + ' ' + str(predict_result) + ' ' + str(i))
            else:
                print(test_fn[i][test_fn[i].rindex('/')+1:] + ' ' + str(predict_result) + ' ' + str(i))
            content_str += test_fn[i][test_fn[i].rindex('/')+1:] + '\t' + str(predict_result)  + '\n'
    
    end_time = time.time()

    f.write(content_str)
    f.close()
    print(str(total_frames_test / (end_time - start_time)) + 'done...........')                  


'''
def simple_test():
    server_test_loc = '/raid/DATASETS/anomaly/UCF_Crimes/Videos/'
    all_dir = os.listdir(server_test_loc)
    for dirs in all_dir:
        frames_dir = server_test_loc + dirs
        test_frames = []
        for files1 in os.listdir(frames_dir):       
            image = cv2.imread(os.path.join(frames_dir, files1))
            image = cv2.resize(image, (107,60))
            image_array = np.array(image) / 255.0
            test_frames.append(image_array)
        test_frames = np.array(test_frames)
        test_frames = np.expand_dims(test_frames,0)
        cc = model.predict(test_frames)
        print(dirs + ' result: ' + str(np.argmax(cc) + 1))
'''

'''
#server_testname_file = '/home/zhen/Documents/Remote/raid/DATASETS/anomaly/Anomaly_Test.txt'
server_testname_file = '/raid/DATASETS/anomaly/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt'
test_file = open(server_testname_file,'r')
test_fn = test_file.read()
test_fn = test_fn.split('\n')
test_fn_dict = {}
for i in range(len(test_fn)):
    if test_fn[i] != '' and test_fn[i].find('/') != -1:
        test_fn_dict[test_fn[i][test_fn[i].find('/')+1:]] = 1
'''


"""
GENERATE LIST of TEST FILES
"""
def test_files():
    test_fn, test_normal_fn, test_abnormal_fn = [],[],[]
    #server_testname_folder = '/raid/DATASETS/anomaly/XD_Violence/testing'
    server_testname_folder = '/media/jtstudents/HDD/.zuble/xdviol/test'
    for root, dirs, files in os.walk(server_testname_folder):
        for file in files:
            if file.find('.mp4') != -1:
                if 'label_A' in file:
                    test_normal_fn.append(os.path.join(root, file))
                else:
                    test_abnormal_fn.append(os.path.join(root, file))
    test_fn = test_normal_fn + test_abnormal_fn
    print("\ntest_fn",np.shape(test_fn),"\ntest_normal_fn",np.shape(test_normal_fn),"\ntest_abnormal_fn",np.shape(test_abnormal_fn))

    return test_fn

"""
GENERATING LIST of TRAIN FILES
"""
def train_files():
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

'''
MODEL TRAIN/VALIDATION 
(silent mode - verbose = 0)
'''
from tqdm.keras import TqdmCallback
import pandas as pd
def model_train():
    
    model = form_model()
    physical_devices = tf.config.list_physical_devices('GPU')
    print("\nAvaiable GPU's",physical_devices)

    #https://keras.io/api/callbacks/model_checkpoint/
    ckpt_path = '/media/jtstudents/HDD/.zuble/zhen_helperino/'+time_str+'_2_4_8_xdviolence_anomaly_{epoch:08d}.h5'
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


test_fn = test_files()
train_fn = train_files()

target_height = 120
target_width = 160
frame_max = 4000

update_index = range(0, len(train_fn))


def get_precision_recall_f1(labels, predictions):
    p = tf.keras.metrics.Precision(thresholds = 0.5)
    p.update_state(labels, predictions)
    p_res = p.result().numpy()
    print("\tPRECISION (%% of True Positive out of all Positive predicted) ",p_res)
    
    r = tf.keras.metrics.Recall(thresholds=0.5)
    r.update_state(labels, predictions)
    r_res = r.result().numpy()
    print("\tRECALL (%% of True Positive out of all actual anomalies) ",r_res)
    
    #https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
    auprc_ap = sklearn.metrics.average_precision_score(labels, predictions)
    aucroc = sklearn.metrics.roc_auc_score(labels, predictions)
    print("\tAP ( AreaUnderPrecisionRecallCurve ) %.4f \n\t AUC-ROC %.4f "% (auprc_ap, aucroc))
    
    #https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
    #import tensorflow_addons as tfa
    #f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5)
    #f1.update_state(labels, predictions)
    f1_res = 2*((p_res*r_res)/(p_res+r_res+K.epsilon()))
    print("\tF1_SCORE (harmonic mean of precision and recall) ",f1_res)
    
    return p_res,r_res,auprc_ap,aucroc,f1_res

def buf_count_newlines_gen(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count

def get_results_from_txt(rslt_path):
    res_txt_fn = []
    res_model_fn = []
    
    for file in os.listdir(rslt_path):
        fname, fext = os.path.splitext(file)
        if fext == ".txt" and file.find('xdviolence') != -1:
            res_txt_fn.append(os.path.join(rslt_path, file))
            res_model_fn.append(fname)
    
    res_txt_fn = sorted(res_txt_fn)
    res_model_fn = sorted(res_model_fn)
    
    i=0
    #res_list_full = [[() for i in range(len(res_txt_fn))] for j in range(buf_count_newlines_gen(res_txt_fn[i]))]
    res_list_max = [[0.0 for i in range(len(res_txt_fn))] for j in range(buf_count_newlines_gen(res_txt_fn[i]))]
    res_list_fn = [['' for i in range(len(res_txt_fn))] for j in range(buf_count_newlines_gen(res_txt_fn[i]))]
    #print('res_list_full',np.shape(res_list_full))
    print('res_list_max',np.shape(res_list_max))
    print('res_list_fn',np.shape(res_list_fn))
    
    n_models = len(res_txt_fn)
    for txt_i in range(n_models):
        print('\nOPENING',res_txt_fn[txt_i])
        txt = open(res_txt_fn[txt_i],'r')
        txt_data = txt.read()
        txt.close()

        video_list = [line.split() for line in txt_data.split("\n") if line]
        
        for video_j in range(len(video_list)):
            aux_line = str(video_list[video_j]).replace('[','').replace(']','').replace(' ','').split('|')
        
            res_list_fn[video_j][txt_i] = aux_line[0]
            res_list_max[video_j][txt_i] = float(aux_line[1])
            
            #aux2_line = aux_line[2].replace(' ','').replace("'","").replace('(','').replace(')','').split(',')
            #print(aux2_line)
            #res_list_full[video_j][txt_i] = aux2_line
    
    
    res_list_labels = [[0 for i in range(len(res_txt_fn))] for j in range(buf_count_newlines_gen(res_txt_fn[i]))]
    n_videos = np.shape(res_list_fn)[0]
    for txt_i in range(n_models):
        for video_j in range(n_videos):
            if 'label_A' not in res_list_fn[video_j][txt_i]:
                res_list_labels[video_j][txt_i] = 1
        
    for i in range(len(res_txt_fn)):
        print("\nresults for",res_model_fn[i])
            
        res_col = [col[i] for col in res_list_max]
        labels_col = [col[i] for col in res_list_labels]
        get_precision_recall_f1(labels_col,res_col)
        
    return res_list_max, res_list_fn, res_list_labels


'''
has_visited = [0 for i in range(len(train_fn))]

while 1:
    index = round(random.random()*len(train_fn))
    if (index == len(train_fn)):
        index = index - 1
    if has_visited[index] == 1:
        if sum(has_visited) != len(train_fn):
            continue
        else:
            break
    else:
        update_index.append(index)
        has_visited[index] = 1
'''

'''
sequences_file = open(time_str+'_xd_seq.txt', 'w')
for se in update_index:
    sequences_file.write(str(se)+'\n')
sequences_file.close()
'''

'''
read_file = open('seq.txt','r')
lines = read_file.read().split('\n')
update_index = []
for line in lines:
    if line != '':
        update_index.append(int(line))
'''


time_str = str(time.time())
print("\ntime_str=",time_str,"\n")        


#model = model_train()

#crime_test(model) 

res_list_max, res_list_fn, res_list_labels  = get_results_from_txt('/raid/DATASETS/.zuble/vigia/zhen++/parameters_results/original_bt')


'''
for i in range(8,11):
    if i < 10:
        para_file_name = '1626947956.798592_2_4_8_xdviolence_anomaly_0000000' + str(i) + '.h5'
    else:
        para_file_name = '1626947956.798592_2_4_8_xdviolence_anomaly_000000' + str(i) + '.h5'
        
model.load_weights(para_file_name)
'''

'''
for i in range(26,51):
    if i < 10:
        para_file_name = '1625759299.9331803_2_4_8_xdviolence_anomaly_0000000' + str(i) + '.h5'
    else:
        para_file_name = '1628764909.0172913_2_4_8_combined_anomaly_000000' + str(i) + '.h5'
    
    #if not os.path.isfile('/home/zhen/anomaly/test/' + para_file_name + '.txt'):
    if not os.path.isfile('/media/jtstudents/HDD/.zuble/xdviol/test/' + para_file_name + '.txt'):
        model.load_weights(para_file_name)
        crime_test()
'''
