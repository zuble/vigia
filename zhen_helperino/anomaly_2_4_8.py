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


frame_no = 8

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
    class_weights = [10,10,10,10,10,10,10,10,10,10,10,10,0.1,10]
    
    optimizer_adam = keras.optimizers.SGD(learning_rate = 0.0002)
    model.compile(optimizer=optimizer_adam, 
                  loss= 'binary_crossentropy', 
                  #loss_weights = class_weights,
                  metrics=['accuracy'])
    return model

def input_video_data(file_name):
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
import time
import random
#from keras.utils import to_categorical
def generate_input():
    #has_visited = [0 for i in range(len(train_filenames))]
    loop_no = 0
    while 1:
        index = update_index[loop_no]
        loop_no += 1
        if loop_no == len(train_filenames):
            loop_no = 0
            
        #index = 0
        batch_frames, batch_frames_flip, total_frames = input_video_data(train_filenames[index])
        #print("frames ready")
        '''
        if 'Abuse' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0])])
        elif 'Arrest' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0])])
        elif 'Arson' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0])])
        elif 'Assault' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0])]) 
        elif 'Burglary' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0])]) 
        elif 'Explosion' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0])])        
        elif 'Fighting' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0])]) 
        elif 'RoadAccidents' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0])]) 
        elif 'Robbery' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0])]) 
        elif 'Shooting' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0])]) 
        elif 'Shoplifting' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0])]) 
        elif 'Stealing' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0])])
        elif 'Normal' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0])]) 
        elif 'Vandalism' in train_filenames[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1])]) 
        '''
        if 'Normal' in train_filenames[index]:
            yield batch_frames, np.array([0])
        else:
            yield batch_frames, np.array([1])

        if 'Normal' in train_filenames[index]:
            yield batch_frames_flip, np.array([0])
        else:
            yield batch_frames_flip, np.array([1])
            
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


def crime_test():
    class_name = ['Abuse','Arrest','Arson','Assault','Burglary',
                  'Explosion','Fighting','RoadAccidents','Robbery','Shooting',
                  'Shoplifting','Stealing','Normal','Vandalism']
    test_root = server_video_loc
    f = open('/home/zhen/anomaly/test/'+ para_file_name + '_ucf.txt', 'w')
    content_str = ''
    start_time = time.time()
    total_frames_test = 0
    for i in range(len(test_name)):
        if test_name[i] != '':
            file_path = os.path.join(test_root, test_name[i])
            if 'Normal' in test_name[i]:
                file_path = os.path.join('/raid/DATASETS/anomaly/UCF_Crimes/', test_name[i])
                            
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
            if class_name[0] in test_name[i]:
                print('Correct: ' + test_name[i] + ' ' + str(predict_result))
            else:
                print('Wrong: ' + test_name[i] + ' ' + str(predict_result))
            content_str += test_name[i] + '\t' + str(predict_result)  + '\n'
    
    end_time = time.time()
               
    f.write(content_str)
    f.close()
    print(str(total_frames_test / (end_time - start_time)) + 'done...........')                  
        

#server_testname_file = '/home/zhen/Documents/Remote/raid/DATASETS/anomaly/Anomaly_Test.txt'
server_testname_file = '/raid/DATASETS/anomaly/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt'
test_file = open(server_testname_file,'r')
test_name = test_file.read()
test_name = test_name.split('\n')
test_name_dict = {}
for i in range(len(test_name)):
    if test_name[i] != '' and test_name[i].find('/') != -1:
        test_name_dict[test_name[i][test_name[i].find('/')+1:]] = 1
        
import os        
total_video_no = 0
train_filenames = []
#server_video_loc = '/home/zhen/Documents/Remote/raid/DATASETS/anomaly/UCF_Crimes/Videos'
server_video_loc = '/raid/DATASETS/anomaly/UCF_Crimes/Videos/'
for root, dirs, files in os.walk(server_video_loc):
     for file in files:
         total_video_no += 1
         if file not in test_name_dict and file.find('.mp4') != -1:
             train_filenames.append(os.path.join(root, file))
             
print(len(train_filenames), total_video_no, len(test_name_dict))

target_height = 120
target_width = 160
frame_max = 4000

has_visited = [0 for i in range(len(train_filenames))]
update_index = []
while 1:
    index = round(random.random()*len(train_filenames))
    if (index == len(train_filenames)):
        index = index - 1
    if has_visited[index] == 1:
        if sum(has_visited) != len(train_filenames):
            continue
        else:
            break
    else:
        update_index.append(index)
        has_visited[index] = 1

'''        
sequences_file = open('seq.txt', 'w')
for se in update_index:
     sequences_file.write(str(se)+'\n')
sequences_file.close()
'''
read_file = open('seq.txt','r')
lines = read_file.read().split('\n')
update_index = []
for line in lines:
    if line != '':
        update_index.append(int(line))

time_str = str(time.time())
print(time_str)

checkpoint = keras.callbacks.ModelCheckpoint(time_str + '-2_4_8_anomaly_{epoch:08d}.h5', period=1)

model = form_model()

para_file_name = '1626544688.4507377-2_4_8_anomaly_00000019.h5'



'''
from tqdm.keras import TqdmCallback
import pandas as pd
#model.load_weights(para_file_name)

history = model.fit_generator(generate_input(), steps_per_epoch=len(train_filenames)*2, 
                    epochs=40, verbose =0, callbacks = [checkpoint, TqdmCallback(verbose=2)])  
model.save(time_str + '_2_4-8_model')
model.save_weights(time_str + '_2_4-8_model.h5')  
hist_df = pd.DataFrame(history.history)
hist_csv_file = time_str + '_history.csv'
with open(hist_csv_file, mode = 'w') as f:
    hist_df.to_csv(f)
'''
import os
for i in range(1,51):
    if i < 10:
        para_file_name = '1626253985.274994_2_4_8_combined_anomaly_0000000' + str(i) + '.h5'
    else:
        para_file_name = '1628764909.0172913_2_4_8_combined_anomaly_000000' + str(i) + '.h5'
    if not os.path.isfile('/home/zhen/anomaly/test/' + para_file_name + '_ucf.txt'):
        model.load_weights(para_file_name)
        crime_test()   

