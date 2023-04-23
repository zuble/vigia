import os , cv2 

import numpy as np
from sklearn.model_selection import train_test_split

from utils import globo

#--------------------------------------------------------#
# XD VIOLENCE 

# TRAIN VALIDATION 

def train_valdt_files():
    """
    GENERATING LIST of TRAIN FILES
    """
    full_train_fn, full_train_normal_fn, full_train_abnormal_fn = [],[],[]
    full_train_labels, full_train_normal_labels, full_train_abnormal_labels = [],[],[]

    for root, dirs, files in os.walk(globo.SERVER_TRAIN_COPY_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                full_train_fn.append(os.path.join(root, file))
                
                if 'label_A' in file:
                    full_train_normal_fn.append(os.path.join(root, file))
                    full_train_normal_labels.append(0)

                else:
                    full_train_abnormal_fn.append(os.path.join(root, file))
                    full_train_abnormal_labels.append(1)
    #BEFORE SPLIT INTO TRAIN+VALD
    print("\nfull_train_fn",np.shape(full_train_fn),"\nfull_train_normal_fn",np.shape(full_train_normal_fn),"\nfull_train_abnormal",np.shape(full_train_abnormal_fn))
    

    #AFTER SPLIT
    valdt_fn, valdt_normal_fn, valdt_abnormal_fn = [],[],[]
    valdt_labels, valdt_normal_labels, valdt_abnormal_labels = [],[],[]

    train_fn, train_normal_fn, train_abnormal_fn = [],[],[]
    train_labels, train_normal_labels, train_abnormal_labels = [],[],[]

    train_fn, valdt_fn = train_test_split(full_train_fn, test_size=0.2,shuffle=False)
    for i in range(len(train_fn)):
        if 'label_A' in train_fn[i]:train_normal_fn.append(train_fn[i]);train_normal_labels.append(0);train_labels.append(0)
        else: train_abnormal_fn.append(train_fn[i]);train_abnormal_labels.append(1);train_labels.append(1)
    
    print("\ntrain_fn",np.shape(train_fn),"\ntrain_normal_fn",np.shape(train_normal_fn),"\ntrain_abnormal_fn",np.shape(train_abnormal_fn))
    
    
    for i in range(len(valdt_fn)):
        if 'label_A' in valdt_fn[i]:valdt_normal_fn.append(valdt_fn[i]);valdt_normal_labels.append(0);valdt_labels.append(0)
        else: valdt_abnormal_fn.append(valdt_fn[i]);valdt_abnormal_labels.append(1);valdt_labels.append(1)   
    
    print("\nvaldt_fn",np.shape(valdt_fn),"\nvaldt_normal_fn",np.shape(valdt_normal_fn),"\nvaldt_abnormal_fn",np.shape(valdt_abnormal_fn))

    return train_fn, train_labels, valdt_fn, valdt_labels

 

# DATASET INFO

def load_xdv_test():
    '''
        load mp4 paths for test videos
        if aac path given, aac test audio paths arre returned
    '''
    mp4_paths, mp4_labels = [],[]
    for root, dirs, files in os.walk(globo.SERVER_TRAIN_COPY_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths.append(os.path.join(root, file))
    mp4_paths.sort()
    for i in range(len(mp4_paths)):            
        if 'label_A' in  mp4_paths[i] : mp4_labels.append(0)
        else:mp4_labels.append(1)
    
    return mp4_paths,mp4_labels

def load_xdv_train():
    mp4_paths, mp4_labels = [],[]
    for root, dirs, files in os.walk(globo.SERVER_TRAIN_COPY_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths.append(os.path.join(root, file))
    mp4_paths.sort()
    for i in range(len(mp4_paths)):            
        if 'label_A' in  mp4_paths[i] : mp4_labels.append(0)
        else:mp4_labels.append(1)
    
    return mp4_paths,mp4_labels


def get_xdv_info(test=False,train=False):
    " 'test' or 'train' "
    
    if test:
        mp4_paths,mp4_labels = load_xdv_test()
        txt_fn = '/raid/DATASETS/.zuble/vigia/zurgb/dataset-xdv-info/test.txt'
    elif train:
        mp4_paths,mp4_labels = load_xdv_train()
        txt_fn = '/raid/DATASETS/.zuble/vigia/zurgb/dataset-xdv-info/train.txt'
    else: raise Exception("not a valid string")
    print(np.shape(mp4_paths))

    data = '';aux=0;total=0;line=''
    for path in mp4_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        video = cv2.VideoCapture(path)
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        video_time = frames/fps
        video.release()

        aux+=1;total+=frames

        line=str(round(frames))+' frames | '+str(round(video_time))+' secs | '+fname+'\n'
        data+=line
        print(line)

    line = "\nmean of frames per video: "+str(total/aux)
    data+=line

    f = open(txt_fn, 'w')        
    f.write(data)
    f.close()  


def get_testxdvanom_info():    
    print('\nOPENING annotations',)
    txt = open('/raid/DATASETS/anomaly/XD_Violence/annotations.txt','r')
    txt_data = txt.read()
    txt.close()

    video_list = [line.split() for line in txt_data.split("\n") if line]
    total_anom_frame_count = 0
    for video_j in range(len(video_list)):
        print(video_list[video_j])
        video_anom_frame_count = 0
        for nota_i in range(len(video_list[video_j])):
            if not nota_i % 2 and nota_i != 0: #i=2,4,6...
                aux2 = int(video_list[video_j][nota_i])
                dif_aux = aux2-int(video_list[video_j][nota_i-1])
                total_anom_frame_count += dif_aux 
                video_anom_frame_count += dif_aux
        print(video_anom_frame_count,'frames | ', "%.2f"%(video_anom_frame_count/24) ,'secs | ', int(video_list[video_j][-1]),'max anom frame\n')
    
    total_secs = total_anom_frame_count/24
    mean_secs = total_secs / len(video_list)
    mean_frames = total_anom_frame_count / len(video_list)
    print("TOTAL OF ", "%.2f"%(total_anom_frame_count),"frames  "\
            "%.2f"%(total_secs), "secs\n"\
            "MEAN OF", "%.2f"%(mean_frames),"frames  "\
            "%.2f"%(mean_secs), "secs per video\n")


def get_index_per_label_from_filelist(file_list):
    '''retrives video indexs per label and all from file list xdv'''
        
    print("\n  get_index_per_label_from_list\n")
    
    labels_indexs={'A':[],'B1':[],'B2':[],'B4':[],'B5':[],'B6':[],'G':[],'BG':[]}
    
    # to get frist label only add _ to all : if 'B1' 'B2' ...
    for video_j in range(len(file_list)):
        
        label_strap = os.path.splitext(os.path.basename(file_list[video_j]))[0].split('label')[1]
        #print(os.path.basename(file_list[video_j]),label_strap)
        
        if 'A' in label_strap: labels_indexs['A'].append(video_j)
        else:
            labels_indexs['BG'].append(video_j)
            if 'B1' in label_strap : labels_indexs['B1'].append(video_j)
            if 'B2' in label_strap : labels_indexs['B2'].append(video_j)
            if 'B4' in label_strap : labels_indexs['B4'].append(video_j)
            if 'B5' in label_strap : labels_indexs['B5'].append(video_j)
            if 'B6' in label_strap : labels_indexs['B6'].append(video_j)
            if 'G'  in label_strap : labels_indexs['G'].append(video_j)
    
    print(  '\tA NORMAL',               len(labels_indexs['A']),\
            '\n\n\tB1 FIGHT',           len(labels_indexs['B1']),\
            '\n\tB2 SHOOT',             len(labels_indexs['B2']),\
            '\n\tB4 RIOT',              len(labels_indexs['B4']),\
            '\n\tB5 ABUSE',             len(labels_indexs['B5']),\
            '\n\tB6 CARACC',            len(labels_indexs['B6']),\
            '\n\tG EXPLOS',             len(labels_indexs['G']),\
            '\n\n\tBG ALL ANOMALIES',   len(labels_indexs['BG']))
    
    return labels_indexs