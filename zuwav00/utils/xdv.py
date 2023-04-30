import os, cv2, numpy as np

from sklearn.model_selection import train_test_split

import utils.globo as globo


def load_test_copy():
    mp4_paths = []
    for root, dirs, files in os.walk(globo.SERVER_TEST_COPY_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths.append(os.path.join(root, file))
    mp4_paths.sort()
    return mp4_paths

def load_train_copy():
    mp4_paths = []
    for root, dirs, files in os.walk(globo.SERVER_TRAIN_COPY_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths.append(os.path.join(root, file))
    return mp4_paths

def load_train_alter():
    mp4_paths = []
    for root, dirs, files in os.walk(globo.SERVER_TRAIN_COPY_ALTER_PATH1):
        if root != globo.SERVER_TRAIN_COPY_ALTER_PATH1 + '/NN' :
            print(root)
            for file in files:
                if file.find('.mp4') != -1:
                 mp4_paths.append(os.path.join(root, file))
    return mp4_paths


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