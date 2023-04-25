import os, cv2, numpy as np

from sklearn.model_selection import train_test_split

import utils.globo as globo



def train_valdt_files(tframes=False):
    """
    GENERATING LIST of TRAIN FILES
    """
    
    full_train_fn, full_train_normal_fn, full_train_abnormal_fn = [],[],[]
    full_train_labels, full_train_normal_labels, full_train_abnormal_labels = [],[],[]

    for root, dirs, files in os.walk(globo.SERVER_TRAIN_COPY_ALTER_PATH1):
        if root != globo.SERVER_TRAIN_COPY_ALTER_PATH1 + '/NN' :
            print(root)
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

    
    if tframes:
        train_tot_frames = []
        for k in range(len(train_fn)):
            video = cv2.VideoCapture(train_fn[k])
            tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
            train_tot_frames.append(tot_frames)
        
        valdt_tot_frames = []
        for j in range(len(valdt_fn)):
            video = cv2.VideoCapture(valdt_fn[j])
            tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
            valdt_tot_frames.append(tot_frames)


    for i in range(len(train_fn)):
        if 'label_A' in train_fn[i]:train_normal_fn.append(train_fn[i]);train_normal_labels.append(0);train_labels.append(0)
        else: train_abnormal_fn.append(train_fn[i]);train_abnormal_labels.append(1);train_labels.append(1)
    
    print("\ntrain_fn",np.shape(train_fn),"\ntrain_normal_fn",np.shape(train_normal_fn),"\ntrain_abnormal_fn",np.shape(train_abnormal_fn))
    
    for i in range(len(valdt_fn)):
        if 'label_A' in valdt_fn[i]:valdt_normal_fn.append(valdt_fn[i]);valdt_normal_labels.append(0);valdt_labels.append(0)
        else: valdt_abnormal_fn.append(valdt_fn[i]);valdt_abnormal_labels.append(1);valdt_labels.append(1)   
    
    print("\nvaldt_fn",np.shape(valdt_fn),"\nvaldt_normal_fn",np.shape(valdt_normal_fn),"\nvaldt_abnormal_fn",np.shape(valdt_abnormal_fn))

    if tframes: return train_fn, train_labels, train_tot_frames, valdt_fn, valdt_labels , valdt_tot_frames
    else: return train_fn, train_labels, valdt_fn, valdt_labels


def test_files():
    """
    GENERATE LIST of train FILES
    """
    test_fn, test_normal_fn, test_abnormal_fn = [],[],[]
    test_labels, test_normal_labels, test_abnormal_labels = [],[],[]


    for root, dirs, files in os.walk(globo.SERVER_TEST_COPY_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                if 'label_A' in file:
                    test_normal_fn.append(os.path.join(root, file))
                    test_normal_labels.append(0)
                else:
                    test_abnormal_fn.append(os.path.join(root, file))
                    test_abnormal_labels.append(1)    
                   
    test_labels = test_normal_labels + test_abnormal_labels                
    test_fn = test_normal_fn + test_abnormal_fn
    
    
    print("\ntest_fn",np.shape(test_fn),"\ntest_normal_fn",np.shape(test_normal_fn),"\ntest_abnormal_fn",np.shape(test_abnormal_fn))
    print("\ntest_labels",np.shape(test_labels),"\ntest_normal_labels",np.shape(test_normal_labels),"\ntest_abnormal_labels",np.shape(test_abnormal_labels))
    print('\n-------------------')
    return test_fn , test_normal_fn , test_abnormal_fn , test_labels 



#test_fn , test_abnormal_fn , test_normal_fn , test_labels = test_files()
#train_fn, train_labels, valdt_fn, valdt_labels = train_valdt_files()
