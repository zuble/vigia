import os, cv2, numpy as np

from sklearn.model_selection import train_test_split

import utils.globo as globo



def train_valdt_files(tframes=False):
    """
    GENERATING LIST of TRAIN FILES
    """
    
    full_train_fn = []
    ff_a , ff_bg = 0 , 0

    for root, dirs, files in os.walk(globo.SERVER_TRAIN_COPY_ALTER_PATH1):
        if root != globo.SERVER_TRAIN_COPY_ALTER_PATH1 + '/NN' :
            print(root)
            for file in files:
                if file.find('.mp4') != -1:
                    full_train_fn.append(os.path.join(root, file))
                    if 'label_A' in file: ff_a += 1   
                    else: ff_bg += 1

    #BEFORE SPLIT INTO TRAIN+VALD
    print("\nfull_train_fn",np.shape(full_train_fn),"\nnormal",ff_a,"\nabnormal",ff_bg)
    
    #AFTER SPLIT
    valdt_fn, valdt_labels = [],[]
    train_fn , train_labels = [],[]
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

    t_a , t_bg = 0 , 0
    for i in range(len(train_fn)):
        if 'label_A' in train_fn[i]: train_labels.append(0); t_a += 1
        else: train_labels.append(1); t_bg += 1
    print("\ntrain_fn",np.shape(train_fn),"\nnormal",t_a,"\nabnormal",t_bg)
    
    v_a , v_bg = 0 , 0
    for i in range(len(valdt_fn)):
        if 'label_A' in valdt_fn[i]: valdt_labels.append(0); v_a += 1
        else: valdt_labels.append(1); v_bg += 1
    print("\nvaldt_fn",np.shape(valdt_fn),"\nnormal",v_a,"\nabnormal",v_bg)

    if tframes: return train_fn, train_labels, train_tot_frames, valdt_fn, valdt_labels , valdt_tot_frames
    else: return train_fn, train_labels, valdt_fn, valdt_labels

## NPY
def create_train_valdt_npy():
    train_fn, train_labels, train_tot_frames, valdt_fn, valdt_labels , valdt_tot_frames = train_valdt_files(True)
    
    data_dict = {
        'train_fn': train_fn,
        'train_labels': train_labels,
        'train_tot_frames': train_tot_frames,
        'valdt_fn': valdt_fn,
        'valdt_labels': valdt_labels,
        'valdt_tot_frames': valdt_tot_frames
    }
    p = os.path.join(globo.ZU_PATH , 'utils/.npy/train_valdt_alter_data_dict.npy')
    print("\nSAVING INTO {}".format(p))
    np.save(p, data_dict)


def load_train_valdt_npy(data_type , filter_frames_under = 0):
    
    if data_type not in ['train', 'valdt']:
        raise ValueError("data_type must be either 'train' or 'valdt'")
    
    p = os.path.join(globo.ZU_PATH , 'utils/.npy/train_valdt_alter_data_dict.npy')
    data = np.load(p, allow_pickle=True).item()
    print(f'\nLOADING {data_type} data',np.shape(data[f'{data_type}_fn']),np.shape(data[f'{data_type}_labels']),np.shape(data[f'{data_type}_tot_frames']),"\n\n")
    
    fn = data[f'{data_type}_fn']
    labels = data[f'{data_type}_labels']
    tot_frames = data[f'{data_type}_tot_frames']
    assert len(fn) == len(labels) == len(tot_frames)
    
    print("\tnormal", sum(1 for label in labels if label == 0))
    print("\tabnormal", sum(1 for label in labels if label == 1))
    
    if filter_frames_under:
        print("FILTERING VIDEO WITH LESS THAN {}".format(filter_frames_under))
        fn, labels, tot_frames = zip(*[(vpath, label, tot_frames) for vpath, label, tot_frames in zip(fn, labels, tot_frames) if tot_frames >= filter_frames_under]) 
        print(f'\nNEW {data_type}',np.shape(fn),np.shape(labels),np.shape(tot_frames),"\n\n")
        print("\tnormal", sum(1 for label in labels if label == 0))
        print("\tabnormal", sum(1 for label in labels if label == 1))
        
    return fn, labels, tot_frames


#################################
## TEST

def test_files(tframes=False):
    """
    GENERATE LIST of train FILES
    """
    test_fn, test_labels = [],[]
  
    for root, dirs, files in os.walk(globo.SERVER_TEST_COPY_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                test_fn.append(os.path.join(root, file))
                if 'label_A' in file: test_labels.append(0)
                else: test_labels.append(1)    
        
    t_a , t_bg = 0 , 0
    for i in range(len(test_labels)):
        if not test_labels[i]: t_a += 1
        else: t_bg += 1
        
    print("\ntrain_fn",np.shape(test_fn),"\nnormal",t_a,"\nabnormal",t_bg)
    print('\n-------------------')
    
    if tframes:
        test_tframes = []
        for k in range(len(test_fn)):
            video = cv2.VideoCapture(test_fn[k])
            tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
            test_tframes.append(tot_frames) 
        assert len(test_fn) == len(test_labels) == len(test_tframes)
        return test_fn , test_labels , test_tframes
    else: 
        assert len(test_fn) == len(test_labels)
        return test_fn , test_labels 

## NPY
def create_test_npy():
    test_fn , test_labels , test_tframes = test_files(True)
    
    data_dict = {
        'test_fn': test_fn,
        'test_labels': test_labels,
        'test_tframes': test_tframes,
    }
    p = os.path.join(globo.ZU_PATH , 'utils/.npy/test_data_dict.npy')
    print("\nSAVING INTO {}".format(p))
    np.save(p, data_dict)

def load_test_npy():
    p = os.path.join(globo.ZU_PATH , 'utils/.npy/test_data_dict.npy')
    data = np.load(p, allow_pickle=True).item()
    print(f'\nLOADING data',np.shape(data['test_fn']),np.shape(data['test_labels']),np.shape(data['test_tframes']),"\n\n")
    
    fn = data['test_fn']
    labels = data['test_labels']
    tframes = data['test_tframes']
    assert len(fn) == len(labels) == len(tframes)
    
    print("\tnormal", sum(1 for label in labels if label == 0))
    print("\tabnormal", sum(1 for label in labels if label == 1))

    return fn, labels, tframes
    
    
#################################
## INFO RETRIEVAL
def get_info_from_infotxt():
    fp = '/raid/DATASETS/anomaly/XD_Violence/training_copy_alter_info/train_alter_sort.txt'
    this = 1000
    a = 0 ; bg = 0
    with open(fp, 'r') as f:
        for i , line in enumerate(f.readlines()):
            if i == 4013: break
            parts = line.strip().split()
            frames = int(parts[0])
            fn = parts[6]
            if frames > this :
                if 'label_A' not in fn:
                    print("BG",frames,fn)
                    bg += 1
                else:
                    print("A",frames,fn)
                    a += 1
                    
    print(bg , a )
    
    
    
####################################

class asgt_from_annotations_xdv:
    def __init__(self):
        self.txt_path = '/raid/DATASETS/anomaly/XD_Violence/annotations.txt'
        self.get_data()
        
        self.video_name=''
        self.video_j = -1
        
    def get_data(self):
        #print('\nOPENING annotations',)
        txt = open(self.txt_path,'r')
        txt_data = txt.read()
        txt.close()
        self.video_list = [line.split() for line in txt_data.split("\n") if line]
    
   
    def get_asgt_per_frame(self,vn):
        
        self.video_name = vn
        self.asgt_per_frame = []
        
        # trys to find index from annotations with same name as input vn
        for video_j in range(len(self.video_list)):
            if str(self.video_list[video_j][0]) == str(self.video_name):
                self.video_j = video_j
                break
            
        # no video with that name in annotations    
        if self.video_j == -1: 
            return self.asgt_per_frame
        else:
            file_path=os.path.join('/raid/DATASETS/anomaly/XD_Violence/testing_copy' , self.video_list[self.video_j][0] + '.mp4')
            #print(file_path)
            video = cv2.VideoCapture(str(file_path))
            nframes_in_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            #print(nframes_in_video)
            
            for i in range(nframes_in_video): self.asgt_per_frame.append(0)
            #print("as_per_frame",np.shape(self.asgt_per_frame))
            
            for nota_i in range(len(self.video_list[self.video_j])):
                if not nota_i % 2 and nota_i != 0: #i=2,4,6...
                    end_anom = int(self.video_list[self.video_j][nota_i])
                    start_anom = int(self.video_list[self.video_j][nota_i-1])
                    for frame in range(start_anom,end_anom):
                        self.asgt_per_frame[frame]=1
                    #print(start_anom,end_anom)
                    
            print("asgt",np.shape(self.asgt_per_frame),"from",self.video_name)        
        return self.asgt_per_frame