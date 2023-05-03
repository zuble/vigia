import os, cv2, numpy as np

from sklearn.model_selection import train_test_split

import utils.globo as globo



## ***************************************** ##
## ORIGINAL XDV

def train_valdt_files(tframes=False):
    """
    GENERATING LIST of TRAIN FILES
    """
    
    full_train_fn, full_train_normal_fn, full_train_abnormal_fn = [],[],[]
    full_train_labels, full_train_normal_labels, full_train_abnormal_labels = [],[],[]

    for root, dirs, files in os.walk(globo.SERVER_TRAIN_COPY_ALTER_PATH1):
        if root != globo.SERVER_TRAIN_COPY_ALTER_PATH1 + '/NN' : ## these are excluded from xdv train dataset
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
    np.save(os.path.join(globo.SERVER_TRAIN_COPY_ALTER_PATH1,'npy/train_valdt_data.npy'), data_dict)

def train_valdt_from_npy():
    data = np.load(os.path.join(globo.SERVER_TRAIN_COPY_ALTER_PATH1,'npy/train_valdt_data.npy'), allow_pickle=True).item()
    print("\n\tTRAIN",np.shape(data["train_fn"]),np.shape(data["train_labels"]),np.shape(data["train_tot_frames"]))
    print("\tVALDT",np.shape(data["valdt_fn"]),np.shape(data["valdt_labels"]),np.shape(data["valdt_tot_frames"]),"\n\n")
    return data


## ***************************************** ##
## EXCLUSIVLY TO TRAIN SINEET NEW CLASSIFIER
## BY USING XDV ANOMALOUS VIDEOS FROM TEST
## WHICH HAVE FRAME LEVEL LABEL ANNOTATIONS 


## 1 gets all frame interval from annotations
def create_train_valdt_test_from_xdvtest_bg():
    
    relevant_time = 4
    
    def get_new_list(list):    
        #OLD ('...v=k7R_Qo-BiAw__#00-10-30_00-10-51_label_B6-0-0.mp4', [(0, 150), (296, 505)])
        #NEW ('...v=k7R_Qo-BiAw__#00-10-30_00-10-51_label_B6-0-0.mp4', [(0, 150, 1), (151, 295, 0), (296, 505, 1)]) 
        
        #OLD ('...v=UK2w9Sh47fM__#1_label_G-0-0.mp4', [(0, 209), (210, 500)], 1198)
        #NEW ('...v=UK2w9Sh47fM__#1_label_G-0-0.mp4', [(0, 209, 1), (210, 500, 1), (501, 1198, 0)])
        
        #OLD ('...v=ZkUciDD55kA__#00-00-00_00-00-30_label_G-0-0.mp4', [(124, 617)], 722)
        #NEW ('...v=ZkUciDD55kA__#00-00-00_00-00-30_label_G-0-0.mp4', [(0, 123, 0), (124, 617, 1), (618, 722, 0)]) 
        
        #OLD ('...v=7rDRFFSUrPI__#00-01-50_00-02-32_label_G-0-0.mp4', [(655, 700), (840, 860), (912, 930)])
        #NEW ('...v=7rDRFFSUrPI__#00-01-50_00-02-32_label_G-0-0.mp4', [(0, 654, 0), (655, 700, 1), (701, 839, 0), (840, 860, 1), (861, 911, 0), (912, 930, 1)]) 
        
        new_list = []
        for video_j in range(len(list)):
            print(list[video_j])
            vpath = list[video_j][0]
            anom_intervals = list[video_j][1]
            
            ## if tot_frames was added to old_list, 
            ## LAST NORMAL frame interval is releveant
            if len(list[video_j]) == 3: frame_max = int(list[video_j][2])
            else: frame_max = anom_intervals[-1][-1]
        
            intervals_with_labels = [];prev_frame = 0
            
            ## Check if INITIAL NORMAL frame interval is relevant
            if anom_intervals[0][0] > fps * relevant_time:
                intervals_with_labels.append((0, anom_intervals[0][0] - 1, 0))
                prev_frame = anom_intervals[0][0] - 1
            
            for interval_i in range(len(anom_intervals)):
                start, end = anom_intervals[interval_i]
                
                if prev_frame < start and interval_i != 0:
                    ## Check if MIDDLE NORMAL interval is relevant
                    if start - prev_frame > fps * relevant_time:
                        intervals_with_labels.append((prev_frame, start - 1, 0))
                    else:
                        # Merge the current anomalous interval with the previous one
                        prev_anom_interval = intervals_with_labels.pop()
                        start = prev_anom_interval[0]
            
                #if prev_frame < start and interval_i != 0:
                #    # Add normal interval
                #    intervals_with_labels.append((prev_frame, start - 1, 0))
                    
                # Add anomalous interval
                intervals_with_labels.append((start, end, 1))
                prev_frame = end + 1

            ## ADD 
            if prev_frame < frame_max:
                intervals_with_labels.append((prev_frame, frame_max, 0))
                
            print((vpath, intervals_with_labels),"\n\t")
            new_list.append((vpath, intervals_with_labels))
            
        return new_list 


    ## from test annotations get line like this one
    ## ('...v=k7R_Qo-BiAw__#00-10-30_00-10-51_label_B6-0-0.mp4', [(0, 150), (296, 505)])
    fp = '/raid/DATASETS/anomaly/XD_Violence/annotations.txt'
    data = []
    with open(fp, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            video_path = os.path.join(globo.SERVER_TEST_COPY_PATH,str(parts[0])+'.mp4')
            video = cv2.VideoCapture(video_path);tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT));fps=int(video.get(cv2.CAP_PROP_FPS));video.release()
            intervals = [(int(parts[i]), int(parts[i + 1])) for i in range(1, len(parts), 2)]
            
            ## appends tot_frames only when ther's a relevant normal interval in the end of video
            if tot_frames - int(intervals[-1][-1]) > fps * relevant_time:
                data.append((video_path, intervals , tot_frames))
            ## otherwise last abnormal end interval is used as the final video interval
            else : data.append((video_path, intervals))
            print(data[-1])
    
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
    new_train = get_new_list(train_data)
    print("\tTRAIN",np.shape(train_data),np.shape(new_train) )
    new_valdt = get_new_list(val_data)
    print("\n\tVALDT",np.shape(val_data),np.shape(new_valdt))
    new_test = get_new_list(test_data)
    print("\n\tTEST",np.shape(test_data),np.shape(new_test))
    
    data_dict = {
        "train" : new_train , 
        "valdt" : new_valdt , 
        "test" : new_test
    }
    np.save(os.path.join(globo.SERVER_TEST_COPY_PATH,'npy/dataset_from_xdvtest_bg_data.npy'), data_dict)
   
    
## just loads it
def train_valdt_test_from_xdvtest_bg_from_npy(printt=False):
    data = np.load(os.path.join(globo.SERVER_TEST_COPY_PATH,'npy/dataset_from_xdvtest_bg_data.npy'), allow_pickle=True).item()
    
    ## checks if all anomalous videos in xdv test are present either in train or valdt or test
    for root, dirs, files in os.walk(globo.SERVER_TEST_COPY_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                if 'label_A' not in file:
                    
                    ok = False    
                    for i in range(len(data)):
                        if not i:typee = "train"
                        elif i == 1:typee = "valdt"
                        else: typee = "test" 
                                  
                        for video_j in range(len(data[typee])):
                            vpath = data[typee][video_j][0]
                            if os.path.join(root, file) == vpath: ok =True
                            
                    if not ok: raise Exception("its not ok")
    
    
    if printt:
        for i in range(len(data)): 
            
            if not i:typee = "train"
            elif i == 1:typee = "valdt"
            else: typee = "test"
            print("\n*****",typee,"******************\n")
            
            anom_cunt , norm_cunt = 0 , 0
            anom_frames_cunt , norm_frames_cunt = 0 , 0
            interval_counts = {}
            for video_j in range(len(data[typee])):
                line = data[typee][video_j]
                vpath = data[typee][video_j][0]
                anom_intervals = data[typee][video_j][1]
                #print(line);print(os.path.basename(vpath))
                #print(anom_intervals, len(anom_intervals))
                
                for i, ai in enumerate(anom_intervals):
                    #print(ai[2],ai)
                    if ai[2] == 1:
                        anom_frames_cunt += ai[1]-ai[0]
                        anom_cunt+=1
                    elif ai[2] == 0:
                        norm_frames_cunt += ai[1]-ai[0]
                        norm_cunt+=1
                #print()
                
                num_intervals = len(anom_intervals)
                if num_intervals not in interval_counts:interval_counts[num_intervals] = 0
                interval_counts[num_intervals] += 1
            
            print("\n1 ANOMALY\n  ",anom_cunt,"total intervals\n  ",anom_frames_cunt,"total frames\n  ",anom_frames_cunt/24/3600,"total hours")
            print("\n0 NORMAL\n  ",norm_cunt,"total intervals\n  ",norm_frames_cunt,"total frames\n  ",norm_frames_cunt/24/3600,"total hours")
            
            sorted_interval_counts = {k: v for k, v in sorted(interval_counts.items())}
            print("\nInterval counts:", sorted_interval_counts)
            
    return data 


## generates sinet outouts for each frame interval and saves
'''
def create_npz_pes_frame_interval():
    import utils.sinet as sinet
    
    CFG_SINET = {
        'sinet_version': 'fsd-sinet-vgg42-tlpf_aps-1',
        'sinet_v': 'sinet42tlpf_aps',
        
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
    
    data=train_valdt_test_from_xdvtest_bg_from_npy(True)
    sinet = sinet.Sinet(CFG_SINET)
    data_dicts= {'train': [], 'valdt': [], 'test': []}

    for i in range(len(data)): 
                
        if not i:typee = "train"
        elif i == 1:typee = "valdt"
        else: typee = "test"
        print("\n*****",typee,"******************\n")
        
        total_intervals = 0
        for video_j in range(len(data[typee])):
            line = data[typee][video_j]
            vpath = data[typee][video_j][0]
            frame_intervals = data[typee][video_j][1] #(sf,ef,label)
            
            total_intervals += len(frame_intervals)
            
            p_es_array = sinet.get_sigmoid_fl(vpath,frame_intervals)
            
            # Store the vpath, frame_interval, p_es_array, and label for each interval
            for k in range(len(p_es_array)):
                data_dicts[typee].append({
                    'vpath': vpath,
                    'frame_interval': frame_intervals[k],
                    'p_es_array': p_es_array[k],
                    'label': frame_intervals[k][2]
                })
                print(vpath,frame_intervals[k],np.shape(p_es_array[k]),frame_intervals[k][2])
            
            #print("\n\n\n")
            #for k in range(len(p_es_array)):
            #    print("interval",k,"@",np.shape(p_es_array[k]),frame_intervals[k][2])
            
        print("\n\n\n***********************\n",total_intervals,len(data_dicts[typee]))
            
    # Save the data for each typee to a single .npz file
    for typee in data_dicts:
        ofn = f"{CFG_SINET['sinet_v']}-fl-{typee}.npz"
        ofp = os.path.join('/raid/DATASETS/.zuble/vigia/zuwav11/aas',ofn)
        print(ofn,'\n',ofp)
        # Save the vpath, frame_interval, p_es_array, and label as .npz files
        np.savez_compressed(ofp, data=data_dicts[typee])
'''

## ***************************************** ##

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

