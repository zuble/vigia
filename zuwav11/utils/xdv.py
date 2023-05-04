import os, cv2, subprocess, numpy as np

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


def create_train_valdt_test_from_xdvtest_bg():
    
    import moviepy.editor as mp
    def is_silent(video_path, threshold=-50.0):
        
        def print_acodec_from_mp4(data, printt=False, only_sr=False):
            out = []
            for i in range(len(data)):
                output = subprocess.check_output('ffprobe -hide_banner -v error -select_streams a:0 -show_entries stream=codec_name,channels,sample_rate,bit_rate -of default=noprint_wrappers=1 -of compact=p=0:nk=1 ' + str('"' + data[i] + '"'), shell=True)
                output = str(output).replace("\\n", "").replace("b", "").replace("'", "").splitlines()[0]
                out.append(output)
                if printt: print(output)
            if only_sr: return int(str(out[0]).split('|')[1])
            else: return out
            
        mp4_fs_aac = print_acodec_from_mp4([video_path], only_sr=True)
        audio = mp.AudioFileClip(filename=video_path, fps=mp4_fs_aac)
    
        if audio is None: audio.close(); return True

        aud_arr = audio.to_soundarray()
        max_volume = np.max(np.abs(np.mean(aud_arr, axis=1).astype(np.float32)))

        # Convert the max volume to dB
        max_volume_db = 20 * np.log10(max_volume)

        audio.close()
        return max_volume_db <= threshold
    

    ## with is_silent this video gets removed , but the logic stays the same
    #OLD v=38GQ9L2meyE__#1_label_B6-0-0 26 80 210 288 377 396 450 517 597 628 650 850 895 973 1106 1226 1330 1400 1490 1675 1713 1820 1890 1980 2025 2100 2177 2290 2375 2450 2579 2655 2657 3045 3091 3170 3259 3475 3571 4060 4143 4288 4364 4450

    #NEW ('/raid/DATASETS/anomaly/XD_Violence/testing_copy/v=38GQ9L2meyE__#1_label_B6-0-0.mp4', [(26, 80, 1), (81, 209, 0), (210, 973, 1), (974, 1105, 0), (1106, 1226, 1), (1227, 1329, 0), (1330, 2450, 1), (2451, 2578, 0), (2579, 4450, 1), (4451, 4650, 0)]) 

    relevant_time = 4
    fp = '/raid/DATASETS/anomaly/XD_Violence/annotations.txt'
    data = []
    
    with open(fp, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            video_path = os.path.join(globo.SERVER_TEST_COPY_PATH,str(parts[0])+'.mp4')
            
            # Check if the video is silent
            if is_silent(video_path):
                print(f"\nSkipping silent video: {video_path}\n")
                continue
            
            video = cv2.VideoCapture(video_path);tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT));fps=int(video.get(cv2.CAP_PROP_FPS));video.release()
            anom_intervals = [(int(parts[i]), int(parts[i + 1])) for i in range(1, len(parts), 2)]
            print(line.replace("\n",""))
            
            intervals_with_labels = [];prev_frame = 0
            ## Check if [0, INITIAL frame from anom_inteval] is relevant
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

                # Add anomalous interval
                intervals_with_labels.append((start, end, 1))
                prev_frame = end + 1
                
            ## appends [end frame of last anomalous interval,totframes] interval if relevant
            if tot_frames - prev_frame > fps * relevant_time:
                intervals_with_labels.append((prev_frame, tot_frames, 0))
                
            data.append((video_path, intervals_with_labels))
            print(data[-1],"\n")

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
    print("\tTRAIN",np.shape(train_data))
    print("\n\tVALDT",np.shape(val_data))
    print("\n\tTEST",np.shape(test_data))
    
    data_dict = {
        "train" : train_data , 
        "valdt" : val_data , 
        "test" : test_data
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
                            
                    if not ok: print(os.path.join(root, file),"not in data")
    
    
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

