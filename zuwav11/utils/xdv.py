import os, cv2, subprocess, random, glob, numpy as np

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
## AS FRAME INTERVALS OF LABEL 1

###############################
## APPROACH 1 
# from test BG created label 0 and 1
# label 1 -> annoted intervals if they are relevant ( > 4sec*24fps)
# label 0 -> outer regions of annoted intervals if they are relevant


## creates the information about dataset 
## '/raid/DATASETS/anomaly/XD_Violence/testing_copy/v=38GQ9L2meyE__#1_label_B6-0-0.mp4', [(26, 80, 1), (81, 209, 0), (210, 973, 1)]
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
    np.save(os.path.join(globo.AAS_PATH,'1_full_interval/dataset_from_xdvtest_bg_data.npy'), data_dict)
    
    
## loads the information to create dataset of type: vpath, [(sf , ef , label),....]
def train_valdt_test_from_xdvtest_bg_from_npy(printt=False):
    data = np.load(os.path.join(globo.AAS_PATH,'1_full_interval/dataset_from_xdvtest_bg_data.npy'), allow_pickle=True).item()
    
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

## dataset created with aas0.py


##############################
## APPROACH 2
# from test BG created label 1 | from train A created label 0
# label 1 -> sub divided all original frame interval annotations into chuck_fsize 
# label 0 -> iterates over all videos until i got the same frame_intervals as label 1
#            and for each video got a random number of frame_intervals between 2 and 4)

def get_frame_intervals_chuncked_from_test_bg(chunck_fsize , debug=False):
    print("\n***********label 1 / BG\n\nget_frame_intervals_chuncked_from_test_bg\n\nCHUNCK_FSIZE",chunck_fsize,"\n\n")
    def create_sub_intervals(start_frame, end_frame, label=1 , ):
        frame_count = end_frame - start_frame
        
        if frame_count <= chunck_fsize:
            return None , None
        
        sub_fintervals , sub_tintervals = [] , []
        current_start = start_frame
        
        while current_start < end_frame:
            current_end = min(current_start + chunck_fsize, end_frame)
            
            # Discard the last interval if its duration doesn't match max_duration
            if current_end == end_frame and (current_end - current_start) != chunck_fsize:
                break
            
            sub_finterval = (current_start, current_end,label)
            sub_fintervals.append(sub_finterval)
            
            sub_tinterval = ( round(current_start/24 , 4) , round(current_end/24 , 4),label)
            sub_tintervals.append(sub_tinterval)
            
            current_start = current_end
        
        return sub_fintervals , sub_tintervals
    
    print('\nOPENING annotations',)
    txt = open('/raid/DATASETS/anomaly/XD_Violence/annotations.txt','r')
    txt_data = txt.read()
    txt.close()

    video_list = [line.split() for line in txt_data.split("\n") if line]

    frame_intervals , time_intervals = [] , []
    interval_count = 0
    for video_j in range(len(video_list)):
        
        sub_fintervals_total , sub_tintervals_total = [] , []
        for nota_i in range(len(video_list[video_j])):
            if not nota_i % 2 and nota_i != 0: #i=2,4,6...
                
                aux2 = int(video_list[video_j][nota_i]) ; aux1 = int(video_list[video_j][nota_i-1])
                aux2t = round(aux2/24, 2) ; aux1t = round(aux1/24, 2)
                dif_aux = aux2-aux1 ; dif_aux_t = round(dif_aux/24,2)
                
                #print(video_list[video_j][nota_i-1],video_list[video_j][nota_i])    
                
                sub_fintervals , sub_tintervals = create_sub_intervals(aux1, aux2)
                if sub_fintervals is not None: 
                    #print(sub_fintervals)
                    sub_fintervals_total.extend(sub_fintervals)
                    sub_tintervals_total.extend(sub_tintervals)        
                    
        if len(sub_fintervals_total) != 0:
            #print(sub_fintervals_total)
            frame_intervals.append((os.path.join(globo.SERVER_TEST_COPY_PATH,video_list[video_j][0]+".mp4"),sub_fintervals_total))
            time_intervals.append((os.path.join(globo.SERVER_TEST_COPY_PATH,video_list[video_j][0]+".mp4"),sub_tintervals_total))
            
            interval_count += np.shape(frame_intervals[-1][1])[0]
            
            if debug:
                print(np.shape(frame_intervals[-1][1])[0])
                print(video_list[video_j])
                print(frame_intervals[-1])
                print(time_intervals[-1],"\n")
    
    print("total intervals",interval_count)
    return frame_intervals , time_intervals , interval_count

def get_frame_intervals_chuncked_from_train_a(total_intervals,chunck_fsize , debug=False):
    print("\n***********\nlabel 0 / A\n\nget_frame_intervals_chuncked_from_train_a\n\nCHUNCK_FSIZE",chunck_fsize,"\n\n")
    ## from a video paths get num_intervals with chunck_fsize
    def get_random_fixed_intervals(video_path, num_intervals):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open the video file {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert int(cap.get(cv2.CAP_PROP_FPS)) == 24
        
        if total_frames < chunck_fsize:
            print(f"Error: Video {video_path} has fewer frames than the fixed interval size.")
            return None
        if debug:print("total_frames",total_frames)
        
        intervals = []
        available_frames = list(range(0, total_frames - chunck_fsize))
        for _ in range(num_intervals):
            if not available_frames:break

            start_frame = random.choice(available_frames)
            end_frame = start_frame + chunck_fsize
            intervals.append((start_frame, end_frame, 0))

            ## Remove frames that are now part of an interval
            available_frames = [frame for frame in available_frames if frame < start_frame or frame >= end_frame]

        cap.release()
        print(intervals)
        ## Sort intervals by start_frame
        return sorted(intervals, key=lambda x: x[0])
    
    ## get vpath list for train label_A
    train_a_vpath = []
    for root, dirs, files in os.walk(globo.SERVER_TRAIN_COPY_ALTER_PATH1):
        if root != globo.SERVER_TRAIN_COPY_ALTER_PATH1 + '/NN' : ## these are excluded from xdv train dataset
            print(root)
            for file in files:
                if file.find('.mp4') != -1 and 'label_A' in file:
                    train_a_vpath.append(os.path.join(root, file))
    print("train_a_vpath",np.shape(train_a_vpath))
      
    ##main loop tha iterates over all videos and gets a betwen 2 and 4 frame_intervals w/ chunck_fsize   
    collected_intervals = []
    total_intervals_collected = 0
    interval_limit = 0
    #while total_intervals_collected < total_intervals and interval_limit < 1000:
    for i,video_path in enumerate(train_a_vpath):
        print('\n',i,video_path)
        
        if total_intervals_collected >= total_intervals: break
        
        num_intervals = random.randint(2, 4)
        if total_intervals_collected + num_intervals > total_intervals:
            num_intervals = total_intervals - total_intervals_collected
        
        print(f"getting {num_intervals} intervals")
        video_intervals = get_random_fixed_intervals(video_path, num_intervals)
        if video_intervals is not None:
            print("\t", video_intervals)
            collected_intervals.append((video_path, video_intervals))
            total_intervals_collected += len(video_intervals)
        
        if total_intervals_collected >= total_intervals: break
        
        interval_limit += 1

    n_a_intervals = sum(np.shape(collected_intervals[k][1])[0] for k in range(len(collected_intervals)))
    print("\ntotal intervals", n_a_intervals)
    return collected_intervals , n_a_intervals


#DATA = xdv.create_train_valdt_test_from_xdvtest_bg_train_a(True)
def create_train_valdt_test_from_xdvtest_bg_train_a(chunck_fsize = 4*24 , debug=False):

    bg_frame_intervals , bg_time_intervals , n_bg_intervals = get_frame_intervals_chuncked_from_test_bg(chunck_fsize , debug)
    a_frame_intervals , n_a_intervals = get_frame_intervals_chuncked_from_train_a(n_bg_intervals,chunck_fsize,debug)
    print("\n\neg bg:",bg_frame_intervals[0] ,\
        "\neg a:",a_frame_intervals[0])
    
    ## from (vpath1,frame_intervals) to (vpath1,frame_interval1,frame_interval2..)..
    n_a_intervals = sum(np.shape(a_frame_intervals[k][1])[0] for k in range(len(a_frame_intervals)))
    n_bg_intervals = sum(np.shape(bg_frame_intervals[k][1])[0] for k in range(len(bg_frame_intervals)))
    data = [(vp, interval) for vp, intervals in bg_frame_intervals + a_frame_intervals for interval in intervals]
    print("\n\nMERGED data intervals",data[0],"\n",np.shape(data)[0],"=",n_a_intervals+n_bg_intervals,"(",n_a_intervals,"+",n_bg_intervals,")")
    print("NORMAL intervals", sum(1 for _, (_, _, label) in data if label == 0))
    print("ABNORMAL intervals", sum(1 for _, (_, _, label) in data if label == 1))
    assert np.shape(data)[0] == n_a_intervals+n_bg_intervals
    
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

    print("\n\nSPLITEED")
    print("\n\tTRAIN",np.shape(train_data)[0],'"videos" with 1 interval each')
    print("\tNORMAL intervals", sum(1 for _, (_, _, label) in train_data if label == 0))
    print("\tABNORMAL intervals", sum(1 for _, (_, _, label) in train_data if label == 1))
    print("\n\tVALDT",np.shape(val_data)[0],'"videos" with 1 interval')
    print("\tNORMAL intervals", sum(1 for _, (_, _, label) in val_data if label == 0))
    print("\tABNORMAL intervals", sum(1 for _, (_, _, label) in val_data if label == 1))
    print("\n\tTEST",np.shape(test_data)[0],'"videos" with 1 interval')
    print("\tNORMAL intervals", sum(1 for _, (_, _, label) in test_data if label == 0))
    print("\tABNORMAL intervals", sum(1 for _, (_, _, label) in test_data if label == 1))

    data_dict = {
        "train" : train_data , 
        "valdt" : val_data , 
        "test" : test_data
    }
    npz_path = os.path.join(globo.AAS_PATH+"/2_sub_interval/temp",f"dataset_from_xdvtest_bg_train_a_data-{chunck_fsize}fi.npy")
    np.save( npz_path , data_dict)
    return npz_path


def load_train_valdt_test_from_xdvtest_bg_train_a(mode , iteration):
    print("\n\tmode",mode,"fi",globo.CFG_SINET['chunck_fsize'],"iter",iteration,"\n")
    
    path = globo.AAS_PATH+"/2_sub_interval/"+str(globo.CFG_SINET['chunck_fsize'])+"/"+str(iteration)+"/*{}*.npz".format(mode)
    npz_path = glob.glob(path)[0] 
    if not os.path.exists(npz_path): raise Exception("npz no existe:", npz_path) 
    
    data = np.load(npz_path, allow_pickle=True)["data"]
    print("\n\n",mode,"DATA LOADED FROM",npz_path,"\n\n")
    return data


#####
def get_testxdvanom_info():    
    print('\nOPENING annotations',)
    txt = open('/raid/DATASETS/anomaly/XD_Violence/annotations.txt','r')
    txt_data = txt.read()
    txt.close()

    video_list = [line.split() for line in txt_data.split("\n") if line]
    total_anom_frame_count = 0
    frame_intervals = []
    for video_j in range(len(video_list)):
        #print(video_list[video_j])
        video_anom_frame_count = 0
        for nota_i in range(len(video_list[video_j])):
            if not nota_i % 2 and nota_i != 0: #i=2,4,6...
                aux2 = int(video_list[video_j][nota_i])
                aux1 = int(video_list[video_j][nota_i-1])
                aux2t = round(aux2/24, 2) ; aux1t = round(aux1/24, 2)
                dif_aux = aux2-aux1
                dif_aux_t = round(dif_aux/24,2)
                total_anom_frame_count += dif_aux 
                video_anom_frame_count += dif_aux
                frame_intervals.append((dif_aux,dif_aux_t,aux1,aux1t,aux2,aux2t,os.path.basename(video_list[video_j][0])))
        #print(video_anom_frame_count,'frames | ', "%.2f"%(video_anom_frame_count/24) ,'secs | ', int(video_list[video_j][-1]),'max anom frame\n')
    
    
    total_secs = total_anom_frame_count/24
    mean_secs = total_secs / len(video_list)
    mean_frames = total_anom_frame_count / len(video_list)
    print("TOTAL OF ", "%.2f"%(total_anom_frame_count),"frames  "\
            "%.2f"%(total_secs), "secs\n"\
            "MEAN OF", "%.2f"%(mean_frames),"frames  "\
            "%.2f"%(mean_secs), "secs per video\n")
    
    
    # Sort the frame_intervals list based on the first element (dif_aux) in each tuple
    sorted_frame_intervals = sorted(frame_intervals, key=lambda x: x[0])
    
    more_then_x = 0
    x = 5
    for i in range(len(sorted_frame_intervals)):
        print(i,sorted_frame_intervals[i])
        vpath = sorted_frame_intervals[i][5]
        count_vpath = 0
        
        if sorted_frame_intervals[i][0] < 135:
            # Count how many times vpath appears in the sorted_frame_intervals at position 5
            count_vpath = sum(1 for interval in sorted_frame_intervals if interval[5] == vpath)
            print(f"video with {count_vpath} intervals\n")
        
        if sorted_frame_intervals[i][1] >= x: more_then_x += 1
        
    print("more_then_x",more_then_x)
    
    # Extract the dif_aux values from frame_intervals
    dif_aux_values = [interval[1] for interval in sorted_frame_intervals]

    # Calculate the mean and median of dif_aux values
    mean_dif_aux = np.mean(dif_aux_values)
    median_dif_aux = np.median(dif_aux_values)

    print("Mean of dif_aux values:", mean_dif_aux)
    print("Median of dif_aux values:", median_dif_aux)


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




def fps_mean():
    test_fp, *_ = test_files()
    fps_count = 0 
    for fp in test_fp:
        cap = cv2.VideoCapture(fp)
        if not cap.isOpened():
            print("Error: Could not open the video file.")
        else:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fps_count += fps
        cap.release()
    print(f"fps_mean {round(fps_count/np.shape(test_fp)[0] , 4)}")