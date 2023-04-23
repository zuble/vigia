import cv2, time , os

import numpy as np
import utils.auxua as aux


# FX RELATED TO PROCESS RESULTS LIST

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
            file_path=os.path.join(aux.SERVER_TEST_PATH , self.video_list[self.video_j][0] + '.mp4')
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
            

def get_as_total_from_res_list(res_list,txt_i,printt=False):
    '''
    retrieves the as score per frame for each video 
    taking as input txt file index from res_list['txt_path']
    '''
    
    as_total = []
    print("as_total from",res_list['txt_path'][txt_i])
    for video_j in range(len(res_list['videopath'])):
        file_path = "".join(res_list['videopath'][video_j][txt_i])
        if printt:print(video_j,file_path+"\n")

        nbatch_in_video = int(res_list['full'][video_j][txt_i][0])
        if printt:print("nbatch_in_video",nbatch_in_video)

        nframes_in_video = int(res_list['full'][video_j][txt_i][(nbatch_in_video*3)-2])
        if printt:print("nframes_in_video",nframes_in_video)

        as_per_frame = []
        for i in range(nframes_in_video): as_per_frame.append(0)
        if printt:print("as_per_frame",np.shape(as_per_frame))

        for batch in range(0,nbatch_in_video):
            # frist batch
            if batch == 0:
                if printt:print("\nbatch",batch)

                if nbatch_in_video==1: end_frame_batch = int(res_list['full'][video_j][txt_i][1])
                else: end_frame_batch = int(res_list['full'][video_j][txt_i][1])-1
                if printt:print("end_frame_batch",end_frame_batch)

                res_batch = '{:.4f}'.format(float(res_list['full'][video_j][txt_i][2]))
                if printt:print("res_batch",res_batch)

                for i in range(end_frame_batch):as_per_frame[i] = res_batch
                #as_per_frame = [res_batch for _ in range(end_frame_batch)]

            else:
                # last batch
                if batch == nbatch_in_video-1:
                    if printt:print("\nbatch",batch)

                    start_frame_batch = int(res_list['full'][video_j][txt_i][batch*3])
                    end_frame_batch = int(res_list['full'][video_j][txt_i][(batch*3)+1])   
                    last_batch_end_frame = int(res_list['full'][video_j][txt_i][(batch*3)-2])
                    res_last_batch = '{:.4f}'.format(float(res_list['full'][video_j][txt_i][(batch*3)-1]))

                    # batch type 2 (last batch has no repetead frames)
                    if start_frame_batch == last_batch_end_frame - 1:

                        if printt:print("(bt2)start_frame_batch",start_frame_batch,"\nend_frame_batch",end_frame_batch)

                        res_batch = '{:.4f}'.format(float(res_list['full'][video_j][txt_i][(batch*3)+2]))
                        if printt:print("res_batch",res_batch)

                        for i in range(start_frame_batch,end_frame_batch):as_per_frame[i] = res_batch

                    # batch type 1 (last batch has frame_max frames)
                    else:
                        if printt:print("(bt1)start_frame_batch",start_frame_batch,"\nend_frame_batch",end_frame_batch,"\nlast_batch_end_frame",last_batch_end_frame-1)

                        res_batch = '{:.4f}'.format(float(res_list['full'][video_j][txt_i][(batch*3)+2]))
                        if printt:print("res_batch",res_batch)

                        for i in range(start_frame_batch,last_batch_end_frame):as_per_frame[i]=(res_last_batch,res_batch)
                        for i in range(last_batch_end_frame,end_frame_batch):as_per_frame[i] = res_batch

                # intermidiate batchs        
                else:
                    if printt:print("\nbatch",batch)

                    start_frame_batch = int(res_list['full'][video_j][txt_i][batch*3])
                    end_frame_batch = int(res_list['full'][video_j][txt_i][(batch*3)+1])-1

                    if printt:print("start_frame_batch",start_frame_batch,"\nend_frame_batch",end_frame_batch)

                    res_batch = '{:.4f}'.format(float(res_list['full'][video_j][txt_i][(batch*3)+2]))
                    if printt:print("res_batch",res_batch)

                    for i in range(start_frame_batch,end_frame_batch):as_per_frame[i] = res_batch

        if printt:print(np.shape(as_per_frame))         
        as_total.append(as_per_frame)        
        if printt:print("\n")
    
    return as_total


def cv2_test_from_aslist(res_list,vas_list,video_j,txt_i):
    global is_quit, is_paused, frame_counter
    
    #as_total = get_as_total_from_rslt_txti(txt_i)
    print("  res_list['full'] for video ",video_j,"| txt",txt_i,res_list['full'][video_j][txt_i])
    
    file_path = "".join(res_list['videopath'][video_j][txt_i])
    strap_video_name = os.path.splitext(res_list['videoname'][video_j][txt_i])[0]
    print("  ",strap_video_name)
    
    video = cv2.VideoCapture(str(file_path))
    window_name = "AVwr."+str(video_j)+": "+os.path.basename(res_list['videopath'][video_j][txt_i])
    cv2.namedWindow(window_name)
    
    # Video information
    fps = video.get(cv2.CAP_PROP_FPS)
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5;thickness = 1;lineType = cv2.LINE_AA
    
    # We can set up keys to pause, go back and forth.
    # **params can be used to pass parameters to key actions.
    def quit_key_action(**params):
        global is_quit
        is_quit = True
    def rewind_key_action(**params):
        global frame_counter
        frame_counter = max(0, int(frame_counter - (fps * 5)))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
    def forward_key_action(**params):
        global frame_counter
        frame_counter = min(int(frame_counter + (fps * 5)), total_frame - 1)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
    def pause_key_action(**params):
        global is_paused
        is_paused = not is_paused
    # Map keys to buttons
    key_action_dict = {
        ord('q'): quit_key_action,
        ord('a'): rewind_key_action,
        ord('d'): forward_key_action,
        ord('s'): pause_key_action,
        ord(' '): pause_key_action
    }
    def key_action(_key):
        if _key in key_action_dict:key_action_dict[_key]()
            
    prev_time = time.time() # Used to track real fps
    is_quit = False         # Used to signal that quit is called
    is_paused = False       # Used to signal that pause is called
    
    # if video is abnormal, it gets the asgt_per_frame
    asgt_per_frame =  []
    if 'label_A' not in strap_video_name:
        asgt = asgt_from_annotations_xdv()
        asgt_per_frame = asgt.get_asgt_per_frame(strap_video_name)
        gt = True
    else: gt = False
    
    frame_counter = 0       # Used to track which frame are we.    
    asgt_atual = 0
    try:
        while video.isOpened():
            # If the video is paused, we don't continue reading frames.
            if is_quit:# Do something when quiting
                break
            elif is_paused:# Do something when paused
                pass
            else:
                sucess, frame = video.read() # Read the frames
                if not sucess:break
                frame_counter = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                #print(frame_counter)

        
                predict_atual = vas_list[frame_counter-1]
                #print(predict_atual)
                if gt:asgt_atual = asgt_per_frame[frame_counter-1]
                
                # Display frame numba/AS/time/fps
                cv2.putText(frame, 'AS '+str(predict_atual), (10, 13), font, fontScale, [0,0,255], thickness,lineType)
                cv2.putText(frame, 'GT '+str(asgt_atual), (10, 33), font, fontScale, [100,250,10], thickness,lineType)

                cv2.putText(frame, '%d' % (frame_counter), (10, int(height)-10), font, fontScale, [60,250,250], thickness,lineType)
                cv2.putText(frame, '%.2f' % (frame_counter/fps)+' s', (60, int(height)-10), font, fontScale, [80,100,250], thickness,lineType)
                new_time = time.time()
                cv2.putText(frame, '%.2f' % (1/(new_time-prev_time))+' fps', (140, int(height)-10), font,fontScale, [0,50,200], thickness,lineType)
                prev_time = new_time
                
            # Display the image
            cv2.imshow(window_name,frame)

            # Wait for any key press and pass it to the key action
            frame_time_ms = int(1000/fps)#print(frame_time_ms,fps)
            key = cv2.waitKey(frame_time_ms)
            key_action(key)
            
    finally:
        video.release()
        cv2.destroyAllWindows()


def init_watch_reslist(res_list, txt_i,watch_this):
    
    def get_fn_index_from_res_list(txt_i):
        '''retrives all video indexs from a rslt.txt 
        'N'
        'A','AB1','AB2','AB4','AB5','AB6','AG',
        'FNB1','FNB2','FNB4','FNB5','FNB6','FNG','FNBG',
        'TPB1','TPB2','TPB4','TPB5','TPB6','TPG','TPBG'
        'TN','FP' '''
        
        print("\nget_fn_index_from_list ",res_list['txt_modelname'][txt_i])

        labels_indexs_fn={  
            'N':[],\
            'A':[],'AB1':[],'AB2':[],'AB4':[],'AB5':[],'AB6':[],'AG':[],\
            'FNB1':[],'FNB2':[],'FNB4':[],'FNB5':[],'FNB6':[],'FNG':[],'FNBG':[],\
            'TPB1':[],'TPB2':[],'TPB4':[],'TPB5':[],'TPB6':[],'TPG':[],'TPBG':[],\
            'TN':[],'FP':[]                                
        }
        
        # to get frist label only add _ to all : if 'B1' 'B2' ...
        for video_j in range(len(res_list['videoname'])):
            label_strap = os.path.splitext(res_list['videoname'][video_j][txt_i])[0].split('label')[1]
            
            if 'label_A' not in res_list['videoname'][video_j][txt_i]:  #ANOMALIES
                labels_indexs_fn['A'].append(video_j)
                #print(res_list['videoname'][video_j][txt_i],label_strap)
                if 'B1' in label_strap : labels_indexs_fn['AB1'].append(video_j)
                if 'B2' in label_strap : labels_indexs_fn['AB2'].append(video_j)
                if 'B4' in label_strap : labels_indexs_fn['AB4'].append(video_j)
                if 'B5' in label_strap : labels_indexs_fn['AB5'].append(video_j)
                if 'B6' in label_strap : labels_indexs_fn['AB6'].append(video_j)
                if 'G' in  label_strap : labels_indexs_fn['AG'].append(video_j)
            
                if float(res_list['max'][video_j][txt_i])<=0.5:  #FALSE NEGATIVE
                    labels_indexs_fn['FNBG'].append(video_j) 
                    if 'B1' in label_strap : labels_indexs_fn['FNB1'].append(video_j)
                    if 'B2' in label_strap : labels_indexs_fn['FNB2'].append(video_j)
                    if 'B4' in label_strap : labels_indexs_fn['FNB4'].append(video_j)
                    if 'B5' in label_strap : labels_indexs_fn['FNB5'].append(video_j)
                    if 'B6' in label_strap : labels_indexs_fn['FNB6'].append(video_j)
                    if 'G' in  label_strap : labels_indexs_fn['FNG'].append(video_j)
                
                else:    #TRUE POSITIVE
                    labels_indexs_fn['TPBG'].append(video_j)
                    if 'label_B1' in res_list['videoname'][video_j][txt_i]:labels_indexs_fn['TPB1'].append(video_j)
                    if 'label_B2' in res_list['videoname'][video_j][txt_i]:labels_indexs_fn['TPB2'].append(video_j)
                    if 'label_B4' in res_list['videoname'][video_j][txt_i]:labels_indexs_fn['TPB4'].append(video_j)
                    if 'label_B5' in res_list['videoname'][video_j][txt_i]:labels_indexs_fn['TPB5'].append(video_j)
                    if 'label_B6' in res_list['videoname'][video_j][txt_i]:labels_indexs_fn['TPB6'].append(video_j)
                    if 'label_G'  in res_list['videoname'][video_j][txt_i]:labels_indexs_fn['TPG'].append(video_j)
                    
            else: # NORMAL
                labels_indexs_fn['N'].append(video_j)
                
                if float(res_list['max'][video_j][txt_i])>0.5:  #FALSE POSITIVE
                    labels_indexs_fn['FP'].append(video_j)                
                else:    #TRUE NEGATIVE
                    labels_indexs_fn['TN'].append(video_j)
        
        print(  '\nANOMALIES',\
            '\n  All        ',len(labels_indexs_fn['A']),\
            '\n  AB1 FIGHT  ',    len(labels_indexs_fn['AB1']),\
            '\n  AB2 SHOOT  ',    len(labels_indexs_fn['AB2']),\
            '\n  AB4 RIOT   ',     len(labels_indexs_fn['AB4']),\
            '\n  AB5 ABUSE  ',    len(labels_indexs_fn['AB5']),\
            '\n  AB6 CARACC ',   len(labels_indexs_fn['AB6']),\
            '\n  AG EXPLOS  ',    len(labels_indexs_fn['AG']),\
                
            '\n\nFALSE NEGATIVES (LABEL1 / AS0)',\
            '\n  FN.B1 FIGHT  ',  len(labels_indexs_fn['FNB1']),\
            '\n  FN.B2 SHOOT  ',  len(labels_indexs_fn['FNB2']),\
            '\n  FN.B4 RIOT   ',   len(labels_indexs_fn['FNB4']),\
            '\n  FN.B5 ABUSE  ',  len(labels_indexs_fn['FNB5']),\
            '\n  FN.B6 CARACC ', len(labels_indexs_fn['FNB6']),\
            '\n  FN.G EXPLOS  ',len(labels_indexs_fn['FNG']),\
            '\n  FN.B+G       ',len(labels_indexs_fn['FNBG']),\
                
            '\n\nTRUE POSITIVES (LABEL1 / AS1)',\
            '\n  TP.B1 FIGHT  ',  len(labels_indexs_fn['TPB1']),\
            '\n  TP.B2 SHOOT  ',  len(labels_indexs_fn['TPB2']),\
            '\n  TP.B4 RIOT   ',   len(labels_indexs_fn['TPB4']),\
            '\n  TP.B5 ABUSE  ',  len(labels_indexs_fn['TPB5']),\
            '\n  TP.B6 CARACC ', len(labels_indexs_fn['TPB6']),\
            '\n  TP.G EXPLOS  ',len(labels_indexs_fn['TPG']),\
            '\n  TP.B+G       ',len(labels_indexs_fn['TPBG']),\
                
            '\n\nNORMALS',\
            '\n  All        ',len(labels_indexs_fn['N']),\
                
            '\n\nFALSE POSITIVES (LABEL0 / AS1)',\
            '\n  FP  ',len(labels_indexs_fn['FP']),\
                
            '\n\nTRUE NEGATIVES (LABEL0 / AS0)',\
            '\n  TN  ',len(labels_indexs_fn['TN']),'\n')
                    
        return labels_indexs_fn
    
    as_total = get_as_total_from_res_list(res_list,txt_i)
    
    labels_indexs_fn=get_fn_index_from_res_list(txt_i)
    
    print('set to watch this',watch_this)
    for labels_2_watch in watch_this:
        print('\n',labels_2_watch)
        for i in range(len(labels_indexs_fn[labels_2_watch])):
            video_j = labels_indexs_fn[labels_2_watch][i]
            print(video_j)
            cv2_test_from_aslist(res_list,as_total[video_j],video_j,txt_i)
            
            
# FX RELATED LIVE PROCESSING




# FX RELATED TO XDVIOLENCE STATITCS

def get_vpath_totfra_fromxdvstatstxt(train_or_test):
    filenames,total_frames = [],[]
    
    if train_or_test == 'train':
        basepath = aux.SERVER_TRAIN_COPY_PATH
        fpp = '/raid/DATASETS/.zuble/vigia/zurgb/dataset-xdv-info/train_sort.txt'
        stopper = 3953
    elif train_or_test == 'test':
        basepath = aux.SERVER_TEST_COPY_PATH
        fpp = '/raid/DATASETS/.zuble/vigia/zurgb/dataset-xdv-info/test_sort.txt'
        stopper = 799
    else: Exception(" 'train' or 'test' ")
    
    with open(fpp, 'r') as f:
        lines = f.readlines()
    
    count=0
    for line in lines:
        if count == stopper: break
        filename = line.rstrip().split(' | ')[-1]  # Get the last element of the split line, which is the filename
        total_frame= line.rstrip().split(' | ')[0].strip("frames")
        filenames.append(os.path.join(basepath,filename+'.mp4'))
        total_frames.append(total_frame)
        count+=1
    #print(filenames)
    return filenames,total_frames