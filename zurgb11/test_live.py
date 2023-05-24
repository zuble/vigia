from utils import globo ,  xdv , tfh5

import threading , cv2 , logging , os , time
from queue import Queue
from collections import deque

import numpy as np
import tensorflow as tf
print("tf",tf.version.VERSION)

tfh5.set_tf_loglevel(logging.ERROR)
tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
tfh5.check_gpu_conn()



model = tf.saved_model.load("/raid/DATASETS/.zuble/vigia/zurgb11/model/model/1683929580.7657886_mobilenetv2_fp16_relu_0.0002_sgd_0_4_4000")


model_in_height = 224 #120 
model_in_width  = 224  #160

test_config = {
    "batch_type" : 2, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : '410' 
    }



@tf.function
def vas_predict(x):
    return model(x, training=False)    

def predict_batch(frame_queque,vas_queque,ref_frame_array,printt):
    
    def assert_batch(ass_list,ref_list):
            print("\nASSERT batch\n",np.shape(ass_list),np.shape(ref_list))
            for ass_frame,ref_frame in zip(ass_list,ref_list):
                assert np.all(ass_frame == ref_frame)

    def clear_queques():
        while not frame_queque.empty():
            frame_queque.get()
        while not vas_queque.empty():
            vas_queque.get()    
    
    ###################################################
    ## BUFFER INFO
    frame_buffer_capacity = int(test_config['frame_max'])
    frame_buffer = deque(maxlen=frame_buffer_capacity)
    frame_buffer_atual = 0
    frame_buffer_step_atual = 0
    frame_buffer_step_len = 72
    ##################################################
    
    vas_atual=expand_time=predict_time=0.0
    
    while True:#not stop_event.is_set():
        video_atual_frame, frame = frame_queque.get()
        
        # checks for close signal
        if video_atual_frame == -1 and frame == None:
            clear_queques()
            break
                
        #prep_frame = cv2.resize(frame, (model_in_width, model_in_height)) / 255
        frame_buffer_atual += 1
        frame_buffer.append(frame)
        #print("buffer ", frame_buffer_atual ,'@',frame_buffer_step_atual)
        
        
        # ASSERT frame
        #print("\nASSERT frame")
        #print(type(frame),np.shape(frame))
        #print(type(ref_frame_array[frame_buffer_atual-1]),np.shape(ref_frame_array[frame_buffer_atual-1]))
        #assert np.all(frame == ref_frame_array[frame_buffer_atual-1])    
        
        
        # different moving window 
        '''seq = [0, 1, 2, 3, 4, 5, 6 ,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        window_size = 10; step_size = 2
        for i in range(0, len(seq) - window_size + 1, step_size):
            print(i)
            #print(seq[i:i+window_size])'''
        
        # check if windows reached its predict frame
        if frame_buffer_atual == frame_buffer_capacity + frame_buffer_step_atual*frame_buffer_step_len:
            
            batch = np.array(frame_buffer.copy())
            
            # ASSERT batch
            if len(ref_frame_array)!=0:
                start_ref = frame_buffer_step_atual*frame_buffer_step_len
                end_ref = start_ref + frame_buffer_capacity
                assert_batch(batch,ref_frame_array[start_ref:end_ref])
            
            frame_buffer_step_atual += 1
            
            
            # PREP BATCH
            t0 = time.time()
            batch = [cv2.resize(frame, (model_in_width, model_in_height)) for frame in batch]
            batch = np.array(batch) / 255.0
            t1 = time.time()
            resiz_time = t1-t0

            t2 = time.time()
            batch = np.expand_dims(np.array(batch).astype(np.float32), axis=0) #witf tf took 1.5 sec for buffer_capacity 50
            t3 = time.time()
            expand_time = t3-t2
            

            # PREDICT
            t4 = time.time()
            vas_atual = vas_predict(batch)[0][0].numpy()
            t5 = time.time()
            predict_time = t5-t4
            
            vas_queque.put((vas_atual)) 
            
            if printt:
                print("\n--- predict at frame %.0f ---"%frame_buffer_atual)
                print("  batch_shape", np.shape(batch))
                print("  resiz_time %.3f" %resiz_time)
                print("  expnd_dims %.3f" %expand_time)
                print("  as",vas_atual,"in",f"{predict_time:.{3}f}","\n")
           
               

class ASPlayer:
    
    def __init__(self,video_path,toassert):
        
        # cv video info
        self.video = cv2.VideoCapture(video_path)
        self.total_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_time_ms = int(1000/self.fps)
        self.width  = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5;self.thickness = 1;self.lineType = cv2.LINE_AA
        self.strap_video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.wn = 'as'+self.strap_video_name
        cv2.namedWindow(self.wn) 

        # make assert per batch or not
        if toassert: ref_frame_array = self.get_array_from_video()
        else: ref_frame_array=[]
            
            
        ''' PREDICT THREAD '''
        # Create the input and as queues
        self.frame_queque = Queue()
        self.vas_queque = Queue()
        
        # Create the prediction thread
        self.prediction_thread = threading.Thread(target=predict_batch, args=(self.frame_queque , self.vas_queque , ref_frame_array,True))
        self.prediction_thread.start()
    
    
        # Ground Truth from annotations
        # if video is abnormal, it gets the asgt_per_frame
        self.asgt_per_frame =  []
        if 'label_A' not in self.strap_video_name:
            asgt = xdv.asgt_from_annotations_xdv()
            self.asgt_per_frame = asgt.get_asgt_per_frame(self.strap_video_name)
            print("asgt_per_frame",np.shape(self.asgt_per_frame))
            self.gt = True
        else: self.gt = False
    
    def get_array_from_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frm = cap.read()
            if not ret:break
            #frm = cv2.resize(frm, (model_in_width, model_in_height)) / 255
            frames.append(frm)
        cap.release()
        return np.array(frames)  

    def play(self):
        
        vas_atual=0.0
        asgt_atual=0
        prev_time = time.time()
        
        ''' READ & DISPLAY VIDEO + FEED QUEQUES'''
        while self.video.isOpened() :
            sucess, frame = self.video.read()
            if sucess:
                video_atual_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            
                # inject frame to queque
                try: self.frame_queque.put_nowait((video_atual_frame, frame))
                except: pass
                
                # Get as if ready
                try: vas_atual = self.vas_queque.get_nowait() # Use this to prefer smooth display over frame/text shift
                except: pass
                
                
                # window prints
                cv2.putText(frame,'AS '+str('%.4f' % (vas_atual)),(10,15),self.font,self.fontScale+0.2,[0,0,255],self.thickness,self.lineType)
                
                if self.gt: asgt_atual = self.asgt_per_frame[video_atual_frame-1]
                cv2.putText(frame,'GT '+str(asgt_atual), (10, 40), self.font,self.fontScale+0.2,[100,250,10],self.thickness,self.lineType)
                
                cv2.putText(frame, '%d' % (video_atual_frame), (10, int(self.height)-10),self.font,self.fontScale,[60,250,250],self.thickness,self.lineType)
                cv2.putText(frame, '%.2f' % (video_atual_frame/self.fps)+' s',(60,int(self.height)-10),self.font,self.fontScale, [80,100,250],self.thickness,self.lineType)
                new_time = time.time()
                cv2.putText(frame, '%.2f' % (1/(new_time-prev_time))+' fps',(140,int(self.height)-10),self.font,self.fontScale,[0,50,200],self.thickness,self.lineType)
                prev_time = new_time
                
                cv2.imshow(self.wn, frame)
                #print("\ndisplay", video_atual_frame)
                
                
                key = cv2.waitKey(self.frame_time_ms)  
                if key == ord('q'): break  # quit
                if key == ord(' '):  # pause
                    while True:
                        key = cv2.waitKey(1)
                        if key == ord(' '):break
        
            else: break
            
            
        ''' CLOSE '''    
        self.video.release()
        cv2.destroyAllWindows()

        print("signal frame queue to close")
        self.frame_queque.put_nowait((-1, None))
        
        print("closing predict thread")
        self.prediction_thread.join()
    
  
  
    
def init_watch_live(watch_this):


    print("\n\nINIT WATCH LIVE")
    def get_index_per_label_from_list(file_list):
        '''retrives video indexs per label or all from xdv_test mp4'''
        
        print("\n  get_index_per_label_from_list")
        
        labels_indexs={'A':[],'B1':[],'B2':[],'B4':[],'B5':[],'B6':[],'G':[],'BG':[]}
        for video_j in range(len(file_list)):
            if 'label_A' in file_list[video_j]:
                labels_indexs['A'].append(video_j)
            else:
                labels_indexs['BG'].append(video_j)
                if 'label_B1' in file_list[video_j]:labels_indexs['B1'].append(video_j)
                if 'label_B2' in file_list[video_j]:labels_indexs['B2'].append(video_j)
                if 'label_B4' in file_list[video_j]:labels_indexs['B4'].append(video_j)
                if 'label_B5' in file_list[video_j]:labels_indexs['B5'].append(video_j)
                if 'label_B6' in file_list[video_j]:labels_indexs['B6'].append(video_j)
                if 'label_G'  in file_list[video_j]:labels_indexs['G'].append(video_j)
        
        return labels_indexs


    test_mp4_paths , *_ = xdv.test_files()
    print('\n  test_mp4_paths',np.shape(test_mp4_paths))


    test_labels_indexs = get_index_per_label_from_list(test_mp4_paths)
    print(  '\tA NORMAL',  len(test_labels_indexs['A']),\
            '\n\n\tB1 FIGHT',  len(test_labels_indexs['B1']),\
            '\n\tB2 SHOOT',  len(test_labels_indexs['B2']),\
            '\n\tB4 RIOT',   len(test_labels_indexs['B4']),\
            '\n\tB5 ABUSE',  len(test_labels_indexs['B5']),\
            '\n\tB6 CARACC', len(test_labels_indexs['B6']),\
            '\n\tG EXPLOS',len(test_labels_indexs['G']),\
            '\n\n\tBG ALL ANOMALIES',len(test_labels_indexs['BG']))


    print('\n  watching',watch_this)
    for labels_2_watch in watch_this:
        for i in range(len(test_labels_indexs[labels_2_watch])):
            path = test_mp4_paths[test_labels_indexs[labels_2_watch][i]]
            print('\n\n##################\n',labels_2_watch,i,path,'\n')
            # real shit
            asplayer = ASPlayer(path,False)
            asplayer.play()
            #break



'''
    A  NORMAL
    B1 FIGHT
    B2 SHOOTING
    B4 RIOT
    B5 ABUSE
    B6 CAR ACCIDENT
    G  EXPLOSION 
    BG ALL ANOMALIES
'''

init_watch_live(watch_this=['B2'])


