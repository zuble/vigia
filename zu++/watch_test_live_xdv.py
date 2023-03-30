import cv2 , logging , os , threading , time
from collections import deque

import numpy as np
import tensorflow as tf

import utils.auxua as aux
import utils.tf_formh5 as tf_formh5
import utils.watch as watch



class ASPlayer:
    def __init__(self, video_path,config,model):
        
        # video info
        self.videopath = video_path
        self.video = cv2.VideoCapture(video_path)
        self.paused = False
        self.video_atual_frame = 0
        self.total_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_time_ms = int(1000/self.fps)
        self.width  = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.wn = 'as'
        
        # buffer info
        self.frame_buffer_capacity = int(config['frame_max'])
        self.frame_buffer = deque(maxlen=self.frame_buffer_capacity)
        self.frame_buffer_atual = 0
        self.frame_buffer_step_atual = 0
        self.frame_buffer_step_len = 32
        self.ref_frame_list = self.save_video_to_list() #create a reference to ve assorted when batch window is ready to process

        # predict thread info
        self.as_buffer = deque(maxlen=1)
        self.close_predict = False
        self.buffer_lock = threading.Lock()
        self.model_thread = threading.Thread(target=self.model_predict)
        self.model_thread.start()
        
        # model info
        self.model = model
        self.model_in_height = 120 
        self.model_in_width = 160
 
    def save_video_to_list(self):
        cap = cv2.VideoCapture(self.videopath)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:break
            frame = cv2.resize(frame, (160, 120)) / 255
            frames.append(frame)
        cap.release()
        return frames       

    def play(self):

        cv2.namedWindow(self.wn) ; as_atual=0

        while self.video.isOpened() :
            sucess, frame = self.video.read()
            if sucess:
                self.video_atual_frame = self.video.get(cv2.CAP_PROP_POS_FRAMES)
                prep_frame = cv2.resize(frame, (self.model_in_width, self.model_in_height)) / 255
                
                with self.buffer_lock:
                    
                    if len(self.frame_buffer) > self.frame_buffer_capacity:
                        #print(len(self.frame_buffer),self.frame_buffer_capacity)
                        self.frame_buffer.popleft()
                    
                    self.frame_buffer.append(prep_frame)
                    self.frame_buffer_atual += 1    
                
                # check if result is ready from model_predict    
                if len(self.as_buffer) != 0:
                    as_atual = self.as_buffer.pop()
                
                if not self.paused:
                    cv2.putText(frame, 'AS:'+str(as_atual), (10, 30), self.font, 0.5, [80,100,250], 2)
                    cv2.putText(frame, 'Frame: %d' % (self.video_atual_frame), (10, 10), self.font, 0.5, [60,250,250], 2)
                    cv2.imshow(self.wn, frame)

                key = cv2.waitKey(self.frame_time_ms)
                # pause
                if key == ord(' '):
                    while True:
                        key = cv2.waitKey(1)
                        if key == ord(' '):break
                # quit            
                if key == ord('q'):break
                
            else:break
            
        self.video.release()
        cv2.destroyAllWindows()
        
        self.close_predict = True
        self.model_thread.join()
    

    @tf.function
    def as_predict(self,x):
        return self.model(x, training=False)
    
    def model_predict(self):    
            
        def assert_both(ass_list,ref_list):
            print(np.shape(ass_list),np.shape(ref_list))
            for ass_frame,ref_frame in zip(ass_list,ref_list):
                assert np.all(ass_frame == ref_frame)
        
        while True:
            with self.buffer_lock:
                if self.frame_buffer_atual == self.frame_buffer_capacity + self.frame_buffer_step_atual*self.frame_buffer_step_len:#self.buffer_size + x*step_len:# #10 , 42 , 74
                    print("\n---predict---")
                    
                    batch = self.frame_buffer.copy()
                    print("\nbatch_shape",np.shape(batch))
                    
                    start_ref = self.frame_buffer_step_atual*self.frame_buffer_step_len
                    end_ref = start_ref + self.frame_buffer_capacity
                    print("ref_list_shape",np.shape(self.ref_frame_list[start_ref:end_ref]),start_ref,end_ref)
                    
                    assert_both(batch,self.ref_frame_list[start_ref:end_ref])
                    
                    print("atual_step",self.frame_buffer_step_atual," | video_atual_frame",self.video_atual_frame," | frame_buffer_atual",self.frame_buffer_atual)
                    self.frame_buffer_step_atual += 1
                    
                    t1 = time.time()
                    batch = np.expand_dims(batch, axis=0) #witf tf took 1.5 sec for buffer_capacity 50
                    t2 = time.time()
                    print("expand_dims",t2-t1,"batch_shape",np.shape(batch))
                    
                    start_predict2 = time.time()
                    prediction2 = self.as_predict(batch)[0][0].numpy()
                    end_predict2 = time.time()
                    self.as_buffer.append(prediction2)
                    time_batch_predict2 = end_predict2 - start_predict2
                    print("as",prediction2,"in",f"{time_batch_predict2:.{3}f}")
                    

            # flag to close thread
            if self.close_predict:return



def init_watch_live( model , test_config , watch_this):
    # frame_max is the window buffer size to predict on

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

    test_mp4_paths , *_ = aux.load_xdv_test()
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

    print('\n  watching',watch_this,'with',test_config)
    for labels_2_watch in watch_this:
        for i in range(len(test_labels_indexs[labels_2_watch])):
            path = test_mp4_paths[test_labels_indexs[labels_2_watch][i]]
            print('\n\n$   ',labels_2_watch,i,path)
            # real shit
            asplayer = ASPlayer(path,test_config,model)
            asplayer.play()



if __name__ == '__main__':
    
    # https://stackoverflow.com/questions/38864711/how-to-optimize-multiprocessing-in-python
    
    
    ''' GPU CONFIGURATION '''

    tf_formh5.set_tf_loglevel(logging.ERROR)
    tf_formh5.tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
    #tf_formh5.set_memory_growth()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    
    ''' INIT TEST MODEL '''

    wght4test_config = {
        "ativa" : 'relu',
        "optima" : 'sgd',
        "batch_type" : 0, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
        "frame_max" : '4000'
    }
    #run["test/config_wght4test"].append(wght4test_config)
    model, model_name = tf_formh5.init_test_model(wght4test_config)
    
    
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
    
    test_config = {
    "batch_type" : 2, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : '240' 
    }
    
    init_watch_live( model , test_config , watch_this=['B1'])
    