import cv2, threading, time, logging, os
from collections import deque

import tensorflow as tf
import numpy as np

import utils.auxua as aux
import utils.tf_formh5 as tf_formh5

class VideoPlayer:
    def __init__(self, video_path,buffer_capacity,ref_frame_list,model):
        
        # video info
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
        self.frame_buffer_capacity = buffer_capacity
        self.frame_buffer = deque(maxlen=self.frame_buffer_capacity)
        self.frame_buffer_atual = 0
        self.frame_buffer_step_atual = 0
        self.frame_buffer_step_len = 32
        self.ref_frame_list = ref_frame_list

        # predict thread info
        #self.as_queque = queue.Queue()
        self.as_buffer = deque(maxlen=1)
        self.close_predict = False
        self.buffer_lock = threading.Lock()
        self.model_thread = threading.Thread(target=self.model_predict)
        self.model_thread.start()
        
        # model info
        self.model = model
        self.model_in_height = 120 
        self.model_in_width = 160
 
    def play(self):
        cv2.namedWindow(self.wn)
        while self.video.isOpened() :
            sucess, frame = self.video.read()
            if sucess:
                self.video_atual_frame = self.video.get(cv2.CAP_PROP_POS_FRAMES)
                
                frame = cv2.resize(frame, (self.model_in_width, self.model_in_height)) / 255
                
                with self.buffer_lock:
                    
                    if len(self.frame_buffer) > self.frame_buffer_capacity:
                        print(len(self.frame_buffer),self.frame_buffer_capacity)
                        self.frame_buffer.popleft()
                    
                    self.frame_buffer.append(frame)
                    self.frame_buffer_atual += 1    
                    
                    #if not self.as_buffer.
                    #    as_atual = self.as_queque.get()   
                #if as_atual is not None:    
                #    cv2.putText(frame, 'AS:'+str(as_atual), (10, 30), self.font, 0.5, [80,100,250], 2)
                
                if not self.paused:
                    #cv2.putText(frame, 'Frame: %d' % (self.video_atual_frame), (10, 10), self.font, 0.5, [60,250,250], 2)
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
                    prediction2 = self.as_predict(batch)
                    end_predict2 = time.time()
                    time_batch_predict2 = end_predict2 - start_predict2
                    print("as",prediction2.numpy(),"in",f"{time_batch_predict2:.{3}f}")
                    
            # flag to close thread
            if self.close_predict:return
            
            
def save_frames_to_list(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:break
        frame = cv2.resize(frame, (160, 120)) / 255
        frames.append(frame)
    cap.release()
    return frames       
        
        
if __name__ == '__main__':
    
    ########## HELPER ############
    # https://stackoverflow.com/questions/38864711/how-to-optimize-multiprocessing-in-python
    
    
    ''' GPU CONFIGURATION '''

    tf_formh5.set_tf_loglevel(logging.ERROR)
    tf_formh5.tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,"
    
    
    
    ''' INIT TEST MODEL '''

    wght4test_config = {
        "ativa" : 'relu',
        "optima" : 'sgd',
        "batch_type" : 0, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
        "frame_max" : '4000'
    }
    #run["test/config_wght4test"].append(wght4test_config)
    model, model_name = tf_formh5.init_test_model(wght4test_config)
    
    
    path1 = '/raid/DATASETS/anomaly/XD_Violence/testing/v=wQrV75N2BrI__#1_label_A.mp4'
    path2 = '/raid/DATASETS/anomaly/XD_Violence/testing/v=qmsQ-obL1Z4__#00-03-26_00-04-04_label_B6-0-0.mp4'
    path3 = '/raid/DATASETS/anomaly/XD_Violence/testing/The.World.Is.Not.Enough.1999__#00-10-22_00-10-40_label_G-0-0.mp4'
    
    #video_player = VideoPlayer(path1,100)
    #video_player = VideoPlayer(path2,100)
    
    ref_frame_list = save_frames_to_list(path2)
    video_player = VideoPlayer(path2,50,ref_frame_list,model)
    print(video_player.frame_time_ms)
    video_player.play()