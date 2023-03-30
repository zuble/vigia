import cv2 , multiprocessing , random , logging , os , time
from time import sleep
from collections import deque

import numpy as np
import tensorflow as tf

import utils.auxua as aux
import utils.tf_formh5 as tf_formh5
import utils.watch as watch


# https://stackoverflow.com/questions/38864711/how-to-optimize-multiprocessing-in-python


class ConsumerPredict(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

        
        # model info
        self.model_in_height = 120 
        self.model_in_width = 160
        
        self.wght4test_config = {
            "ativa" : 'relu',
            "optima" : 'sgd',
            "batch_type" : 0, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
            "frame_max" : '4000' }
        self.model,self.model_name = self.init_model()
        
        # test config
        self.test_config = {
            "batch_type" : 2, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
            "frame_max" : '240' }
        
        # buffer info
        self.frame_buffer_capacity = int(self.test_config['frame_max'])
        self.frame_buffer = deque(maxlen=self.frame_buffer_capacity)
        self.frame_buffer_atual = 0
        self.frame_buffer_step_atual = 0
        self.frame_buffer_step_len = 32
        self.as_atual = 0
        
        # timers
        self.expand_time = self.predict_time = 0.0
            
    def init_model(self):
        
        ''' GPU CONFIGURATION '''
        tf_formh5.set_tf_loglevel(logging.ERROR)
        tf_formh5.tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
        tf_formh5.set_memory_growth()
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ["CUDA_VISIBLE_DEVICES"]="2"
        
        
        ''' INIT TEST MODEL '''
        #run["test/config_wght4test"].append(wght4test_config)
        model, model_name = tf_formh5.init_test_model(self.wght4test_config)
        return model, model_name    
        
    
    
    
    
    def run(self):
        
        def assert_both(ass_list,ref_list):
            print(np.shape(ass_list),np.shape(ref_list))
            for ass_frame,ref_frame in zip(ass_list,ref_list):
                assert np.all(ass_frame == ref_frame)
        
        while True:
            frameNum, frame = self.task_queue.get()
            if frameNum == -1 and frame == None:break
            prep_frame = cv2.resize(frame, (self.model_in_width, self.model_in_height)) / 255
            
            if len(self.frame_buffer) > self.frame_buffer_capacity:
                print(len(self.frame_buffer),self.frame_buffer_capacity)
                self.frame_buffer.popleft()
            
            self.frame_buffer.append(prep_frame)
            self.frame_buffer_atual += 1    
            
            
            if self.frame_buffer_atual == self.frame_buffer_capacity + self.frame_buffer_step_atual*self.frame_buffer_step_len:#self.buffer_size + x*step_len:# #10 , 42 , 74
                print("\n---predict---")
                
                batch = self.frame_buffer.copy()
                print("\nbatch_shape",np.shape(batch))
                
                # ASSERT
                #start_ref = self.frame_buffer_step_atual*self.frame_buffer_step_len
                #end_ref = start_ref + self.frame_buffer_capacity
                #print("ref_list_shape",np.shape(self.ref_frame_list[start_ref:end_ref]),start_ref,end_ref)
                
                #assert_both(batch,self.ref_frame_list[start_ref:end_ref])
                
                
                #print("atual_step",self.frame_buffer_step_atual," | video_atual_frame",frameNum," | frame_buffer_atual",self.frame_buffer_atual)
                self.frame_buffer_step_atual += 1
                
                t1 = time.time()
                batch = np.expand_dims(batch, axis=0) #witf tf took 1.5 sec for buffer_capacity 50
                t2 = time.time()
                self.expand_time = t2-t1
                #print("expand_dims",expand_time,"batch_shape",np.shape(batch))
                
                t3 = time.time()
                self.as_atual = self.as_predict(batch)[0][0].numpy()
                t4 = time.time()
                self.predict_time = t4-t3
                #print("as",self.as_atual,"in",f"{time_batch_predict2:.{3}f}")
                
            '''# Do computations on image
            # Simulate a processing longer than image fetching
            m = random.randint(0, 1000000)
            while m >= 0:
                m -= 1'''
                
            # Put result in queue
            self.result_queue.put((str(frameNum),self.as_atual,self.expand_time,self.predict_time))
           


def ASPlayer_wConsumer(video_path):
    
    ''' ConsumerPredict '''   
    # No more than one pending task
    tasks = multiprocessing.Queue(1)
    results = multiprocessing.Queue()
    # Init and start consumer
    consumer = ConsumerPredict(tasks,results)
    consumer.start()


    # video info
    video = cv2.VideoCapture(video_path)
    video_atual_frame1 = 0
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_time_ms = int(1000/fps)
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    font = cv2.FONT_HERSHEY_SIMPLEX
    wn = 'as'; cv2.namedWindow(wn) ; 
    
    as_atual = consumer_atual_frame = 0
    expand_time = predict_time = 0.0
    while video.isOpened():
        sucess, frame = video.read()
        if sucess:
            video_atual_frame1 += 1
            video_atual_frame2 = video.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Put image in task queue if empty
            try: tasks.put_nowait((video_atual_frame1, frame))
            except: pass
            
            # Get result if ready
            try:
                # text = results.get(timeout=0.4)  # Use this if processing is fast enough
                consumer_atual_frame,as_atual,expand_time,predict_time = results.get_nowait() # Use this to prefer smooth display over frame/text shift
            except: pass

            #if tasks.empty():
            #    tasks.put_nowait((video_atual_frame1, frame))
            #else:
            #    as_atual = results.get_nowait()
            
            
            # Add last available text to last image and display
            print("display:", video_atual_frame1,video_atual_frame2, "| got image ", consumer_atual_frame)
            print("expand:",expand_time,"predict",predict_time)
            # Showing the frame with all the applied modifications
            cv2.putText(frame,str(as_atual),(10,25), font, 1,(255,0,0),2)
            cv2.imshow(wn, frame)
            
            key = cv2.waitKey(frame_time_ms)     
            if key == ord('q'):
                tasks.put_nowait((-1, None))
                consumer.terminate()
                break
            if key == ord(' '):
                while True:
                    key = cv2.waitKey(1)
                    if key == ord(' '):break
    
        else:
            #tasks.put_nowait((-1, None))
            #consumer.join()
            #tasks.close()
            #results.close()
            tasks.put_nowait((-1, None))
            consumer.terminate()
            break

    video.release()
    cv2.destroyAllWindows()   
        
            
def init_watch_live(watch_this):
    '''
        frame_max is the window buffer size to predict on
    '''
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

    print('\n  watching',watch_this)
    for labels_2_watch in watch_this:
        for i in range(len(test_labels_indexs[labels_2_watch])):
            path = test_mp4_paths[test_labels_indexs[labels_2_watch][i]]
            print('\n\n##################\n',labels_2_watch,i,path,'\n')
            # real shit
            ASPlayer_wConsumer(path)
    return       
    

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


init_watch_live(watch_this=['B5'])
    
    
    
    