import threading , cv2 , logging , os , time
from queue import Queue
from collections import deque

import numpy as np
import tensorflow as tf

import utils.tf_formh5 as tf_formh5
import utils.auxua as aux


''' GPU CONFIGURATION '''
#tf_formh5.set_tf_loglevel(logging.ERROR)
#tf_formh5.tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
#tf_formh5.set_memory_growth()
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ["CUDA_VISIBLE_DEVICES"]="1,2"


''' INIT TEST MODEL '''
wght4test_config = {
    "ativa" : 'relu',
    "optima" : 'sgd',
    "batch_type" : 0, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : '4000' 
}
#run["test/config_wght4test"].append(wght4test_config)
model, model_name = tf_formh5.init_test_model(wght4test_config)


model_in_height = 120 
model_in_width = 160

test_config = {
    "batch_type" : 2, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : '240' 
    }



@tf.function
def as_predict(x):
    return model(x, training=False)    


def predict_batch(frame_queque,as_queque,stop_event):
    
    def clear_queques():
        while not frame_queque.empty():
            frame_queque.get()
        while not as_queque.empty():
            as_queque.get()    
    
    
    # buffer info
    frame_buffer_capacity = int(test_config['frame_max'])
    frame_buffer = deque(maxlen=frame_buffer_capacity)
    frame_buffer_atual = 0
    frame_buffer_step_atual = 0
    frame_buffer_step_len = 32
    
    while True:#not stop_event.is_set():
        video_atual_frame, frame = frame_queque.get()
        if video_atual_frame == -1 and frame == None:
            clear_queques()
            break
        
        if len(frame_buffer) > frame_buffer_capacity:
            #print(len(frame_buffer),frame_buffer_capacity)
            frame_buffer.popleft()
        
        prep_frame = cv2.resize(frame, (model_in_width, model_in_height)) / 255
        frame_buffer.append(prep_frame)
        frame_buffer_atual += 1    
        
        if frame_buffer_atual == frame_buffer_capacity + frame_buffer_step_atual*frame_buffer_step_len:#buffer_size + x*step_len:# #10 , 42 , 74
            print("\n---predict---")
            
            batch = frame_buffer.copy()
            print("\nbatch_shape",np.shape(batch))
            
            # ASSERT
            #start_ref = frame_buffer_step_atual*frame_buffer_step_len
            #end_ref = start_ref + frame_buffer_capacity
            #print("ref_list_shape",np.shape(ref_frame_list[start_ref:end_ref]),start_ref,end_ref)
            
            #assert_both(batch,ref_frame_list[start_ref:end_ref])
            
            
            #print("atual_step",frame_buffer_step_atual," | video_atual_frame",frameNum," | frame_buffer_atual",frame_buffer_atual)
            frame_buffer_step_atual += 1
            
            t1 = time.time()
            batch = np.expand_dims(batch, axis=0) #witf tf took 1.5 sec for buffer_capacity 50
            t2 = time.time()
            expand_time = t2-t1
            #print("expand_dims",expand_time,"batch_shape",np.shape(batch))
            
            #t3 = time.time()
            #as_atual = as_predict(batch)[0][0].numpy()
            #t4 = time.time()
            #predict_time = t4-t3
            #print("as",as_atual,"in",f"{time_batch_predict2:.{3}f}")
           
                
        as_queque.put((video_atual_frame,frame_buffer_step_atual))    


def ASPlayer2(video_path):

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
    
    
    ''' PROCESS + PREDICT THREADS'''
    # Create the input , output and as queues
    frame_queque = Queue()
    as_queque = Queue()
    
    stop_event = threading.Event()

    # Create the prediction thread
    prediction_thread = threading.Thread(target=predict_batch, args=(frame_queque , as_queque , stop_event))
    prediction_thread.start()
    

    ''' READ VIDEO + FEED QUEQUES'''
    as_atual=frame_buffer_step_atual=frame_buffer_atual_frame=step_atual=0
    while video.isOpened() :
        sucess, frame = video.read()
        if sucess:
            video_atual_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
           
            try: frame_queque.put_nowait((video_atual_frame, frame))
            except: pass
            
            # Get result if ready
            try: frame_buffer_atual_frame,frame_buffer_step_atual = as_queque.get_nowait() # Use this to prefer smooth display over frame/text shift
            except: pass
            
            # Add last available text to last image and display
            print("display:", video_atual_frame, "| got ", frame_buffer_atual_frame ,'at',frame_buffer_step_atual)
            
            # Showing the frame with all the applied modifications
            cv2.putText(frame,str(as_atual),(10,25), font, 1,(255,0,0),2)
            cv2.imshow(wn, frame)
            
            key = cv2.waitKey(frame_time_ms)  
            # quit   
            if key == ord('q'): break
            # pause
            if key == ord(' '):
                while True:
                    key = cv2.waitKey(1)
                    if key == ord(' '):break
    
        else: break
        
        
    ''' CLOSE '''    
    video.release()
    cv2.destroyAllWindows()

    print("signal frame queue to close")
    frame_queque.put_nowait((-1, None))
    
    print("closing predict thread")
    prediction_thread.join()
    
    
    
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
            ASPlayer2(path)


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


