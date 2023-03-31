import threading , cv2 , logging , os , time
from queue import Queue
from collections import deque

import numpy as np
import tensorflow as tf

import utils.tf_formh5 as tf_formh5
import utils.auxua as aux


# https://stackoverflow.com/questions/38864711/how-to-optimize-multiprocessing-in-python


''' GPU CONFIGURATION '''
tf_formh5.set_tf_loglevel(logging.ERROR)
tf_formh5.tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
tf_formh5.set_memory_growth()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"


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
    "frame_max" : '500' 
    }



@tf.function
def as_predict(x):
    return model(x, training=False)    

def predict_batch(frame_queque,as_queque,ref_frame_array):
    
    def assert_batch(ass_list,ref_list):
            print("\nASSERT batch\n",np.shape(ass_list),np.shape(ref_list))
            for ass_frame,ref_frame in zip(ass_list,ref_list):
                assert np.all(ass_frame == ref_frame)

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
    
    as_atual=expand_time=predict_time=0.0
    
    while True:#not stop_event.is_set():
        video_atual_frame, frame = frame_queque.get()
        
        # checks for close signal
        if video_atual_frame == -1 and frame == None:
            clear_queques()
            break
                
        #prep_frame = cv2.resize(frame, (model_in_width, model_in_height)) / 255
        frame_buffer_atual += 1
        frame_buffer.append(frame)
        print("buffer ", frame_buffer_atual ,'@',frame_buffer_step_atual)
        
        
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
            #start_ref = frame_buffer_step_atual*frame_buffer_step_len
            #end_ref = start_ref + frame_buffer_capacity
            #assert_batch(batch,ref_frame_array[start_ref:end_ref])
            
            frame_buffer_step_atual += 1
            
            
            # PREP BATCH
            t0 = time.time()
            batch = [cv2.resize(frame, (model_in_width, model_in_height)) for frame in batch]
            batch = np.array(batch) / 255.0
            t1 = time.time()
            resiz_time = t1-t0

            t2 = time.time()
            batch = np.expand_dims(batch, axis=0) #witf tf took 1.5 sec for buffer_capacity 50
            t3 = time.time()
            expand_time = t3-t2
            

            # PREDICT
            t4 = time.time()
            as_atual = as_predict(batch)[0][0].numpy()
            t5 = time.time()
            predict_time = t5-t4
            
            as_queque.put((as_atual)) 
            
            print("\n--- predict at frame %.0f ---"%frame_buffer_atual)
            print("  batch_shape", np.shape(batch))
            print("  resiz_time %.3f" %resiz_time)
            print("  expnd_dims %.3f" %expand_time)
            print("  as",as_atual,"in",f"{predict_time:.{3}f}","\n")
           
               


#ref frame list
def save_video_to_array(pathh):
    cap = cv2.VideoCapture(pathh)
    frames = []
    while True:
        ret, frm = cap.read()
        if not ret:break
        #frm = cv2.resize(frm, (model_in_width, model_in_height)) / 255
        frames.append(frm)
    cap.release()
    return np.array(frames)  


def ASPlayer2(video_path):

    # cv video info
    video = cv2.VideoCapture(video_path)
    video_atual_frame1 = 0
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_time_ms = int(1000/fps)
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    font = cv2.FONT_HERSHEY_SIMPLEX
    wn = 'as'
    cv2.namedWindow(wn) 
    
       
    ref_frame_array = save_video_to_array(video_path)
    
    
    ''' PREDICT THREAD '''
    # Create the input and as queues
    frame_queque = Queue()
    as_queque = Queue()

    # Create the prediction thread
    prediction_thread = threading.Thread(target=predict_batch, args=(frame_queque , as_queque , ref_frame_array))
    prediction_thread.start()
    

    ''' READ VIDEO + FEED QUEQUES'''
    as_atual=0.0
    while video.isOpened() :
        sucess, frame = video.read()
        if sucess:
            video_atual_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
           
            # inject frame to queque
            try: frame_queque.put_nowait((video_atual_frame, frame))
            except: pass
            
            # Get as if ready
            try: 
                as_atual = as_queque.get_nowait() # Use this to prefer smooth display over frame/text shift
            except: pass
            

            cv2.putText(frame,str(as_atual),(10,25), font, 1,(255,0,0),2)
            cv2.imshow(wn, frame)
            print("\ndisplay", video_atual_frame)
            

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

init_watch_live(watch_this=['B1'])


