import cv2 , multiprocessing , random , logging , os
from time import sleep

import numpy as np

import utils.auxua as aux
import utils.tf_formh5 as tf_formh5
import utils.watch as watch

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.join_flag = False
        # Other initialization stuff

    def run(self):
        while True:
            frameNum, frameData = self.task_queue.get()
            if frameNum == -1 and frameData == None:break
            
            
            # Do computations on image
            # Simulate a processing longer than image fetching
            m = random.randint(0, 1000000)
            while m >= 0:
                m -= 1
            # Put result in queue
            self.result_queue.put("result from image " + str(frameNum))
 
 
 
class ASPlayer_wConsumer:
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

        
    def play(self):        
        # No more than one pending task
        tasks = multiprocessing.Queue(1)
        results = multiprocessing.Queue()
        
        # Init and start consumer
        consumer = Consumer(tasks,results)
        consumer.start()

        #Creating window and starting video capturer from camera
        cv2.namedWindow(self.wn) ; as_atual=0
        
        # Dummy int to represent frame number for display
        frameNum = 0
        # String for result
        text = None
        font = cv2.FONT_HERSHEY_SIMPLEX
        while self.video.isOpened():
            sucess, frame = self.video.read()
            if sucess:
                self.video_atual_frame += 1
                
                # Put image in task queue if empty
                try: tasks.put_nowait((frameNum, frame))
                except: pass
                
                # Get result if ready
                try:
                    # text = results.get(timeout=0.4)  # Use this if processing is fast enough
                    text = results.get_nowait() # Use this to prefer smooth display over frame/text shift
                except: pass

                # Add last available text to last image and display
                print("display:", frameNum, "|", text)
                
                # Showing the frame with all the applied modifications
                cv2.putText(frame,text,(10,25), font, 1,(255,0,0),2)
                cv2.imshow(self.wn, frame)
                
                key = cv2.waitKey(self.frame_time_ms)     
                if key == ord('q'):
                    tasks.put_nowait((-1, None))
                    consumer.join()
                    tasks.close()
                    results.close()
                    break
                if key == ord(' '):
                    while True:
                        key = cv2.waitKey(1)
                        if key == ord(' '):break
        
            else:
                tasks.put_nowait((-1, None))
                consumer.join()
                tasks.close()
                results.close()
                break

        self.video.release()
        cv2.destroyAllWindows()   
        
            

def init_watch_live( model , test_config , watch_this):
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

    print('\n  watching',watch_this,'with',test_config)
    for labels_2_watch in watch_this:
        for i in range(len(test_labels_indexs[labels_2_watch])):
            path = test_mp4_paths[test_labels_indexs[labels_2_watch][i]]
            print('\n\n##################\n',labels_2_watch,i,path,'\n')
            # real shit
            asplayer = ASPlayer_wConsumer(path,test_config,model)
            asplayer.play()


if __name__ == '__main__':
    
    ########## HELPER ############
    # https://stackoverflow.com/questions/38864711/how-to-optimize-multiprocessing-in-python
    
    
    ''' GPU CONFIGURATION '''

    tf_formh5.set_tf_loglevel(logging.ERROR)
    tf_formh5.tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
    #tf_formh5.set_memory_growth()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    
    
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
    
    init_watch_live( model , test_config , watch_this=['B5'])
    