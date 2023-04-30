import os, random, logging , cv2 , csv

#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print("tf",tf.version.VERSION)
#from tensorflow import keras
from tqdm.keras import TqdmCallback

from utils import globo ,  xdv , tfh5


''' GPU CONFIGURATION '''
os.environ["CUDA_VISIBLE_DEVICES"]="3"

tfh5.set_tf_loglevel(logging.ERROR)
tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.

tfh5.check_gpu_conn()

#tfh5.set_memory_growth()
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
#tfh5.limit_gpu_gb(2)


''' TRAIN & VALDT '''
#train_fn, train_labels, train_tot_frames, valdt_fn, valdt_labels , valdt_tot_frames = xdv.train_valdt_files(tframes=True)
train_fp, train_labl, valdt_fp, valdt_labl = xdv.train_valdt_files()


''' CONFIGS '''
train_config = {
    "frame_step":2, #24 fps -> 12
    
    "in_height":120,
    "in_width":160,
    
    "batch_size":1,
    "augment":True,
    "shuffle":False,
    
    "ativa" : 'relu',
    "optima" : 'sgd',
    "batch_type" : 0,   # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : 8000,
    "ckpt_start" : f"{0:0>8}",  #used in train_model: if 00000000 start from scratch, else start from ckpt with config stated
    
    "epochs" : 30
}


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, vpath_list, label_list, config , mode , debug = False , printt = True):
        
        self.mode = mode
        if mode == 'valdt'  : self.valdt = True ; self.train = False
        elif mode == 'train': self.train = True ; self.valdt = False
        else: raise Exception("mode can be 'train' or 'valdt' ")
        print("\n\nDataGen",mode,self.train,self.valdt)
        
        
        self.vpath_list = vpath_list
        self.len_vpath_list = len(self.vpath_list)
        self.label_list = label_list
        print("vpath , label",self.len_vpath_list,(len(label_list)))
        
        
        self.frame_step = config["frame_step"]
        self.maxpool3_min_tframes = 21 * self.frame_step
        
        self.batch_size = config["batch_size"]
        self.frame_max = config["frame_max"]
        
        self.in_height = config["in_height"]
        self.in_width = config["in_width"]
        
        self.augment = config["augment"]
        self.shuffle = config["shuffle"]
        
        
        #self.indices = np.arange(self.len_vpath_list)
        if self.augment and self.train: self.lleenn = self.len_vpath_list * 2
        else: self.lleenn = self.len_vpath_list

        self.debug = debug
        self.printt = printt
 
    def skip_frames(self,cap,fs):
        start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print("skip_start",start_frame)
        while True:
            success = cap.grab()
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if not success or curr_frame - start_frame >= fs:break
        
        if not success:return success, None, start_frame + fs

        success, image = cap.retrieve()
        return success, image, curr_frame        
    
    def showfr(self, fr1):
        for frame in fr1:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(0)
            if key == ord('q'): break  # quit
        cv2.destroyAllWindows()
        
        
    def __len__(self):
        if self.debug: print("\n\n__len__",self.mode,"= n batchs = ",self.lleenn, " w/ '1' vid_frames each")
        return self.lleenn
        
        
    def __getitem__(self, idx):
        ## idx 0 - flipp False - vpath[0] 
        ## idx 1 - flipp True - vpath[0] 
        
        ## flipp flag
        if self.train and self.augment: i = idx // 2 ; flipp = idx % 2 == 1
        else: i = idx; flipp = False
        
        batch_frames , batch_labels = [] , [] 
        
        vpath = self.vpath_list[i]
        label = self.label_list[i]
        if not label:label_str=str('NORMAL')
        else:label_str=str('ABNORMAL')
        
        
        video = cv2.VideoCapture(vpath)
        if video.isOpened(): pass
        else: print(f"Failed to open video: {vpath}")
        
        tframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
         
        # Check if the video has enough frames so shape isnt -1
        if tframes >= self.maxpool3_min_tframes:
            
            if label == 0 and tframes > self.frame_max:
                vid_start_idx = random.randint(0, tframes - self.frame_max)
                vid_end_idx = vid_start_idx + self.frame_max
                video.set(cv2.CAP_PROP_POS_FRAMES, vid_start_idx)
            else: vid_start_idx = 0; vid_end_idx = tframes
            frame_step = self.frame_step
            
        else: vid_start_idx = 0; vid_end_idx = tframes; frame_step = 1
        
        
        frames = []
        curr_frame = 0
        success, frame = video.read()
        for j in range(vid_end_idx - vid_start_idx):
            
            if not success or curr_frame > vid_end_idx: 
                if self.debug: print(f"Frame read failed at idx: {j}, curr_frame: {curr_frame}, vid_end_idx: {vid_end_idx}")
                break
            
            frame = cv2.resize(frame, (self.in_width, self.in_height))
            frame_arr = np.array(frame)/255.0
            frames.append(frame_arr)
            ## jumps the next frame wo decoding
            success, frame, curr_frame = self.skip_frames(video,frame_step)
            if self.debug: print(f"Frame read successful at idx: {j}, curr_frame: {curr_frame}, success: {success}")
            
             
        frames_arr = np.array(frames)
        
        #self.showfr(frames_arr)
        if flipp:frames_arr = np.flip(frames_arr, axis=2)
        #self.showfr(frames_arr)
        
        batch_frames.append(frames_arr)
        batch_labels.append(label)
        
        X = np.array(batch_frames).astype(np.float32)
        y = np.array(batch_labels).astype(np.float32)
         
         
        ## prints
        if self.debug:print(f"\n********** {self.mode}_{i} **** {label_str} ***************\n" \
                            f"    {tframes} @ {os.path.basename(vpath)}\n"\
                            f"    vid_idx {vid_start_idx} {vid_end_idx}\n\n"
                            f"    X  w/ flip {flipp}\n"
                            f"    {frames_arr.shape}, dtype: {frames_arr.dtype}\n"
                            f"    {X.shape}, dtype: {X.dtype}\n"
                            f"    y {y}\n")
        elif self.printt:print(  f"\n\n\n£££ {self.mode}_{i} * {label_str} * {tframes} @ {vpath}\n"
                                f"    vid_idx {vid_start_idx} {vid_end_idx}\n"
                                f"    X {X.shape}  w/ flip {flipp} @{X.dtype} , y {y}")
        
        return X , y

   
          
if __name__ == "__main__":
    
    ''' DATA '''
    
    ## dummy GENERATOR
    
    #t = train_fp[:4]   ;   v = valdt_fp[:4] 
    #tl = train_labl[:4] ; vl = valdt_labl[:4]
    #train_generator = DataGen(t, tl, train_config , 'train' )
    #valdt_generator = DataGen(v, vl, train_config , 'valdt' )

    ## real GERADOR
    train_generator = DataGen(train_fp, train_labl, train_config, 'train')
    valdt_generator = DataGen(valdt_fp, valdt_labl, train_config, 'valdt')
    

    #for step in range(NUM_STEPS):
    #    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
    #        train_data = next(dataset)
    #        train_step(train_data)

    #raise Exception("o") 
    
    
    ''' MODEL '''
    model,model_name = tfh5.form_model(train_config)
    
    ## CLBK's
    ckpt_clbk = tfh5.ckpt_clbk(model_name)
    early_stop_clbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    tqdm_clbk = TqdmCallback(train_config["epochs"],len(train_fp)*2,train_config["batch_size"],verbose=1)
    #tnsrboard_clbk = tfh5.tnsrboard_clbk(model_name,10,60)
    
    ''' FIT '''
    history = model.fit(train_generator, 
                        epochs = train_config["epochs"] ,
                        steps_per_epoch = len(train_fp) * 2,
                        
                        verbose=2,
                        
                        validation_data = valdt_generator ,
                        validation_steps = len(valdt_fp),
                        
                        use_multiprocessing = True , 
                        workers = 16 ,
                        callbacks=[ckpt_clbk , early_stop_clbk , tqdm_clbk] # , 
                    )


    ''' SAVINGS '''
    model.save(globo.MODEL_PATH + model_name + '.h5')
    model.save(globo.MODEL_PATH + model_name )
    
    model.save_weights(globo.WEIGHTS_PATH + model_name + '_weights.h5')
    
    hist_csv_file = globo.HIST_PATH + model_name + '_history.csv'
    with open(hist_csv_file, 'w', newline='') as file:writer = csv.writer(file);writer.writerow(history.history.keys());writer.writerows(zip(*history.history.values()))
        