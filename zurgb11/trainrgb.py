import os, random, logging , cv2 , csv , time

from utils import globo ,  xdv , tfh5

import numpy as np
from tqdm.keras import TqdmCallback

import tensorflow as tf
tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print("\n\nPRECISION POLICY FLOAT16\n\n")

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
#https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/keras/applications/mobilenet_v2/preprocess_input
#https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/keras/applications/Xception


''' TRAIN & VALDT '''
## TO DO 
## INHERIT TRAIN AND VALDT FUNCTION LOAD FROM NPY @ ZUWAV1
train_fp, train_labl, valdt_fp, valdt_labl = xdv.train_valdt_files()


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, mode , printt = True):
        
        #self.vpath_list , self.label_list , self.tframes_list = xdv.load_train_valdt_npy(self.mode,self.frame_max)
        
        self.mode = mode
        if mode == 'valdt'  : 
            self.valdt = True ; self.train = False
            self.vpath_list = valdt_fp
            self.len_vpath_list = len(self.vpath_list)
            self.label_list = valdt_labl     
        elif mode == 'train': 
            self.train = True ; self.valdt = False
            self.vpath_list = train_fp
            self.len_vpath_list = len(self.vpath_list)
            self.label_list = train_labl    
        else: raise Exception("mode can be 'train' or 'valdt' ")
        print("\n\nDataGen",mode,self.train,self.valdt)
        
        
        self.dummy = globo.CFG_RGB_TRAIN["dummy"]
        if self.dummy:
            self.vpath_list = self.vpath_list[:self.dummy]
            self.len_vpath_list = len(self.vpath_list)
            self.label_list = self.label_list[:self.dummy]
        print("vpath , label",self.len_vpath_list,(len(self.label_list)))
        
        
        ## preprocessing fx
        if globo.CFG_RGB_TRAIN["backbone"] == "original":
            self.prep_fx = self.normalize
        elif globo.CFG_RGB_TRAIN["backbone"] == "mobilenetv2":
            self.prep_fx = mobilenet_v2_preprocess_input
        elif globo.CFG_RGB_TRAIN["backbone"] == "xception":
            self.prep_fx = xception_preprocess_input
        
            
        self.frame_step = globo.CFG_RGB_TRAIN["frame_step"]
        self.maxpool3_min_tframes = 21 * self.frame_step
        
        self.batch_size = globo.CFG_RGB_TRAIN["batch_size"]
        self.frame_max = globo.CFG_RGB_TRAIN["frame_max"]
        
        self.in_height = globo.CFG_RGB_TRAIN["in_height"]
        self.in_width = globo.CFG_RGB_TRAIN["in_width"]
        
        self.augment = globo.CFG_RGB_TRAIN["augment"]
        self.shuffle = globo.CFG_RGB_TRAIN["shuffle"]
        
        
        #self.indices = np.arange(self.len_vpath_list)
        if self.augment and self.train: self.lleenn = self.len_vpath_list * 2
        else: self.lleenn = self.len_vpath_list

        self.debug = globo.CFG_RGB_TRAIN["debug"]
        self.printt = printt
 
 
    def normalize(self,x): return x/255.0
    
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
        ## idx 3 - flipp False - vpath[1] ..
        
        ## flipp flag
        if self.train and self.augment: 
            i = idx // 2 ; flipp = idx % 2 == 1
        else: i = idx; flipp = False
        
        
        vpath = self.vpath_list[i]
        label = self.label_list[i]
        #tframes = self.tframes_list[i]
        label_str = 'NORMAL' if not label else 'ABNORMAL'
        

        ## tries to open video , if not attempts 3 times w/ delay
        vc_attmp = 0 ; max_attmp = 3 ; delay = 4 ; video_opened = False
        while vc_attmp < max_attmp and not video_opened:
            video = cv2.VideoCapture(vpath)
            video_opened = video.isOpened()
            if not video_opened:
                vc_attmp += 1
                print(f"\nAttempt {vc_attmp}: Failed to open video: {vpath}")
                time.sleep(delay)
                continue
        ## after failed 3 times, return zeros 
        if not video_opened:
            print(f"\nSkipping video: {vpath}")
            return  np.expand_dims(np.zeros((self.maxpool3_min_tframes, self.in_height, self.in_width, 3), dtype=np.float32) , 0) ,\
                    np.expand_dims(np.array(label, dtype=np.float32) , 0)
        
       
        ## Check if the video has enough frames so shape isnt -1
        tframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if tframes >= self.maxpool3_min_tframes:
            
            ## chunks input to frame_max
            if tframes > self.frame_max:
                
                if label == 0 : 
                    vid_start_idx = random.randint(0, tframes - self.frame_max)
                    video.set(cv2.CAP_PROP_POS_FRAMES, vid_start_idx)
                elif label == 1 : vid_start_idx = 0
                vid_end_idx = vid_start_idx + self.frame_max
            
            ## full
            else: vid_start_idx = 0 ; vid_end_idx = tframes
            
            frame_step = self.frame_step
            
        else: vid_start_idx = 0 ; vid_end_idx = tframes ; frame_step = 1
        
        
        frames = []
        curr_frame = 0
        success, frame = video.read()
        #for j in range(vid_end_idx - vid_start_idx):
        while True:
                
            if not success or curr_frame > vid_end_idx: 
                #if self.debug: print(f"Frame read failed at idx: {j}, curr_frame: {curr_frame}, vid_end_idx: {vid_end_idx}")
                break
            
            frame = cv2.resize(frame, (self.in_width, self.in_height))
            frame_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_prep = self.prep_fx(frame_rgb)    ## uint8 -> float32
            frames.append(frame_prep)
            
            ## jumps the next frame wo decoding
            success, frame, curr_frame = self.skip_frames(video,frame_step)
            #if self.debug: print(f"Frame read successful at idx: {j}, curr_frame: {curr_frame}, success: {success}")
        
        frames_arr = np.stack(frames)
   
        #self.showfr(frames_arr)
        if flipp:
            frames_arr = np.flip(frames_arr, axis=2)
            #flipud : Flip an array vertically (axis=0).
            #fliplr : Flip an array horizontally (axis=1).
        #self.showfr(frames_arr)
        
        X = np.expand_dims(frames_arr, 0) #.astype(np.float32)
        y = np.expand_dims(np.array(label).astype(np.float32), 0)
         
         
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
    
    ''' DATA GERADOR '''
    train_generator = DataGen( 'train' )
    valdt_generator = DataGen( 'valdt' )
    

    ''' MODEL '''
    model,model_name = tfh5.form_model(globo.CFG_RGB_TRAIN)
    
    #model = tfh5.load_h5(model,globo.WEIGHTS_PATH,globo.CFG_RGB_TRAIN)
    #model.load_weights('/raid/DATASETS/.zuble/vigia/zurgb11/model/ckpt/1682641424.8587277_relu_sgd_0_2_8000/1682641424.8587277_relu_sgd_0_2_8000_ckpt-20.h5')
    
    
    ''' CLBK's '''
    if globo.CFG_RGB_TRAIN["dummy"] : clbks = []
    else:
        #tnsrboard_clbk = tfh5.tnsrboard_clbk(model_name,10,60)
        ckpt_clbk = tfh5.ckpt_clbk(model_name)
        early_stop_clbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
        tqdm_clbk = TqdmCallback(globo.CFG_RGB_TRAIN["epochs"],len(train_fp)*2,globo.CFG_RGB_TRAIN["batch_size"],verbose=1)
        csv_clbk = tf.keras.callbacks.CSVLogger(filename= globo.HIST_PATH + model_name + '_history.csv')
        clbks = [ ckpt_clbk , tqdm_clbk , early_stop_clbk , csv_clbk]
    
    
    ''' FIT '''
    history = model.fit(train_generator, 
                        epochs = globo.CFG_RGB_TRAIN["epochs"] ,
                        #steps_per_epoch = len(train_fp) * 2,
                        
                        verbose=2,
                        
                        validation_data = valdt_generator ,
                        #validation_steps = len(valdt_fp),
                        
                        use_multiprocessing = True , 
                        workers = 2 ,
                        callbacks = clbks
                    )

    ''' SAVINGS '''
    if not globo.CFG_RGB_TRAIN["dummy"]:
        model.save(globo.MODEL_PATH + model_name + '.h5')
        model.save(globo.MODEL_PATH + model_name )
        
        model.save_weights(globo.WEIGHTS_PATH + model_name + '_weights.h5')
        
        #hist_csv_file = globo.HIST_PATH + model_name + '_history.csv'
        #with open(hist_csv_file, 'w', newline='') as file:writer = csv.writer(file);writer.writerow(history.history.keys());writer.writerows(zip(*history.history.values()))