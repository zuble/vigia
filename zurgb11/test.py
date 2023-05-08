import os, random, logging , cv2 , csv , time

#import matplotlib.pyplot as plt
import numpy as np
from tqdm.keras import TqdmCallback

''' GPU CONFIGURATION '''
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
import tensorflow as tf
print("tf",tf.version.VERSION)
#from tensorflow import keras

from utils import globo ,  xdv , tfh5


tfh5.set_tf_loglevel(logging.ERROR)
tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
tfh5.check_gpu_conn()
#tfh5.set_memory_growth()
#tfh5.limit_gpu_gb(2)


''' TRAIN & VALDT '''
## TO DO 
## INHERIT TRAIN AND VALDT FUNCTION LOAD FROM NPY @ ZUWAV1
#train_fn, train_labels, train_tot_frames, valdt_fn, valdt_labels , valdt_tot_frames = xdv.train_valdt_files(tframes=True)
train_fp, train_labl, valdt_fp, valdt_labl = xdv.train_valdt_files()


''' CONFIGS '''
CFG_RGB = {
    "frame_step":2, #fstep=2 : 24 fps -> 12 , =4 : -> 
    
    "in_height":120,
    "in_width":160,
    
    "batch_size":1,
    "augment":True,
    "shuffle":False,
    
    "ativa" : 'relu',
    "optima" : 'sgd',
    "batch_type" : 0,   # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : 4000,
    "ckpt_start" : f"{0:0>8}",  #used in train_model: if 00000000 start from scratch, else start from ckpt with config stated
    
    "epochs" : 30
}



''' TEST FX '''

@tf.function
def vas_predict(model,x):
    return model(x, training=False)    

def input_test_video_data(file_name,config,batch_no=0):
    #file_name = 'C:\\Bosch\\Anomaly\\training\\videos\\13_007.avi'
    #file_name = '/raid/DATASETS/anomaly/UCF_Crimes/Videos/Training_Normal_Videos_Anomaly/Normal_Videos308_x264.mp4'
    video = cv2.VideoCapture(file_name)
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    #mtcnn_detector = mtcnn.mtcnn.MTCNN()
    divid_no = 1
    frame_max = config["frame_max"]
    batch_type = config["batch_type"]
    
    # define the nmbers of batchs to divid atual video (divid_no)
    if total_frame > int(frame_max):
        total_frame_int = int(total_frame)
        if total_frame_int % int(frame_max) == 0:
            divid_no = int(total_frame / int(frame_max))
        else:
            divid_no = int(total_frame / int(frame_max)) + 1


    #updates the start frame to 0,frame_max*1,frame_max*2... excluding the last batch
    passby = 0
    if batch_no != divid_no - 1:
        while video.isOpened and passby < int(frame_max) * batch_no:
            passby += 1
            success, image = video.read()
            if success == False:
                break
            
    #updates the last batch starting frame 
    else:
        if batch_type==1:
            #print("1")
            while video.isOpened and passby < total_frame - int(frame_max):
                passby += 1
                success, image = video.read()
                if success == False:
                    break
        #last batch must have >= frame_max/10 otherwise it falls back to batch_type 1
        if batch_type==2 and total_frame - (int(frame_max) * batch_no) >= int(frame_max)*0.1:
            #print("2")
            while video.isOpened and passby < int(frame_max) * batch_no:
                passby += 1
                success, image = video.read()
                if success == False:
                    break
        else:
            while video.isOpened and passby < total_frame - int(frame_max):
                passby += 1
                success, image = video.read()
                if success == False:
                    break

            
    batch_frames, batch_imgs = [], []
    counter = 0
    
    while video.isOpened:               
        success, image = video.read()
        if success == False:
            break
        batch_imgs.append(image)
        
        image_rsz = cv2.resize(image, (in_width, in_height))
        image_array = np.array(image_rsz)/255.0 #normalize
        batch_frames.append(image_array)
        
        counter += 1
        if counter > int(frame_max):
            break
            
    video.release()
    
    #batch_frames_tensor = tf.convert_to_tensor(batch_frames)
    ##print("\tshap tensor",tf.shape( tf.expand_dims(batch_frames_tensor,0) ) )
    #print("\t-batch",batch_no,"[",passby,", ... ] ", batch_frames_tensor.get_shape().as_list() )    

    batch_frames = np.array(batch_frames)
    #print("\tshap NP ARRAY",np.shape( np.expand_dims(batch_frames,0) ))
    print("\t-batch",batch_no,"[",passby,", ... ] ",batch_frames.shape)    

    #return tf.expand_dims(batch_frames_tensor,0), batch_imgs, divid_no, total_frame, passby, fps
    return np.expand_dims(batch_frames,0), batch_imgs, divid_no, total_frame, passby, fps

def test_model(model,model_name,config,files=test_fn):
    print("\n\nTEST MODEL\n")

    # rslt txt file creation
    txt_path = globo.RSLT_PATH+model_name+'-'+str(config["batch_type"])+'_'+str(config["frame_max"])+'.txt'
    if os.path.isfile(txt_path):raise Exception(txt_path,"eriste")
    else: print("\tSaving @",txt_path,"\n")
    
    f = open(txt_path, 'w')
    
    content_str = ''
    total_frames_test = 0
    
    predict_total = [] #to output predict in vizualizer accordingly to the each batch prediction
    predict_max = 0     #to print the max predict related to the file in test
    predict_total_max = [] #to perform the metrics
    
    start_test = time.time()
    for i in range(len(files)):
        if files[i] != '':
            file_path = files[i]
            predict_result = () #to save predictions per file
            time_batch_predict = time_video_predict = 0.0

            #the frist 4000 frames from actual test video                
            batch_frames, batch_imgs, divid_no, total_frames,start_frame, fps = input_test_video_data(file_path,config)
            video_time = total_frames/fps
            total_frames_test += total_frames
            
            #prediction on frist batch
            start_predict1 = time.time()
            #predict_aux = model.predict(batch_frames)[0][0]
            
            predict_aux = vas_predict(model,batch_frames)[0][0].numpy() #using tf.function
            #predict_aux = model(batch_frames,training=False)[0][0]
            end_predict1 = time.time()
            time_video_predict = time_batch_predict = end_predict1-start_predict1
            
            predict_max = predict_aux
            predict_result = (divid_no,start_frame+batch_frames.shape[1],predict_max)
            #print(predict_result,batch_frames.shape)
            
            high_score_patch = 0
            print("\t ",predict_max,"%"," in ","{:.4f}".format(time_batch_predict)," secs")
            
            #when batch_frames (input video) has > frame_max frames
            patch_num = 1
            while patch_num < divid_no:
                batch_frames, batch_imgs, divid_no, total_frames,start_frame, fps = input_test_video_data(file_path,config,patch_num)
                
                #nésimo batch prediction
                start_predict2 = time.time()
                #predict_aux = model.predict(batch_frames)[0][0]
                predict_aux = vas_predict(model,batch_frames)[0][0].numpy() #using tf.function
                end_predict2 = time.time()
                time_batch_predict = end_predict2 - start_predict2
                time_video_predict += time_batch_predict

                if predict_aux > predict_max:
                    predict_max = predict_aux
                    high_score_patch = patch_num
                
                predict_result += (start_frame,start_frame+batch_frames.shape[1], predict_aux)
                #print(predict_result)
                
                print("\t ",predict_aux,"%"," in ","{:.4f}".format(time_batch_predict)," secs")  
                patch_num += 1
            
            predict_total.append(predict_result)
            predict_total_max.append(predict_max)
            print("\n\t",predict_total[i])
            
            if 'label_A' in files[i]:
                print('\nNORM',str(i),':',f'{total_frames:.0f}',"@",f'{fps:.0f}',"fps =",f'{video_time:.2f}',"sec\n\t",
                        files[i][files[i].rindex('/')+1:],
                        "\n\t "+str(predict_max),"% @batch",high_score_patch,"in",str(time_video_predict),"seconds\n",
                        "----------------------------------------------------\n")
            else:
                print('\nABNORM',str(i),':',f'{total_frames:.0f}',"@",f'{fps:.0f}',"fps =",f'{video_time:.2f}',"sec\n\t",
                        files[i][files[i].rindex('/')+1:],
                        "\n\t"+str(predict_max),"% @batch",high_score_patch,"in",str(time_video_predict),"seconds\n",
                        "----------------------------------------------------\n")
                
            content_str += files[i][files[i].rindex('/')+1:] + '|' + str(predict_total_max[i]) + '|' + str(predict_total[i])  + '\n'
            
    end_test = time.time()
    time_test = end_test - start_test

    f.write(content_str)
    f.close()
    print("\nDONE\n\ttotal of",str(total_frames_test),"frames processed in",time_test," seconds",
            "\n\t"+str(total_frames_test / time_test),"frames per second",
            "\n\n********************************************************",
            "\n\n********************************************************")                  

    #remove white spaces in file , for further easier reading
    with open(txt_path, 'r+') as f:txt=f.read().replace(' ', '');f.seek(0);f.write(txt);f.truncate()
    aux.sort_files(txt_path) #sort alphabetcly #also done in get_rslts_from_txt
    run["test/model/rslt"].upload(txt_path)
    
    return predict_total_max, predict_total


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, mode , dummy = 0 , debug = False , printt = True):
        
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
        
        if dummy:
            self.vpath_list = self.vpath_list[:dummy]
            self.len_vpath_list = len(self.vpath_list)
            self.label_list = self.label_list[:dummy]

        print("vpath , label",self.len_vpath_list,(len(self.label_list)))
        
        self.frame_step = CFG_RGB["frame_step"]
        self.maxpool3_min_tframes = 21 * self.frame_step
        
        self.batch_size = CFG_RGB["batch_size"]
        self.frame_max = CFG_RGB["frame_max"]
        
        self.in_height = CFG_RGB["in_height"]
        self.in_width = CFG_RGB["in_width"]
        
        self.augment = CFG_RGB["augment"]
        self.shuffle = CFG_RGB["shuffle"]
        
        
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
        ## idx 3 - flipp False - vpath[1] ..
        
        ## flipp flag
        if self.train and self.augment: 
            i = idx // 2 ; flipp = idx % 2 == 1
        else: i = idx; flipp = False
        
        
        vpath = self.vpath_list[i]
        label = self.label_list[i]
        if not label:label_str=str('NORMAL')
        else:label_str=str('ABNORMAL')
        

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
            
            if label == 0 and tframes > self.frame_max:
                vid_start_idx = random.randint(0, tframes - self.frame_max)
                vid_end_idx = vid_start_idx + self.frame_max
                video.set(cv2.CAP_PROP_POS_FRAMES, vid_start_idx)
            
            else: vid_start_idx = 0 ; vid_end_idx = tframes
            
            frame_step = self.frame_step
            
        else: vid_start_idx = 0 ; vid_end_idx = tframes ; frame_step = 1
        
        
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
        
        
        #batch_frames , batch_labels = [] , [] 
        
        #batch_frames.append(frames_arr)
        #batch_labels.append(label)
        
        #X = np.array(batch_frames).astype(np.float32)
        #y = np.array(batch_labels).astype(np.float32) 
        
        
        X = np.expand_dims(np.array(frames_arr).astype(np.float32), 0)
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
    
    ''' DATA GERADORES '''
    
    ## dummy 
    #train_generator = DataGen( 'train' , 8)
    #valdt_generator = DataGen( 'valdt' , 8)

    ## real 
    train_generator = DataGen( 'train' )
    valdt_generator = DataGen( 'valdt' )
    
    
    ''' INIT TEST MODEL '''
    wght4test_config = {
        "ativa" : 'leakyrelu',
        "optima" : 'sgd',
        "batch_type" : 0, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
        "frame_max" : '4000'
    }
    model, model_name = tfh5.init_test_model(wght4test_config,from_path=globo.MODEL_PATH)
    
    ''' TEST '''
    test_config = {
        "batch_type" : 2, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
        "frame_max" : '3000' 
    }
    predict_total_max, predict_total = test_model(model,model_name,test_config)
        