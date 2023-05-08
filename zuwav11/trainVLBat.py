import os, time, logging
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print("tf",tf.version.VERSION)

#from tensorflow import keras
from tqdm.keras import TqdmCallback

from utils import globo , xdv , tfh5 , sinet


''' GPU CONFIGURATION '''
os.environ["CUDA_VISIBLE_DEVICES"]="2"
tfh5.set_tf_loglevel(logging.ERROR)
tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
#tfh5.set_memory_growth()
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
#tfh5.limit_gpu_gb(2)


''' TRAIN & VALDT '''
#train_fn, train_labels, train_tot_frames, valdt_fn, valdt_labels , valdt_tot_frames = xdv.train_valdt_files(tframes=True)
#train_fp, train_labl, valdt_fp, valdt_labl = xdv.train_valdt_files()
tv_dict=xdv.train_valdt_from_npy()


''' CONFIGS '''
CFG_SINET = {
    
    'sinet_version': 'fsd-sinet-vgg42-tlpf_aps-1',
    
    'graph_filename' : os.path.join(globo.FSDSINET_PATH,"fsd-sinet-vgg42-tlpf_aps-1.pb"),
    'metadata_file'  : os.path.join(globo.FSDSINET_PATH,"fsd-sinet-vgg42-tlpf_aps-1.json"),
    
    'audio_fs_input':22050,
    'batchSize' : 64,
    'lastPatchMode': 'repeat',
    'patchHopSize' : 50,
    
    
    'anom_labels' : ["Alarm","Boom","Crowd","Dog","Drill","Explosion","Fire","Gunshot and gunfire","Hammer","Screaming","Screech",\
                    "Shatter","Shout","Siren","Slam","Squeak","Yell"],
    'anom_labels_i' : [4,18,51,59,65,72,78,92,94,145,146,147,148,152,154,161,198],
    
    'anom_labels2' : ["Boom","Explosion","Fire","Gunshot and gunfire","Screaming",\
                    "Shout","Siren","Yell"],
    'anom_labels_i2' : [18,72,78,92,147,148,152,198],
    
    'full_or_max' : 'max', #chose output to (timesteps,labels_total) ot (1,labels_total)
    'labels_total' : 200
    
}
CFG_WAV= {
    
    "full_or_max" : CFG_SINET["full_or_max"],
    "sinet_aas_len" : CFG_SINET["labels_total"], 
    
    "shuffle" : False,
    
    "ativa" : 'relu',
    "optima" : 'sgd',
    "batch_type" : 0,   # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : 8000, # used to trim the normal videos, label 0 
    "ckpt_start" : f"{0:0>8}",  #used in train_model: if 00000000 start from scratch, else start from ckpt with CFG_WAV stated
    
    "epochs" : 1,
    "batch_size" : 32
    
}


''' Data Gerador w train_xdv '''
class DataGenOrigBatch(tf.keras.utils.Sequence):
    def __init__(self,vpath_list, label_list , tot_frames_list = [] , mode = 'train' , debug = False):
        
        self.mode = mode
        if mode == 'valdt' : self.valdt = True ;  self.train = False
        else: self.train = True ; self.valdt = False
        print("\n\n\n\tDataGen",mode,self.train,self.valdt)
        
        
        self.vpath_list = vpath_list
        self.len_vpath_list = len(self.vpath_list)
        self.label_list = label_list
        self.tot_frames_list = tot_frames_list
        print("\tvpath , label , tot_frames",self.len_vpath_list,(len(label_list)),len(tot_frames_list))
        
        
        self.batch_size = CFG_WAV["batch_size"]
        self.frame_max = CFG_WAV["frame_max"]
        self.shuffle = CFG_WAV["shuffle"]
        print("\tbatch_size",self.batch_size,"frame_max",self.frame_max,"shuffle",self.shuffle)
        
        self.sinet = sinet.Sinet(CFG_SINET)
        
        self.debug = debug
        print("\n\n\n")
 
    def __len__(self):
        if self.debug: print("\n\n__len__",self.mode,"= n batchs = ",int(np.ceil(self.len_vpath_list / float(self.batch_size))), " w/ ",self.batch_size," vid_frames each")
        return int(np.ceil(self.len_vpath_list / float(self.batch_size)))
        
    
    def call_get_sigmoid(self, i, p_es, vpath_list, sframe_list, eframe_list, debug):
        print("\t call for ",i , os.path.basename(vpath_list[i]),sframe_list[i],eframe_list[i])
        p_es[i , :] = self.sinet.get_sigmoid(vpath_list[i],sframe_list[i],eframe_list[i],debug=debug)

    
    def __getitem__(self, idx):
         
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.len_vpath_list)
        current_batch_size = end_idx - start_idx
        
        print(  f"\n\n***************** START BATCH  {idx} [{start_idx} , {end_idx}] @ size {current_batch_size} ********************"\
                f"\n£££ {self.mode} w/ {self.sinet.model_config['full_or_max']} mode" )
        

        ## determine the frame window for each video in batch
        print("\n###frame windowing")
        sf_list , ef_list = [] , []
        for i in range(current_batch_size):
            
            vpath =  self.vpath_list[start_idx + i]
            label = self.label_list[start_idx + i]
            tot_frame = self.tot_frames_list[start_idx + i]
            
            if label == 0 and tot_frame > self.frame_max:  # Normal video w/ > frame_max frames
                start_frame = np.random.randint(0, tot_frame - self.frame_max )
                end_frame = start_frame + self.frame_max
            else:  # Abnormal/Normal video w/ < frame_max frames
                start_frame = 0
                end_frame = -1  # Use -1 to indicate no restriction on end frame
                
            print("\t",i,os.path.basename(vpath),"label",label,"tott_frame",tot_frame,start_frame,end_frame)
            
            sf_list.append(start_frame)
            ef_list.append(end_frame)
            
        if self.debug:print("\tstart_frames , end_frames",np.shape(sf_list),np.shape(ef_list))
        
        
        # Preallocate p_es array
        p_es = np.empty((current_batch_size , CFG_SINET["labels_total"]), dtype=np.float32)
        if self.debug:print("\n\tP_ES @ data_gen b4",p_es.shape,"\n")
        
        
        # Calculate p_es in parallel
        with ThreadPoolExecutor() as executor:
            executor.map(lambda i: self.call_get_sigmoid(i, p_es, self.vpath_list[start_idx:end_idx], sf_list, ef_list, True), range(current_batch_size))
        
        if self.debug:print("\n\tP_ES @ data_gen after",p_es.shape,"\n")
        
        
        # Preallocate X_batch and y_batch arrays
        X_batch = np.empty((current_batch_size, 1, p_es.shape[1]), dtype=np.float32)
        y_batch = np.empty((current_batch_size, 1), dtype=np.float32)
        if self.debug: print(  f"\n\tBATCH @ data_gen b4\n\tX {X_batch.shape} @{X_batch.dtype} , y {y_batch} , {y_batch.shape} @{y_batch.dtype}\n\n")
        
        
        # Iterate over the samples in the batch
        for i in range(current_batch_size):
            
            vpath = os.path.basename(self.vpath_list[start_idx + i])
            label = self.label_list[start_idx + i]
            tot_frame = self.tot_frames_list[start_idx + i]
            if not label:label_str=str('NORMAL')
            else:label_str=str('ABNORMAL')
            
            ## prints
            #if self.debug:
            print(  f"\n\n### batching X & y {idx}"\
                    f"\n\t {vpath}\n\t{self.mode} {start_idx + i} **** {label_str}"\
                    f"\n\t p_es {start_idx + i} {p_es[i].shape} @{p_es[i].dtype} , y {label}"\
                    f"\n\t 72 Explosion {p_es[i,72]}")

            X_batch[i] = np.expand_dims(np.array(p_es[i]).astype(np.float32), 0)
            y_batch[i] = np.expand_dims(np.array(label).astype(np.float32), 0)

        if self.debug: print(  f"\n\tBATCH @ data_gen after\n\tX {X_batch.shape} @{X_batch.dtype} , y {y_batch} , {y_batch.shape} @{y_batch.dtype}\n\n")
        
        print("\n\n***************** END BATCH ",idx,"********************")
        
        return X_batch , y_batch


    ## FOOD 4 TOUGHT
    ## https://github.com/MTG/essentia/issues/1268
'''    
    As the Essentia algorithms are not thread-safe, you should avoid using multithreading (e.g., ThreadPoolExecutor). 
    Instead, you can use the concurrent.futures.ProcessPoolExecutor to provide process-based parallelization. 
    This will ensure that each function call will be executed in a separate process, avoiding issues with thread safety.

    Here's the updated snippet of your code with ProcessPoolExecutor:
    python

    from concurrent.futures import ProcessPoolExecutor

    # ...

    # Calculate p_es in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(lambda i: self.call_get_sigmoid(i, p_es, self.vpath_list[start_idx:end_idx], sf_list, ef_list, True), range(current_batch_size))

    However, using the ProcessPoolExecutor requires that the functions and their arguments are picklable. 
    One solution is to modify your call_get_sigmoid function so that it returns the computed p_es value for each index i instead of modifying it in-place. 
    You can then assign the results to the p_es array after collecting them from the executor.

    Here's the updated call_get_sigmoid function:
    python

    def call_get_sigmoid(self, i, vpath, start_frame, end_frame, debug):
        print("\t call for ", i, os.path.basename(vpath), start_frame, end_frame)
        return i, self.sinet.get_sigmoid(vpath, start_frame, end_frame, debug=debug)

    And here's the updated parallel calculation of p_es:
    python

    # Calculate p_es in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(lambda i: self.call_get_sigmoid(i, self.vpath_list[start_idx + i], sf_list[i], ef_list[i], True), range(current_batch_size)))

    # Assign the results to p_es
    for i, p_es_i in results:
        p_es[i, :] = p_es_i

    With these changes, your code should now use process-based parallelization, which avoids the thread safety issues with the Essentia algorithms.
    
    '''
   
          
if __name__ == "__main__":
    
    ''' DATA '''
    
    ## dummy GENERATOR
    #dmy = 4
    #train_generator = DataGen(tv_dict['train_fn'][0:dmy], tv_dict['train_labels'][0:dmy] , tv_dict['train_tot_frames'][0:dmy] , 'train')
    #valdt_generator = DataGen(tv_dict['valdt_fn'][0:dmy], tv_dict['valdt_labels'][0:dmy],  tv_dict['valdt_tot_frames'][0:dmy] , 'valdt')


    ## real GERADOR
# change this to only take a single dictionary
    train_generator = DataGenOrigBatch(tv_dict['train_fn'], tv_dict['train_labels'] , tv_dict['train_tot_frames'] , 'train')
    valdt_generator = DataGenOrigBatch(tv_dict['valdt_fn'], tv_dict['valdt_labels'],  tv_dict['valdt_tot_frames'] , 'valdt')
    
    
    ## TF.DATA FROM GENERATOR
    '''
    def data_gen_wrapper(data_gen):
        for i in range(len(data_gen)):
            yield data_gen[i]
        
    output_types = (tf.float32)
    output_shapes = (
        tf.TensorShape((None , CFG_WAV["sinet_aas_len"])),
        tf.TensorShape((None,))
    )
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_gen_wrapper(train_generator),
        output_types=output_types,
        output_shapes=output_shapes
    )
    valdt_dataset = tf.data.Dataset.from_generator(
        lambda: data_gen_wrapper(valdt_generator),
        output_types=output_types,
        output_shapes=output_shapes
    )
    '''
    
    ''' MODEL WAV'''
    
    ## SINGLE
    model,model_name = tfh5.form_model_wav(CFG_WAV)
    
    ## CLBK's
    #ckpt_clbk = tfh5.ckpt_clbk(model_name)
    #early_stop_clbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    tqdm_clbk = TqdmCallback(CFG_WAV["epochs"], len(tv_dict['train_fn']), CFG_WAV["batch_size"])
    clbks = [ tqdm_clbk ]
    
    ''' FIT '''
    history = model.fit(train_generator, 
                        epochs = CFG_WAV["epochs"] ,
                        #steps_per_epoch = len(train_generator),
                        
                        verbose=2,
                        
                        validation_data = valdt_generator,
                        #validation_steps = len(valdt_fp),
                        
                        #use_multiprocessing = True , 
                        #workers = 8 ,
                        callbacks=clbks
                    )


    ''' SAVINGS '''
    #model.save(globo.MODEL_PATH + model_name + '.h5')
    #model.save(globo.MODEL_PATH + model_name )
    #
    #model.save_weights(globo.WEIGHTS_PATH + model_name + '_weights.h5')
    #
    #hist_csv_file = globo.HIST_PATH + model_name + '_history.csv'
    #with open(hist_csv_file, 'w', newline='') as file:writer = csv.writer(file);writer.writerow(history.history.keys());writer.writerows(zip(*history.history.values()))
        