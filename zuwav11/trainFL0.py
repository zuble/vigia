import os, time, random, logging , datetime , cv2 , csv , subprocess , json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print("tf",tf.version.VERSION)

#from tensorflow import keras
from tqdm.keras import TqdmCallback

from utils import globo , xdv , tfh5 , sinet


''' GPU CONFIGURATION '''
os.environ["CUDA_VISIBLE_DEVICES"]="1"
tfh5.set_tf_loglevel(logging.ERROR)
tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
#tfh5.set_memory_growth()
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
#tfh5.limit_gpu_gb(2)


''' TRAIN & VALDT '''
TV1_DICT=xdv.train_valdt_test_from_xdvtest_bg_from_npy()


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
    "frame_max" : 8000,
    "ckpt_start" : f"{0:0>8}",  #used in train_model: if 00000000 start from scratch, else start from ckpt with CFG_WAV stated
    
    "epochs" : 1,
    "batch_size" : 1
    
}


''' Data Gerador w/o batch on xdv_test set'''
class DataGenFL0(tf.keras.utils.Sequence):
    def __init__(self, mode = 'train' , dummy = 0 , debug = False):
        
        #self.data = np.load(f"data_{typee}.npz", allow_pickle=True)["data"]
            
        self.mode = mode
        if mode == 'valdt' : 
            self.valdt = True ;  self.train = False
            self.data = TV1_DICT['valdt']
            self.len_data = len(self.data)
        elif mode == 'train': 
            self.train = True ; self.valdt = False
            self.data = TV1_DICT['train']
            self.len_data = len(self.data)
        if dummy:
            self.data = self.data[:dummy]
            self.len_data = len(self.data)
            
        print("\n\nDataGen",mode,self.train,self.valdt)
        print("vpath , label",self.len_vpath_list,(len(self.label_list)))
        
    
        self.batch_size = 1
        self.frame_max = CFG_WAV["frame_max"]
        self.shuffle = CFG_WAV["shuffle"]
        
        self.sinet = sinet.Sinet(CFG_SINET)
        
        self.debug = debug

 
    def __len__(self):
        if self.debug: print("\n\n__len__",self.mode,"= n batchs = ",int(np.ceil(self.len_data / float(self.batch_size))), " w/ ",self.batch_size," vid_frames each")
        return int(np.ceil(self.len_data / float(self.batch_size)))
        
        
    def __getitem__(self, idx):

        sample = self.data[idx]
        vpath = sample['vpath']
        frame_interval = sample['frame_interval']
        p_es_array = sample['p_es_array']
        label = sample['label']
        print(vpath,"\n",frame_interval,np.shape(p_es_array),label)
        '''
        data_dicts[typee].append({
                'vpath': vpath,
                'frame_interval': frame_intervals[k],
                'p_es_array': p_es_array[k],
                'label': frame_intervals[k][2]
        })
        '''
        
        
        X = np.expand_dims(np.array(p_es_array).astype(np.float32),0)
        y = np.expand_dims(np.array(label).astype(np.float32),0)
         
         
        ## prints
        #if self.debug:print(f"\n********** {self.mode}_{idx} **** {label_str} ***************\n")
        #print( f"\n\n\n£££ {self.mode}_{self.sinet.model_config['full_or_max']}_{idx} * {label_str} @ {os.path.basename(vpath)}\n"
        #            f"    X {X.shape} @{X.dtype} , y {y} , {y.shape} @{y.dtype}\n\n")
        
        return X , y

   
          
if __name__ == "__main__":
    
    ''' DATA '''
    
    ## dummy GENERATOR
    train_generator = DataGenOrig('train' , 4 , True)
    valdt_generator = DataGenOrig('valdt' , 4 , True)
    
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
    #tqdm_clbk = TqdmCallback(CFG_WAV["epochs"], len(train_fp), CFG_WAV["batch_size"])
    
    
    ''' FIT '''
    history = model.fit(train_generator, 
                        epochs = CFG_WAV["epochs"] ,
                        steps_per_epoch = len(train_generator),
                        
                        verbose=2,
                        
                        #validation_data = valdt_generator,
                        #validation_steps = len(valdt_fp),
                        
                        #use_multiprocessing = True , 
                        #workers = 8 #,
                        #callbacks=[ckpt_clbk , early_stop_clbk , tqdm_clbk ]
                    )


    ''' SAVINGS '''
    #model.save(globo.MODEL_PATH + model_name + '.h5')
    #model.save(globo.MODEL_PATH + model_name )
    #
    #model.save_weights(globo.WEIGHTS_PATH + model_name + '_weights.h5')
    #
    #hist_csv_file = globo.HIST_PATH + model_name + '_history.csv'
    #with open(hist_csv_file, 'w', newline='') as file:writer = csv.writer(file);writer.writerow(history.history.keys());writer.writerows(zip(*history.history.values()))
        