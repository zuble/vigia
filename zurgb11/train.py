# %%
import os, time, random, logging , datetime , cv2 , csv , subprocess
import numpy as np

import tensorflow as tf
print("tf",tf.version.VERSION)
from tensorflow import keras

from utils import globo ,  xdv , tfh5


''' GPU CONFIGURATION '''

tfh5.set_tf_loglevel(logging.ERROR)
tfh5.tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
tfh5.set_memory_growth()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

# %%
''' TRAIN & VALDT '''
#train_fn, train_labels, train_tot_frames, valdt_fn, valdt_labels , valdt_tot_frames = xdv.train_valdt_files(tframes=True)
train_fn, train_labels, valdt_fn, valdt_labels = xdv.train_valdt_files()

update_index_train = range(0, len(train_fn))
update_index_valdt = range(0, len(valdt_fn))

# %%
''' CONFIGS '''

train_config = {
    "frame_step":2, #24 fps -> 12
    
    "in_height":120,
    "in_width":160,
    
    "batch_size":1,
    "augment":True,
    "shuffle":False,
    
    "ativa" : 'leakyrelu',
    "optima" : 'sgd',
    "batch_type" : 0,   # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : 8000,
    "ckpt_start" : f"{0:0>8}",  #used in train_model: if 00000000 start from scratch, else start from ckpt with config stated
    
    "epochs" : 1
}


# %%
class PrintDataCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        # Access the input data shape of the current batch
        input_batch_shape = self.model.get_layer('print_input_shape').output

        # Run the model on a dummy input to get the actual input shape
        input_shape = self.model.predict(np.zeros((1, *input_batch_shape.shape)))

        # Print the input shape
        print("Input shape:\n", input_shape)

# %%
class DataGen(keras.utils.Sequence):
    def __init__(self, vpath_list, label_list, config , valdt=False):
        
        self.valdt = valdt
        self.vpath_list = vpath_list
        self.label_list = label_list
        
        print(len(vpath_list),(len(label_list)))
        
        self.batch_size = config["batch_size"]
        self.frame_max = config["frame_max"]
        
        self.in_height = config["in_height"]
        self.in_width = config["in_width"]
        
        self.augment = config["augment"]
        self.shuffle = config["shuffle"]
        
        self.len_vpath_list = len(self.vpath_list)
        #self.indices = np.arange(self.len_vpath_list)

        self.frame_step = config["frame_step"]
    
    
    def skip_ms(self,cap):
        start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print("skip_start",start_frame)
        while True:
            success = cap.grab()
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if not success or curr_frame - start_frame >= self.frame_step:break
        
        if not success:return success, None, start_frame + self.frame_step

        success, image = cap.retrieve()
        return success, image, curr_frame        
        
        
    def __len__(self):
        if not self.valdt: print("\n\n__len__ = n batchs = ",int(np.ceil(self.len_vpath_list / float(self.batch_size ))) ," w/ '2' vid each")
        else: print("\n\n__len__ = n batchs = ",int(np.ceil(self.len_vpath_list / float(self.batch_size ))) ," w/ 1 vid each")
        return self.len_vpath_list
           
    def __getitem__(self, idx):
        #batch_indices = self.indices[idx * self.batch_size : (idx+1) * self.batch_size]
        print("\n\nbatch_indx",idx)
        
        batch_frames , batch_frames_flip , batch_labels = [] , [] , []
        
        #for i, index in enumerate(batch_indices):
        #vpath = self.vpath_list[index]
        #label = self.label_list[index] 
        vpath = self.vpath_list[idx]
        label = self.label_list[idx]
    
        video = cv2.VideoCapture(vpath)
        tframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("\n*************",vpath , label , tframes)
        
        ## if normal > frame_max picks random frame_max W
        if label == 0 and tframes > self.frame_max :
            start_index = random.randint(0, tframes - self.frame_max)
            end_index = start_index + self.frame_max
            video.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        ## else ingests full video
        else: 
            start_index = 0
            end_index = tframes
        
        print("sstart_index,end_index",start_index , end_index)
        
        frames = []
        curr_frame = 0
        success, frame = video.read()
        for j in range(end_index - start_index):
            
            if not success or curr_frame > end_index: break
            
            frame = cv2.resize(frame, (self.in_width, self.in_height))
            frame_arr = np.array(frame)/255.0
            frames.append(frame_arr)
            
            ## jumps the next frame wo decoding
            success, frame, curr_frame = self.skip_ms(video)
            #print("skip_end",curr_frame)
                
        
        frames_arr = np.array(frames)
        frames_arr_flip = np.flip(frames_arr, axis=2)
        print("frames",frames_arr.shape,frames_arr_flip.shape)

        batch_frames.append(frames_arr)
        batch_frames_flip.append(frames_arr_flip)
        batch_labels.append(label)
        
        XN = np.array(batch_frames).astype(np.float32)
        XF = np.array(batch_frames_flip).astype(np.float32)
        y = np.array(batch_labels).astype(np.float32)
        

        if self.valdt or not self.augment:
            print("valdt")
            print("XN ",XN.dtype,XN.shape )
            print("y",y.shape)
            return XN , y
        elif self.augment:
            print("augment , train")
            X = np.concatenate([XN, XF], axis=0)
            Y = np.concatenate([y, y], axis=0)
            print("XN ",XN.dtype,XN.shape )
            print("XF ",XF.dtype,XF.shape )
            print("X ",X.dtype,X.shape )
            print("Y ",Y.dtype,Y.shape)
            return X , Y

    #def on_epoch_end(self):
    #    if self.shuffle:
    #        np.random.shuffle(self.indexes)
    


# %% [markdown]
# If there are only 8 videos being fed to the training phase, and batch_size is set to 1 with augment enabled, then the generator will yield 16 batches for each epoch of training, as each video will be flipped horizontally to create a second batch. This means that each video will be processed twice per epoch, once in its original orientation and once flipped horizontally.
# 
# After all of the training batches have been processed, the fit method will move onto the validation data, which is processed separately using a different generator (valdt_generator).
# 
# If augment is set to False, then each video will only yield one batch, regardless of the batch_size. So in this case, with a batch_size of 1, the generator would yield 8 batches for training before moving onto the validation data.

# %%
train_generator = DataGen(train_fn, train_labels, train_config)

valdt_generator = DataGen(valdt_fn, valdt_labels, train_config , True)

## len(train_fn) / batch_size = number of video per batch = __len__
## if batch_size 1 , each batch contains a video
## if augmt =True & batch_size 1 , each batch contains "2" videos

model,model_name = tfh5.form_model(train_config)

history = model.fit(train_generator, 
                    epochs = train_config["epochs"] ,
                    steps_per_epoch = len(train_fn),
                    
                    verbose=2,
                    
                    validation_data = valdt_generator ,
                    validation_steps = len(valdt_fn),
                    
                    use_multiprocessing = True , 
                    workers = 32 #,
                    #callbacks=[print_data_callback]
                  )

# Save the history to a CSV file
hist_csv_file = globo.HIST_PATH + model_name + '_history.csv'
with open(hist_csv_file, 'w', newline='') as file:writer = csv.writer(file);writer.writerow(history.history.keys());writer.writerows(zip(*history.history.values()))
    
