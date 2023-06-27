import globo , utils

import cv2 , glob , os , random , time
import numpy as np

import tensorflow as tf
tf.debugging.set_log_device_placement(False) 

from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input


DEBUG = False

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
## heavily based in VideoFrameGenerator@Patrice Ferlet <patrice.ferlet@smile.fr>
class DataGen(tf.keras.utils.Sequence):
    
    def __init__(self, ds:dict, cfg:dict, injext:bool=False, shuffle:bool=True):
        
        self.ds = ds
        
        self.backbone = cfg["backbone"]
        if self.backbone == "mobilenetv2":
            self.preprocess_input = mobilenet_v2_preprocess_input
        elif self.backbone == "xception":
            self.preprocess_input = xception_preprocess_input
        else: raise Exception("no backbone named assim")
        print(self.preprocess_input)
        
        self.in_shape = cfg["in_shape"]
        self.batch_size = cfg["batch_size"]        
        self.frame_max  = cfg["frame_max"]    
        self.frame_step = cfg["frame_step"]  
        
        if cfg["transform"]:
            self.transform = keras.preprocessing.image.ImageDataGenerator.get_random_transform
            #self.transform = keras.preprocessing.image.ImageDataGenerator(rotation_range=5, horizontal_flip=True)
        else: self.transform = None
        print("Transform",self.transform)
        
        self.shuffle = shuffle
        
        self.list = [] ## (classes,vpaths)
        self.__list()
        self.total_video_time = 0
        
        self.injext  = injext
        if self.injext:
            self.data = []
            self.__injext_ds()
            

    def __len__(self):
        return self.__filecount//self.batch_size
        
        
    def __getitem__(self, idx):
        
        i0 , i1 = idx*self.batch_size , (idx+1)*self.batch_size
        if DEBUG: print(f'__get_item__init{idx} [{i0} : {i1}]')
        
        if self.injext: data = self.data[i0,i1]
        else: data = self.__open_batch(i0,i1)
    
        X, Y = self.__transform(data)
        if DEBUG: print(f'__get_item__endit{idx} X {np.shape(X)} , Y {np.shape(Y)}\n\n')
        return X, Y


    def __list(self):
        self.classes = glob.glob(os.path.join(self.ds, '*'))
        self.classes = [os.path.basename(c) for c in self.classes]
        self.__filecount = len(glob.glob(os.path.join(self.ds, '*/*')))
        
        for classname in self.classes:
            files = glob.glob(os.path.join(self.ds, classname, '*'))
            for file in files:
                self.list.append((classname, file))  
            print(f'    {classname} {len(files)} videos')
        print("\n__list" , np.shape(self.list) , self.__filecount)
        
        if globo.ARGS.dummy:
            self.list = self.list[:globo.ARGS.dummy]
            self.__filecount = globo.ARGS.dummy
        
        if self.shuffle:random.shuffle(self.list)

    
    def __skip_frames(self,video):
        """ skips frames without decoding """
        for _ in range(self.frame_step - 1):
            success = video.grab()
            if not success: break
        curr_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        return curr_frame
        
    def __showfr(self, fr1):
        for frame in fr1:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(0)
            if key == ord('q'): break  # quit
        cv2.destroyAllWindows()
    
    
    def __getrawvid(self,vpath,label):
        
        idx = self.list.index((label,vpath))
        print(label,vpath,idx)
        
        
    def __injext_ds(self):
        """ Inject all frames in memory """
        t = time.time()
        for i in range(self.__filecount):
            self.data.append(self.__open_batch(i,i+1))
            print('\rinjexted %d/%d' % (i+1, self.__filecount), end='')
            
        tt = time.time()
        print("\n\nINJECT DONE IN",str(tt-t),"SECS")
        if self.shuffle:random.shuffle(self.data)
     
     
    def __open_batch(self, i0 , i1):
        data = []
        if DEBUG: print(f'   __open_batch : {i0} {i1}')
        
        for i in range(i0,i1):
            classname , file = self.list[i]
            
            video = cv2.VideoCapture(file)
            tframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            
            ## set random starting frame
            max_start_frame = tframes - self.frame_max * self.frame_step
            if max_start_frame > 0:
                start_frame = np.random.randint(0, max_start_frame)
            else : start_frame = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if DEBUG: curr_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            
            frames = []
            if DEBUG: print(f'\t{i} {os.path.basename(file)}@{classname} \n\t  {tframes} frames , {fps}fps')
            for _ in range(self.frame_max):
                success, frame = video.read()
                if not success: break

                #print(f"Frame read successful, curr_frame: {curr_frame}, success: {success}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.in_shape[:2])
                frames.append(frame)

                curr_frame = self.__skip_frames(video)

            video.release()
            frames = np.stack(frames)
            #self.__showfr(frames)    

            frames = self.preprocess_input(frames)  ## uint8 -> float32
            data.append((classname, frames))
            
            #self.total_video_time += int(tframes/fps)
            if DEBUG: print(f"\t  [{start_frame}:{start_frame+self.frame_max}] -> {np.shape(frames)},{frames.dtype}")

        #print(f'\nTOTAL OF {self.total_video_time} SECS')
        return data
    
    def __transform(self, batch):
        """ Make random transformation based on ImageGenerator arguments"""
        
        T = self.transform
        X, Y = [], []
        for y, images in batch:
            Y.append(self.classes.index(y)) # label
            x = []
            for img in images:
                if T:x.append(self.transform.apply_transform(img, T))
                else:x.append(img)
            X.append(x)
        return np.array(X), keras.utils.to_categorical(Y, num_classes=len(self.classes))
   


def create_tf_dataset(what):
    
    if what == 'train':
        cfg = globo.CFG_TRAIN
        data = DataGen(globo.ARGS.ds['train'],cfg)
    elif what == 'test':
        cfg = globo.CFG_TEST
        data = DataGen(globo.ARGS.ds['test'],cfg)    
    
    def generator():
        for idx in range(len(data)):
            X, Y = data[idx]
            yield X, Y

    output_types = (tf.float32, tf.int32)
    output_shapes = (tf.TensorShape([None,cfg["frame_max"],*cfg["in_shape"]]), tf.TensorShape([None, cfg["out_nclasses"]]))

    
    data_tf = tf.data.Dataset.from_generator(generator, output_types=output_types, output_shapes=output_shapes)
    #data_tf = data_tf.batch(cfg["batch_size"])
    
    data_tf = data_tf.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 
    
    print(f'\nCREATED {what} TF DATASET WITH {output_shapes},{output_types}')
    return data_tf


if __name__ == "__main__":
    
    v1 = DataGen(globo.ARGS.ds['train'] , globo.CFG_TRAIN )
    x , y = v1.__getitem__(0)
    print(np.shape(x) , np.shape(y))
    
    #train_tfds = create_tf_dataset('train')                        
    #for batch_index, (X, Y) in enumerate(train_tfds):
    #    print(f"Batch {batch_index}: X shape: {X.shape}, Y shape: {Y.shape}")
    #    break
    
    #test_tfds = create_tf_dataset('test')
    #for batch_index, (X, Y) in enumerate(test_tfds):
    #    print(f"Batch {batch_index}: X shape: {X.shape}, Y shape: {Y.shape}")
    #    break

    #utils.get_ds_info(globo.ARGS.ds)
