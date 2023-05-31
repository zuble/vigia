import cv2 , glob , os , random , time
import numpy as np

import tensorflow as tf
tf.debugging.set_log_device_placement(False) 

from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input



# author: Patrice Ferlet <patrice.ferlet@smile.fr>
# licence: MIT
class VideoFrameGenerator(keras.utils.Sequence):
    '''
        Video frame generator generates batch of frames from a video directory. Videos should be
        classified in classes directories. E.g:
            videos/class1/file1.avi
            videos/class1/file2.avi
            videos/class2/file3.avi
    '''
    def __init__(self, from_dir, backbone, batch_size=8, shape=(299, 299, 3),
                 shuffle=True, transform:keras.preprocessing.image.ImageDataGenerator=None
                ):
        """
        Create a Video Frame Generator with data augmentation.

        Usage example:
        gen = VideoFrameGenerator('./out/videos/',
            batch_size=5,
            transform=keras.preprocessing.image.ImageDataGenerator(rotation_range=5, horizontal_flip=True))

        Arguments:
        - from_dir: path to the data directory where resides videos,
            videos should be splitted in directories that are name as labels
        - batch_size: number of videos to generate
        - shuffle: boolean, shuffle data at start and after each epoch
        - transform: a keras ImageGenerator configured with random transformations
            to apply on each frame. Each video will be processed with the same
            transformation at one time to not break consistence.
        """

        self.from_dir = from_dir
        self.backbone = backbone
        self.batch_size = batch_size
        self.target_shape = shape
        self.shuffle = shuffle
        self.transform = transform

        if self.backbone == "mobilenetv2":
            self.prep_fx = mobilenet_v2_preprocess_input
        elif self.backbone == "xception":
            self.prep_fx = xception_preprocess_input
        print(self.prep_fx)


        # the list of classes, built in __list_all_files
        self.classes = []
        self.files = []
        self.data = []

        # prepare the list
        self.__filecount = 0
        self.__list_all_files()


    def __len__(self):
        """ Length of the generator
        Warning: it gives the number of loop to do, not the number of files or
        frames. The result is number_of_video/batch_size. You can use it as
        `step_per_epoch` or `validation_step` for `model.fit_generator` parameters.
        """
        return self.__filecount//self.batch_size

    def __getitem__(self, index):
        """ Generator needed method - return a batch of `batch_size` video
        block with `self.nbframe` for each
        """
        indexes = self.data[index*self.batch_size:(index+1)*self.batch_size]
        print("__get_item__ [",index*self.batch_size,":",(index+1)*self.batch_size,"]")
        X, Y = self.__data_aug(indexes)
        return X, Y

    def on_epoch_end(self):
        """ When epoch has finished, random shuffle images in memory """
        if self.shuffle:
            random.shuffle(self.data)

    def __list_all_files(self):
        """ List and inject images in memory """
        self.classes = glob.glob(os.path.join(self.from_dir, '*'))
        self.classes = [os.path.basename(c) for c in self.classes]
        self.__filecount = len(glob.glob(os.path.join(self.from_dir, '*/*')))

        #for classs in self.classes:print(classs)
        
        i = 1
        print("Inject frames in memory, could take a while...")
        for classname in self.classes:
            files = glob.glob(os.path.join(self.from_dir, classname, '*'))
            #for f in files:print(f)
            
            for file in files:
                print('\rProcessing file %d/%d' % (i, self.__filecount), end='')
                i+=1
                #t = time.time()
                self.__openframe(classname, file)
                #tt = time.time()
                #print(str(tt-t))
                #break
            #break

        if self.shuffle:random.shuffle(self.data)
    
    
    def __showfr(self, fr1):
        """ simple frame viewer"""
        for frame in fr1:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(0)
            if key == ord('q'): break  # quit
        cv2.destroyAllWindows()
    
    def __skip_frames(self,cap,frame_skip=3):
        """ skips frames without decoding"""
        start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print("skip_start",start_frame)
        while True:
            success = cap.grab()
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not success or curr_frame - start_frame >= frame_skip:break
        if not success:return success, None, start_frame + frame_skip
        success, image = cap.retrieve()
        return success, image, curr_frame 
    
    
    def __openframe(self, classname, file):
        """Append ORIGNALS frames in memory, transformations are made on the fly"""
        frames = []
        video = cv2.VideoCapture(file)
        tframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame = 0
        success, frame = video.read()
        #for j in range(vid_end_idx - vid_start_idx):
        while True:   
            if not success or curr_frame > tframes: 
                #print(f"Frame read failed, curr_frame: {curr_frame}")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_shape[:2])
            #frame = self.prep_fx(frame)
            frames.append(frame)

            success, frame, curr_frame = self.__skip_frames(video)
            #print(f"Frame read successful, curr_frame: {curr_frame}, success: {success}")
        
        video.release()
        
        frames = np.stack(frames)
        #self.__showfr(frames)
        frames = self.prep_fx(frames)  ## uint8 -> float32
        self.data.append((classname, frames))
        
        print(f"\n__openframe from {tframes} -> {np.shape(frames)},{frames.dtype}")


    def __data_aug(self, batch):
        """ Make random transformation based on ImageGenerator arguments"""
        T = None
        if self.transform: T = self.transform.get_random_transform(self.target_shape[:2])

        X, Y = [], []
        for y, images in batch:
            Y.append(self.classes.index(y)) # label
            x = []
            for img in images:
                if T:x.append(self.transform.apply_transform(img, T))
                else:x.append(img)
            X.append(x)
        return np.array(X), keras.utils.to_categorical(Y, num_classes=len(self.classes))


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
## one at a time
class DataGen(tf.keras.utils.Sequence):
    
    def __init__(self, from_dir, backbone, batch_size=8, shape=(299, 299, 3), nbframe = 32,
                shuffle=True, transform:keras.preprocessing.image.ImageDataGenerator=None
            ):
        
        self.from_dir = from_dir
        self.backbone = backbone
        self.batch_size = batch_size
        self.nbframe = nbframe
        self.target_shape = shape
        self.shuffle = shuffle
        self.transform = transform

        
        if self.backbone == "mobilenetv2":
            self.prep_fx = mobilenet_v2_preprocess_input
        elif self.backbone == "xception":
            self.prep_fx = xception_preprocess_input
        else: raise Exception("no backbone named assim")
        print(self.prep_fx)

        self.list = []
        self.__list_all_files()
        self.total_video_time = 0


    def __len__(self):
        return self.__filecount//self.batch_size
        
        
    def __getitem__(self, idx):
        i0 , i1 = idx*self.batch_size , (idx+1)*self.batch_size
        print("__get_item__ [",idx*self.batch_size,":",(idx+1)*self.batch_size,"]")
        X, Y = self.__open_batch(i0,i1)
        return X, Y  


    def __list_all_files(self):
        self.classes = glob.glob(os.path.join(self.from_dir, '*'))
        self.classes = [os.path.basename(c) for c in self.classes]
        self.__filecount = len(glob.glob(os.path.join(self.from_dir, '*/*')))
        
        for classname in self.classes:
            files = glob.glob(os.path.join(self.from_dir, classname, '*'))
            for file in files:
                self.list.append((classname, file))  
            print(f'{classname} {len(files)} videos')

        print("\n__list_all_files" , np.shape(self.list) , self.__filecount)
        
        if self.shuffle:random.shuffle(self.list)
    
    
    def __skip_frames(self,cap,frame_skip=3):
        """ skips frames without decoding"""
        start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print("skip_start",start_frame)
        while True:
            success = cap.grab()
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not success or curr_frame - start_frame >= frame_skip:break
        if not success:return success, None, start_frame + frame_skip
        success, image = cap.retrieve()
        return success, image, curr_frame       
    
    def __showfr(self, fr1):
        for frame in fr1:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(0)
            if key == ord('q'): break  # quit
        cv2.destroyAllWindows()
        
    def __open_batch(self, i0 , i1):
        data = []
        print(f'__open_batch : {i0} {i1}')
        
        for i in range(i0,i1):
            classname , file = self.list[i]
            frames = []
            video = cv2.VideoCapture(file)
            tframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            curr_frame = 0
            success, frame = video.read()
            #for j in range(vid_end_idx - vid_start_idx):
            while True:   
                if not success or curr_frame > tframes: 
                    #print(f"Frame read failed, curr_frame: {curr_frame}")
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_shape[:2])
                #frame = self.prep_fx(frame)
                frames.append(frame)

                success, frame, curr_frame = self.__skip_frames(video,frame_skip=0)
                #print(f"Frame read successful, curr_frame: {curr_frame}, success: {success}")
            video.release()
            
            frames = np.stack(frames)
            
            self.__showfr(frames)
            step = len(frames)//self.nbframe
            frames = frames[::step]
            if len(frames) >= self.nbframe:
                frames = frames[:self.nbframe]
            self.__showfr(frames)
            
            frames = self.prep_fx(frames)  ## uint8 -> float32
            data.append((classname, frames))
            
            self.total_video_time += int(tframes/fps)
            print(f"\t{i} {tframes} -> {np.shape(frames)},{frames.dtype}")
        
        X, Y = self.__data_aug(data)
        print(f'\nTOTAL OF {self.total_video_time} SECS')
        return X , Y
    
    
    def __data_aug(self, batch):
        """ Make random transformation based on ImageGenerator arguments"""
        T = None
        if self.transform: T = self.transform.get_random_transform(self.target_shape[:2])

        X, Y = [], []
        for y, images in batch:
            Y.append(self.classes.index(y)) # label
            x = []
            for img in images:
                if T:x.append(self.transform.apply_transform(img, T))
                else:x.append(img)
            X.append(x)
        return np.array(X , dtype=object), keras.utils.to_categorical(Y, num_classes=len(self.classes))


if __name__ == "__main__":
    import globo
    
    #v1 = VideoFrameGenerator(globo.UCF101['train_path'] , globo.CFG_TRAIN["backbone"] , 1)
    #x , y = v1.__getitem__(0)
    #print(np.shape(x) , np.shape(y))
    
    
    v2 = DataGen(globo.UCF101['train_path'] , globo.CFG_TRAIN["backbone"] , 4)
    for bi in range(v2.__len__()):
        x , y = v2.__getitem__(bi)
        break