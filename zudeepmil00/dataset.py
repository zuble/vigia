import globo
import numpy as np , os , tensorflow as tf
from utils import *

class Dataset:
    def __init__(   self, is_normal=True , is_test = False , \
                    inject = False , ):
        
        self.features = globo.ARGS.features
        self.is_normal = is_normal
        self.is_test = is_test
        self.inject = inject
        self.debug = globo.ARGS.debug
        dummy = globo.ARGS.dummy
        
        ## defines lists to load features
        if   self.features == 'i3ddeepmil' : flists = globo.UCFCRIME_I3DDEEPMIL_LISTS
        elif self.features == 'i3drtfm': flists = globo.UCFCRIME_I3DRTFM_LISTS
        elif self.features == 'c3d' : flists = globo.UCFCRIME_C3DRAW_LISTS
        
        
        ## test done without 32-segmentation of features
        ## only working with deepmil features atm
        if self.is_test:
            
            self.segments32 = False
            
            if self.is_normal :
                self.list_file = flists["test"]
                self.list = list(open(self.list_file))[140:]
                print("test normal list",self.list_file,np.shape(self.list) , self.list[0])
            else:
                self.list_file = flists["test"]
                self.list = list(open(self.list_file))[:140]
                print("test abnormal list",self.list_file,np.shape(self.list) , self.list[138])
    
        ## train done with 32-segmentation of features
        else :
            self.segments32 = True
                
            if self.is_normal:
                self.list_file = flists["train_normal"]
                self.list = list(open(self.list_file))
                print("normal list",self.list_file,np.shape(self.list))
        
            else:
                self.list_file = flists["train_abnormal"]
                self.list = list(open(self.list_file))
                print("abnormal list",self.list_file,np.shape(self.list))
        
        
        if dummy: 
            self.list = self.list[:dummy]
            print("dummy",np.shape(self.list))

        if self.inject: 
            self.data = []
            self.__load_full() 
    
    
    ## theres tf l2norm , test the two
    def l2norm(self,x):
        return x/np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
         
            
    def __load_file(self,index):
        
        fpath = self.list[index]
        base_fn = os.path.splitext(os.path.basename(fpath))[0]
        

        if self.features == 'i3ddeepmil':
            '''
                each .npy in list is the basename for the 10 crops video features
                divide this to either interpolate each crop separality , all together
                to l2norm before or after interpolate
                and to return multicrop or just one crop 
            '''
            ncrops = 10
            dir = os.path.dirname(fpath)
            base_fn = os.path.splitext(os.path.basename(fpath))[0]
            features = []
            
            for i in range(ncrops):
                crop_fn = f"{base_fn}__{i}.npy" if i > 0 else f"{base_fn}.npy"
                crop_path = os.path.join(dir , crop_fn)
                
                feature_crop10 = np.load(crop_path)  ## (timesteps, 1024)
                if self.debug:
                    print(f'\t{crop_fn} {np.shape(feature_crop10)} {feature_crop10.dtype}')

                ## l2norm ncrop features
                #feature_crop10 = self.l2norm(feature_crop10)

                ## interpolate ncrop
                #feature_crop10 = process_feat(feature_crop10, 32)  ## (32, 1024)
                
                ## l2norm ncrop divided features
                #feature_crop10 = self.l2norm(feature_crop10)
                
                features.append(feature_crop10)
                
        elif self.features == 'i3drtfm':
            '''
                each .npy in list have all 10crop (t,ncrops,features)
                they are not l2norm , but performed better wo (mil-bert)
            '''
            
            fpath = fpath.strip('\n')
            features = np.load(fpath)
            features = features.transpose(1, 0, 2)  # (10, t, 2048)
            
            if self.debug: print(f'\t{fpath} {np.shape(features)} {features.dtype}')
            
            
        if self.segments32: #=train
            
            ## l2norm ncrops features
            #features = self.l2norm(np.asarray(features))
            
            ## interpolate all ncrops features at once
            features = segment_feat_crop(np.asarray(features) ) ## (10 , 32 , 2048)
            
            ## l2norm ncrop divided features
            #features = self.l2norm(features)
        
        print(f'Loading {index} {base_fn}  {np.shape(features)}')
        
        return features


    def __load_full(self):
        for index , fpath in enumerate(self.list):
            video_features = self.__load_file(index)
            self.data.append(video_features)
            
          
    def __getitem__(self, index):
        
        if self.inject:
            features = self.data[index]
            print(f'\n{index} {np.shape(features)} ')
        
        else:
            print(f'\nData __get_item__ {index}')
            features = self.__load_file(index)
            
        return features


    def __len__(self):
        return len(self.list)



def get_tfdataset(is_train = True):
    
    def dataset_generator_wrapper(dataset_instance):
            for idx in range(len(dataset_instance)):
                yield dataset_instance[idx]
                
    if is_train:
        
        normal_dataset = Dataset()
        abnormal_dataset = Dataset(is_normal = False)
        
        output_types = tf.float32
        output_shapes = tf.TensorShape([globo.NCROPS , globo.NSEGMENTS , globo.NFEATURES])
        normal_tf_dataset = tf.data.Dataset.from_generator(
            dataset_generator_wrapper,
            output_types=output_types, output_shapes=output_shapes,
            args=(normal_dataset,))
        abnormal_tf_dataset = tf.data.Dataset.from_generator(
            dataset_generator_wrapper,
            output_types=output_types, output_shapes=output_shapes,
            args=(abnormal_dataset,))
        
        normal_tf_dataset = normal_tf_dataset.batch(globo.ARGS.batch_size)
        abnormal_tf_dataset = abnormal_tf_dataset.batch(globo.ARGS.batch_size)
        
        num_iterations = min(len(normal_dataset), len(abnormal_dataset)) // globo.ARGS.batch_size
        print(len(normal_dataset), len(abnormal_dataset))
        print(f'num_iterations {num_iterations}')        
        
        return normal_tf_dataset , abnormal_tf_dataset , num_iterations
    
    else: ## test no need to construct generator

        normal_dataset = Dataset(is_test=True)
        abnormal_dataset = Dataset(is_test=True , is_normal = False)
    
        num_iterations = len(normal_dataset) + len(abnormal_dataset)
        print(len(normal_dataset), len(abnormal_dataset))
        print(f'num_iterations {num_iterations}')

        return normal_dataset , abnormal_dataset , num_iterations
    
    

## how mil bert return features of RTFM @ MIL-BERT dataset.py
'''
def get_item_UCF_Crime_RTFM(idx): 
    npy_file = self.data_list[idx][:-1]
    npy_file = npy_file.split('/')[-1] 
    npy_file = os.path.join(self.path,'UCF_Train_ten_crop_i3d',npy_file[:-4]+'_i3d.npy')
    #print(npy_file) 

    features = np.load(npy_file)

    if not self.multiCrop: 
        #take the first crop only 
        features = features[:,0:1] 
        features = np.transpose(features,(1,0,2)) 
        if self.L2Norm==2: #L2 norm every feature 
            features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  

        features = process_feat(features, 32)
        if self.L2Norm>0: #L2 norm divided features 
            features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
        features = np.squeeze(features,0) 
    else: 
        features = np.transpose(features,(1,0,2)) #ncrops x t x 2048 
        if self.L2Norm==2: #L2 norm every feature 
            features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
        features = process_feat(features, 32) #ncrops x 32 x 2048 
        if self.L2Norm>0: #L2 norm divided features 
            features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  

    #print(features.shape) 
    return features 
'''

      
if __name__ == "__main__":
    ## train normal
    #dd = Dataset(debug = True)
    #dd0 = dd.__getitem__(2)
    
    ## test abnormal
    #ee = Dataset(features ='i3d_deepmil' , is_normal=False , is_test=True)
    #ee0 = ee.__getitem__(2)
    
    normal_tf_dataset , abnormal_tf_dataset , num_iterations = get_tfdataset()

    for normal_in in normal_tf_dataset:
        print(np.shape(normal_in))
        break