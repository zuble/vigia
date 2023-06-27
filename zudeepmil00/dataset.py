import globo , os
import numpy as np , tensorflow as tf
from sklearn.preprocessing import normalize
from utils import *


class Dataset(tf.keras.utils.Sequence):

    def __init__(self, args , is_normal: bool, is_test: bool, debug: bool = False):
        self.args = args
        print("\nARGS",args)
        
        self.is_normal = is_normal
        self.is_test = is_test
        self.debug = debug

        flists = globo.FEATURE_LISTS[self.args.features]

        if is_test: 
            self.list_file = flists["test"]
            self.segments32 = False
            self.list = list(open(self.list_file))
            self.list = self.list[140:] if is_normal else self.list[:140]
        else: 
            self.list_file = flists["train_normal"] if is_normal else flists["train_abnormal"]
            self.segments32 = True
            self.list = list(open(self.list_file))
            
            
        print("LIST_FILE",self.list_file , np.shape(self.list))
        
        
        if args.dummy: self.list = self.list[:args.dummy]


    def l2norm(self, x): return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)


    def __load_file(self, index):
        fpath = self.list[index]
        base_fn = os.path.splitext(os.path.basename(fpath))[0]

        ## load features from .npy
        if self.args.features == 'i3ddeepmil':
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
             
        elif self.args.features == 'i3drtfm':
            '''
                each .npy in list have all 10crop (t,ncrops,features)
                they are not l2norm , but performed better wo (mil-bert)
            '''
            
            fpath = fpath.strip('\n')
            features = np.load(fpath)
            features = features.transpose(1, 0, 2)  # (10, t, 2048)
            
            if self.debug: print(f'\t{fpath} {np.shape(features)} {features.dtype}')
            
        elif self.args.features == 'c3d' or 'i3d':
            fpath = fpath.strip('\n')
            features = np.load(fpath)
            
            if self.debug: print(f'\t{fpath} {np.shape(features)} {features.dtype}')


        if self.segments32: ## train
            
            if self.args.l2norm: ## raw
                features = self.l2norm(np.asarray(features))
                #features2 = normalize(features, axis=1)
                #print("\tL2NORM same ?",np.allclose(features,features2))
            
            if self.args.features == 'c3d' or 'i3d': ## no crops
                features = segment_feat(np.asarray(features) )
            else:
                features = segment_feat_crop(np.asarray(features) )
        
            if self.args.l2norm == 2: ## segmt 
                features = self.l2norm(features)
        
        
        elif self.args.l2norm: ## test raw
            features = self.l2norm(np.asarray(features))
            

        print(f'Loading {index} {base_fn}  {np.shape(features)}')
        return features


    def __getitem__(self, index): 
        return self.__load_file(index)

    def __len__(self): 
        return len(self.list)



def get_tfslices(train = True):

    def create_tf_dataset(dataset: Dataset, batch_size: int = None):
        tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
        if batch_size: tf_dataset = tf_dataset.batch(batch_size)
        return tf_dataset

    if train:
        normal_dataset = Dataset(globo.ARGS, is_normal=True, is_test=False)
        abnormal_dataset = Dataset(globo.ARGS, is_normal=False, is_test=False)

        normal_tf_dataset = create_tf_dataset(normal_dataset, globo.ARGS.batch_size)
        abnormal_tf_dataset = create_tf_dataset(abnormal_dataset, globo.ARGS.batch_size)

        num_iterations = len(normal_dataset) + len(abnormal_dataset)
        print(f'num_iterations {num_iterations}')

        ## maybe 
        # normal_tf_dataset.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)

        
        return normal_tf_dataset , abnormal_tf_dataset , num_iterations
    
    else: ## test no need to construct generator

        normal_dataset = Dataset(globo.ARGS, is_normal=True, is_test=True)
        abnormal_dataset = Dataset(globo.ARGS, is_normal=False, is_test=True)
    
        num_iterations = len(normal_dataset) + len(abnormal_dataset)
        print(len(normal_dataset), len(abnormal_dataset))
        print(f'num_iterations {num_iterations}')

        return normal_dataset , abnormal_dataset , num_iterations


if __name__ == "__main__":

    normal_tf_dataset , abnormal_tf_dataset , num_iterations = get_tfslices(True)
    #normal_tf_dataset , abnormal_tf_dataset , num_iterations = get_tfslices()

    for normal_in in normal_tf_dataset:
        print(np.shape(normal_in))
        break



'''
def get_tfgen(is_train = True ):
    
    def dataset_generator_wrapper(dataset_instance):
            for idx in range(len(dataset_instance)):
                yield dataset_instance[idx]
                
    if is_train:
        
        normal_dataset = Dataset(globo.ARGS, is_normal=True, is_test=False)
        abnormal_dataset = Dataset(globo.ARGS, is_normal=False, is_test=False)
        
        output_types = tf.float32
        if globo.ARGS.features == 'c3d' : output_shapes = tf.TensorShape([ globo.NSEGMENTS , globo.NFEATURES])
        else: output_shapes = tf.TensorShape([globo.NCROPS , globo.NSEGMENTS , globo.NFEATURES])
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
'''

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