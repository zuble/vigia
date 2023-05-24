import globo
import tensorflow as tf
import numpy as np

class Model1Crop(tf.keras.Model):
    def __init__(self):
        super(Model1Crop, self).__init__()

        self.nfeatures = globo.NFEATURES

        ##in bert-rtfm fc2 with no actiavation perfomed better

        self.fc1=tf.keras.layers.Dense( 512, activation='relu', input_shape=(self.nfeatures,),
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                        bias_initializer='zeros')
        
        self.fc2=tf.keras.layers.Dense( 32, activation=None,
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                        bias_initializer='zeros')
        
        self.fc3=tf.keras.layers.Dense( 1, activation='sigmoid',
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                        bias_initializer='zeros')
        
        self.dropout=tf.keras.layers.Dropout(0.6)
        
    def call(self, inputs):
        
        debug = True
        x0 = inputs
        
        bs, ts, feat = tf.shape(x0)
        if debug: print("\nModel1Crop inputs = ",bs.numpy(),ts.numpy(),feat.numpy())
        
        x = self.fc1(x0)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        if debug: print("scores final = ", x.shape)
    
        return x


class ModelMultiCrop(tf.keras.Model):
    def __init__(self , nfeatues):
        super(ModelMultiCrop, self).__init__()
        
        self.nfeatures = nfeatues

        ##in bert-rtfm fc2 with no actiavation perfomed better

        self.fc1=tf.keras.layers.Dense( 512, activation='relu', input_shape=(self.nfeatures,),
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                        bias_initializer='zeros')
        
        self.fc2=tf.keras.layers.Dense( 32, activation=None,
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                        bias_initializer='zeros')
        
        self.fc3=tf.keras.layers.Dense( 1, activation='sigmoid',
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                        bias_initializer='zeros')
        
        self.dropout=tf.keras.layers.Dropout(0.6)
        
        '''
        # Initialize weights using the Xavier uniform initialization and biases to zero
        self.fc1.build(input_shape=(None, n_features))
        self.fc1.set_weights([tf.keras.initializers.GlorotUniform()(self.fc1.get_weights()[0].shape), tf.zeros_like(self.fc1.get_weights()[1])])
        self.fc2.build(input_shape=(None, 512))
        self.fc2.set_weights([tf.keras.initializers.GlorotUniform()(self.fc2.get_weights()[0].shape), tf.zeros_like(self.fc2.get_weights()[1])])
        self.fc3.build(input_shape=(None, 32))
        self.fc3.set_weights([tf.keras.initializers.GlorotUniform()(self.fc3.get_weights()[0].shape), tf.zeros_like(self.fc3.get_weights()[1])])'''


    def call(self, inputs):
        
        x = self.fc1(inputs)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


'''  
if __name__ == "__main__":

    from loss import *

    bs = 4 
    loss_obj = RankingLoss( bs )
    
    data_in0 = np.ones(( bs*2 , globo.NSEGMENTS , globo.NFEATURES),np.float32)
    model0 = Model1Crop()
    scores0 = model0(data_in0)
    loss0 = loss_obj(tf.zeros_like(scores0), scores0)
    
    data_in1 = np.ones(( bs*2 , globo.NCROPS , globo.NSEGMENTS , globo.NFEATURES),np.float32)
    model1 = ModelMultiCrop()
    scores1 = model1(data_in1)
    loss1 = loss_obj(tf.zeros_like(scores1), scores1)
'''
    