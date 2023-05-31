import globo
import tensorflow as tf
import numpy as np


class ModelMLP(tf.keras.Model):
    def __init__(self , nfeatues):
        super(ModelMLP, self).__init__()
        
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


 
if __name__ == "__main__":

    from loss import *

    bs = 1
    loss_obj = RankingLoss( bs )
    
    zero = np.zeros(( bs , globo.NSEGMENTS , globo.NFEATURES),np.float32)
    one = np.ones(( bs , globo.NSEGMENTS , globo.NFEATURES),np.float32)
    data_in = tf.concat((zero, one), axis=0)

    model = ModelMLP(globo.NFEATURES)
    scores = model(data_in)
    print(scores)
    loss = loss_obj(tf.zeros_like(scores), scores)
    print(loss)

    