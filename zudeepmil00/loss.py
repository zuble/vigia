import globo
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np

class RankingLoss(tf.keras.losses.Loss):
    def __init__(self , batch_size = globo.ARGS.batch_size , lambda1=8e-5, lambda2=8e-5 , lossfx = 'milbert'):
        super().__init__()

        self.batch_size = batch_size
         
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.lossfx = lossfx
        self.nsegments = 32
        
        print(f'\nRankingLoss {self.lossfx}\n')


    def smooth(self, arr , deepmil=False):
        if deepmil:
            arr2 = np.zeros_like(arr)
            arr2[:-1] = arr[1:]
            arr2[-1] = arr[-1]
        else: 
            arr2 = tf.concat([arr[1:], arr[-1:]], axis=0)
            
        loss = tf.reduce_sum(tf.square(arr2 - arr[:-1]))
        return self.lambda1 * loss
    
    def sparsity(self, arr):
        loss = tf.reduce_sum(arr)
        return self.lambda2 * loss


    def ranking(self, scores):
        #bs, ts, *_ = tf.shape(scores) ## (bs, 32)
        #print("\nRANKING_LOSS",bs.numpy(),ts.numpy(),tf.shape(scores))
        
        ## deepmil original
        if self.lossfx == 'deepmil':
            
            batch_size = self.batch_size
            
            scores = tf.reshape(scores, (-1, 1)) ## (bs * 32)
            #print("reshape =",tf.shape(scores))
            
            loss = tf.constant(0.0, dtype=tf.float32)
            for i in range(batch_size):
                startn = i * 32
                endn = (i + 1) * 32
                #print(f'startn {startn} , endn {endn}')
                
                starta = (i * 32 + batch_size * 32)
                enda = (i + 1) * 32 + batch_size * 32
                #print(f'starta {starta} , enda {enda}')
                
                maxn = tf.reduce_max(scores[ startn : endn ])
                maxa = tf.reduce_max(scores[ starta : enda ])
                #print(maxn,maxn)
                
                tmp = tf.nn.relu(1.0 - maxa + maxn)
                loss += tmp
                loss += self.smooth(scores[ starta : enda ] , True)
                loss += self.sparsity(scores[ starta : enda ])
            
            return loss / batch_size
    

        ## mil-bert
        elif self.lossfx == 'milbert':
            
            batch_size = self.batch_size
            
            loss = tf.constant(0.0, dtype=tf.float32)
            loss_intra = tf.constant(0.0, dtype=tf.float32)
            sparsity = tf.constant(0.0, dtype=tf.float32)
            smooth = tf.constant(0.0, dtype=tf.float32)
            
            ## each batch is ( 2 * bs , 32 , 1)
            ## frist bs of 32 scores are normal , second is abnormal
            for i in range(batch_size):
                #print(f'i: {i}, batch_size: {batch_size}, scores.shape: {scores.shape}')

                normal_index = tf.random.shuffle(tf.range(self.nsegments))
                y_normal  = tf.gather(scores[i], normal_index)
                y_normal_max = tf.reduce_max(y_normal)
                y_normal_min = tf.reduce_min(y_normal)
                
                anomaly_index = tf.random.shuffle(tf.range(self.nsegments))
                y_anomaly = tf.gather(scores[i+ batch_size], anomaly_index)
                y_anomaly_max = tf.reduce_max(y_anomaly)
                y_anomaly_min = tf.reduce_min(y_anomaly)
            
                loss += tf.nn.relu(1.0 - y_anomaly_max + y_normal_max)
                sparsity += tf.reduce_sum(y_anomaly) * self.lambda2
                smooth += tf.reduce_sum(tf.square(scores[i + batch_size, :31] - scores[i + batch_size, 1:32])) * self.lambda1
                
            return (loss + sparsity + smooth) / batch_size


        ## espana sultani replica
        elif self.lossfx == 'espana':
        
            scores = K.reshape(scores, [-1]) ## bs * 32
            print('loss scores' , np.shape(scores))
            n_exp = int(self.batch_size / 2)

            max_scores_list = []
            z_scores_list = []
            temporal_constrains_list = []
            sparsity_constrains_list = []
            
            for i in range(0, n_exp, 1):

                video_predictions = scores[ i*32 :(i+1)*32]

                max_scores_list.append(K.max(video_predictions))
                temporal_constrains_list.append(
                    K.sum(K.pow(video_predictions[1:] - video_predictions[:-1], 2))
                )
                sparsity_constrains_list.append(K.sum(video_predictions))

            for j in range(n_exp, 2*n_exp, 1):
                video_predictions = scores[j*32:(j+1)*32]
                max_scores_list.append(K.max(video_predictions))


            max_scores = K.stack(max_scores_list)
            temporal_constrains = K.stack(temporal_constrains_list)
            sparsity_constrains = K.stack(sparsity_constrains_list)
            
            for ii in range(0, n_exp, 1):
                max_z = K.maximum(1 - max_scores[:n_exp] + max_scores[n_exp+ii], 0)
                z_scores_list.append(K.sum(max_z))

            z_scores = K.stack(z_scores_list)
            z = K.mean(z_scores)

            return z + \
                0.00008*K.sum(temporal_constrains) + \
                0.00008*K.sum(sparsity_constrains)
    
    
    def call(self, y_true, y_pred):
        return self.ranking(y_pred)
    
    
if __name__ == "__main__":
    
    ''' LOSS FUNCTIONS VERSUS '''

    BATCH_SIZE = 4
    lambda1 = lambda2 = 8e-5
    nrm = tf.zeros([BATCH_SIZE, 32, 1])
    #abn = tf.ones([BATCH_SIZE, 32, 1])
    abn = tf.random.uniform([BATCH_SIZE, 32, 1] , maxval = 1)
    scores = tf.concat((nrm, abn), axis=0)
    print(scores.shape)


    def smoothfx(arr , deepmil=False):
        if deepmil:
            arr2 = np.zeros_like(arr)
            arr2[:-1] = arr[1:]
            arr2[-1] = arr[-1]
        else: 
            arr2 = tf.concat([arr[1:], arr[-1:]], axis=0)
        
        loss = tf.reduce_sum(tf.square(arr2 - arr))
        return lambda1 * loss


    def sparsityfx( arr):
        loss = tf.reduce_sum(arr)
        return lambda1 * loss


    def loss_deepmil(scores):
        print("\n\nDEEPMIL")
        scores = tf.reshape(scores, (-1, 1))
        #print("reshape =",scores.shape)

        loss1 = tf.constant(0.0, dtype=tf.float32)
        loss2 = tf.constant(0.0, dtype=tf.float32)
        
        l1 = 0.0 ; l2 = 0.0
        for i in range(BATCH_SIZE):
            startn = i * 32
            endn = (i + 1) * 32
            #print(f'startn {startn} , endn {endn}')
            
            starta = (i * 32 + BATCH_SIZE * 32)
            enda = (i + 1) * 32 + BATCH_SIZE * 32
            #print(f'starta {starta} , enda {enda}')
            
            maxn = tf.reduce_max(scores[ startn : endn ])
            maxa = tf.reduce_max(scores[ starta : enda ])
            #print(maxn,maxn)
            
            tmp = tf.nn.relu(1.0 - maxa + maxn)
            loss1 += tmp
            loss1 += sparsityfx(scores[ starta : enda ])
            loss1 += smoothfx(scores[ starta : enda ] , False)
            
            loss2 += tmp
            loss2 += sparsityfx(scores[ starta : enda ])
            loss2 += smoothfx(scores[ starta : enda ] , True)
            
        l1 = (loss1 / BATCH_SIZE).numpy()
        l2 = (loss2 / BATCH_SIZE).numpy()
        print("l1",l1,"l2",l2)
        print("abs dif = ",abs(l1 - l2) , "\nisclose?" ,np.isclose(l1 , l2))
        return l1 , l2


    def loss_milbert(scores):
        print("\nMILBERT")
        loss = tf.constant(0.0, dtype=tf.float32)
        sparsity = tf.constant(0.0, dtype=tf.float32)
        smooth = tf.constant(0.0, dtype=tf.float32)
        l=0.0
        for i in range(BATCH_SIZE):
            #print(i)
            normal_index = tf.random.shuffle(tf.range(32))
            y_normal  = tf.gather(scores[i], normal_index)
            y_normal_max = tf.reduce_max(y_normal)
            y_normal_min = tf.reduce_min(y_normal)
            #print("normal",normal_index ,'\n', y_normal ,'\n', y_normal_max.numpy() ,'\n', y_normal_min.numpy())
            
            #print(str(i + BATCH_SIZE))
            anomaly_index = tf.random.shuffle(tf.range(32))
            y_anomaly = tf.gather(scores[i + BATCH_SIZE], anomaly_index)
            y_anomaly_max = tf.reduce_max(y_anomaly)
            y_anomaly_min = tf.reduce_min(y_anomaly)
            #print("abnormal",anomaly_index ,'\n', y_anomaly ,'\n', y_anomaly_max.numpy() ,'\n', y_anomaly_min.numpy())
            
            ## original milbert
            ## sparsity uses anomaly scores shuffled
            ## smooth uses original anomaly scores
            loss += tf.nn.relu(1.0 - y_anomaly_max + y_normal_max) 
            sparsity += tf.reduce_sum(y_anomaly) * lambda1
            smooth += tf.reduce_sum(tf.square(scores[i + BATCH_SIZE, :31] - scores[i + BATCH_SIZE, 1:32])) * lambda1
            
        l = ((loss + sparsity + smooth ) / BATCH_SIZE).numpy()

        print("l",l)
        return l 
        
        
    l1 , l11 = loss_deepmil(scores)
    l2  = loss_milbert(scores)
    #abs(l1 - l2)
    print('\n', np.isclose(l1 , l11 , l2) )

