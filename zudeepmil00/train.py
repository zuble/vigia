import numpy as np
import tensorflow as tf

DEBUG = True

def reshape_in(x):
    '''
        as the input format is cat( (bs_normal , ncrops , 32 , feat) , (bs_abnormal , ncrops , 32 , feat) )
        reshape so model processes all arrays of features
    '''
    
    bs, ncrops, ts, feat = x.shape
    if DEBUG: tf.print("\nModelMultiCrop inputs = ", bs, ncrops, ts, feat)

    x = tf.reshape(x, (-1, ts, feat))  # ( bs * ncrops , ts , features)
    if DEBUG: tf.print("inputs reshape = ", x.shape)

    return x, bs, ncrops


def reshape_out(x,bs,ncrops):
    '''
        as the output is a score for each feature array
        reshape so each crop from each batch_normal and abnormal is "exposed"
    '''
    
    if DEBUG: print("scores = ", x.shape) ## (bs * ncrops , 32 ,1)
    
    ## get scores for each crop
    x = tf.reshape(x, (bs, ncrops, -1)) ## ( bs , ncrops , 32)
    if DEBUG: print("scores reshape = ", x.shape)
    
    ##########################################
    ## mean across the ncrops
    x = tf.reduce_mean(x, axis=1) ## (bs , 32)
    if DEBUG: print("scores mean = ", x.shape)
    ##########################################
    
    x = tf.expand_dims(x, axis=2) ## (bs , 32 , 1)
    if DEBUG: print("scores final = ", x.shape , x.dtype) 

    return x


## with generator
def train_gen(model, normal_loader, abnormal_loader, num_iterations, optimizer, loss_obj, ncrops):
    ''' 
        if features are divided in ncrops 
        the input data need to be reshaped into ( bs * ncrops , ts , features) before feed to model 
    '''
    
    losses = []
    for i, (normal_in, abnormal_in) in enumerate(zip(normal_loader, abnormal_loader)):
        
        if i >= num_iterations: break

        data_in = tf.concat((normal_in, abnormal_in), axis=0)
        if ncrops: data_in , bs , ncrops = reshape_in(data_in)
        
        with tf.GradientTape() as tape:
            scores = model(data_in)
            if ncrops: scores = reshape_out(scores, bs , ncrops)
            loss = loss_obj(tf.zeros_like(scores), scores)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        losses.append(loss)

        if i % 10 == 0:
            print(f"\nTRAIN_GEN{i}\nnormal_in abnormal_in : {normal_in.shape}{normal_in.dtype} {abnormal_in.shape}{abnormal_in.dtype}")
            print(f"data_in : {data_in.shape}")
            print("train_step scores",np.shape(scores))
            print(f'Iteration {i}: Loss {loss:.4f}\n')

    return losses



## with tf.dataset
def train_tfdata(model, normal_dataset, abnormal_dataset, num_iterations, optimizer, loss_obj, batch_size):
    
    losses = []
    for i in range(num_iterations):
        
        normal_indices = np.random.choice(len(normal_dataset), size=batch_size, replace=False)
        abnormal_indices = np.random.choice(len(abnormal_dataset), size=batch_size, replace=False)
        
        normal_features = np.stack([normal_dataset[j] for j in normal_indices])
        abnormal_features = np.stack([abnormal_dataset[j] for j in abnormal_indices])
        
        
        normal_in = tf.convert_to_tensor(normal_features, dtype=tf.float32)
        abnormal_in = tf.convert_to_tensor(abnormal_features, dtype=tf.float32)

        data_in = tf.concat((normal_in, abnormal_in), axis=0)
    
        with tf.GradientTape() as tape:
            scores = model(data_in)
            loss = loss_obj(tf.zeros_like(scores), scores)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        losses.append(loss)

        if i % 10 == 0:
            print(f"\n{i}\nnormal_in abnormal_in : {normal_in.shape} {abnormal_in.shape}")
            print(f"data_in : {data_in.shape}")
            print("train_step scores",np.shape(scores))
            print(f'Iteration {i}: Loss {loss:.4f}')

    return losses