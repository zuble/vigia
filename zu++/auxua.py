import os , cv2

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix
from IPython import display

import tensorflow as tf
from keras import backend as K

#--------------------------------------------------------#


def rename_files(path,old,new,dry_run=True):
    #path = weights_path
    for root, dirs, files in os.walk(path):
        for fil in files:
            if old in fil:
                new_fil=fil.replace(old,new)
                if dry_run:print("\nold fn",os.path.join(root,fil),"\nnew fn",os.path.join(root,new_fil),"\n")
                else:os.rename(os.path.join(root,fil),os.path.join(root,new_fil))
                    
def rename_filesndfolders(path, old_string, new_string,dry_run=True):
    for root, dirs, files in os.walk(path):
        for name in files + dirs:
            if old_string in name:
                new_name = name.replace(old_string, new_string)
                if dry_run:print("src",os.path.join(root, name),"\ndst",os.path.join(root, new_name),"\n")
                else:os.rename(os.path.join(root, name), os.path.join(root, new_name))
                
def rename_folders(path, old_string, new_string,dry_run=True):
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)) and old_string in folder:
            new_folder = folder.replace(old_string, new_string)
            if dry_run:print("src",os.path.join(path, folder),"\ndst",os.path.join(path, new_folder),"\n")
            else:os.rename(os.path.join(path, folder), os.path.join(path, new_folder))        
   
            
#--------------------------------------------------------#


def get_xdv_info(path,test=False,train=False):
    data = '';aux=0;total=0;line='';txt_fn='';
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.find('.mp4') != -1:
                fname, fext = os.path.splitext(file)
                video = cv2.VideoCapture(os.path.join(root,file))
                frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = video.get(cv2.CAP_PROP_FPS)
                video_time = frames/fps
                video.release()

                aux+=1;total+=frames

                line=str(round(frames))+' frames | '+str(round(video_time))+' secs | '+fname+'\n'
                data+=line
                print(line)

    line = "\nmean of frames per video: "+str(total/aux)
    data+=line
    if test: txt_fn = '/raid/DATASETS/.zuble/vigia/aux/.xdv_info/test.txt'
    if train: txt_fn = '/raid/DATASETS/.zuble/vigia/aux/.xdv_info/train.txt'
    f = open(txt_fn, 'w')        
    f.write(data)
    f.close()  


def get_testxdvanom_info():    
    print('\nOPENING annotations',)
    txt = open('/raid/DATASETS/anomaly/XD_Violence/annotations.txt','r')
    txt_data = txt.read()
    txt.close()

    video_list = [line.split() for line in txt_data.split("\n") if line]
    total_anom_frame_count = 0
    for video_j in range(len(video_list)):
        print(video_list[video_j])
        video_anom_frame_count = 0
        for nota_i in range(len(video_list[video_j])):
            if not nota_i % 2 and nota_i != 0: #i=2,4,6...
                aux2 = int(video_list[video_j][nota_i])
                dif_aux = aux2-int(video_list[video_j][nota_i-1])
                total_anom_frame_count += dif_aux 
                video_anom_frame_count += dif_aux
        print(video_anom_frame_count,'frames | ', "%.2f"%(video_anom_frame_count/24) ,'secs | ', int(video_list[video_j][-1]),'max anom frame\n')
    
    total_secs = total_anom_frame_count/24
    mean_secs = total_secs / len(video_list)
    mean_frames = total_anom_frame_count / len(video_list)
    print("TOTAL OF ", "%.2f"%(total_anom_frame_count),"frames  "\
            "%.2f"%(total_secs), "secs\n"\
            "MEAN OF", "%.2f"%(mean_frames),"frames  "\
            "%.2f"%(mean_secs), "secs per video\n")
#--------------------------------------------------------#


""" 
METRICS/RESULTS CALCULUS
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data 
"""

def plot_cm(name,labels,predictions,p=0.5):
    '''
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#download_the_kaggle_credit_card_fraud_data_set
    '''
    predictions = np.array(predictions)
    cm = confusion_matrix(labels, predictions > p)
    #plt.clf()
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(name+'.png',facecolor='white', transparent=False)
    plt.show()

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, color=colors[0], linestyle='--')
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,80])
    plt.ylim([20,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(name+'.png',facecolor='white', transparent=False)
    plt.show()
    
def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    #plt.clf()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.plot(precision, recall, label=name, linewidth=2, color=colors[0])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim([-0.5,100.5])
    plt.ylim([-0.5,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(name+'.png',facecolor='white', transparent=False)
    plt.show()

def get_precision_recall_f1(labels, predictions):
    p = tf.keras.metrics.Precision(thresholds = 0.5)
    p.update_state(labels, predictions)
    p_res = p.result().numpy()
    print("\tPRECISION (%% of True Positive out of all Positive predicted) ",p_res)
    
    r = tf.keras.metrics.Recall(thresholds=0.5)
    r.update_state(labels, predictions)
    r_res = r.result().numpy()
    print("\tRECALL (%% of True Positive out of all actual anomalies) ",r_res)
    
    #https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
    auprc_ap = sklearn.metrics.average_precision_score(labels, predictions)
    aucroc = sklearn.metrics.roc_auc_score(labels, predictions)
    print("\tAP ( AreaUnderPrecisionRecallCurve ) %.4f \n\t AUC-ROC %.4f "% (auprc_ap, aucroc))
    
    #https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
    #import tensorflow_addons as tfa
    #f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5)
    #f1.update_state(labels, predictions)
    f1_res = 2*((p_res*r_res)/(p_res+r_res+K.epsilon()))
    print("\tF1_SCORE (harmonic mean of precision and recall) ",f1_res)
    
    return p_res,r_res,auprc_ap,aucroc,f1_res

#https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697
def buf_count_newlines_gen(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count

def get_results_from_txt(rslt_path):
    res_txt_fn = []
    res_model_fn = []
    
    for file in os.listdir(rslt_path):
        fname, fext = os.path.splitext(file)
        if fext == ".txt" and file.find('weights') != -1:
            res_txt_fn.append(os.path.join(rslt_path, file))
            res_model_fn.append(fname)
    
    res_txt_fn = sorted(res_txt_fn)
    res_model_fn = sorted(res_model_fn)
    
    i=0
    res_list_full = [[() for i in range(len(res_txt_fn))] for j in range(buf_count_newlines_gen(res_txt_fn[i]))]
    res_list_max = [[0.0 for i in range(len(res_txt_fn))] for j in range(buf_count_newlines_gen(res_txt_fn[i]))]
    res_list_fn = [['' for i in range(len(res_txt_fn))] for j in range(buf_count_newlines_gen(res_txt_fn[i]))]
    #print(np.shape(res_list_full))
    
    n_models = len(res_txt_fn)
    for txt_i in range(n_models):
        print('\nOPENING',res_txt_fn[txt_i])
        txt = open(res_txt_fn[txt_i],'r')
        txt_data = txt.read()
        txt.close()

        video_list = [line.split() for line in txt_data.split("\n") if line]
        #print(video_list)
        for video_j in range(len(video_list)):
            aux_line = str(video_list[video_j]).replace('[','').replace(']','').replace(' ','').split('|')
            #print(aux_line[1])
            res_list_fn[video_j][txt_i] = aux_line[0]
            res_list_max[video_j][txt_i] = float(aux_line[1])
            
            aux2_line = aux_line[2].replace(' ','').replace("'","").replace('(','').replace(')','').split(',')
            #print(aux2_line)
            res_list_full[video_j][txt_i] = aux2_line
    
    
    res_list_labels = [[0 for i in range(len(res_txt_fn))] for j in range(buf_count_newlines_gen(res_txt_fn[i]))]
    n_videos = np.shape(res_list_fn)[0]
    for txt_i in range(n_models):
        for video_j in range(n_videos):
            if 'label_A' not in res_list_fn[video_j][txt_i]:
                res_list_labels[video_j][txt_i] = 1
        
    for i in range(len(res_txt_fn)):
        print("\nresults for",res_model_fn[i])
            
        res_col = [col[i] for col in res_list_max]
        labels_col = [col[i] for col in res_list_labels]
        get_precision_recall_f1(labels_col,res_col)
        
    return res_list_full, res_list_max, res_list_fn, res_list_labels


#------------------------------------------------------------#


""" GELU """
#https://keras.io/guides/distributed_training/
#strategy = tf.distribute.MirroredStrategy()
#print("Number of devices: {}".format(strategy.num_replicas_in_sync))
'''Everything that creates variables should be under the strategy scope.In general this is only model construction & `compile()` '''
#with strategy.scope():
#    model_gelu = form_model(ativa = 'gelu')
#model_gelu = train_model(model_gelu,'_3gelu_xdviolence',weights_path)


''' load model from model_save '''
#weights_fn, weights_path = find_h5(model_path,'_3gelu_xdviolence')
#print(weights_fn,weights_path)

##load model error with activation
#model_gelu = keras.models.load_model(weights_path[0],custom_objects={'gelu': Activation(gelu)})


''' create model arch and loads weights '''
#model_gelu = form_model(ativa = tf.keras.activations.gelu)
#model_gelu = form_model(ativa = "gelu",optima='sgd')

#weights_fn, weights_path = find_h5(weights_path,'_3gelu_sgd_1_xdviolence')
#print(weights_fn,weights_path)

#https://stackoverflow.com/questions/72524486/i-get-this-error-attributeerror-nonetype-object-has-no-attribute-predict
#model_gelu.load_weights(str(weights_path[0]))


#--------------------------------------------------------------#


""" ZHEN .h5 FILES """
#def test_zhen_h5():
#    model = form_model(ativa='relu',optima='sgd')
#    weights_fn, weights_path = find_h5(weights_path,'_2_4_8_xdviolence_model_weights')
#    onev_fn, y_onev_labels,None = test_files(onev = 10)
#    
#    
#
#    for i in range(len(weights_fn)):
#        print("\n\nLoading weights from",weights_fn[i],"\n")
#        model.load_weights(weights_path[i])
#        weight_fn,weight_ext = os.path.splitext(str(weights_fn[i]))
#        
#        y_test_pred, predict_total = test_model(model, onev_fn, rslt_path, model_weight_fn = weight_fn)
#        plot_cm(rslt_path+'/1V/'+weight_fn+'_CM'+str(batch_type),y_onev_labels, y_test_pred)
#        plot_roc(rslt_path+'/1V/'+weight_fn+'_ROC'+str(batch_type), y_onev_labels , y_test_pred)
#        plot_prc(rslt_path+'/1V/'+weight_fn+'_PRC'+str(batch_type), y_onev_labels, y_test_pred)
#        get_precision_recall_f1(y_onev_labels, y_test_pred)
#        #watch_test(predict_total,onev_fn)
#        
#    ''' 
#    =1 last batch has 4000 frames 
#    =2 last batch has no repetead frames 
#    '''
#batch_type = 2
#print("\n\n\tBATCH TYPE",batch_type)

#test_zhen_h5()

#res_list_full,res_list_max,rest_list_fn,res_list_labels = get_results_from_txt()