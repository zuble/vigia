import os , cv2 , logging

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import confusion_matrix
from IPython import display
from prettytable import PrettyTable

import tensorflow as tf
from keras import backend as K

import neptune


''' PATH VARS '''
BASE_VIGIA_DIR = "/raid/DATASETS/.zuble/vigia"

SERVER_TRAIN_PATH = '/raid/DATASETS/anomaly/XD_Violence/training/'
SERVER_TEST_PATH = '/raid/DATASETS/anomaly/XD_Violence/testing'
SERVER_TEST_AUD_ORIG_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing/original'
SERVER_TEST_AUD_MONO_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing/mono'

MODEL_PATH = BASE_VIGIA_DIR+'/zu++/model/model/'
CKPT_PATH = BASE_VIGIA_DIR+'/zu++/model/ckpt/'
HIST_PATH = BASE_VIGIA_DIR+'/zu++/model/hist/'
RSLT_PATH = BASE_VIGIA_DIR+'/zu++/model/rslt/'
WEIGHTS_PATH = BASE_VIGIA_DIR+'/zu++/model/weights/'

#--------------------------------------------------------#
def set_2wdpath_var():
    BASE_VIGIA_DIR = "/media/jtstudents/HDD/.zuble/vigia"
    
    #SERVER_TRAIN_PATH = '/home/zhen/Documents/Remote/raid/DATASETS/anomaly/UCF_Crimes/Videos'
    SERVER_TRAIN_PATH = '/media/jtstudents/HDD/.zuble/xdviol/train'
    SERVER_TEST_PATH = '/media/jtstudents/HDD/.zuble/xdviol/test'


    WEIGHTS_PATH = BASE_VIGIA_DIR+"/zhen++/parameters_saved"
    RSLT_PATH = BASE_VIGIA_DIR+'zhen++/parameters_results/original_bt'
    

#--------------------------------------------------------#
# GPU TF CONFIGURATION

#https://www.tensorflow.org/guide/gpu 
#https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth

def set_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("\nAvaiable GPU's",gpus)
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)

    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus));
    for i in range(len(gpus)) :print(str(gpus[i]))



#--------------------------------------------------------#
# FILE ND FOLDERS NAMING

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


def sort_txt_abc(fldr_or_file):

    if os.path.isfile(fldr_or_file) and os.path.split(fldr_or_file)[1] == ".txt":
        cmd = "sort "+fldr_or_file+" -o "+fldr_or_file
        os.system(str(cmd))
        print(cmd)
    else:
        for file in os.listdir(fldr_or_file):
            fname, fext = os.path.splitext(file)
            if fext == ".txt":
                cmd = "sort "+os.path.join(fldr_or_file, file)+" -o "+os.path.join(fldr_or_file, file)
                os.system(str(cmd))
                print(cmd)

      

#--------------------------------------------------------#
# XD VIOLENCE DATASET INFO

def load_xdv_test(aac_path):
    mp4_paths, mp4_labels = [],[]
    for root, dirs, files in os.walk(SERVER_TEST_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths.append(os.path.join(root, file))
    mp4_paths.sort()
    for i in range(len(mp4_paths)):            
        if 'label_A' in  file:mp4_labels.append(0)
        else:mp4_labels.append(1)
    
    print('acc_path',aac_path)
    aac_paths, aac_labels = [],[]                            
    for root, dirs, files in os.walk(aac_path):
        for file in files:
            if file.find('.aac') != -1:
                aac_paths.append(os.path.join(root, file))
    aac_paths.sort()
    for i in range(len(aac_paths)):               
        if 'label_A' in  file:aac_labels.append(0)
        else:aac_labels.append(1)                
    
    return mp4_paths,mp4_labels,aac_paths,aac_labels

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
# RESULTS AND HISTOGRAMA PRINT AND PLOTS

""" 
METRICS/RESULTS CALCULUS
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data 
"""

def plot_cm(name,labels,predictions,threshold=0.5):
    '''
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#download_the_kaggle_credit_card_fraud_data_set
    '''
    predictions = np.array(predictions)
    cm = confusion_matrix(labels, predictions > threshold)
    #plt.clf()
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(threshold))
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

def get_cm_accuracy_precision_recall_f1(labels, predictions,printt=True):
    #plot_cm()

    acc = tf.metrics.BinaryAccuracy()
    acc.update_state(labels, predictions)
    acc_res = acc.result().numpy()
    if printt:print("\tACCURACY  ",acc_res)

    pre = tf.keras.metrics.Precision(thresholds = 0.5)
    pre.update_state(labels, predictions)
    pre_res = pre.result().numpy()
    if printt:print("\tPRECISION (%% of True 1 out of all Positive predicted) ",pre_res)
    
    rec = tf.keras.metrics.Recall(thresholds=0.5)
    rec.update_state(labels, predictions)
    rec_res = rec.result().numpy()
    if printt:print("\tRECALL (%% of True Positive out of all actual anomalies) ",rec_res)
    
    #https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
    auprc_ap = sklearn.metrics.average_precision_score(labels, predictions)
    aucroc = sklearn.metrics.roc_auc_score(labels, predictions)
    if printt:print("\tAUPRC ( AreaUnderPrecisionRecallCurve ) %.4f \n\t AUC-ROC %.4f "% (auprc_ap, aucroc))
    
    #https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
    #import tensorflow_addons as tfa
    #f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5)
    #f1.update_state(labels, predictions)
    f1_res = 2*((pre_res*rec_res)/(pre_res+rec_res+K.epsilon()))
    if printt:print("\tF1_SCORE (harmonic mean of precision and recall) ",f1_res)
    
    return (acc_res,pre_res,rec_res,auprc_ap,aucroc,f1_res)


def buf_count_newlines_gen(fname):
    #https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count

def get_results_from_txt(fldr_or_file):
    
    res_path = [];res_model_fn = [] 
    #FILE
    if os.path.isfile(fldr_or_file):
        fname, fext = os.path.splitext(fldr_or_file)
        res_path.append(os.path.join(fldr_or_file))
        res_model_fn.append(fname)
        table = False
    # FOLDER
    else:
        for file in os.listdir(fldr_or_file):
            if os.path.splitext(file)[1] == ".txt": #and file.find('weights') != -1
                res_path.append(os.path.join(fldr_or_file, file))
        table = True
        res_path = sorted(res_path)
        for path in res_path:res_model_fn.append(os.path.splitext(os.path.basename(path))[0])
    
    # save into matrix all predictions info from all txt files within fx input
    total_txt = len(res_path) ;i=0
    res_list_full = [[() for i in range(total_txt)] for j in range(buf_count_newlines_gen(res_path[i]))]
    res_list_max = [[0.0 for i in range(total_txt)] for j in range(buf_count_newlines_gen(res_path[i]))]
    res_list_fn = [['' for i in range(total_txt)] for j in range(buf_count_newlines_gen(res_path[i]))]
    #print(np.shape(res_list_full))
    
    for txt_i in range(total_txt):
        print('OPENING',res_path[txt_i])
        txt = open(res_path[txt_i],'r')
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
    

    res_list_labels = [[0 for i in range(total_txt)] for j in range(buf_count_newlines_gen(res_path[i]))]
    total_videos = np.shape(res_list_fn)[0]
    for txt_i in range(total_txt):
        for video_j in range(total_videos):
            #list inti all zero so only care to anom/1
            if 'label_A' not in res_list_fn[video_j][txt_i]: 
                res_list_labels[video_j][txt_i] = 1

    # PRINTS
    def strip_res_model_fn(fn):
        tokens = []
        for i in range(len(fn)):
            aux = fn[i].split("_",1)
            tokens.append(str(aux[1]))
        return tokens
    res_model_fn_strap = strip_res_model_fn(res_model_fn)
    
    # text
    for i in range(total_txt):
        print("\nresults for",res_model_fn_strap[i])
        res_col = [col[i] for col in res_list_max]
        labels_col = [col[i] for col in res_list_labels]
        ress = get_cm_accuracy_precision_recall_f1(labels_col,res_col)

    # table only when all rslt fld as input
    if table:
        table = PrettyTable()
        table.add_column("MODEL", res_model_fn_strap)
        metrics = ["Accuracy","Precision","Recall","AUPRC-AP","AUROC","F1-score"]
        for i in range(len(metrics)):
            new_tbl_col = []
            for j in range(total_txt):
                res_col = [col[j] for col in res_list_max]
                labels_col = [col[j] for col in res_list_labels]
                ress = get_cm_accuracy_precision_recall_f1(labels_col,res_col,False)
                new_tbl_col.append("{:.4f}".format(ress[i]))
            table.add_column(metrics[i], new_tbl_col)
        print(table)
    
    return res_list_full,res_list_max,res_list_fn,res_list_labels,res_path,res_model_fn,res_model_fn_strap


#------------------------------------------------------------#
# hist.csv

def count_files(directory):
    return len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

def strip_model_fn(fn):
    tokens = fn.split("_")
    return str(tokens[1]+'_'+tokens[2]+'_'+tokens[3]+'_'+tokens[4])


# save/run not implemented in versus false
def get_histplot_wo_val_from_csv(fldr_or_file,versus=False,save=False,show=True,run=None):
    
    '''
    fldr_or_file: takes into account subfolders
    versus:
        -false
            plots 1 metric per model separatly , 
            except for tp,fp,tn,fn (all4in1plot)
        -true
            plots 1 metric for all model history csv 
    '''
    
    strings=["loss","accuracy","precision","recall","auc","prc","tp","fp","tn","fn"]
    csv_path=[]
    csv_fn=[]

    if not versus:

        if os.path.isfile(fldr_or_file):
            fname, fext = os.path.splitext(fldr_or_file)
            print(fname)
            csv_path.append(fldr_or_file)
            csv_fn.append(fname)
        else:
            for root, dirs, files in os.walk(fldr_or_file):
                for file in files:    
                    fname, fext = os.path.splitext(file)
                    if fext == ".csv":
                        print(fname)
                        csv_path.append(os.path.join(root,file))
                        csv_fn.append(fname)

        #1 PLOT PER METRICS PER MODEL
        for csv in range(len(csv_path)):
            data = pd.read_csv(csv_path[csv]) # read csv file

            #single metrics per plot
            for i in range(0,6):
                plt.plot(data[strings[i]],label=strings[i])
                #plt.plot(data[strings[str]]) # 4validation
                plt.xlabel('epochs');plt.ylabel(strings[i])
                plt.legend();plt.title(csv_fn[csv])
                plt.show()

            # tp , fn ,tn ,fn all in 1 plot
            for j in range(6,10):plt.plot(data[strings[j]]) 
            plt.xlabel('epochs')
            plt.ylabel('videos')
            plt.legend([strings[6],strings[7],strings[8],strings[9]])
            plt.title(csv_fn[csv])
            plt.show()

    else:
        if os.path.isfile(fldr_or_file): 
            raise Exception("must be folder to print the metrics versus per model")
 
        for root, dirs, files in os.walk(fldr_or_file):
            for file in files:    
                fname, fext = os.path.splitext(file)
                if fext == ".csv":
                    print(fname)
                    csv_path.append(os.path.join(root,file))
                    csv_fn.append(fname)
          
        for i in range(0,6):
            for csv in range(len(csv_path)):  
                data = pd.read_csv(csv_path[csv])
                label = strip_model_fn(csv_fn[csv])
                plt.plot(data[strings[i]],label=label)
              
            plt.xlabel('epochs');plt.ylabel(strings[i])
            plt.legend();plt.title(strings[i]+' VS')
            
            if run:run['train/hist_'+strings[i]+' VS'].upload(neptune.types.File.as_image(plt.gcf()))
            if save: plt.savefig(os.path.join(os.path.dirname(csv_path[0]),strings[i]+'.png'))
            if show:plt.show();


# to test versus plots            
def get_histplot_from_csv(fldr_or_file,versus=False,save=False,show=True,run=None):
    
    strings=["loss","accuracy","precision","recall","auc","prc","tp","fp","tn","fn",\
            "val_loss","val_accuracy","val_precision","val_recall","val_auc","val_prc","val_tp","val_fp","val_tn","val_fn"]
    csv_path=[]
    csv_fn=[]
    plt.style.use('bmh')
    
    if not versus:

        if os.path.isfile(fldr_or_file):
            fname, fext = os.path.splitext(fldr_or_file)
            print(fname)
            csv_path.append(os.path.join(fldr_or_file))
            csv_fn.append(fname)
        else:
            for root, dirs, files in os.walk(fldr_or_file):
                for file in files:    
                    fname, fext = os.path.splitext(file)
                    if fext == ".csv":
                        print(fname)
                        csv_path.append(os.path.join(fldr_or_file,file))
                        csv_fn.append(fname)
                break

        #1 PLOT loss+val_loss / acc+val_acc /... PER MODEL
        for csv in range(len(csv_path)):
            data = pd.read_csv(csv_path[csv]) # read csv file

            #single metrics per plot
            for i in range(0,6):
                plt.plot(data[strings[i]],label=strings[i])
                plt.plot(data[strings[i+10]],label=strings[i+10]) # 4validation
                plt.xlabel('epochs');plt.ylabel(strings[i])
                plt.legend();plt.title(csv_fn[csv])
                plt.show()

            # tp,fn,tn,fn + it's val all in 1 plot
            strs=[]
            for j in range(6,10):
                plt.plot(data[strings[j]]);strs.append(strings[j])
                plt.plot(data[strings[j+10]]);strs.append(strings[j+10])
            plt.xlabel('epochs')
            plt.ylabel('videos')
            plt.legend(strs)
            plt.title(csv_fn[csv])
            plt.show()


    # PLOT SAME METRICS TOGETHER FOR ALL MODEL HISTORY .csv
    else:
        if os.path.isfile(fldr_or_file): 
            raise Exception("must be folder to print the metrics versus per model")
        elif count_files(fldr_or_file) == 1:
            raise Exception("must have more than 1 history to print the metrics versus per model")
        
        for root, dirs, files in os.walk(fldr_or_file):
            for file in files:    
                fname, fext = os.path.splitext(file)
                if fext == ".csv":
                    print(fname)
                    csv_path.append(os.path.join(fldr_or_file,file))
                    csv_fn.append(fname)
            break

        #for i in list(range(0,6)) + list(range(10,16)):
        #    for csv in range(len(csv_path)):  
        #        data = pd.read_csv(csv_path[csv])
        #        label = strip_model_fn(csv_fn[csv])
        #        #print(csv_path[csv]+"\n"+label)
        #        plt.plot(data[strings[i]],label=label)
        #      
        #    plt.xlabel('epochs');plt.ylabel(strings[i])
        #    plt.legend();plt.title(strings[i]+' VS')
        #    
        #    if run:run['train/hist_'+strings[i]+' VS'].upload(neptune.types.File.as_image(plt.gcf()))
        #    if save: plt.savefig(os.path.join(os.path.dirname(csv_path[0]),strings[i]+'.png'))
        #    if show:plt.show();


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

#--------------------------------------------------------------#
# def generate_input(data):for the other dataset zhen used

'''
        if 'Abuse' in train_fn[index]:
            yield batch_frames,  np.array([np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0])])
        elif 'Arrest' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0])])
        elif 'Arson' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0])])
        elif 'Assault' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0])]) 
        elif 'Burglary' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0])]) 
        elif 'Explosion' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0])])        
        elif 'Fighting' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0])]) 
        elif 'RoadAccidents' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0])]) 
        elif 'Robbery' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0])]) 
        elif 'Shooting' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0])]) 
        elif 'Shoplifting' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0])]) 
        elif 'Stealing' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0])])
        elif 'Normal' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0])]) 
        elif 'Vandalism' in train_fn[index]:
            yield batch_frames,  np.array([np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1])]) 
        '''