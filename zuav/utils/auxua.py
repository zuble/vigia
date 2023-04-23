import os , cv2 

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import confusion_matrix , accuracy_score , f1_score , precision_score , average_precision_score , roc_auc_score , recall_score
from sklearn.model_selection import train_test_split
from IPython import display
from prettytable import PrettyTable

#import tensorflow as tf
from keras import backend as K


from utils import globo


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

    if os.path.isfile(fldr_or_file) and os.path.splitext(fldr_or_file)[1] == ".txt":
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
# RESULTS AND HISTOGRAMA PRINT AND PLOTS

""" 
METRICS/RESULTS CALCULUS
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data 
"""

def plot_cm(name,labels,predictions,threshold=0.5,save=False):
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
    if save: plt.savefig(name+'.png',facecolor='white', transparent=False)
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

def get_cm_accuracy_precision_recall_f1(model_name,labels, predictions,printt=True,plot=False,save_plot=False):
    

    predictions_binary = [1 if p >= 0.5 else 0 for p in predictions]
    
    if plot:
        plot_cm(str(model_name),labels, predictions_binary, save=save_plot)
    
    #acc = tf.metrics.BinaryAccuracy()
    #acc.update_state(labels, predictions)
    #acc_res = acc.result().numpy()
    acc_res = accuracy_score(labels, predictions_binary)
    if printt:print("\tACCURACY  %.4f"% acc_res)

    #pre = tf.keras.metrics.Precision(thresholds = 0.5)
    #pre.update_state(labels, predictions)
    #pre_res = pre.result().numpy()
    pre_res = precision_score(labels, predictions_binary)
    if printt:print("\tPRECISION (%% of True 1 out of all Positive predicted) %.4f"% pre_res)
    
    #rec = tf.keras.metrics.Recall(thresholds=0.5)
    #rec.update_state(labels, predictions)
    #rec_res = rec.result().numpy()
    rec_res = recall_score(labels, predictions_binary)
    if printt:print("\tRECALL (%% of True Positive out of all actual anomalies) ",rec_res)
    
    #https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
    auprc_ap = average_precision_score(labels, predictions)
    aucroc = roc_auc_score(labels, predictions)
    if printt:print("\tAUPRC ( AreaUnderPrecisionRecallCurve ) %.4f \n\t AUC-ROC %.4f "% (auprc_ap, aucroc))
    
    #https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
    #import tensorflow_addons as tfa
    #f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5)
    #f1.update_state(labels, predictions)
    f1_res1 = f1_score(labels, predictions_binary)
    f1_res2 = 2*((pre_res*rec_res)/(pre_res+rec_res+K.epsilon()))
    if printt:print("\tF1_SCORE (harmonic mean of precision and recall)  %.4f %.4f "% (f1_res1,f1_res2))
    
    return (acc_res,pre_res,rec_res,auprc_ap,aucroc,f1_res1)


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

def get_results_from_txt(fldr_or_file,printt=True,plot=False,save_plot=False):
    
    res_path = [];res_model_fn = [] 
    #FILE
    if os.path.isfile(fldr_or_file):
        if printt:print("FILE")
        fname, fext = os.path.splitext(fldr_or_file)
        sort_txt_abc(fldr_or_file)
        res_path.append(os.path.join(fldr_or_file))
        res_model_fn.append(os.path.splitext(os.path.basename(fldr_or_file))[0])
        tablee = False
    # FOLDER
    else:
        if printt:print("FOLDER")
        for file in os.listdir(fldr_or_file):
            if os.path.splitext(file)[1] == ".txt" and os.path.getsize(os.path.join(fldr_or_file, file)) != 0: #and file.find('weights') != -1
                res_path.append(os.path.join(fldr_or_file, file))
                sort_txt_abc(os.path.join(fldr_or_file, file))
        tablee = True
        res_path = sorted(res_path)
        for path in res_path:res_model_fn.append(os.path.splitext(os.path.basename(path))[0])
    #print(res_path,res_model_fn)
    
    
    # save into matrix all predictions info from all txt files within fx input
    total_txt = len(res_path) ;i=0
    res_list_full = [[() for i in range(total_txt)] for j in range(buf_count_newlines_gen(res_path[i]))]
    res_list_max = [[0.0 for i in range(total_txt)] for j in range(buf_count_newlines_gen(res_path[i]))]
    res_list_fn = [['' for i in range(total_txt)] for j in range(buf_count_newlines_gen(res_path[i]))]
    res_list_path = [['' for i in range(total_txt)] for j in range(buf_count_newlines_gen(res_path[i]))]
    #print(np.shape(res_list_full))
    
    for txt_i in range(total_txt):
        #print('OPENING',res_path[txt_i])
        txt = open(res_path[txt_i],'r')
        txt_data = txt.read()
        txt.close()

        video_list = [line.split() for line in txt_data.split("\n") if line]
        #print(video_list)
        for video_j in range(len(video_list)):
            aux_line = str(video_list[video_j]).replace('[','').replace(']','').replace(' ','').split('|')
            #print('aux_line[0]',aux_line[0].replace("'",""))
            #res_list_fn[video_j][txt_i] = aux_line[0]
            res_list_fn[video_j][txt_i] = aux_line[0].replace("'","")
            res_list_path[video_j][txt_i] = globo.SERVER_TEST_COPY_PATH+'/'+aux_line[0].replace("'","")
            #print(globo.SERVER_TEST_COPY_PATH+'/'+aux_line[0].replace("'",""))
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
        if printt:print("\nresults for",res_model_fn_strap[i])
        res_col = [col[i] for col in res_list_max]
        labels_col = [col[i] for col in res_list_labels]
        #for ii in range(len(res_col)):
        #    if(res_col[ii]<0.5)and(labels_col[ii]==1):print(res_col[ii])
        ress = get_cm_accuracy_precision_recall_f1(res_model_fn_strap[i],labels_col,res_col,printt,plot,save_plot)


    # table only when all rslt fld as input
    if tablee:
        table = PrettyTable()
        table.add_column("MODEL", res_model_fn_strap)
        metrics = ["Accuracy","Precision","Recall","AUPRC-AP","AUROC","F1-score"]
        for i in range(len(metrics)):
            new_tbl_col = []
            for j in range(total_txt):
                res_col = [col[j] for col in res_list_max]
                labels_col = [col[j] for col in res_list_labels]
                ress = get_cm_accuracy_precision_recall_f1('',labels_col,res_col,printt=False)
                new_tbl_col.append("{:.4f}".format(ress[i]))
            table.add_column(metrics[i], new_tbl_col)
        if printt:print(table)
    
    return res_list_full,res_list_max,res_list_fn,res_list_path,res_list_labels,res_path,res_model_fn,res_model_fn_strap
    #return res_list

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
            
            #if run:run['train/hist_'+strings[i]+' VS'].upload(neptune.types.File.as_image(plt.gcf()))
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


