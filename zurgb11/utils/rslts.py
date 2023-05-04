import os

import matplotlib.pyplot as plt
import pandas as pd

#------------------------------------------------------------#
# hist.csv

def count_files(directory):
    return len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

def strip_model_fn(fn):
    tokens = fn.split("_")
    return str(tokens[1]+'_'+tokens[2]+'_'+tokens[3]+'_'+tokens[4])

def get_histplot_from_csv(fldr_or_file,versus=False,save=False,show=True):
    
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