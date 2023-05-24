import os , json
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

''' PATH VARS '''
BASE_VIGIA_DIR = "/raid/DATASETS/.zuble/vigia"

## with time stats wrong
SERVER_TRAIN_PATH = '/raid/DATASETS/anomaly/XD_Violence/training/'
SERVER_TEST_PATH = '/raid/DATASETS/anomaly/XD_Violence/testing'

## with time stats right
SERVER_TRAIN_COPY_PATH = '/raid/DATASETS/anomaly/XD_Violence/training_copy'
SERVER_TEST_COPY_PATH =  '/raid/DATASETS/anomaly/XD_Violence/testing_copy'

## alter cut 
SERVER_TRAIN_COPY_ALTER_PATH1 = "/raid/DATASETS/anomaly/XD_Violence/training_copy_alter"
SERVER_TRAIN_COPY_ALTER_PATH2 = SERVER_TRAIN_COPY_ALTER_PATH1 + '/CUT'


MODEL_PATH = BASE_VIGIA_DIR+'/zuwav11/model/model/'
CKPT_PATH = BASE_VIGIA_DIR+'/zuwav11/model/ckpt/'
HIST_PATH = BASE_VIGIA_DIR+'/zuwav11/model/hist/'
RSLT_PATH = BASE_VIGIA_DIR+'/zuwav11/model/rslt/'
WEIGHTS_PATH = BASE_VIGIA_DIR+'/zuwav11/model/weights/'

AAS_PATH = BASE_VIGIA_DIR+'/zuwav11/aas'
FSDSINET_PATH = '/raid/DATASETS/.zuble/vigia/zuwav00/fsd-sinet-essentia/models'
YAMMET_PATH = '/raid/DATASETS/.zuble/vigia/zuwav00/audioset-yamet'

''' CONFIGS '''
# fsd-sinet-vgg42-tlpf_aps-1 : best 
# fsd-sinet-vgg42-tlpf-1
# fsd-sinet-vgg42-aps-1
# fsd-sinet-vgg41-tlpf-1 : lighter 

CFG_SINET = {
    'sinet_version': 'fsd-sinet-vgg42-tlpf_aps-1',
    
    'graph_filename' : os.path.join(FSDSINET_PATH,"fsd-sinet-vgg42-tlpf_aps-1.pb"),
    'metadata_file'  : os.path.join(FSDSINET_PATH,"fsd-sinet-vgg42-tlpf_aps-1.json"),
    'labels' :  json.load(open(os.path.join(FSDSINET_PATH,"fsd-sinet-vgg42-tlpf_aps-1.json"), "r"))["classes"],
    
    'audio_fs_input':22050,
    'batchSize' : 64,
    'lastPatchMode': 'repeat',
    'patchHopSize' : 50,
    
    
    'anom_labels' : ["Alarm","Boom","Crowd","Dog","Drill","Explosion","Fire","Gunshot and gunfire","Hammer","Screaming","Screech",\
                    "Shatter","Shout","Siren","Slam","Squeak","Yell"],
    'anom_labels_i' : [4,18,51,59,65,72,78,92,94,145,146,147,148,152,154,161,198],
    
    'anom_labels2' : ["Boom","Explosion","Fire","Gunshot and gunfire","Screaming",\
                    "Shout","Siren","Yell"],
    'anom_labels_i2' : [18,72,78,92,147,148,152,198],
    
    'chunck_fsize': 10 * 24 , 
    'full_or_max' : 'full', #chose output to (timesteps,labels_total) ot (1,labels_total)
    'labels_total' : 8  # anom_labels2 : 8
}


CFG_YAMMET = {
    'graph_filename' : '/raid/DATASETS/.zuble/vigia/zuwav00/audioset-yamet/audioset-yamnet-1.pb', 
    
    'metadata_file' : '/raid/DATASETS/.zuble/vigia/zuwav00/audioset-yamet/audioset-yamnet-1.json',
    'labels' :  json.load(open('/raid/DATASETS/.zuble/vigia/zuwav00/audioset-yamet/audioset-yamnet-1.json', "r"))["classes"],
    
    'anom_labels_i' : [317,318,319,390,394,420,421,422,423,424,425,426,427,428,429,430],
    
    'audio_fs_input' : 16000,
    'full_or_max' : 'full'
}


CFG_URBNET = {
    'graph_filename' : '/raid/DATASETS/.zuble/vigia/zuwav00/us8k-urbnet/urbansound8k-musicnn-msd-1.pb', 
    
    'metadata_file' : '/raid/DATASETS/.zuble/vigia/zuwav00/us8k-urbnet/urbansound8k-musicnn-msd-1.json',
    'labels' :  json.load(open('/raid/DATASETS/.zuble/vigia/zuwav00/us8k-urbnet/urbansound8k-musicnn-msd-1.json', "r"))["classes"],
    
    'anom_labels_i' : [6],
    
    'audio_fs_input' : 16000,
    'full_or_max' : 'full'
}

'''
both models trained in AudioSet(yammet) and its derivative FSD50k(sinnet)

Explosion :The sound of a rapid increase in volume and release of energy in an extreme manner, usually with the generation of high temperatures and the release of gases. 
    Gunshot, Gunfire : The sound of the discharge of a firearm, or multiple such discharges. 
        Machine gun : Sounds of a fully automatic mounted or portable firearm, designed to fire bullets in quick succession from an ammunition belt or magazine, typically at a rate of 300 to 1800 rounds per minute.     
        Fusillade : The sound of the simultaneous and continuous firing of a group of firearms on command.
        Artillery fire : The sound of large military weapons built to fire munitions far beyond the range and power of infantry's small arms.         
        Cap gun : The sound of a toy gun that creates a loud sound simulating a gunshot and a puff of smoke when a small percussion cap is exploded.
    
    Fireworks : Sounds of a class of low explosive pyrotechnic devices used for aesthetic and entertainment purposes. 
        Firecracker : The sound of a small explosive device primarily designed to produce a large amount of noise, especially in the form of a loud bang. 
    
    Burst, Pop : The sound of an enclosure holding gas or liquid coming open suddenly and violently. 

    Eruption : The sound of a volcano violently ejecting ash and lava, or a similar event in which large volumes of material are explosively released. 

    Boom : A deep prolonged loud noise. 

FSD50K contains: Explosion , Gunshot and gunfire , Fireworks , Boom

+++
UrbanSound8K : air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, street music

'''
##############################################3


CFG_WAV_TRAIN= {
    "arch" : 'topgurlmax', #c1d , lstm , topgurlmax
    
    "lstm_units" : 128,
    
    "sinet_aas_len" : CFG_SINET["labels_total"], 
    "sinet_fi" : CFG_SINET["chunck_fsize"], 
    "sinet_fi_iter" : 1, # (96)-1 | (240)-1,2

    "sigm_norm" : False,
    "mm_norm" : False, #MinMaxNorm
    "anom_filter" : True,
    
    "shuffle" : True, # on epoch end
    
    "ativa" : 'relu', # relu , gelu , leakyrelu
    "optima" : 'nadam', # sgd , adam , adamamsgrad , nadam
    "lr": 1e-2,
    "lr_agenda" : False, #only valid for sgd 

    "epochs" : 200,
    "batch_size" : 8
}


CFG_WAV_TEST= {
    "arch" : 'lstm', #c1d , lstm , topgurlmax
    "sinet_aas_len" : CFG_SINET["labels_total"], 
    "timesteps" : 8,

    "ativa" : 'relu',
    "optima" : 'adam',
    "lr": 1e-4,
    
    #"sigm_norm" : False,
    #"mm_norm" : False, #MinMaxNorm
}