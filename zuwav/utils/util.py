import os

SERVER_TEST_PATH = '/raid/DATASETS/anomaly/XD_Violence/testing'
SERVER_TEST_AUD_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing'


def load_xdv_test():
    mp4_fn, mp4_labels = [],[]
    for root, dirs, files in os.walk(SERVER_TEST_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_fn.append(os.path.join(root, file))
                if 'label_A' in  file:mp4_labels.append(0)
                else:mp4_labels.append(1)
                
    aac_fn, aac_labels = [],[]                            
    for root, dirs, files in os.walk(SERVER_TEST_AUD_PATH):
        for file in files:
            if file.find('.aac') != -1:
                aac_fn.append(os.path.join(root, file))
                if 'label_A' in  file:aac_labels.append(0)
                else:aac_labels.append(1)                
    
    return mp4_fn,mp4_labels,aac_fn,aac_labels


def print_acodec_from_mp4(data):
    for i in range(len(data)):
        print(os.system('ffprobe -hide_banner -v error -select_streams a:0 -show_entries stream=codec_name,channels,sample_rate,bit_rate -of default=noprint_wrappers=1 -of compact=p=0:nk=1 '+str('"'+data[i]+'"')))
    #os.system('ffprobe -hide_banner -show_streams -select_streams a:0 -i '+str('"'+test_fn[1]+'"'))
    #os.system('ffprobe -hide_banner '+str('"'+test_fn[1]+'"'))


def conv_mp4_to_aac(data,dry_run=True):
    print(SERVER_TEST_AUD_PATH)
    for i in range(len(data)):
        out_fn = os.path.join(SERVER_TEST_AUD_PATH,os.path.splitext(os.path.basename(data[i]))[0]+'.aac')
        print("\nconverting ",os.path.basename(data[i]),'\n\t\t\t',os.path.basename(out_fn))

        if not dry_run:
            command = "ffmpeg -nostats -hide_banner -v warning -i "+str('"'+data[i]+'"')+" -c:a copy "+str('"'+out_fn+'"')
            os.system(command)
