import os, subprocess
from essentia.standard import *

SERVER_TEST_PATH = '/raid/DATASETS/anomaly/XD_Violence/testing_copy'
SERVER_TEST_AUD_ORIG_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing/original'
SERVER_TEST_AUD_MONO_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing/mono'

#------------------------------------#
def load_xdv_test(aac_path):
    mp4_paths, mp4_labels = [],[]
    for root, dirs, files in os.walk(SERVER_TEST_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths.append(os.path.join(root, file))
    mp4_paths.sort()
    for i in range(len(mp4_paths)):            
        if 'label_A' in mp4_paths[i]:mp4_labels.append(0)
        else:mp4_labels.append(1)
    
    print('acc_path',aac_path)
    aac_paths, aac_labels = [],[]                            
    for root, dirs, files in os.walk(aac_path):
        for file in files:
            if file.find('.aac') != -1:
                aac_paths.append(os.path.join(root, file))
    aac_paths.sort()
    for i in range(len(aac_paths)):               
        if 'label_A' in aac_paths[i] : aac_labels.append(0)
        else:aac_labels.append(1)                
    
    return mp4_paths,mp4_labels,aac_paths,aac_labels

def get_index_per_label_from_filelist(file_list):
    '''retrives video indexs per label and all from file list xdv'''
        
    print("\n  get_index_per_label_from_list\n")
    
    labels_indexs={'A':[],'B1':[],'B2':[],'B4':[],'B5':[],'B6':[],'G':[],'BG':[]}
    
    # to get frist label only add _ to all : if 'B1' 'B2' ...
    for video_j in range(len(file_list)):
        
        label_strap = os.path.splitext(os.path.basename(file_list[video_j]))[0].split('label')[1]
        #print(os.path.basename(file_list[video_j]),label_strap)
        
        if 'A' in label_strap: labels_indexs['A'].append(video_j)
        else:
            labels_indexs['BG'].append(video_j)
            if 'B1' in label_strap : labels_indexs['B1'].append(video_j)
            if 'B2' in label_strap : labels_indexs['B2'].append(video_j)
            if 'B4' in label_strap : labels_indexs['B4'].append(video_j)
            if 'B5' in label_strap : labels_indexs['B5'].append(video_j)
            if 'B6' in label_strap : labels_indexs['B6'].append(video_j)
            if 'G'  in label_strap : labels_indexs['G'].append(video_j)
    
    print(  '\tA NORMAL',               len(labels_indexs['A']),\
            '\n\n\tB1 FIGHT',           len(labels_indexs['B1']),\
            '\n\tB2 SHOOT',             len(labels_indexs['B2']),\
            '\n\tB4 RIOT',              len(labels_indexs['B4']),\
            '\n\tB5 ABUSE',             len(labels_indexs['B5']),\
            '\n\tB6 CARACC',            len(labels_indexs['B6']),\
            '\n\tG EXPLOS',             len(labels_indexs['G']),\
            '\n\n\tBG ALL ANOMALIES',   len(labels_indexs['BG']))
    
    return labels_indexs


#---------------------------------------------#


from pylab import plot, show, figure, imshow
#%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

def plot_mfcc_melbands(audio):
    
    w = Windowing(type = 'hann')
    spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = MFCC()
    
    mfccs = []
    melbands = []
    melbands_log = []

    logNorm = UnaryOperator(type='log')
    
    for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)
        melbands_log.append(logNorm(mfcc_bands))

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    mfccs = essentia.array(mfccs).T
    melbands = essentia.array(melbands).T
    melbands_log = essentia.array(melbands_log).T

    # and plot
    imshow(melbands[:,:], aspect = 'auto', origin='lower', interpolation='none')
    plt.title("Mel band spectral energies in frames")
    show()

    imshow(melbands_log[:,:], aspect = 'auto', origin='lower', interpolation='none')
    plt.title("Log-normalized mel band spectral energies in frames")
    show()

    imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')
    plt.title("MFCCs in frames")
    show()



def print_acodec_from_mp4(data,printt=False):
    out=[]
    for i in range(len(data)):
        output = subprocess.check_output('ffprobe -hide_banner -v error -select_streams a:0 -show_entries stream=codec_name,channels,sample_rate,bit_rate -of default=noprint_wrappers=1 -of compact=p=0:nk=1 '+str('"'+data[i]+'"'), shell=True)
        output = str(output).replace("\\n","").replace("b","").replace("'","").splitlines()[0]
        out.append(output)
        if printt:print(output)
    #os.system('ffprobe -hide_banner -show_streams -select_streams a:0 -i '+str('"'+test_fn[1]+'"'))
    #os.system('ffprobe -hide_banner '+str('"'+test_fn[1]+'"'))
    return out

# convert mp4/multiple audio channels to mp4+mono https://superuser.com/questions/1711628/5-1-to-mono-using-ffmpeg
def conv_mp4_to_aac(mp4_paths,dest_path,channels,dry_run=True):
    print(dest_path)
    for path in mp4_paths:
        out_fn = os.path.join(dest_path,os.path.splitext(os.path.basename(path))[0]+'.aac')
        print("\nconverting ",os.path.basename(path),'\n\t\t\t',os.path.basename(out_fn))
        print("\n\nffmpeg -nostats -hide_banner -v warning -i "+str('"'+path+'"')+" -ac "+str(channels)+' '+str('"'+out_fn+'"'))
        if not dry_run:
            command = "ffmpeg -nostats -hide_banner -v warning -i "+str('"'+path+'"')+" -ac "+str(channels)+' '+str('"'+out_fn+'"')
            os.system(command)


import moviepy.editor as mp
import cv2 , datetime
'''
test_mp4_paths,test_mp4_labels,test_aac_paths,test_aac_labels = load_xdv_test(SERVER_TEST_AUD_ORIG_PATH)
for path in test_mp4_paths:
    recreate_mp4_with_right_duration(path)
'''
def get_total_time(path):
        videocv = cv2.VideoCapture(path)
        total_frames = int(videocv.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(videocv.get(cv2.CAP_PROP_FPS))
        total_time = total_frames/fps
        videocv.release()
        return total_time
    
def recreate_mp4_with_right_duration(path):
    print('\n#--------------------------------------------------------------#')
    print('\nold',path)
    total_time = get_total_time(path)
    print("total_time_old",total_time)
    
    aux_fn = os.path.splitext(os.path.basename(path))[0]+'_1.mp4'
    dir = os.path.dirname(path)
    aux_path = os.path.join(dir,aux_fn)
    print('\naux',aux_path)
    os.rename(path,aux_path)

    command = "ffmpeg -nostats -hide_banner -v warning -i "+str('"'+aux_path+'"')+" -ss 0 -t "+str(total_time)+' '+str('"'+path+'"')
    print('\n',command)
    os.system(command)
    
    print("deleteing aux")
    os.remove(aux_path)

    total_time = get_total_time(path)
    print("\ntotal_time_new",total_time)

    # shit with the codecs fml
    #videomp = mp.VideoFileClip(path).subclip(0,total_time)
    #videomp.write_videofile(new_path)
