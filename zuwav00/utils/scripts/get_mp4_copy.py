import os , cv2 , numpy as np

SERVER_TRAIN_COPY_PATH = '/raid/DATASETS/anomaly/XD_Violence/training_copy'
SERVER_TEST_COPY_PATH =  '/raid/DATASETS/anomaly/XD_Violence/testing_copy'

def load_test_copy():
    mp4_paths = []
    for root, dirs, files in os.walk(SERVER_TEST_COPY_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths.append(os.path.join(root, file))
    mp4_paths.sort()
    return mp4_paths
def load_train_copy():
    mp4_paths = []
    for root, dirs, files in os.walk(SERVER_TRAIN_COPY_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths.append(os.path.join(root, file))
    return mp4_paths


def get_total_time(path):
        videocv = cv2.VideoCapture(path)
        total_frames = int(videocv.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(videocv.get(cv2.CAP_PROP_FPS))
        total_time = total_frames/fps
        videocv.release()
        return total_time
    
def recreate_mp4_with_right_duration(path, dry_run=True):
    print('\n#--------------------------------------------------------------#')
    print('\nold',path)
    total_time = get_total_time(path)
    print("total_time_old",total_time)
    
    aux_fn = os.path.splitext(os.path.basename(path))[0]+'_1.mp4'
    dir = os.path.dirname(path)
    aux_path = os.path.join(dir,aux_fn)
    print('\naux',aux_path)
    if not dry_run: os.rename(path,aux_path)
    
    command = "ffmpeg -nostats -hide_banner -v warning -i "+str('"'+aux_path+'"')+" -ss 0 -t "+str(total_time)+' '+str('"'+path+'"')
    print('\n',command)
    if not dry_run: os.system(command)
    
    print("\ndel aux")
    if not dry_run:
        os.remove(aux_path)
        total_time = get_total_time(path)
        print("\ntotal_time_new",total_time)

    # shit with the codecs fml
    #videomp = mp.VideoFileClip(path).subclip(0,total_time)
    #videomp.write_videofile(new_path)


mp4_paths_train = load_train_copy()
print(np.shape(mp4_paths_train))

for path in mp4_paths_train:
    recreate_mp4_with_right_duration(path,False)