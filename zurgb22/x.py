from utils import globo , xdv

import cv2
import numpy as np , os 

RESOLUTION = 224
channels = 3
frame_count = 16
features_per_bag = 32


fn, labels, tframes = xdv.load_test_npy()


def view_clips(clips, delay_between_frames=int(1000/24), pause_between_clips=1000):
    for clip_idx, clip in enumerate(clips):
        print(f"Displaying clip {clip_idx + 1}/{len(clips)}")
        for frame in clip:
            cv2.imshow("Clip Frame", frame)
            key = cv2.waitKey(delay_between_frames)
            if key == ord("q"):  # Quit
                break

        #print(f"SPACE to view next clip || Q to quit")
        #while True:
        #    key = cv2.waitKey(pause_between_clips)
        #    if key == ord(" "):  # Next clip
        #        break
        #    if key == ord("q"):  # Quit
        #        return
    cv2.destroyAllWindows()

## how data is feed to extract features of UCF-Crime
## from  tfm-anomaly-detection/proposal/utils/video_util.py
def sliding_window(arr, size, stride):
    num_chunks = int((len(arr) - size) / stride) + 2
    result = []
    print("num_chuncks",num_chunks,len(arr))
    for i in range(0,  num_chunks * stride, stride):
        if len(arr[i:i + size]) > 0:
            result.append(arr[i:i + size])
    return result #np.array(result)    

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (cap.isOpened()):
        sucess, frame = cap.read()
        if not sucess: break
        frame = cv2.resize(frame, (RESOLUTION,RESOLUTION))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames    

def get_video_clips(video_path):
    frames = get_video_frames(video_path)
    clips = sliding_window(frames, frame_count, frame_count)
    return clips, len(frames)


for i in range(len(fn)):
    #if "label_A" not in f:
    clips , frames = get_video_clips(fn[i+100])
    view_clips(clips)
    