import numpy as np
import cv2
import time

def quit_key_action(**params):
    global is_quit
    is_quit = True
def rewind_key_action(**params):
    global frame_counter
    frame_counter = max(0, int(frame_counter - (fps * 5)))
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
def forward_key_action(**params):
    global frame_counter
    frame_counter = min(int(frame_counter + (fps * 5)), total_frame - 1)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
def pause_key_action(**params):
    global is_paused
    is_paused = not is_paused
# Map keys to buttons
key_action_dict = {
    ord('q'): quit_key_action,
    ord('a'): rewind_key_action,
    ord('d'): forward_key_action,
    ord('s'): pause_key_action,
    ord(' '): pause_key_action
}
def key_action(_key):
    if _key in key_action_dict:
        key_action_dict[_key]()


base_vigia_dir = "/media/jtstudents/HDD/.zuble/vigia"
rslt_path = base_vigia_dir+'/zhen++/parameters_results'
        
batch_type = 1
batch_no=0
target_height = 120
target_width = 160
frame_max = 4000

#input
file_name = '/media/jtstudents/HDD/.zuble/xdviol/test/v=lcBUb7EOQ4o__#1_label_G-0-0.mp4'
video = cv2.VideoCapture(file_name)
total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)
video_time = total_frame / fps

#output
final_path = rslt_path+'/1V/best_batch/test_BB.mp4'
out = cv2.VideoWriter(final_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (target_width,target_height), False)


batch_frames = [] 
while video.isOpened:
    
    success, frame = video.read()
    if success == False:
        break
    
   
    
    #print("image_shape",np.shape(frame)) 
    
    frame = cv2.resize(frame, (target_width, target_height))
    #image_array = np.array(frame)/255.0 #normalize
    
    batch_frames.append(frame)
    
    #data = frame.astype('uint8') * 255
    #data = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #cv2.imshow('frame',frame)
    
    cv2.imshow('frame', frame.astype('uint8') * 255)

    # Wait for any key press and pass it to the key action
    key = cv2.waitKey(int(1000/fps))
    key_action(key)
    
  
#print("image_shape",np.shape(batch_frames)) 
#for i in range(len(batch_frames)):
#    print("frame",i)
#    data = batch_frames[i].astype('uint8') * 255
#    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
#    out.write(data)
#out.release()



prev_time = time.time() # Used to track real fps
frame_counter = 0       # Used to track which frame are we.
is_quit = False         # Used to signal that quit is called
is_paused = False       # Used to signal that pause is called

try:
    while cap.isOpened():
        # If the video is paused, we don't continue reading frames.
        if is_quit:
            # Do something when quiting
            break
        elif is_paused:
            # Do something when paused
            pass
        else:
            ret, frame = cap.read() # Read the frames

            if not ret:
                break

            frame_counter = int(cap.get(cv2.CAP_PROP_POS_FRAMES))