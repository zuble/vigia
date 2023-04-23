import cv2 , os
from utils import auxua as aux , tf_formh5 , watch


def video_player(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    wn = os.path.splitext(os.path.basename(video_path))[0]
    cv2.namedWindow(wn) 
    
    # Get the total number of frames and fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_time_ms = int(1000/fps)
    
    # Initialize variables for tracking playback
    paused = False
    frame_number = 0
    jump_size = int(fps / 2)  # Jump size is half a second
    playback_speed = 1  # 1 is normal speed, negative values play backwards

    # Start the video playback loop
    while True:
        # Check if the video has ended
        ret, frame = cap.read()
        if not ret:
            break

        # Display the current frame number and total frames
        cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the current time
        seconds = frame_number / fps
        cv2.putText(frame, f"Time: {seconds:.2f} seconds", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow(wn,frame)

        # Handle key presses
        key = cv2.waitKey(frame_time_ms)
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # pause
            while True:
                key = cv2.waitKey(1)
                if key == ord(' '):break
        elif key == ord('a'):  # Jump backwards
            frame_number -= jump_size
            if frame_number < 0:
                frame_number = 0
        elif key == ord('d'):  # Jump forwards
            frame_number += jump_size
            if frame_number >= total_frames:
                frame_number = total_frames - 1
        elif key == ord('s'):  # Slow down playback
            playback_speed /= 2
        elif key == ord('w'):  # Speed up playback
            playback_speed *= 2

        # Update the frame number
        if not paused:
            frame_number += playback_speed

            # Make sure the frame number is within the bounds of the video
            if frame_number >= total_frames:
                frame_number = total_frames - 1
            elif frame_number < 0:
                frame_number = 0

            # Set the video position to the current frame number
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Release the video capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    
    paths,total_frames = watch.get_vpath_totfra_fromxdvstatstxt('train')
    for i in range(len(total_frames)):
        if int(total_frames[i]) > 4000:
            print(i,paths[i],total_frames[i])
        else:break
        
    #for path in paths:
    #    player = video_player(path)