import cv2
import time
def watch_test():
    global is_paused
    for i in range(3):
        base_vigia_dir = "/media/jtstudents/HDD/.zuble/vigia"
        rslt_path = base_vigia_dir+'/zhen++/parameters_results'
                
        batch_type = 1
        batch_no=0
        target_height = 120
        target_width = 160
        frame_max = 4000

        #input
        file_path = '/media/jtstudents/HDD/.zuble/xdviol/test/v=lcBUb7EOQ4o__#1_label_G-0-0.mp4'
        video = cv2.VideoCapture(str(file_path))

        window_name = "anoml vwr"
        cv2.namedWindow(window_name)

        # Video information
        fps = video.get(cv2.CAP_PROP_FPS)
        width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
        font = cv2.FONT_HERSHEY_SIMPLEX


        # We can set up keys to pause, go back and forth.
        # **params can be used to pass parameters to key actions.
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


        # We can setup mouse actions
        #def click_action(event, x, y, flags, param):
        #    if event == cv2.EVENT_LBUTTONDOWN:
        #        for item in data_frames[str(frame_counter)]:
        #                item_id = item['visibleObjectId']
        #                if frame_counter in data_objects[item_id]:
        #                    bbox = data_objects[item_id][frame_counter]['rectangle']
        #                    if bbox[0]['x'] < x < bbox[0]['x']+bbox[0]['w'] \
        #                            and bbox[0]['y'] < y < bbox[0]['y']+bbox[0]['h']:
        #                        print("Info about item:", item)
        #cv2.setMouseCallback(window_name, click_action)

        prev_time = time.time() # Used to track real fps
        frame_counter = 0       # Used to track which frame are we.
        is_quit = False         # Used to signal that quit is called
        is_paused = False       # Used to signal that pause is called

        try:
            while video.isOpened():
                # If the video is paused, we don't continue reading frames.
                if is_quit:
                    # Do something when quiting
                    break
                elif is_paused:
                    # Do something when paused
                    pass
                else:
                    sucess, frame = video.read() # Read the frames

                    if not sucess:
                        break

                    frame_counter = int(video.get(cv2.CAP_PROP_POS_FRAMES))

                    # for current frame, check all visible items
                    #if str(frame_counter) in data_frames:
                    #    for item in data_frames[str(frame_counter)]:
                    #        item_id = item['visibleObjectId']
                    #        color = colormap.get_color(item_id)
                    #    
                    #        # for visible item, get position at current frame and paint rectangle in
                    #        if frame_counter in data_objects[item_id]:
                    #            bbox = data_objects[item_id][frame_counter]['rectangle']
                    #            x1 = bbox[0]['x']
                    #            y1 = bbox[0]['y']
                    #            x2 = x1 + bbox[0]['w']
                    #            y2 = y1 + bbox[0]['h']
                    #            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    #            cv2.putText(frame, str(item_id[:3]), (x1, y1-10), font, 0.5, color, 2)

                    # Display fps and frame count
                    new_time = time.time()
                    cv2.putText(frame, 'FPS: %.2f' % (1/(new_time-prev_time)), (10, 10), font, 0.5, [0,50,200], 2)
                    prev_time = new_time
                    cv2.putText(frame, 'Frame: %d' % (frame_counter), (int(width*2/8), 10), font, 0.5, [60,250,250], 2)
                    cv2.putText(frame, 'Time: %f' % (frame_counter/fps), (int(width*4/8), 10), font, 0.5, [100,250,10], 2)

                # Display the image
                cv2.imshow(window_name,frame)

                # Wait for any key press and pass it to the key action
                frame_time_ms = int(1000/fps)
                key = cv2.waitKey(frame_time_ms)
                key_action(key)
                
        finally:
            video.release()
            cv2.destroyAllWindows()
    
watch_test()