# %%
import streamlit as st
import os

# %%
SERVER_TEST_PATH = '/raid/DATASETS/anomaly/XD_Violence/testing'
SERVER_TEST_AUD_ORIG_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing/original'
SERVER_TEST_AUD_MONO_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing/mono'

# %%
def load_xdv_test(accc_path=SERVER_TEST_AUD_ORIG_PATH):
    mp4_paths, mp4_labels = [],[]
    for root, dirs, files in os.walk(SERVER_TEST_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths.append(os.path.join(root, file))
    mp4_paths.sort()
    for i in range(len(mp4_paths)):            
        if 'label_A' in  file:mp4_labels.append(0)
        else:mp4_labels.append(1)
    

    aac_paths, aac_labels = [],[]                            
    for root, dirs, files in os.walk(accc_path):
        for file in files:
            if file.find('.aac') != -1:
                aac_paths.append(os.path.join(root, file))
    aac_paths.sort()
    for i in range(len(aac_paths)):               
        if 'label_A' in  file:aac_labels.append(0)
        else:aac_labels.append(1)                
    
    return mp4_paths,mp4_labels,aac_paths,aac_labels

#mp4_paths,mp4_labels,aac_paths,aac_labels = load_xdv_test()
#video = True

mp4_paths,mp4_labels,aac_paths,aac_labels = load_xdv_test(accc_path=SERVER_TEST_AUD_MONO_PATH)
video = False

# %% MP4 VIEWER

if video:
    # strip paths to have video name per labels
    mp4_paths_strip, mp4B1fight, mp4B2shoot, mp4B4riot, mp4B5abuse, mp4B6caracc, mp4Gexplosion = [],[],[],[],[],[],[]
    for i in range(len(mp4_paths)):
        mp4_paths_strip.append(os.path.basename(mp4_paths[i]))
        if 'B1' in mp4_paths[i]: mp4B1fight.append(os.path.basename(mp4_paths[i]))
        if 'B2' in mp4_paths[i]: mp4B2shoot.append(os.path.basename(mp4_paths[i]))
        if 'B4' in mp4_paths[i]: mp4B4riot.append(os.path.basename(mp4_paths[i]))
        if 'B5' in mp4_paths[i]: mp4B5abuse.append(os.path.basename(mp4_paths[i]))
        if 'B6' in mp4_paths[i]: mp4B6caracc.append(os.path.basename(mp4_paths[i]))
        if 'G' in mp4_paths[i]: mp4Gexplosion.append(os.path.basename(mp4_paths[i]))
        

    with st.sidebar:

        anom_type = st.selectbox(
            'anom type',
            ('ALL','B1 - FIGHT', 'B2 - SHOOT', 'B4 - RIOT','B5 - ABUSE','B6 - CAR ACIDENTE','G - EXPLSION')
        )
        
        if anom_type == 'ALL':file_selected = st.radio('ALL',mp4_paths_strip)
        if anom_type == 'B1 - FIGHT':file_selected = st.radio('B1 - FIGHT',mp4B1fight)
        if anom_type == 'B2 - SHOOT':file_selected = st.radio('B2 - SHOOT',mp4B2shoot)
        if anom_type == 'B4 - RIOT':file_selected = st.radio('B4 - RIOT',mp4B4riot)
        if anom_type == 'B5 - ABUSE':file_selected = st.radio('B5 - ABUSE',mp4B5abuse)
        if anom_type == 'B6 - CAR ACIDENTE':file_selected = st.radio('B6 - CAR ACIDENTE',mp4B6caracc)
        if anom_type == 'G - EXPLSION':file_selected = st.radio('G - EXPLSION',mp4Gexplosion)

    st.write(file_selected)


    video_file = open(SERVER_TEST_PATH+'/'+file_selected, 'rb')
    video_bytes = video_file.read() #reading the file
    st.video(video_bytes) #displaying the video

else:
    
    with st.sidebar:
        file_selected = st.radio('ALL',aac_paths)
        
    st.write(file_selected)


    aac_file = open(file_selected, 'rb')
    aac_bytes = aac_file.read() #reading the file
    st.audio(aac_bytes, format='audio/aac')