import globo , glob , os

'''
    As features are 10-crop , save only the base filename without __n
    them in the data loader for each file in .list , iterate over itself and other 9 parts
'''

def create_i3ddeepmil_lists():
    feat_dict = globo.UCFCRIME_I3DDEEPMIL_FPATHS
    list_dict = globo.UCFCRIME_I3DDEEPMIL_LISTS
    
    for type, dir in feat_dict.items():

        if type == 'train_abnormal' or 'test':
            vpaths = glob.glob(dir + "/**/*.npy" , recursive=True)
            
        elif type == 'train_normal':
            vpaths = glob.glob(dir + "/*.npy" , recursive=True)
    
        vpaths.sort()
        
        print(f'\n****************\n\nProcessing {type} @ {dir}\n\n{len(vpaths)/10} vids\n\nsaving into {list_dict[type]}\n\n')
        
        with open(list_dict[type], 'w+') as f:
            for vpath in vpaths[::10]:
                print(f'{vpath}')
                
                newline = vpath+'\n'
                f.write(newline)

def create_i3drtfm_lists():
    feat_dict = globo.UCFCRIME_I3DRTFM_FPATHS
    list_dict = globo.UCFCRIME_I3DRTFM_LISTS

    for type, dir in feat_dict.items():

        if type == 'train_abnormal' or 'test':
            vpaths = glob.glob(dir + "/**/*.npy" , recursive=True)
            
        elif type == 'train_normal':
            vpaths = glob.glob(dir + "/*.npy" , recursive=True)

        vpaths.sort()

        print(f'\n****************\n\nProcessing {type} @ {dir}\n\n{len(vpaths)} vids\n\nsaving into {list_dict[type]}\n\n')
        
        with open(list_dict[type], 'w+') as f:
            for vpath in vpaths:
                print(f'{vpath}')
                
                newline = vpath+'\n'
                f.write(newline)

def compare_lists():
    
    def paths_list_2_fn(list,what='original'):
        aaa = []
        for line in list:
            p = os.path.basename(line.strip('\n'))
            if what == 'original' : f = os.path.splitext(p)[0].replace("__0","") ## only original
            elif what == 'rtfm' : f = os.path.splitext(p)[0].replace("_i3d","") ## only rtfm
            #print(f'{f}')
            aaa.append(f)
        #print("\n\n")
        return aaa


    list_dict = globo.UCFCRIME_I3DRTFM_LISTS


    ## TEST
    ## original deepmil
    aaa = list(open('list/original/ucf-c3d-test.list'))
    aaa = paths_list_2_fn(aaa)

    zzz = list(open(globo.UCFCRIME_I3DDEEPMIL_LISTS["test"]))
    zzz.sort()
    zzz = paths_list_2_fn(zzz)

    print(len(aaa) , len(zzz))
    print("TEST LISTS MATCH" , set(aaa) == set(zzz) ,"\n") 


    ## TRAIN
    ## orignal deepmil
    bbb = list(open('list/original/ucf-c3d.list'))
    bbb = paths_list_2_fn(bbb)
    print(bbb[809],bbb[810],"\n\n")

    rrr = list(open(list_dict["train_abnormal"]))
    www = list(open(list_dict["train_normal"]))
    ttt = paths_list_2_fn(rrr+www , 'rtfm')

    print(len(bbb) , len(ttt))
    print("abnormal",len(rrr),"normal",len(www),"train")
    print("TRAIN LISTS MATCH" , set(bbb) == set(ttt) ,"\n") 
    


create_i3ddeepmil_lists()
create_i3drtfm_lists()
compare_lists()