import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from scipy.linalg import LinAlgError
from scipy.spatial.distance import cdist
import pandas as pd
from modelsgelu import *
import numpy as np
import optuna
import random
lb = np.array([4]*28+[175]*2+[35]*2)
ub = np.array([0]*28+[125]*2+[1]*2)
Material_lab = pd.read_csv("./ml.csv",header=None)
nvar = len(lb)
pop = np.random.randint(ub,lb+1 ,(100,nvar))
skin_seq = pop[:,:16]
stringer_seq = pop[:,16:28]
other_feas = np.zeros_like(a=0,shape=(100, 14),dtype=np.float32)
skin_seq_length = np.zeros(100)
strigner_seq_length = np.zeros(100)
rou_skin = np.zeros(100)
rou_stringer = np.zeros(100)
for i in range(100): # type: ignore
    skin_seq_ = skin_seq[i,:]
    decode_skin = np.array([x for x in skin_seq_ if x!=0])
    skin_seq_length[i] = len(decode_skin)
    decode_skin = np.pad(decode_skin, (0,16-len(decode_skin)), "constant", constant_values=0)
    skin_seq[i] = decode_skin
    stringer_seq_ = stringer_seq[i,:]
    decode_stringer = np.array([x for x in stringer_seq_ if x!=0])
    strigner_seq_length[i] = len(decode_stringer)
    decode_stringer = np.pad(decode_stringer, (0,12-len(decode_stringer)), "constant", constant_values=0)
    stringer_seq[i] = decode_stringer
    skin_material = pop[:,28:][i,2]
    stringer_material = pop[:,28:][i,3]
    rou_skin[i] = Material_lab.iloc[skin_material-1][6]
    rou_stringer[i] = Material_lab.iloc[stringer_material-1][6]
    thickness = np.array([pop[:,28:][i,0]/1000 ,pop[:,28:][i,1]/1000])
    skin_material = np.array(Material_lab.iloc[skin_material-1][0:6])
    stringer_material = np.array(Material_lab.iloc[stringer_material-1][0:6])
    other_fea = np.concatenate([skin_material,stringer_material,thickness])  
    other_feas[i] = other_fea



def determing_seq(seq):
    pops = seq.shape[0]
    index = np.zeros(pops)
    for i in range(pops):
        seq_ = seq[i]
        decode_seq = np.array([x for x in seq_ if x!=0])
        seq_len = len(decode_seq)
        index[i] = 0
        if sum(decode_seq==2) != sum(decode_seq==4) or sum(decode_seq==1)<0.1*seq_len or sum(decode_seq==2)<0.1*seq_len or sum(decode_seq==3)<0.1*seq_len or sum(decode_seq==4)<0.1*seq_len or decode_seq[-3] == decode_seq[-2] or decode_seq[-3] == decode_seq[-1]:
            index[i] = 1
            break
        for j in range(seq_len-1):
            if abs(decode_seq[j]-decode_seq[j+1]) == 2:
                index[i] = 1
                break
            
        for k in range(seq_len-4):
            if decode_seq[k] == decode_seq[k+1] or decode_seq[k] == decode_seq[k+2] or decode_seq[k] == decode_seq[k+3] or decode_seq[k] == decode_seq[k+4]:
                index[i] = 1
                break
    return index
determing_seq(skin_seq)