import torch
import pandas as pd
from  torch.utils.data import Dataset,DataLoader
import os
import torch.nn as nn
import numpy as np

def get_layup(num,index):
    skin_layerup_list = []
    stringer_layerup_list = []
    dis_feas = np.zeros(num*14).reshape(num,14)
    path = "E://second paper//ququfeatures"
    for i in range(num):
        if i not in index:
            csv_file = os.path.join(path,"stacking_seqs{}.csv".format(i+1))
            dis_file = os.path.join(path,"discreate_fea{}.csv".format(i+1))
            a = pd.read_csv(dis_file, header=None)
            dis_feas[i] = a
            largest_column_count =0
            with open(csv_file, 'r') as temp_f:
                lines = temp_f.readlines()
                for l in lines:
                    column_count = len(l.split(',')) + 1
                    largest_column_count = column_count if largest_column_count < column_count else largest_column_count
            temp_f.close()
            column_names = [i for i in range(0, largest_column_count)]
            data = pd.read_csv(csv_file, header=None, names=column_names)
            skin_layerup = data.iloc[0,:]
            stringer_layerup = data.iloc[1,:]
            skin_layerup = skin_layerup.dropna()
            stringer_layerup = stringer_layerup.dropna()
            skin_layerup = torch.tensor(skin_layerup.values, dtype=torch.float32)
            stringer_layerup = torch.tensor(stringer_layerup, dtype=torch.float32)
            skin_layerup_list.append(skin_layerup)
            stringer_layerup_list.append(stringer_layerup)
        else:
            continue
    dis_feas = torch.tensor(dis_feas, dtype= torch.float32)
    return skin_layerup_list,stringer_layerup_list, dis_feas
class MinimalDataset(Dataset):
    def __init__(self, skin_data, stringer_data, buckleload, otherfeatures):
        self.skin_data = skin_data
        self.stringer_data = stringer_data
        self.otherfeatures = otherfeatures
        self.buckleload = buckleload
    def __getitem__(self, index):
        return [self.skin_data[index], self.stringer_data[index],self.otherfeatures[index], self.buckleload[index]]
    def __len__(self):
        return len(self.skin_data) 
def collate_fn(data):
    skin_data = []
    stringer_data = []
    other_features = []
    buckleload = []
    for i in range(len(data)):
        #将0，45，-45，90 化为index 分别对应 1，2，3，4 再embedding
        data[i][0] = torch.tensor([torch.where(a==0, 1, torch.where(a==45, 2, torch.where(a == -45, 3, 4))) for a in data[i][0]])
        data[i][1] =torch.tensor( [torch.where(a==0, 1, torch.where(a==45, 2, torch.where(a == -45, 3, 4))) for a in data[i][1]])
        skin_data.append(data[i][0])
        stringer_data.append(data[i][1])
        other_features.append(data[i][2])
        buckleload.append(data[i][3])
    skin_data = [sq for sq in skin_data]
    stringer_data = [sq for sq in stringer_data]
    skin_data = torch.nn.utils.rnn.pad_sequence(skin_data, batch_first=True, padding_value=0.0) 
    stringer_data = torch.nn.utils.rnn.pad_sequence(stringer_data, batch_first=True, padding_value=0.0) 
    other_features = torch.stack(other_features)
    buckleload = torch.stack(buckleload)
    return skin_data,  stringer_data,other_features, buckleload
def get_data_iter(DATASET, BATCH_SIZE):
    return DataLoader(DATASET, BATCH_SIZE, collate_fn= collate_fn, shuffle=True)
def guiyi(x,min,max):
    return (x-min)/(max-min)
def get_loader(batch_size):
    buckleload = pd.read_csv("E://second paper//buckling_load//buckling_load.csv", header=None)
    index = buckleload[(buckleload[0]>600)|
                       (buckleload[0]<100)|
                       (buckleload[1]>800)|
                       (buckleload[1]<100)].index
    buckleload.drop(index)
    buckleload = torch.tensor(buckleload.values, dtype=torch.float32)
    skin_layerup_list, striger_layerup_list,otherfeatures = get_layup(3000,index)
    E1_max = 210
    E1_min = 25
    E2_max = 80
    E2_min = 5
    G12_max = 7.5
    G12_min = 2.4
    G13_max = 7.5
    G13_min = 2.4
    G23_max = 6.5
    G23_min = 2.5
    V12_max = 0.37
    V12_min = 0.015
    thickness_max = 0.2
    thickness_min = 0.1
    otherfeatures[0] = guiyi(otherfeatures[0], E1_min, E1_max)
    otherfeatures[1] = guiyi(otherfeatures[1], E2_min, E2_max)
    otherfeatures[2] = guiyi(otherfeatures[2], G12_min, G12_max)
    otherfeatures[3] = guiyi(otherfeatures[3], G13_min, G13_max)
    otherfeatures[4] = guiyi(otherfeatures[4], G23_min, G23_max)
    otherfeatures[5] = guiyi(otherfeatures[5], V12_min, V12_max)
    otherfeatures[6] = guiyi(otherfeatures[6], E1_min, E1_max)
    otherfeatures[7] = guiyi(otherfeatures[7], E2_min, E2_max)
    otherfeatures[8] = guiyi(otherfeatures[8], G12_min, G12_max)
    otherfeatures[9] = guiyi(otherfeatures[9], G13_min, G13_max)
    otherfeatures[10] = guiyi(otherfeatures[10], G23_min, G23_max)
    otherfeatures[11] = guiyi(otherfeatures[11], V12_min, V12_max)
    otherfeatures[12] = guiyi(otherfeatures[12], thickness_min, thickness_max)
    otherfeatures[13] = guiyi(otherfeatures[13], thickness_min, thickness_max)
    dataset = MinimalDataset(skin_layerup_list, striger_layerup_list, buckleload=buckleload, otherfeatures=otherfeatures) 
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainitor = get_data_iter(train_dataset, batch_size)
    testitor = get_data_iter(test_dataset, batch_size)
    return trainitor, testitor