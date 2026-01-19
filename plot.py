import pandas as pd
import matplotlib.pyplot as plt
from modelsgelu import *
from data import *
import numpy as np
import optuna
import random

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)
seed_everything(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_study = optuna.study.load_study("selus", "sqlite:///optuna.db")
best_params = load_study.best_params
model_dim = best_params["model_dim"]
# batch_size = best_params["batch_size"]
batch_size = 128
lr = best_params["lr"]
skin_hidden_size = best_params["skin_hidden_size"]
stringer_hidden_size = best_params["stringer_hidden_size"]
otherfeatures_hidden_size = best_params["otherfeatures_hidden_size"]
traindataitor, testdataitor = get_loader(batch_size)
net = parallel_net(model_dim,skin_hidden_size,stringer_hidden_size,otherfeatures_hidden_size)
bucking_loads = np.zeros(1)
preds = np.zeros(1)
model_dict = torch.load("./checkpoint5.pt")
net.load_state_dict(model_dict)
net.to(device)
net.eval()
for data in testdataitor:
    skin_data = data[0].to(device)
    stirnger_data = data[1].to(device)
    other_features = data[2].to(device)
    bucking_load = data[3][:,0].squeeze().to(device)
    pred = net(skin_data, stirnger_data, other_features).squeeze()
    bucking_load = bucking_load.to("cpu").squeeze()
    pred = pred.to("cpu")
    bucking_load = bucking_load.detach().numpy()
    pred = pred.detach().numpy()
    if any(np.zeros(1)):
        bucking_loads = bucking_load
        preds = pred
    else:
        bucking_loads = np.concatenate([bucking_loads,bucking_load])
        preds = np.concatenate([preds,pred])

bucking_loads = pd.DataFrame(bucking_loads)
bucking_loads.to_csv("./testdata.csv")
preds = pd.DataFrame(preds)
preds.to_csv("./preds.csv")
