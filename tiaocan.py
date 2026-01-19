import torch
import torch.nn as nn
from modelsgelu import *
import optuna
from data import *
import datetime
import os,sys,time
from tqdm import tqdm
from copy import deepcopy
from torchmetrics.regression.mape import MeanAbsolutePercentageError 
import matplotlib.pyplot as plt
import numpy as np
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

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(str(info) + "\n")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial):
    model_dim = trial.suggest_categorical("model_dim", [2048, 4096])
    stringer_hidden_size = trial.suggest_categorical("stringer_hidden_size",[1024, 2048])
    skin_hidden_size = trial.suggest_categorical("skin_hidden_size",[1024, 2048])
    otherfeatures_hidden_size = trial.suggest_categorical("otherfeatures_hidden_size",[2048, 4096])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log = True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6,1e-4,log=True)
    traindataitor, testdataitor = get_loader(128)
    net = parallel_net(model_dim,skin_hidden_size, stringer_hidden_size,otherfeatures_hidden_size)
    critizer = nn.MSELoss().to(device)
    number = trial.number
    metrics_dict = {"mape": MeanAbsolutePercentageError().to(device)}
    ckpt_path = 'checkpoint{}.pt'.format(number)# monitor = "val_r2"
    monitor = "val_loss"
    patience = 200
    mode = "min"
    history = {}
    optimizer = torch.optim.AdamW(net.parameters(),lr=lr , weight_decay=weight_decay)
    epoches = 5000
    for epoch in range(1, epoches + 1):
        printlog("Epoch {0} / {1}".format(epoch, epoches))
        net.to(device)

        # 1，train -------------------------------------------------
        net.train()
        total_loss, step = 0, 0
        loop = tqdm(enumerate(traindataitor), total=len(traindataitor))
        train_metrics_dict = deepcopy(metrics_dict)

        for i, batch in loop:
            skin_data = batch[0].to(device)
            stirnger_data = batch[1].to(device)
            other_features = batch[2].to(device)
            # bucking_load = batch[3].to(device)
            bucking_load = batch[3][:,0].squeeze().to(device)
            preds = net(skin_data, stirnger_data, other_features).squeeze()
            loss = critizer(preds, bucking_load).to(device)
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # metrics
            step_metrics = {"train_" + name: metric_fn(preds, bucking_load).item()
                            for name, metric_fn in train_metrics_dict.items()}

            step_log = dict({"train_loss": loss.item()}, **step_metrics)

            total_loss += loss.item()

            step += 1
            if i != len(traindataitor) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {"train_" + name: metric_fn.compute().item()
                                for name, metric_fn in train_metrics_dict.items()}
                epoch_log = dict({"train_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in train_metrics_dict.items():
                    metric_fn.reset()

        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        net.eval()

        total_loss, step = 0, 0
        loop = tqdm(enumerate(testdataitor), total=len(testdataitor))

        val_metrics_dict = deepcopy(metrics_dict)

        with torch.no_grad():
            for i, batch in loop:


                # forward
                skin_data = batch[0].to(device)
                stirnger_data = batch[1].to(device)
                other_features = batch[2].to(device)
                # bucking_load = batch[3].to(device)
                bucking_load = batch[3][:,0].squeeze().to(device)
                preds = net(skin_data, stirnger_data, other_features).squeeze()
                loss = critizer(preds, bucking_load)

                # metrics
                step_metrics = {"val_" + name: metric_fn(preds, bucking_load).item()
                                for name, metric_fn in val_metrics_dict.items()}

                step_log = dict({"val_loss": loss.item()}, **step_metrics)

                total_loss += loss.item()
                step += 1
                if i != len(testdataitor) - 1:
                    loop.set_postfix(**step_log)
                else:
                    epoch_loss = (total_loss / step)
                    epoch_metrics = {"val_" + name: metric_fn.compute().item()
                                    for name, metric_fn in val_metrics_dict.items()}
                    epoch_log = dict({"val_loss": epoch_loss}, **epoch_metrics)
                    loop.set_postfix(**epoch_log)

                    for name, metric_fn in val_metrics_dict.items():
                        metric_fn.reset()

        epoch_log["epoch"] = epoch
        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(net.state_dict(), ckpt_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                                                            arr_scores[best_score_idx]), file=sys.stderr)
        if len(arr_scores) - best_score_idx > patience:
            print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor, patience), file=sys.stderr)
            break
        net.load_state_dict(torch.load(ckpt_path))
        trial.report(history[monitor][epoch-1], epoch)
    
    dfhistory = pd.DataFrame(history)



    dfhistory.to_csv("./loss{}.csv".format(number))    
    return arr_scores[best_score_idx]
storage_name = "sqlite:///optuna.db"    
study = optuna.create_study(direction="minimize",study_name="selus", sampler=optuna.samplers.TPESampler(), storage=storage_name,load_if_exists=True)
study.optimize(objective, n_trials=50)



