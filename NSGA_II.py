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
import itertools
# load net and material lab
def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)
seed_everything(1)
load_study = optuna.study.load_study("selus", "sqlite:///optuna.db")
best_params = load_study.best_params
model_dim = best_params["model_dim"]
# batch_size = best_params["batch_size"]
lr = best_params["lr"]
skin_hidden_size = best_params["skin_hidden_size"]
stringer_hidden_size = best_params["stringer_hidden_size"]
otherfeatures_hidden_size = best_params["otherfeatures_hidden_size"]
net = parallel_net(model_dim,skin_hidden_size,stringer_hidden_size,otherfeatures_hidden_size)
model_dict = torch.load("./checkpoint5.pt")
net.load_state_dict(model_dict)
net.eval()
Material_lab = pd.read_csv("./ml.csv",header=None)
def cal_obj(pop, nobj):
    #use net solve buckling load
    #decode -> solve
    npops = pop.shape[0]
    skin_seq = pop[:,:16]
    stringer_seq = pop[:,16:28]
    other_feas = np.zeros_like(a=0, shape=(npops, 14), dtype=np.float32)
    skin_seq_length = np.zeros(npops)
    strigner_seq_length = np.zeros(npops)
    rou_skin = np.zeros(npops)
    rou_stringer = np.zeros(npops)
    for i in range(npops): # decode
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
        skin_material = int(pop[:,28:][i,2])
        stringer_material = int(pop[:,28:][i,3])
        rou_skin[i] = Material_lab.iloc[skin_material-1][6]
        rou_stringer[i] = Material_lab.iloc[stringer_material-1][6]
        thickness = np.array([pop[:,28:][i,0]/1000 ,pop[:,28:][i,1]/1000])
        skin_material = np.array(Material_lab.iloc[skin_material-1][0:6])
        stringer_material = np.array(Material_lab.iloc[stringer_material-1][0:6])
        other_fea = np.concatenate([skin_material,stringer_material,thickness])  
        other_feas[i] = other_fea    
    # mass
    index_seq  = determing_seq(skin_seq)
    index_stringer = determing_seq(stringer_seq)
    index = np.logical_or(index_seq, index_stringer)
    mass = (0.479+0.000057692*rou_stringer*4*strigner_seq_length*2*other_feas[:,-1]+0.000422451*rou_skin*skin_seq_length*2*other_feas[:,-2]).reshape(-1,1)

    mass[index]+=10000
    skin_seq = torch.tensor(skin_seq, dtype=torch.int32)
    stringer_seq = torch.tensor(stringer_seq, dtype=torch.int32)
    other_feas = torch.tensor(other_feas,dtype=torch.float32)
    preds = -1*net(skin_seq, stringer_seq, other_feas).detach().numpy().reshape(-1,1)
    preds[index]+=10000
    objs = np.concatenate([mass, preds],axis=1)
    return objs
def initiate_pop(pop,num):
    number = [1,2,3,4]
    combain = np.array(list(itertools.product(number,repeat=10)))
    index = np.zeros(combain.shape[0], dtype=np.int32)
    for i in range(combain.shape[0]):
        seq = combain[i]
        if sum(seq==2) != sum(seq==3) or sum(seq==1)<int(0.1*10) or sum(seq==2)<int(0.1*10) or sum(seq==3)<int(0.1*10) or sum(seq==4)<int(0.1*10) or seq[0] == 1 or seq[0] == 4 :
            index[i] = 1
            continue
        for j in range(10-1):
            if seq[j] == 1 and seq[j+1] == 4:
                index[i] = 1
                continue
            if seq[j] == 4 and seq[j+1] == 1:
                index[i] = 1
                continue
            if seq[j] == 2 and seq[j+1] == 3:
                index[i] = 1
                continue
            if seq[j] == 3 and seq[j+1] == 2:
                index[i] = 1   
                continue
        for k in range(10-4):
            if seq[k] == seq[k+1] and seq[k] == seq[k+2] and seq[k] == seq[k+3] and seq[k] == seq[k+4]:
                index[i] = 1
    seq_ = combain[index==0]
    index_seq = np.random.choice(np.array(range(seq_.shape[0])),pop)
    seq = seq_[index_seq]
    padding = np.zeros((pop,num-10),dtype=np.int32)
    seq = np.hstack((seq,padding))
    return seq
def determing_seq(seq):
    pops = seq.shape[0]
    index = np.zeros(pops , dtype=np.int32)
    for i in range(pops):
        seq_ = seq[i]
        decode_seq = np.array([x for x in seq_ if x!=0])
        seq_len = len(decode_seq)
        index[i] = 0
        
        if sum(decode_seq==2) != sum(decode_seq==3) or sum(decode_seq==1)<int(0.1*seq_len) or sum(decode_seq==2)<int(0.1*seq_len) or sum(decode_seq==3)<int(0.1*seq_len) or sum(decode_seq==4)<int(0.1*seq_len) or decode_seq[0] == 1 or decode_seq[0] == 4 :
            index[i] = 1
            continue

        for j in range(seq_len-1):
            if decode_seq[j] == 1 and decode_seq[j+1] == 4:
                index[i] = 1
                continue
            if decode_seq[j] == 4 and decode_seq[j+1] == 1:
                index[i] = 1
                continue
            if decode_seq[j] == 2 and decode_seq[j+1] == 3:
                index[i] = 1
                continue
            if decode_seq[j] == 3 and decode_seq[j+1] == 2:
                index[i] = 1
                continue
            
        for k in range(seq_len-4):
            if decode_seq[k] == decode_seq[k+1] and decode_seq[k] == decode_seq[k+2] and decode_seq[k] == decode_seq[k+3] and decode_seq[k] == decode_seq[k+4]:
                index[i] = 1
            
    return index

def selection(pop, rank, cd, pc):
    # improved binary tournament selection
    (npop, dim) = pop.shape
    nm = int(npop * pc)
    nm = nm if nm % 2 == 0 else nm + 1
    mating_pool = np.zeros((nm, dim))
    for i in range(nm):
        [ind1, ind2] = np.random.choice(npop, 2)
        if rank[ind1] < rank[ind2]:
            mating_pool[i] = pop[ind1]
        elif rank[ind1] == rank[ind2]:
            mating_pool[i] = pop[ind1] if cd[ind1] > cd[ind2] else pop[ind2]
        else:
            mating_pool[i] = pop[ind2]
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, dim) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, dim))
    mu = np.random.random((nm, dim))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    for i in offspring:
        for j in range(len(i)):
            i[j] = round(i[j])
    return offspring


def mutation(pop, lb, ub, pm, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, dim)) < pm / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    for i in pop:
        for j in range(len(i)):
            i[j] = round(i[j])
    return pop


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def crowding_distance(objs, pfs):
    # crowding distance
    (npop, nobj) = objs.shape
    cd = np.zeros(npop)
    for key in pfs.keys():
        pf = pfs[key]
        temp_obj = objs[pf]
        fmin = np.min(temp_obj, axis=0)
        fmax = np.max(temp_obj, axis=0)
        df = fmax - fmin
        for i in range(nobj):
            if df[i] != 0:
                rank = np.argsort(temp_obj[:, i])
                cd[pf[rank[0]]] = np.inf
                cd[pf[rank[-1]]] = np.inf
                for j in range(1, len(pf) - 1):
                    cd[pf[rank[j]]] += (objs[pf[rank[j + 1]], i] - objs[pf[rank[j]], i]) / df[i]
    return cd


def nd_cd_sort(pop, objs, rank, cd, npop):
    # sort the population according to the Pareto rank and crowding distance
    temp_list = []
    for i in range(len(pop)):
        temp_list.append([pop[i], objs[i], rank[i], cd[i]])
    temp_list.sort(key=lambda x: (x[2], -x[3]))
    next_pop = np.zeros((npop, pop.shape[1]))
    next_objs = np.zeros((npop, objs.shape[1]))
    next_rank = np.zeros(npop)
    next_cd = np.zeros(npop)
    for i in range(npop):
        next_pop[i] = temp_list[i][0]
        next_objs[i] = temp_list[i][1]
        next_rank[i] = temp_list[i][2]
        next_cd[i] = temp_list[i][3]
    return next_pop, next_objs, next_rank, next_cd


def main(npop, iter, lb, ub, pc=1, eta_c=20, pm=0.1, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: upper bound
    :param ub: lower bound
    :param pc: crossover probability
    :param eta_c: spread factor distribution index
    :param pm: mutation probability
    :param eta_m: perturbance factor distribution index
    :return:
    """
    # Step 1. Initialization
    dim = len(lb)  # dimension
    skin_seq = initiate_pop(npop, 16)
    stiffener_pop = initiate_pop(npop, 12)
    tANDm_pop = np.random.randint(np.array([125,125,1,1]), np.array([176,176,36,36]), (npop,4))  # population
    pop = np.hstack((skin_seq,stiffener_pop, tANDm_pop))
    objs = cal_obj(pop, dim)  # objectives
    [pfs, rank] = nd_sort(objs)  # Pareto rank
    cd = crowding_distance(objs, pfs)  # crowding distance

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 20 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Selection + crossover + mutation
        mating_pool = selection(pop, rank, cd, pc)
        offspring = crossover(mating_pool, lb, ub, eta_c)
        offspring = mutation(offspring, lb, ub, pm, eta_m)
        new_objs = cal_obj(offspring, dim  )
        pop = np.concatenate((pop, offspring), axis=0)
        objs = np.concatenate((objs, new_objs), axis=0)
        [pfs, rank] = nd_sort(objs)
        cd = crowding_distance(objs, pfs)
        [pop, objs, rank, cd] = nd_cd_sort(pop, objs, rank, cd, npop)
        print(objs[rank == 1].shape[0])
        pf = objs[np.where(rank == 1)]
        pf = pd.DataFrame(pf, columns=["object1", "object2"])
        pf.to_csv("./nsga_ii{}.csv".format(t))
        if objs[rank == 1].shape[0] == npop:
            break

    # Step 3. Sort the results
    pf = objs[np.where(rank == 1)]
    pf = pd.DataFrame(pf, columns=["object1", "object2"])
    objs = pd.DataFrame(objs, columns=["object1", "object2"])
    pf.to_csv("./proto_resultnsga_ii.csv")
    popf = pop [rank == 1]
    popf = pd.DataFrame(popf)
    objs.to_csv("./all_resultnsga_ii.csv")
    popf.to_csv("./protocanshunsga_ii.csv")


if __name__ == '__main__':
    main(500, 300, np.array([0]*28+[125]*2+[1]*2), np.array([4]*28+[175]*2+[35]*2))
