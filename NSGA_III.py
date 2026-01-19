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

def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1) #type a num


def combination(n, m):
    # choose m elements from an n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))

def reference_points(npop, nvar):
    # calculate approximately npop uniformly distributed reference points on nvar dimensions
    h1 = 0
    while combination(h1 + nvar, nvar - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < nvar:
        h2 = 0
        while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * nvar)
            points = np.concatenate((points, temp_points), axis=0)
    return points


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 0
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
        #确定 pareto解
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

def selection(pop, pc, rank, k=2):
    # binary tournament selection
    (npop, nvar) = pop.shape
    nm = int(npop * pc)
    nm = nm if nm % 2 == 0 else nm + 1
    mating_pool = np.zeros((nm, nvar))
    for i in range(nm):
        [ind1, ind2] = np.random.choice(npop, k, replace=False)
        if rank[ind1] <= rank[ind2]:
            mating_pool[i] = pop[ind1]
        else:
            mating_pool[i] = pop[ind2]
    return mating_pool


def crossover(mating_pool, lb, ub, pc, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > pc, (1, nvar))] = 1
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
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < pm / nvar
    mu = np.random.random((npop, nvar))
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


def environmental_selection(pop, objs, zmin, npop, V):
    # NSGA-III environmental selection
    pfs, rank = nd_sort(objs)
    nobj = objs.shape[1]
    selected = np.full(pop.shape[0], False)
    ind = 0
    while np.sum(selected) + len(pfs[ind]) <= npop:
        selected[pfs[ind]] = True
        ind += 1
    K = npop - np.sum(selected)

    # select the remaining K solutions
    objs1 = objs[selected]
    objs2 = objs[pfs[ind]]
    npop1 = objs1.shape[0]
    npop2 = objs2.shape[0]
    nv = V.shape[0]
    temp_objs = np.concatenate((objs1, objs2), axis=0)
    t_objs = temp_objs - zmin

    # extreme points
    extreme = np.zeros(nobj)
    w = 1e-6 + np.eye(nobj)
    for i in range(nobj):
        extreme[i] = np.argmin(np.max(t_objs / w[i], axis=1))

    # intercepts
    try:
        hyperplane = np.matmul(np.linalg.inv(t_objs[extreme.astype(int)]), np.ones((nobj, 1)))
        if np.any(hyperplane == 0):
            a = np.max(t_objs, axis=0)
        else:
            a = 1 / hyperplane
    except LinAlgError:
        a = np.max(t_objs, axis=0)
    t_objs /= a.reshape(1, nobj)

    # association
    cosine = 1 - cdist(t_objs, V, 'cosine')
    distance = np.sqrt(np.sum(t_objs ** 2, axis=1).reshape(npop1 + npop2, 1)) * np.sqrt(1 - cosine ** 2)
    dis = np.min(distance, axis=1)
    association = np.argmin(distance, axis=1)
    temp_rho = dict(Counter(association[: npop1]))
    rho = np.zeros(nv)
    for key in temp_rho.keys():
        rho[key] = temp_rho[key]

    # selection
    choose = np.full(npop2, False)
    v_choose = np.full(nv, True)
    while np.sum(choose) < K:
        temp = np.where(v_choose)[0]
        jmin = np.where(rho[temp] == np.min(rho[temp]))[0]
        j = temp[np.random.choice(jmin)]
        I = np.where(np.bitwise_and(~choose, association[npop1:] == j))[0]
        if I.size > 0:
            if rho[j] == 0:
                s = np.argmin(dis[npop1 + I])
            else:
                s = np.random.randint(I.size)
            choose[I[s]] = True
            rho[j] += 1
        else:
            v_choose[j] = False
    selected[np.array(pfs[ind])[choose]] = True
    return pop[selected], objs[selected], rank[selected]


def main(npop, iter, lb, ub, nobj=2, pc=1, pm=1, eta_c=30, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space
    :param pc: crossover probability (default = 1)
    :param pm: mutation probability (default = 1)
    :param eta_c: spread factor distribution index (default = 30)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    skin_seq = initiate_pop(npop, 16)
    stiffener_pop = initiate_pop(npop, 12)
    tANDm_pop = np.random.randint(np.array([125,125,1,1]), np.array([176,176,36,36]), (npop,4))
    pop = np.hstack((skin_seq,stiffener_pop, tANDm_pop))
    # pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives
    V = reference_points(npop, nobj)  # reference vectors
    zmin = np.min(objs, axis=0)  # ideal points
    [pfs, rank] = nd_sort(objs)  # Pareto rank

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 10 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation
        mating_pool = selection(pop, pc, rank)
        off = crossover(mating_pool, lb, ub, pc, eta_c)
        off = mutation(off, lb, ub, pm, eta_m)
        off_objs = cal_obj(off, nobj)

        # Step 2.2. Environmental selection
        zmin = np.min((zmin, np.min(off_objs, axis=0)), axis=0)
        pop, objs, rank = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), zmin, npop, V)
        print(objs[rank == 0].shape[0])
        pf = objs[np.where(rank == 0)]
        pf = pd.DataFrame(pf, columns=["object1", "object2"])
        pf.to_csv("./nsga_iii{}.csv".format(t))
        if objs[rank == 0].shape[0] == npop:
            break
    # Step 3. Sort the results
    pf = objs[rank == 0]
    pf = pd.DataFrame(pf, columns=["object1", "object2"])
    objs = pd.DataFrame(objs, columns=["object1", "object2"])
    pf.to_csv("./proto_resultfinal.csv")
    popf = pop [rank == 0]
    popf = pd.DataFrame(popf)
    objs.to_csv("./all_resultfinal.csv")
    popf.to_csv("./protocanshufinal.csv")


if __name__ == '__main__':
    main(500, 3000, np.array([0]*28+[125]*2+[1]*2), np.array([4]*28+[175]*2+[35]*2))
# 3.78 391KN