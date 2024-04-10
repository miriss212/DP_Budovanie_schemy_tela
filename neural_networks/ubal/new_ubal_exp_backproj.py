import json, random
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('D:/DP_Budovanie_schemy_tela')

import data_processing.ubal.data_util as du
from myubal import MyUBal
from UBAL_numpy import Sigmoid, SoftMax, UBAL
from sklearn.model_selection import train_test_split

suffix = ".baldata"
hidden_layer = 200
betas = [1.0, 1.0, 0.9]
gammasF = [float("nan"), 1.0, 1.0]
gammasB = [0.9, 1.0, float("nan")]
# betas=[0.0, 1.0, 0.0]
# gammasF=[float("nan"), 1.0, 1.0]
# gammasB=[1.0, 1.0, float("nan")]
alpha = 0.6
sigmoid = Sigmoid()
softmax = SoftMax()
act_fun_F = [sigmoid, sigmoid, softmax]
act_fun_B = [sigmoid, sigmoid, sigmoid]
init_w_mean=0.0
init_w_variance=1.0
no_samples = 15
k = 13
epochs = 120
# trainingName = "rh"
trainingName = "lh"
# kwta = True
kwta = False
figname = "backward_projection_" + trainingName

def train_ubal(dataset, epochs, arch):
    model = MyUBal(inp=arch[0], hid=arch[1], out=arch[2], betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha)
    bal_results = model.train_minimal(dataset, max_epoch=epochs)
    return bal_results, model.network


if __name__ == "__main__":
    with open("ubal_data/ubal_data_leyla/" + trainingName + suffix) as f:
        data = json.load(f)
    labeled = [(du.kwta_per_bodypart(np.array(i['pos']),k=k,continuous=True), np.array(i["touch"])) for i in data]
    du.analyze_data(labeled)

    # res, ubal_net = train_ubal(labeled, epochs, [len(labeled[0][0]), hidden_layer, len(labeled[0][1])])
    # np.savez("weights/trained_new_{}_{}.npz".format(trainingName, time.time()), wtsF=ubal_net.weightsF, wtsB=ubal_net.weightsB)

    # loaded_wts = np.load("weights/trained_new_rh_1662821287.0330048.npz", allow_pickle=True)
    loaded_wts = np.load("weights/trained_new_lh_1662819134.3687465.npz", allow_pickle=True)
    # loaded_wts = np.load("weights/trained_new_lh_1658140056.048678.npz", allow_pickle=True)
    # loaded_wts = np.load("weights/trained_new_lh_1658147878.66412.npz", allow_pickle=True)
    ubal_net = UBAL([len(labeled[0][0]), hidden_layer, len(labeled[0][1])], act_fun_F, act_fun_B, alpha, init_w_mean, init_w_variance, betas, gammasF, gammasB)
    ubal_net.weightsF = loaded_wts["wtsF"]
    ubal_net.weightsB = loaded_wts["wtsB"]

    # rand_index = np.random.randint(len(data_lh))
    indeces = random.sample(range(len(labeled)), no_samples)
    # kwta = True
    # print(indeces)
    # du.visualize(labeled, no_samples, "random_lf", indeces)
    selected = [labeled[i] for i in indeces]
    projected = []
    s = 1
    for X,y in selected:
        print("sample ",s)
        s = s + 1
        y = np.expand_dims(np.transpose(y), axis=1)
        backward = ubal_net.activation_BP_last(y)
        backward = np.transpose(np.squeeze(backward))
        y = np.transpose(np.squeeze(y))
        # kwta
        if kwta:
            backward = du.kwta_per_bodypart(backward, k=k, continuous=True)
        projected.append((X, y, backward))
    if kwta:
        figname = figname + "_kwta"
    du.visualize_backproj(projected, no_samples, figname, range(no_samples))