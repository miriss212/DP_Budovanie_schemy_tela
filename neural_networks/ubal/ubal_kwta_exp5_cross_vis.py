import json, random
import numpy as np
import matplotlib.pyplot as plt
import random
import data_processing.ubal.data_util as du
from myubal import MyUBal
from UBAL_numpy import Sigmoid, SoftMax, UBAL
from sklearn.model_selection import train_test_split


hidden_layer=100
betas=[0.0, 1.0, 0.0]
gammasF=[float("nan"), 1.0, 1.0]
gammasB=[1.0, 1.0, float("nan")]
alpha = 0.3
sigmoid = Sigmoid()
softmax = SoftMax()
act_fun_F = [sigmoid, sigmoid, softmax]
act_fun_B = [sigmoid, sigmoid, sigmoid]
init_w_mean=0.0
init_w_variance=1.0
no_samples = 3


def train_ubal(dataset, epochs, arch):
    model = MyUBal(inp=arch[0], hid=arch[1], out=arch[2], betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha)
    bal_results = model.train_minimal(dataset, max_epoch=epochs)
    return bal_results, model.network


if __name__ == "__main__":
    suffix = ".baldata"

    with open("../ubal_data/ubal_data_correct_touch/" + "lh" + suffix) as f:
        data = json.load(f)
    # apply kwta per body part
    k = 16
    data_lh = [(du.kwta_per_bodypart(np.array(i['pos']),k=k,continuous=True), np.array(i["touch"])) for i in data]
    arch_lh = len(data_lh[0][0]), hidden_layer, len(data_lh[0][1])
    # du.analyze_data(labeled)
    # res, trained_net_lh = train_ubal(data_lh, 150, arch_lh)
    # np.savez("weights/trained_LH_all.npz", wtsF=trained_net_lh.weightsF, wtsB=trained_net_lh.weightsB)
    loaded_wts = np.load("weights/trained_LH_all.npz", allow_pickle=True)
    ubal_lh = UBAL(arch_lh, act_fun_F, act_fun_B, alpha, init_w_mean, init_w_variance, betas, gammasF, gammasB)
    ubal_lh.weightsF = loaded_wts["wtsF"]
    ubal_lh.weightsB = loaded_wts["wtsB"]

    with open("../ubal_data/ubal_data_correct_touch/" + "lf" + suffix) as f:
        data2 = json.load(f)
    # apply kwta per body part
    k = 16
    data_lf = [(du.kwta_per_bodypart(np.array(i['pos']), k=k, continuous=True), np.array(i["touch"])) for i in data2]
    arch_lf = len(data_lf[0][0]), hidden_layer, len(data_lf[0][1])
    # res2, trained_net_lf = train_ubal(data_lf, 150, arch_lf)
    # np.savez("weights/trained_LF_all.npz", wtsF=trained_net_lf.weightsF, wtsB=trained_net_lf.weightsB)
    loaded_wts = np.load("weights/trained_LF_all.npz", allow_pickle=True)
    ubal_lf = UBAL(arch_lh, act_fun_F, act_fun_B, alpha, init_w_mean, init_w_variance, betas, gammasF, gammasB)
    ubal_lf.weightsF = loaded_wts["wtsF"]
    ubal_lf.weightsB = loaded_wts["wtsB"]

    # rand_index = np.random.randint(len(data_lh))
    indeces = []
    data_x_lh,data_y_lh = du.split_data(data_lh)
    data_x_lf,data_y_lf = du.split_data(data_lf)
    print(data_x_lf.shape)
    for i in range(len(data_lf)):
        X, y = data_lf[i]
        found = 0
        for j in range(len(data_lh)):
            if list(data_x_lh[j]) == list(X):
                found = found + 1
        if found == 0:
            indeces.append(i)
        else:
            print("pattern from LF found in LH")
    print(indeces)

    du.visualize(data_lf, no_samples, "random_lf", indeces[0:no_samples])
    pred_impossible = []
    rand_data_lf = [data_lf[i] for i in indeces]
    s = 1
    for X,y in rand_data_lf:
        print("sample ",s)
        s = s + 1
        X = np.expand_dims(np.transpose(X), axis=1)
        prediction,net = ubal_lh.activation_fp_last_with_net(X)
        prediction = np.transpose(np.squeeze(prediction))
        net = np.transpose(np.squeeze(net))
        targ_winners = np.argmax(y, axis=0)
        pred_winners = np.argmax(prediction, axis=0)
        print(targ_winners)
        print(pred_winners)
        print(prediction[pred_winners])
        print(prediction)
        print(net)
        print(net.shape)
        print(np.argmax(net))
        print(net[np.argmax(net)])
        pred_impossible.append((X, prediction))

    du.visualize(pred_impossible, no_samples, "prediction_impossible", range(no_samples))

    # rand_data_lh = [data_lh[i] for i in indeces]
    # s = 1
    # for X,y in rand_data_lh:
    #     print("sample ",s)
    #     s = s + 1
    #     X = np.expand_dims(np.transpose(X), axis=1)
    #     prediction,net = ubal_lf.activation_fp_last_with_net(X)
    #     prediction = np.transpose(np.squeeze(prediction))
    #     net = np.transpose(np.squeeze(net))
    #     targ_winners = np.argmax(y, axis=0)
    #     pred_winners = np.argmax(prediction, axis=0)
    #     print(targ_winners)
    #     print(pred_winners)
    #     print(prediction[pred_winners])
    #     print(prediction)
    #     print(net)
    #     print(np.argmax(net))
    #     print(net[np.argmax(net)])
    #     pred_impossible.append((X, prediction))
    # du.visualize(pred_impossible, 2*no_samples, "prediction_impossible", range(2*no_samples))