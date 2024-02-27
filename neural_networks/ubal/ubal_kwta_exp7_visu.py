import json, random
import numpy as np
import matplotlib.pyplot as plt
import random
import data_processing.ubal.data_util as du
from myubal import MyUBal
from UBAL_numpy import Sigmoid, SoftMax, UBAL
from sklearn.model_selection import train_test_split


betas=[0.0, 1.0, 0.0]
gammasF=[float("nan"), 1.0, 1.0]
gammasB=[1.0, 1.0, float("nan")]
alpha = 0.6
sigmoid = Sigmoid()
softmax = SoftMax()
act_fun_F = [sigmoid, sigmoid, softmax]
act_fun_B = [sigmoid, sigmoid, sigmoid]
init_w_mean=0.0
init_w_variance=1.0
no_samples = 5
hidden_layer = 200


def visualize_one_line_all(data_propri, data_touch, figname="default"):
    num_row = 1
    num_col = 8
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col, num_row))
    parts = []
    parts.append(data_propri[0:64])  # left hand
    parts.append(data_propri[64:172])  # right hand
    parts.append(data_propri[172:236])  # left fore
    parts.append(data_propri[236:344])  # right fore
    parts += data_touch
    names = ["LH-prop", "LF-prop", "RH-prop", "RF-prop", "LH-touch", "LF-touch", "RH-touch","RF-touch"]
    for idx, pattern in enumerate(parts):
        # print(pattern)
        sidex, sidey = 9, 12
        if len(pattern) == 64:
            sidex, sidey = 8, 8
        elif len(pattern) == 49:
            sidex, sidey = 7, 7
        pattern = pattern.reshape(sidex, sidey)
        ax = axes[idx]
        ax.imshow(pattern, cmap='viridis', interpolation='nearest')
        ax.set_title(names[idx])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # plt.imshow(pattern, cmap='gray')
    plt.tight_layout()
    fig_name = "visualize_ubal_data_" + figname
    plt.savefig(fig_name+".png", format="png")
    plt.savefig(fig_name+".pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    suffix = ".baldata"
    k = 13
    # with open("../ubal_data/ubal_data_correct_touch/" + "rh" + suffix) as f:
    #     data = json.load(f)
    # data_train_loaded_rh, data_test_loaded_rh = train_test_split(data, test_size=0.2, random_state=42)
    # with open('weights/rh-testdata.baldata', "w") as out:
    #     out.write(str(json.dumps(data_test_loaded_rh, indent=4)))
    # with open("../ubal_data/ubal_data_correct_touch/" + "rf" + suffix) as f:
    #     data = json.load(f)
    # data_train_loaded_2, data_test_loaded_2 = train_test_split(data, test_size=0.2, random_state=42)
    # with open('weights/rf-testdata.baldata', "w") as out:
    #     out.write(str(json.dumps(data_test_loaded_2, indent=4)))

    # with open("weights/rh-testdata.baldata") as f:
    #     data_test_loaded = json.load(f)
    # data_train = [(du.kwta_per_bodypart(np.array(i['pos']), k=k, continuous=True), np.array(i["touch"])) for i in data_train_loaded_rh]
    # data_test = [(du.kwta_per_bodypart(np.array(i['pos']), k=k, continuous=True), np.array(i["touch"])) for i in data_test_loaded_rh]
    # data_train2 = [(du.kwta_per_bodypart(np.array(i['pos']), k=k, continuous=True), np.array(i["touch"])) for i in data_train_loaded_2]
    # data_test2 = [(du.kwta_per_bodypart(np.array(i['pos']), k=k, continuous=True), np.array(i["touch"])) for i in data_test_loaded_2]
    # arch = len(data_train2[0][0]), hidden_layer, len(data_train2[0][1])
    # du.analyze_data(data_rh)
    # model = MyUBal(inp=arch[0], hid=arch[1], out=arch[2], betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha, testData=data_test)
    # bal_results = model.train(data_train, max_epoch=120)
    # np.savez("weights/trained_RH.npz", wtsF=model.network.weightsF, wtsB=model.network.weightsB)
    # model2 = MyUBal(inp=arch[0], hid=arch[1], out=arch[2], betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha, testData=data_test2)
    # bal_results = model2.train(data_train2, max_epoch=120)
    # np.savez("weights/trained_RF.npz", wtsF=model2.network.weightsF, wtsB=model2.network.weightsB)

    with open("weights/rf-testdata.baldata") as f:
        data_test_loaded = json.load(f)
    data_test = [(du.kwta_per_bodypart(np.array(i['pos']), k=k, continuous=True), np.array(i["touch"])) for i in data_test_loaded]
    with open("weights/lh-testdata.baldata") as f:
        data_test_loaded = json.load(f)
    data_test2 = [(du.kwta_per_bodypart(np.array(i['pos']), k=k, continuous=True), np.array(i["touch"])) for i in data_test_loaded]
    name = "lh-rf"
    # loaded_wts = np.load("weights/trained_RH.npz", allow_pickle=True)
    # ubal_rh = UBAL((len(data_test[0][0]), hidden_layer, len(data_test[0][1])), act_fun_F, act_fun_B, alpha, init_w_mean, init_w_variance, betas, gammasF, gammasB)
    # ubal_rh.weightsF = loaded_wts["wtsF"]
    # ubal_rh.weightsB = loaded_wts["wtsB"]
    # loaded_wts = np.load("weights/trained_LH.npz", allow_pickle=True)
    # ubal_2 = UBAL((len(data_test2[0][0]), hidden_layer, len(data_test2[0][1])), act_fun_F, act_fun_B, alpha, init_w_mean, init_w_variance, betas, gammasF, gammasB)
    # ubal_2.weightsF = loaded_wts["wtsF"]
    # ubal_2.weightsB = loaded_wts["wtsB"]

    err = 0
    err_indices = []
    ok_indeces = []
    predictions = []
    found_same = []
    for i in range(len(data_test)):
        X, y = data_test[i]
        for j in range(len(data_test2)):
            XX, _ = data_test2[j]
            if list(XX) == list(X):
                found_same.append((i,j))
        # X = np.expand_dims(np.transpose(X), axis=1)
        # pred = ubal_rh.activation_fp_last(X)
        # pred = np.transpose(np.squeeze(pred))
        # targ_winners = np.argmax(y, axis=0)
        # pred_winners = np.argmax(pred, axis=0)
        # if not np.all(targ_winners == pred_winners):
        #     err += 1
        #     err_indices.append(i)
        # else:
        #     ok_indeces.append(i)
        # predictions.append(pred)
    # print(err_indices)
    # print(ok_indeces)
    # print("Test error was: ", (err / len(data_test_rh)) * 100)
    print(len(found_same)," / ",len(data_test))

    # index_r = 58
    # index_l = 55
    indeces = random.choice(found_same)
    print(indeces)
    data_all_touch = []
    rpropri, rtouch = data_test[indeces[0]]
    lpropri, ltouch = data_test2[indeces[1]]
    #lh
    # data_all_touch.append(np.zeros(7*7))
    data_all_touch.append(ltouch)
    # lf
    data_all_touch.append(np.zeros(9*12))
    # data_all_touch.append(ltouch)
    # rh
    # data_all_touch.append(rtouch)
    data_all_touch.append(np.zeros(7*7))
    # rf
    # data_all_touch.append(np.zeros(9*12))
    data_all_touch.append(rtouch)
    visualize_one_line_all(lpropri, data_all_touch, name)

    # visualize(data_test, 11, "targets_rh", err_indices)

    # du.visualize(data_lf, no_samples, "random_lf", indeces)
    # pred_impossible = []
    # rand_data_lf = [data_lf[i] for i in indeces]
    # s = 1
    # for X,y in rand_data_lf:
    #     print("sample ",s)
    #     s = s + 1
    #     X = np.expand_dims(np.transpose(X), axis=1)
    #     prediction,net = ubal_lh.activation_fp_last_with_net(X)
    #     prediction = np.transpose(np.squeeze(prediction))
    #     net = np.transpose(np.squeeze(net))
    #     targ_winners = np.argmax(y, axis=0)
    #     pred_winners = np.argmax(prediction, axis=0)
    #     print(targ_winners)
    #     print(pred_winners)
    #     print(prediction[pred_winners])
    #     print(prediction)
    #     print(net)
    #     print(net.shape)
    #     print(np.argmax(net))
    #     print(net[np.argmax(net)])
    #     pred_impossible.append((X, prediction))

    # du.visualize(pred_impossible, no_samples, "prediction_impossible", range(no_samples))