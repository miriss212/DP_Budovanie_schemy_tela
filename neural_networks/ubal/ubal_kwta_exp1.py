import json, random
import numpy as np
import matplotlib.pyplot as plt
import random
import data_processing.ubal.data_util as du
from myubal import MyUBal, balSuportLearning
from myubal import BalResults
from sklearn.model_selection import train_test_split


def train_ubal(labeled, epochs, test_ratio=0.2, hidden_layer=100, betas=[0.0, 1.0, 0.0],
               gammasF=[float("nan"), 1.0, 1.0], gammasB=[1.0, 1.0, float("nan")]):
    data_train,data_test = train_test_split(labeled, test_size=test_ratio)
    # print(np.shape(data_train))
    # print(np.shape(data_test))

    # print("\n::::: TRAIN data :::::")
    # du.analyze_data(data_train)
    # print("\n::::: TEST data :::::")
    # du.analyze_data(data_test)



    # betas = [1.0, 1.0, 0.9]
    # gammasF = [float("nan"), 1.0, 1.0]
    # gammasB = [0.9, 1.0, float("nan")]
    alpha = 0.3
    inp, hid, out = len(labeled[0][0]), hidden_layer, len(labeled[0][1])
    model = MyUBal(inp=inp, hid=hid, out=out, betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha,
                   testData=data_test)
    # print("inp:{} hid:{} out:{}".format(inp, hid, out))
    bal_results = model.train(data_train, max_epoch=epochs)

    # Plot
    fig, errplot = plt.subplots()
    errplot.set_title("classification error [%]")
    errplot.set_ylim([0, 100])
    epochs = list(range(len(bal_results.ClassificationErrorsPercentages())))
    errplot.plot(epochs,bal_results.ClassificationErrorsPercentages())
    errplot.plot(epochs,bal_results.ClassificationErrorsPercentagesTest())
    plt.savefig("FINAL_{0}-ce".format(trainingName))
    print("SAVED SUCC")
    plt.show()

    # Post-test
    err = 0
    for X, y in data_test:
        X = np.expand_dims(np.transpose(X), axis=1)
        prediction = model.network.activation_fp_last(X)
        prediction = np.transpose(np.squeeze(prediction))
        prediction = balSuportLearning(prediction)
        if not np.all(prediction == y):
            err += 1

    testingSucc = -1
    if len(data_test) != 0:
        testingSucc = 100 - ((err / len(data_test)) * 100)
        print("testing: succ", testingSucc, "%")

    err = 0
    for X, y in data_train:
        X = np.expand_dims(np.transpose(X), axis=1)
        prediction = model.network.activation_fp_last(X)
        prediction = np.transpose(np.squeeze(prediction))
        prediction = balSuportLearning(prediction)
        if not np.all(prediction == y):
            err += 1

    trainingAcc = 100 - ((err / max(1, len(data_train))) * 100)
    print("training succ: ", trainingAcc, "%")
    return bal_results, trainingAcc, testingSucc


if __name__ == "__main__":
    suffix = ".baldata"
    trainingName = "lh"
    # trainingName = "rh"
    # trainingName = "lf" # THIS IS GOOD DONT RETRAIN
    # trainingName = "rf"  #
    # trainingName = "lf_rh"
    # trainingName = "all"  #

    with open("../ubal_data/ubal_data_leyla/" + trainingName + suffix) as f:
        data = json.load(f)
    tested_k = [1,5,9,12,16]
    for k in tested_k:
        print("Training with k =",k)
        # apply kwta per body part
        labeled = [(du.kwta_per_bodypart(np.array(i['pos']),k,True), np.array(i["touch"])) for i in data]
        du.analyze_data(labeled)
        res, trainAcc, testAcc = train_ubal(labeled, 150, 0.2)

