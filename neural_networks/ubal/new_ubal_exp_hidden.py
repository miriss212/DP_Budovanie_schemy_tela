import json
import numpy as np
import matplotlib.pyplot as plt
import data_processing.ubal.data_util as du
from myubal import MyUBal, balSuportLearning
from sklearn.model_selection import KFold, StratifiedKFold


def train_ubal_crossval(dataset, epochs, folds=5, hidden_layer=200, alpha = 0.6,
                        # betas=[1.0, 1.0, 0.9], gammasF = [float("nan"), 1.0, 1.0], gammasB = [0.9, 1.0, float("nan")]):
                        betas=[0.0, 1.0, 0.0], gammasF=[float("nan"), 1.0, 1.0], gammasB=[1.0, 1.0, float("nan")]):
    acc_train_per_run = []
    acc_test_per_run = []
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    f = 0
    for train_index, test_index in kfold.split(dataset):
        f += 1
        print("Training Fold ",f)
        data_train = np.take(dataset, train_index, axis=0)
        data_test = np.take(dataset, test_index, axis=0)

        print("::::: TRAIN data :::::")
        du.analyze_data(data_train)
        print("::::: TEST data :::::")
        du.analyze_data(data_test)

        inp, hid, out = len(dataset[0][0]), hidden_layer, len(dataset[0][1])
        model = MyUBal(inp=inp, hid=hid, out=out, betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha,
                       testData=data_test)
        net = model.train_minimal(data_train, max_epoch=epochs)
        # Post-test
        err_train = 0
        for X, y in data_train:
            X = np.expand_dims(np.transpose(X), axis=1)
            prediction = model.network.activation_fp_last(X)
            prediction = np.transpose(np.squeeze(prediction))
            # prediction = balSuportLearning(prediction)
            targ_winners = np.argmax(y, axis=0)
            pred_winners = np.argmax(prediction, axis=0)
            if not np.all(targ_winners == pred_winners):
                err_train += 1
        trainingAcc = 100 - ((err_train / max(1, len(data_train))) * 100)
        print("training accuracy: ", trainingAcc, "%")
        acc_train_per_run.append(trainingAcc)
        err_test = 0
        for X, y in data_test:
            X = np.expand_dims(np.transpose(X), axis=1)
            prediction = model.network.activation_fp_last(X)
            prediction = np.transpose(np.squeeze(prediction))
            # prediction = balSuportLearning(prediction)
            targ_winners = np.argmax(y, axis=0)
            pred_winners = np.argmax(prediction, axis=0)
            if not np.all(targ_winners == pred_winners):
                err_test += 1
            # print("T:",y)
            # print("P",prediction)
        testingSucc = 100 - ((err_test / len(data_test)) * 100)
        print("testing: accuracy", testingSucc, "%")
        acc_test_per_run.append(testingSucc)

    return acc_train_per_run, acc_test_per_run


if __name__ == "__main__":
    suffix = ".baldata"
    # trainingName = "lh"
    # trainingName = "rh"
    # trainingName = "lf" # THIS IS GOOD DONT RETRAIN
    trainingName = "rf"  #
    # trainingName = "lf_rh"
    # trainingName = "all"  #

    with open("../ubal_data/ubal_data_leyla/" + trainingName + suffix) as f:
        data = json.load(f)
    k = 13
    epcs = 100
    results_train = []
    results_test = []
    hidden_sizes = [100,200]
    for hs in hidden_sizes:
        print(":::::::::: Training with hidden size =",hs," ::::::::::")
        labeled = [(du.kwta_per_bodypart(np.array(i['pos']),k=k,continuous=True), np.array(i["touch"])) for i in data]
        acc_train_list, acc_test_list = train_ubal_crossval(labeled, epochs=epcs, hidden_layer=hs)
        results_train.append(sum(acc_train_list)/len(acc_train_list))
        results_test.append(sum(acc_test_list)/len(acc_test_list))
        print()

    print(hidden_sizes)
    print(results_train)
    print(results_test)