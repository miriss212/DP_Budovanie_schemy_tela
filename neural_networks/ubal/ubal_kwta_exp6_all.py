import json
import numpy as np
import matplotlib.pyplot as plt
import data_processing.ubal.data_util as du
from myubal import MyUBal, balSuportLearning
from sklearn.model_selection import KFold, StratifiedKFold
import statistics


def train_ubal_crossval(dataset, epochs, folds=5, repetitions=10, hidden_layer=100, alpha = 0.3, betas=[0.0, 1.0, 0.0],
                        gammasF=[float("nan"), 1.0, 1.0], gammasB=[1.0, 1.0, float("nan")]):
    du.analyze_data(dataset)
    print()
    acc_train_per_run = []
    acc_test_per_run = []
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    f = 0
    for train_index, test_index in kfold.split(dataset):
        # for i in range(repetitions):
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
        model.train(data_train, max_epoch=epochs)
        # Post-test
        err_train = 0
        for X, y in data_train:
            X = np.expand_dims(np.transpose(X), axis=1)
            prediction = model.network.activation_fp_last(X)
            # prediction = balSuportLearning(prediction)
            # targ_winners = np.argmax(y, axis=0)
            # pred_winners = np.argmax(prediction, axis=0)
            # if not np.all(targ_winners == pred_winners):
            if not np.all(du.binarize(prediction) == y):
                err_train += 1
        trainingAcc = 100 - ((err_train / max(1, len(data_train))) * 100)
        print("training accuracy: ", trainingAcc, "%")
        acc_train_per_run.append(trainingAcc)
        err_test = 0
        for X, y in data_test:
            X = np.expand_dims(np.transpose(X), axis=1)
            prediction = model.network.activation_fp_last(X)
            # prediction = np.transpose(np.squeeze(prediction))
            # prediction = balSuportLearning(prediction)
            # targ_winners = np.argmax(y, axis=0)
            # pred_winners = np.argmax(prediction, axis=0)
            # if not np.all(targ_winners == pred_winners):
            if not np.all(du.binarize(prediction) == y):
                err_test += 1
            # print("T:",y)
            # print("P",prediction)
        testingSucc = 100 - ((err_test / len(data_test)) * 100)
        print("testing: accuracy", testingSucc, "%")
        acc_test_per_run.append(testingSucc)

    return acc_train_per_run, acc_test_per_run


if __name__ == "__main__":
    with open("../ubal_data/ubal_data_new/all.baldata") as f:
    # with open("../ubal_data/ubal_data_new/lh.baldata") as f:
    # with open("../ubal_data/ubal_data_new/rh.baldata") as f:
        data = json.load(f)
    tested_k = [5]
    results_train = []
    results_test = []
    for k in tested_k:
        # print(":::::::::: Training with k =",k," ::::::::::")
        # apply kwta per body part
        labeled = [(
            du.kwta_per_bodypart(np.array(i['pos']),k,continuous=False),
            du.kwta_per_bodypart(np.array(i["touch"]),k,hand=(7,7),farm=(9,12),continuous=False)
        ) for i in data]
        betas = [0.2, 0.8, 0.0]
        gammasF = [float("nan"), 0.0, 0.5]
        gammasB = [0.0, 0.5, float("nan")]
        acc_train_list, acc_test_list = train_ubal_crossval(labeled, hidden_layer=120, epochs=250, betas=betas,
                                                            gammasF=gammasF,gammasB=gammasB)
        # acc_train_list, acc_test_list = train_ubal_crossval(labeled, hidden_layer=200, epochs=120, folds=5)
        # print("Finished training with k =", k)
        # print("Average training accuracy: ",(sum(acc_train_list)/len(acc_train_list)))
        # print("Average test accuracy: ",(sum(acc_test_list)/len(acc_test_list)))

        # test_propri, test_touch = labeled[0]
        # prediction = nets_list[0].activation_fp_last(test_touch)
        # prediction = np.transpose(np.squeeze(prediction))
        # prediction = balSuportLearning(prediction)
        # print(prediction)
        # exit()
        results_train.append((statistics.mean(acc_train_list),statistics.stdev(acc_train_list)))
        results_test.append((statistics.mean(acc_test_list),statistics.stdev(acc_test_list)))

    print(tested_k)
    print(results_train)
    print(results_test)