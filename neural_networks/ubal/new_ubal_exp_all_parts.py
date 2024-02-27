import json
import numpy as np
import time
import matplotlib.pyplot as plt
import data_processing.ubal.data_util as du
from myubal import MyUBal, balSuportLearning
from sklearn.model_selection import KFold, StratifiedKFold
import statistics
from sklearn.metrics import fbeta_score, f1_score


def train_ubal_crossval(dataset, epochs, folds=5, repetitions=5, hidden_layer=200, alpha = 0.6,
                        betas=[1.0, 1.0, 0.9], gammasF = [float("nan"), 1.0, 1.0], gammasB = [0.9, 1.0, float("nan")]):
    acc_train_per_run = []
    acc_test_per_run = []
    for i in range(repetitions):
        print("repetition ", i)
        kfold = KFold(n_splits=folds, shuffle=True)
        f = 0
        for train_index, test_index in kfold.split(dataset):
            f += 1
            print("Training Fold ",f)
            data_train = np.take(dataset, train_index, axis=0)
            data_test = np.take(dataset, test_index, axis=0)
            inp, hid, out = len(dataset[0][0]), hidden_layer, len(dataset[0][1])
            model = MyUBal(inp=inp, hid=hid, out=out, betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha,
                           testData=data_test)
            bal_results = model.train(data_train, max_epoch=epochs)
            # Post-test
            err_train = 0
            for X, y in data_train:
                X = np.expand_dims(np.transpose(X), axis=1)
                prediction = model.network.activation_fp_last(X)
                prediction = np.transpose(np.squeeze(prediction))
                targ_winners = np.argmax(y, axis=0)
                pred_winners = np.argmax(prediction, axis=0)
                if not np.all(targ_winners == pred_winners):
                    err_train += 1
            trainingAcc = ((err_train / len(data_train)) * 100)
            # print("training accuracy: ", trainingAcc, "%")
            acc_train_per_run.append(trainingAcc)
            err_test = 0
            for X, y in data_test:
                X = np.expand_dims(np.transpose(X), axis=1)
                prediction = model.network.activation_fp_last(X)
                prediction = np.transpose(np.squeeze(prediction))
                targ_winners = np.argmax(y, axis=0)
                pred_winners = np.argmax(prediction, axis=0)
                if not np.all(targ_winners == pred_winners):
                    err_test += 1
            testingSucc = ((err_test / len(data_test)) * 100)
            # print("testing: accuracy", testingSucc, "%")
            acc_test_per_run.append(testingSucc)
    return acc_train_per_run, acc_test_per_run


if __name__ == "__main__":
    suffix = ".baldata"
    body_parts_names = ["lh","rh","lf","rf"]
    acc_train = []
    acc_test = []

    epcs = 50

    starttime = time.time()

    for body_part in body_parts_names:
        print()
        print("Training body part ",body_part)
        with open("../ubal_data/ubal_data_leyla/" + body_part + suffix) as f:
            data = json.load(f)
        k = 13
        # apply kwta per body part
        labeled = [(du.kwta_per_bodypart(np.array(i['pos']),k=k,continuous=True), np.array(i["touch"])) for i in data]
        acc_train_list, acc_test_list = train_ubal_crossval(labeled, hidden_layer=200, epochs=epcs)
        # acc_train.append((statistics.mean(acc_train_list),statistics.stdev(acc_train_list)))
        # acc_test.append((statistics.mean(acc_test_list),statistics.stdev(acc_test_list)))
        print("Acc train all:", acc_train_list)
        print("Acc test all:", acc_test_list)

        print(body_parts_names)
        print("Acc train: ",acc_train)
        print("Acc test: ",acc_test)

    du.note_end_time(starttime)
