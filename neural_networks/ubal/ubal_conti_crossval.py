import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('D:/DP_Budovanie_schemy_tela')
import data_processing.ubal.data_util as du
from myubal import MyUBal, balSuportLearning
from sklearn.model_selection import KFold, StratifiedKFold


def train_ubal_crossval(dataset, epochs, folds=5, repetitions=1, hidden_layer=100, alpha = 0.3, betas=[0.0, 1.0, 0.0],
                        gammasF=[float("nan"), 1.0, 1.0], gammasB=[1.0, 1.0, float("nan")]):
    acc_train_per_run = []
    acc_test_per_run = []
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    f = 0
    dataset = np.array(dataset, dtype=object)
    data_X = dataset[:,[0]]
    data_y = dataset[:,[1]]
    data_y_strat = np.argmax(data_y, axis=1)
    for train_index, test_index in kfold.split(data_X, data_y_strat):
        for i in range(repetitions):
            f += 1
            print("Training Fold ",f)
            data_train = np.hstack((np.take(data_X, train_index, axis=0), np.take(data_y, train_index, axis=0)))
            data_test = np.hstack((np.take(data_X, test_index, axis=0), np.take(data_y, test_index, axis=0)))
            inp, hid, out = len(dataset[0][0]), hidden_layer, len(dataset[0][1])
            model = MyUBal(inp=inp, hid=hid, out=out, betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha,
                           testData=data_test)
            bal_results = model.train(data_train, max_epoch=epochs)
            print(bal_results)
            # Plot
            fig, errplot = plt.subplots()
            errplot.set_title("classification error [%]")
            errplot.set_ylim([0, 100])
            epochs_list = list(range(len(bal_results.ClassificationErrorsPercentages())))
            errplot.plot(epochs_list, bal_results.ClassificationErrorsPercentages())
            errplot.plot(epochs_list, bal_results.ClassificationErrorsPercentagesTest())
            plt.savefig("FINAL_{0}-ce".format(trainingName))
            # print("SAVED SUCC")
            plt.show()
            # Post-test
            err_train = 0
            for X, y in data_train:
                X = np.expand_dims(np.transpose(X), axis=1)
                prediction = model.network.activation_fp_last(X)
                prediction = np.transpose(np.squeeze(prediction))
                prediction = balSuportLearning(prediction)
                if not np.all(prediction == y):
                    err_train += 1
            trainingAcc = 100 - ((err_train / max(1, len(data_train))) * 100)
            print("training accuracy: ", trainingAcc, "%")
            acc_train_per_run.append(trainingAcc)
            err_test = 0
            for X, y in data_test:
                X = np.expand_dims(np.transpose(X), axis=1)
                prediction = model.network.activation_fp_last(X)
                prediction = np.transpose(np.squeeze(prediction))
                prediction = balSuportLearning(prediction)
                if not np.all(prediction == y):
                    err_test += 1
            testingSucc = 100 - ((err_test / len(data_test)) * 100)
            print("testing: accuracy", testingSucc, "%")
            acc_test_per_run.append(testingSucc)

    return acc_train_per_run, acc_test_per_run


if __name__ == "__main__":
    suffix = ".baldata"
    trainingName = "lh"
    # trainingName = "rh"
    # trainingName = "lf" # THIS IS GOOD DONT RETRAIN
    # trainingName = "rf"  #
    # trainingName = "lf_rh"
    # trainingName = "all"  #

    with open("ubal_data/ubal_data_leyla/" + trainingName + suffix) as f:
        data = json.load(f)
    labeled = [(du.normalize_and_invert(np.array(i['pos'])), np.array(i["touch"])) for i in data]
    betas = [1.0, 1.0, 0.9]
    gammasF = [float("nan"), 1.0, 1.0]
    gammasB = [0.98, 1.0, float("nan")]
    # acc_train_list, acc_test_list = train_ubal_crossval(labeled, 200)
    acc_train_list, acc_test_list = train_ubal_crossval(labeled, 10, alpha=0.1, betas=betas, gammasF=gammasF, gammasB=gammasB)
    print("Average training accuracy: ",(sum(acc_train_list)/len(acc_train_list)))
    print("Average test accuracy: ",(sum(acc_test_list)/len(acc_test_list)))