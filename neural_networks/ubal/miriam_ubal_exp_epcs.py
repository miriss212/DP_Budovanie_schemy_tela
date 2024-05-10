import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('D:/DP_Budovanie_schemy_tela')
import data_processing.ubal.data_util as du
from myubal import MyUBal, balSuportLearning
from sklearn.model_selection import KFold, StratifiedKFold
import statistics
import pickle

def train_ubal_crossval(dataset, epochs, folds=5, repetitions=10, hidden_layer=100, alpha = 0.3, betas=[0.0, 1.0, 0.0],
                        gammasF=[float("nan"), 1.0, 1.0], gammasB=[1.0, 1.0, float("nan")]):
    acc_train_per_run = []
    acc_test_per_run = []
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    f = 0
    for train_index, test_index in kfold.split(dataset):
    #     for i in range(repetitions):
        f += 1
        print("Training Fold ",f)
        data_train = [dataset[i] for i in train_index]
        data_test = [dataset[i] for i in test_index]

        print("::::: TRAIN data :::::")
        du.analyze_data(data_train)
        print("::::: TEST data :::::")
        du.analyze_data(data_test)

        inp, hid, out = len(dataset[0][0]), hidden_layer, len(dataset[0][1])
        model = MyUBal(inp=inp, hid=hid, out=out, betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha,
                       testData=data_test)
        bal_results = model.train_minimal(data_train, max_epoch=epochs)
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
    model_names = ["lf", "rf", "rh", "lh"]
    alphas = [0.01, 0.1, 0.05, 0.5, 0.3]
    hidden_layers = [100, 150, 200, 250, 300]
    
    results_dict = {}

    for trainingName in model_names:
        with open("ubal_data/ubal_data_leyla_2/" + trainingName + suffix) as f:
            data = json.load(f)

        test_epcs = [50]
        betas = [1.0, 1.0, 0.9]
        gammasF = [float("nan"), 1.0, 1.0]
        gammasB = [0.9, 1.0, float("nan")]
        results_train = []
        results_test = []

        for alpha in alphas:
            for hidden_layer in hidden_layers:
                results_train = []
                results_test = []

                for e in test_epcs:
                    labeled = [(du.kwta_per_bodypart(np.array(i['pos']), k=13, continuous=True), np.array(i["touch"])) for i in data]
                    acc_train_list, acc_test_list = train_ubal_crossval(labeled, hidden_layer=hidden_layer, epochs=e, folds=2, repetitions=5,
                                                                        betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha)
                    results_train.append((statistics.mean(acc_train_list), statistics.stdev(acc_train_list)))
                    results_test.append((statistics.mean(acc_test_list), statistics.stdev(acc_test_list)))

                results_dict[(trainingName, alpha, hidden_layer)] = {epoch: {'train': results_train[i], 'test': results_test[i]} for i, epoch in enumerate(test_epcs)}

    with open('results_all_grid_hidden_layer_alphas.pkl', 'wb') as f:
        pickle.dump(results_dict, f)