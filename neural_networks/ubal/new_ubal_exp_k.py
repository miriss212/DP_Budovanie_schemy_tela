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
import time

def train_ubal_crossval(dataset, epochs, folds=5, repetitions=10, hidden_layer=100, alpha = 0.6, betas=[0.0, 1.0, 0.0],
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
    # trainingName = "lh"
    trainingName = "rh"
    # trainingName = "lf" # THIS IS GOOD DONT RETRAIN
    #trainingName = "rf"  # nemame u mateja
    # trainingName = "lf_rh"
    # trainingName = "all"  #

    #with open("ubal_data/ubal_data_matej/" + trainingName + suffix) as f:
    with open("ubal_data/ubal_data_leyla_2/" + trainingName + suffix) as f:
        data = json.load(f)
    print(len(data))

    betas = [1.0, 1.0, 0.9]
    gammasF = [float("nan"), 1.0, 1.0]
    gammasB = [0.9, 1.0, float("nan")]
    epcs = 100
    results_train = []
    results_test = []
    tested_k = [1,6,9,13]
    #tested_k = [1,6]
   
    results_dict = {}
    for k in tested_k:
        labeled = [(du.kwta_per_bodypart(np.array(i['pos']),k=k,continuous=True), np.array(i["touch"])) for i in data]
        acc_train_list, acc_test_list = train_ubal_crossval(labeled, hidden_layer=200, epochs=epcs, folds=2, repetitions=5,
                                                            betas=betas, gammasF=gammasF, gammasB=gammasB)
        #results_train.append((statistics.mean(acc_train_list),statistics.stdev(acc_train_list)))
        #results_test.append((statistics.mean(acc_test_list),statistics.stdev(acc_test_list)))
        mean_train_accuracy = statistics.mean(acc_train_list)
        mean_test_accuracy = statistics.mean(acc_test_list)
        std_train_accuracy =statistics.stdev(acc_train_list)
        std_test_accuracy = statistics.stdev(acc_test_list)
        results_dict[k] = (mean_train_accuracy, mean_test_accuracy, std_train_accuracy, std_test_accuracy)
        print()
       

   
    pickle_file_path = f"data/results/pickles/results_{trainingName}_{time.time()}.pkl"
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(results_dict, pickle_file)

    print("Results saved to:", pickle_file_path)