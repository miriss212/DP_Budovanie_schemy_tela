import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from myubal import MyUBal, balSuportLearning
from sklearn.model_selection import StratifiedKFold
import time

sys.path.append('D:/DP_Budovanie_schemy_tela')
import data_processing.ubal.data_util as du

def train_ubal_crossval(dataset, epochs, folds=2, repetitions=1, hidden_layer=100, alpha=0.3, betas=[0.0, 1.0, 0.0],
                        gammasF=[float("nan"), 1.0, 1.0], gammasB=[1.0, 1.0, float("nan")]):
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    dataset = np.array(dataset, dtype=object)
    data_X = dataset[:,[0]]
    data_y = dataset[:,[1]]
    data_y_strat = np.argmax(data_y, axis=1)
    results = []  # Store results for each fold
    for train_index, test_index in kfold.split(data_X, data_y_strat):
        fold_results = {'params': {'alpha': alpha, 'betas': betas, 'gammasF': gammasF, 'gammasB': gammasB},
                        'train_accuracies': [], 'test_accuracies': []}
        for i in range(repetitions):
            data_train = np.hstack((np.take(data_X, train_index, axis=0), np.take(data_y, train_index, axis=0)))
            data_test = np.hstack((np.take(data_X, test_index, axis=0), np.take(data_y, test_index, axis=0)))
            inp, hid, out = len(dataset[0][0]), hidden_layer, len(dataset[0][1])
            model = MyUBal(inp=inp, hid=hid, out=out, betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha,
                        testData=data_test)
            _, acc_train_epoch, acc_test_epoch = model.train_minimal(data_train, max_epoch=epochs)
            fold_results['train_accuracies'].append(acc_train_epoch)
            fold_results['test_accuracies'].append(acc_test_epoch)
        results.append(fold_results)
    return results




if __name__ == "__main__":


    suffix = ".baldata"
    model_names = ["lf", "rf", "rh", "lh"]  # List of model names
    #model_names = ["lf"]  # List of model names
    hidden_layer_sizes = [100, 150, 200, 250, 300]  # List of hidden layer sizes
    learning_rates = [0.01, 0.1, 0.05, 0.5, 0.3]  # List of learning rates
    #hidden_layer_sizes = [100, 150]  # List of hidden layer sizes
    #learning_rates = [0.01, 0.1] # List of learning rates
    results_dict = {}  # Dictionary to store results for each model

    for trainingName in model_names:
        with open("ubal_data/ubal_data_leyla_2/" + trainingName + suffix) as f:
            data = json.load(f)
        labeled = [(du.normalize_and_invert(np.array(i['pos'])), np.array(i["touch"])) for i in data]
        betas = [1.0, 1.0, 0.9]
        gammasF = [float("nan"), 1.0, 1.0]
        gammasB = [0.98, 1.0, float("nan")]
        for hidden_layer_size in hidden_layer_sizes:
            for learning_rate in learning_rates:
                results = train_ubal_crossval(labeled, epochs=2, alpha=learning_rate, betas=betas, gammasF=gammasF, gammasB=gammasB, hidden_layer=hidden_layer_size)
                results_dict[trainingName] = results
                results_dict[(trainingName, hidden_layer_size, learning_rate)] = results

    # Save all data in one pickle file
    with open(f'data/results/ubal_results_conti_{hidden_layer_sizes}_{learning_rates}.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

