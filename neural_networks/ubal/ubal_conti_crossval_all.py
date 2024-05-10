import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from myubal import MyUBal, balSuportLearning
from sklearn.model_selection import StratifiedKFold

sys.path.append('D:/DP_Budovanie_schemy_tela')
import data_processing.ubal.data_util as du

def train_ubal_crossval(dataset, epochs, folds=5, repetitions=1, hidden_layer=200, alpha=0.3, betas=[0.0, 1.0, 0.0],
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
    #model_names = ["lf", "rf", "rh", "lh"]  # List of model names
    model_names = ["lf"]
    results_dict = {}  # Dictionary to store results for each model

    for trainingName in model_names:
        with open("ubal_data/ubal_data_leyla_2/" + trainingName + suffix) as f:
            data = json.load(f)
        labeled = [(du.normalize_and_invert(np.array(i['pos'])), np.array(i["touch"])) for i in data]
        betas = [1.0, 1.0, 0.9]
        gammasF = [float("nan"), 1.0, 1.0]
        gammasB = [0.98, 1.0, float("nan")]
        results = train_ubal_crossval(labeled, epochs=10, alpha=0.1, betas=betas, gammasF=gammasF, gammasB=gammasB)
        results_dict[trainingName] = results

    # Save all data in one pickle file
    with open('ubal_results_combined_2.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

    fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    fig.suptitle('UBal Results')

    for i, trainingName in enumerate(model_names):
        for fold_result in results_dict[trainingName]:
            train_accuracies = np.mean(fold_result['train_accuracies'], axis=0)
            test_accuracies = np.mean(fold_result['test_accuracies'], axis=0)
            axes[i].plot(train_accuracies, label='Train', marker='o')
            axes[i].plot(test_accuracies, label='Test', marker='o')
            axes[i].set_title(f'{trainingName} Accuracies')
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel('Accuracy (%)')
            axes[i].legend()
            axes[i].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("ubal_results_combined.pdf", format="pdf")
    plt.show()
