import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('D:/DP_Budovanie_schemy_tela')
import data_processing.ubal.data_util as du
from myubal import MyUBal, balSuportLearning
from sklearn.model_selection import KFold, StratifiedKFold
import pickle


def train_ubal_crossval(dataset, epochs, folds=5, repetitions=1, hidden_layer=100, alpha = 0.3, betas=[0.0, 1.0, 0.0],
                        gammasF=[float("nan"), 1.0, 1.0], gammasB=[1.0, 1.0, float("nan")]):
    
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    f = 0
    dataset = np.array(dataset, dtype=object)
    data_X = dataset[:,[0]]
    data_y = dataset[:,[1]]
    data_y_strat = np.argmax(data_y, axis=1)


    avg_training_accuracy = []
    avg_testing_accuracy = []

    for train_index, test_index in kfold.split(data_X, data_y_strat):
        acc_train_per_run = []
        acc_test_per_run = []
        training_accuracies_per_index = [[] for _ in range(repetitions)]
        testing_accuracies_per_index = [[] for _ in range(repetitions)]
        
        for i in range(repetitions):
            f += 1
            print("Training Fold ",f)
            data_train = np.hstack((np.take(data_X, train_index, axis=0), np.take(data_y, train_index, axis=0)))
            data_test = np.hstack((np.take(data_X, test_index, axis=0), np.take(data_y, test_index, axis=0)))
            inp, hid, out = len(dataset[0][0]), hidden_layer, len(dataset[0][1])
            model = MyUBal(inp=inp, hid=hid, out=out, betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha,
                           testData=data_test)
            bal_results, acc_train_epoch = model.train_minimal(data_train, max_epoch=epochs)
            #acc_train_per_run.append(acc_train_epoch)
            # Save training and testing accuracies for each repetition
            training_accuracies_per_index[i] = [acc_train_epoch]
            #print(f"res from train_minimal: {acc_train_epoch}")
            err_test, len_test = model.test()
            
            acc_test_per_run.append((len_test - err_test) / len_test * 100)
            print(f"test accuracy:{acc_test_per_run}")
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
            #acc_train_per_run.append(trainingAcc)
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
            #acc_test_per_run.append(testingSucc)
            avg_training_accuracy.append(np.mean(training_accuracies_per_index, axis=0))
        avg_testing_accuracy.append(acc_test_per_run)
        print(avg_testing_accuracy)

    return trainingAcc, testingSucc, avg_training_accuracy, avg_testing_accuracy


def train_ubal_crossval2(dataset, epochs, folds=5, repetitions=1, hidden_layer=100, alpha=0.3, betas=[0.0, 1.0, 0.0],
                    gammasF=[float("nan"), 1.0, 1.0], gammasB=[1.0, 1.0, float("nan")]):
    
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    f = 0
    dataset = np.array(dataset, dtype=object)
    data_X = dataset[:,[0]]
    data_y = dataset[:,[1]]
    data_y_strat = np.argmax(data_y, axis=1)

    results = []  # Store results for each fold

    for train_index, test_index in kfold.split(data_X, data_y_strat):
        fold_results = {'params': {'alpha': alpha, 'betas': betas, 'gammasF': gammasF, 'gammasB': gammasB},
                        'train_accuracies': [], 'test_accuracies': []}
        
        for i in range(repetitions):
            f += 1
            print("Training Fold ", f)
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
    #trainingName = "lh"
    #trainingName = "rh"
    #trainingName = "lf" # THIS IS GOOD DONT RETRAIN
    trainingName = "rf"  #
    #trainingName = "all"  #

    with open("ubal_data/ubal_data_leyla_2/" + trainingName + suffix) as f:
        data = json.load(f)
    labeled = [(du.normalize_and_invert(np.array(i['pos'])), np.array(i["touch"])) for i in data]
    betas = [1.0, 1.0, 0.9]
    gammasF = [float("nan"), 1.0, 1.0]
    gammasB = [0.98, 1.0, float("nan")]
    # acc_train_list, acc_test_list = train_ubal_crossval(labeled, 200)
    results = train_ubal_crossval2(labeled, 5, alpha=0.1, betas=betas, gammasF=gammasF, gammasB=gammasB)
    
    results_file_path = f"data/results/ubal_results_{trainingName}.pickle"

    with open(results_file_path, "wb") as f:
        pickle.dump(results, f)

    for fold_result in results:
        test_accuracies = fold_result['test_accuracies']
        print(f"Test accuracies for this fold: {test_accuracies}")

    avg_test_accuracies = []
    for epoch_index in range(len(results[0]['test_accuracies'][0])):      
        sum_at_index_test = 0

        for fold_result in results:
            test_accuracies = fold_result['test_accuracies']
            value_at_index_test = test_accuracies[0][epoch_index]  # Assuming the test_accuracies list has only one sublist
            sum_at_index_test += value_at_index_test

        avg_value_at_index_test = sum_at_index_test / len(results)
        avg_test_accuracies.append(avg_value_at_index_test)


    for fold_result in results:
        train_accuracies = fold_result['train_accuracies']
        print(f"Train accuracies for this fold: {train_accuracies}")


    avg_train_accuracies = []
    
    for epoch_index in range(len(results[0]['train_accuracies'][0])):      
        sum_at_index_train = 0
        sum_at_index_test = 0

        for fold_result in results:
            train_accuracies = fold_result['train_accuracies']
            value_at_index_train = train_accuracies[0][epoch_index]  # Assuming the train_accuracies list has only one sublist
            sum_at_index_train += value_at_index_train
            
        avg_value_at_index_train = sum_at_index_train / len(results)
        avg_train_accuracies.append(avg_value_at_index_train)

    print("Average train accuracies:", avg_train_accuracies)
    print("Average test accuracies:", avg_test_accuracies)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_accuracies, label='Average Training Accuracy', marker='o')
    plt.plot(avg_test_accuracies, label='Average Testing Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{trainingName}')
    plt.legend()
    plt.grid(True)
    plt.savefig("FINAL_{0}-ce".format(trainingName))
    plt.savefig(f"data/results/ubal_conti_{trainingName}.pdf", format="pdf")
    plt.show()  
    
    #print("Average training accuracy: ",(sum(acc_train_list)/len(acc_train_list)))
    #print("Average test accuracy: ",(sum(acc_test_list)/len(acc_test_list)))
