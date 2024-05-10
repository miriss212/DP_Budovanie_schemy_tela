
import sys
import json
import os
import numpy as np
from som import SOM, SOMSaver
from train import train_new_som

def load_data(data_source):
    with open(data_source, 'r') as f:
        data = json.load(f)
    return data

def create_experiment_sets(left_data, separator):

    left_hand = np.array([i["leftHandPro"][5:] for i in left_data])
    left_forearm = np.array([i["leftHandPro"][:5] for i in left_data])

    # Training sets
    training_set_left_hand = np.array(left_hand[:separator, :]).T
    training_set_left_forearm = np.array(left_forearm[:separator, :]).T

    # Testing sets
    testing_set_left_hand = np.array(left_hand[separator:, :]).T
    testing_set_left_forearm = np.array(left_forearm[separator:, :]).T

    return training_set_left_hand, training_set_left_forearm, testing_set_left_hand, testing_set_left_forearm

def run_experiment(alpha_s, alpha_f, lambda_s, lambda_f, eps, separator):
    training_set_left_hand, training_set_left_forearm, testing_set_left_hand, testing_set_left_forearm = create_experiment_sets(left_data, separator)

    ROWS_HAND = 8
    COLS_HAND = 8

    ROWS_FORE = 9
    COLS_FORE = 12

    left_som_hand = train_new_som(ROWS_HAND, COLS_HAND, training_set_left_hand)
    som_saver.save_som(som=left_som_hand, name="left_hand")

    left_som_forearm = train_new_som(ROWS_FORE, COLS_FORE, training_set_left_forearm)
    som_saver.save_som(som=left_som_forearm, name="left_forearm")

    # Record Results
    error_LH_train = left_som_hand.quant_err()
    error_LH_test = left_som_hand.quant_err(testing_set_left_hand)
    dw_LH_train = left_som_hand.winner_diff(data=training_set_left_hand)
    dw_LH_test = left_som_hand.winner_diff(testing_set_left_hand)
    entropy_LH_train = left_som_hand.compute_entropy(data=training_set_left_hand)
    entropy_LH_test = left_som_hand.compute_entropy(data=testing_set_left_hand)

    error_LF_train = left_som_forearm.quant_err()
    error_LF_test = left_som_forearm.quant_err(testing_set_left_forearm)
    dw_LF_train = left_som_forearm.winner_diff(data=training_set_left_forearm)
    dw_LF_test = left_som_forearm.winner_diff(testing_set_left_forearm)
    entropy_LF_train = left_som_forearm.compute_entropy(data=training_set_left_forearm)
    entropy_LF_test = left_som_forearm.compute_entropy(data=testing_set_left_forearm)

    # vyslekdy do subora
    results_filename = f"experiment_results_{alpha_s}_{alpha_f}_{lambda_s}_{lambda_f}_{eps}.txt"
    with open(results_filename, 'w') as f:
        f.write('Quantization error of left hand: {} [train], {} [test]\n'.format(error_LH_train, error_LH_test))
        f.write('Winner differentiation of left hand: {} [train], {} [test]\n'.format(dw_LH_train, dw_LH_test))
        f.write('Enthropy of left hand: {} [train], {} [test]\n'.format(entropy_LH_train, entropy_LH_test))
        f.write('Quantization error of left forearm: {} [train], {} [test]\n'.format(error_LF_train, error_LF_test))
        f.write('Winner differentiation of left forearm: {} [train], {} [test]\n'.format(dw_LF_train, dw_LF_test))
        f.write('Enthropy of left forearm: {} [train], {} [test]\n'.format(entropy_LF_train, entropy_LF_test))

if __name__ == "__main__":
    data_source = "data/data/data_leyla.data"
    np.set_printoptions(threshold=sys.maxsize)
    currentDirectory = os.getcwd()
    print(currentDirectory)
    left_data = load_data(data_source)

    separator = int(len(left_data) * 0.8)
    som_saver = SOMSaver()

    alpha_s_values = [0.1, 0.5, 1.0]
    alpha_f_values = [0.001, 0.01, 0.1]
    lambda_s_values = [0.1, 0.5, 1.0]
    lambda_f_values = [0.1, 1.0, 2.0]
    eps_values = [10, 20, 30]

    for alpha_s in alpha_s_values:
        for alpha_f in alpha_f_values:
            for lambda_s in lambda_s_values:
                for lambda_f in lambda_f_values:
                    for eps in eps_values:
                        run_experiment(alpha_s, alpha_f, lambda_s, lambda_f, eps, separator)
