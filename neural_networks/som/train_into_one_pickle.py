import sys
import numpy as np
import json
import os
import pickle

try:
    from .som_ukladacia_do_pickle import *
except Exception: #ImportError
    from som_ukladacia_do_pickle import *

def humanreadible_runtime(runtime):
    m, s = divmod(runtime, 60)
    h, m = divmod(m, 60)
    return '{:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s))

def L_p(u, v, p, axis=0):
    return np.sum(np.abs(u-v)**p, axis=axis) ** (1/p)

def L_2(u, v, axis=0):
    return L_p(u, v, p=2, axis=axis)

def train_new_som(rows, cols, inputsData):
    metric = L_2
    (dim, count) = inputsData.shape

    top_left = np.array((0, 0))
    bottom_right = np.array((rows - 1, cols - 1))
    lambda_s = metric(top_left, bottom_right) * 0.3

    model = SOM(dim, rows, cols, inputsData)
    training_parameters = {
        "metric": metric,
        "alpha_s": 0.02,
        "alpha_f": 0.01,
        "lambda_s": lambda_s,
        "lambda_f": 1,
        "eps": 100,
        "trace_interval": 10
    }
    model.train(inputsData, **training_parameters)


    return model

if __name__ == "__main__":
    data_source = "data/data/data_matej_new.data"
    np.set_printoptions(threshold=sys.maxsize)
    currentDirectory = os.getcwd()
    f = open(data_source)
    data = json.load(f)
    f.close()

    left = np.array([i["leftHandPro"] for i in data])
    left_hand = np.array(left[:, 5:])
    left_forearm = np.array(left[:, :5])

    (row_dim, count) = left_forearm.shape
    (count_whole, _) = left.shape
    separator = int(count_whole * 0.8)

    training_set_left_hand = np.array(left_hand[:separator, :]).T
    training_set_left_forearm = np.array(left_forearm[:separator, :]).T

    testing_set_left_hand = np.array(left_hand[separator:, :]).T
    testing_set_left_forearm = np.array(left_forearm[separator:, :]).T

    som_saver = SOMSaver()
    ROWS_HAND = 4 
    COLS_HAND = 4 
    ROWS_FORE = 4
    COLS_FORE = 4

    sim_start_time = time.time()
    left_som_forearm = train_new_som(ROWS_FORE, COLS_FORE, training_set_left_forearm)
    sim_end_time = time.time()
    print("Total training time forearm: {} seconds".format(humanreadible_runtime(sim_end_time - sim_start_time)))
    som_saver.save_som(som=left_som_forearm, name="left_forearm")

    sim_start_time = time.time()
    left_som_hand = train_new_som(ROWS_HAND, COLS_HAND, training_set_left_hand)
    sim_end_time = time.time()
    print("Total training time hand: {} seconds".format(humanreadible_runtime(sim_end_time - sim_start_time)))
    som_saver.save_som(som=left_som_hand, name="left_hand")

    error_LF_train = left_som_forearm.quant_err()
    error_LF_test = left_som_forearm.quant_err(testing_set_left_forearm)
    error_LH_train = left_som_hand.quant_err()
    error_LH_test = left_som_hand.quant_err(testing_set_left_hand)

    dw_LF_train = left_som_forearm.winner_diff(data=training_set_left_forearm)
    dw_LF_test = left_som_forearm.winner_diff(testing_set_left_forearm)
    dw_LH_train = left_som_hand.winner_diff(data=training_set_left_hand)
    dw_LH_test = left_som_hand.winner_diff(testing_set_left_hand)

    entropy_LF_train = left_som_forearm.compute_entropy(data=training_set_left_forearm)
    entropy_LF_test = left_som_forearm.compute_entropy(data=testing_set_left_forearm)
    entropy_LH_train = left_som_hand.compute_entropy(data=training_set_left_hand)
    entropy_LH_test = left_som_hand.compute_entropy(data=testing_set_left_hand)

    # Save the SOMs and their quantization error arrays into a pickle file
    soms_and_errors = {
        "left_forearm": {
            "som": left_som_forearm,
            "quant_error_array": left_som_forearm.quant_error_array
        },
        "left_hand": {
            "som": left_som_hand,
            "quant_error_array": left_som_hand.quant_error_array
        }
    }

    with open("soms_and_errors.pickle", "wb") as f:
        pickle.dump(soms_and_errors, f)
