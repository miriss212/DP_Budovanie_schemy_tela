import sys
import numpy as np
import json
import os

try:
    from .som import *
except Exception: #ImportError
    from som import *

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
    # Define the parameters
    training_parameters = {
        "metric": metric,
        "alpha_s": 1,
        "alpha_f": 0.1,
        "lambda_s": lambda_s,
        "lambda_f": 1,
        "eps": 2500,#toto ovplyvnuje pocet epoch
        "trace_interval": 10
    }
    
    model.train(inputsData, **training_parameters)
    

    if inputsData is not None:
        quantization_error_test = model.quant_err()
        print(quantization_error_test)
        winner_diff_test = model.winner_diff()
        print(winner_diff_test)
        entropy_test = model.compute_entropy()
        print(entropy_test)
    else:
        quantization_error_test = None
        winner_diff_test = None
        entropy_test = None
    
   
    return model


def train_new_som_lr(rows, cols, inputsData, alpha_s, alpha_f):
    metric = L_2
    (dim, count) = inputsData.shape

    top_left = np.array((0, 0))
    bottom_right = np.array((rows - 1, cols - 1))
    lambda_s = metric(top_left, bottom_right) * 0.3

    model = SOM(dim, rows, cols, inputsData)
    # Define the parameters
    training_parameters = {
        "metric": metric,
        "alpha_s": alpha_s,
        "alpha_f": alpha_f,
        "lambda_s": lambda_s,
        "lambda_f": 1,
        "eps": 50,#toto ovplyvnuje pocet epoch
        "trace_interval": 10
    }
    
    model.train(inputsData, **training_parameters)

    if inputsData is not None:
        quantization_error_test = model.quant_err()
        print(quantization_error_test)
        winner_diff_test = model.winner_diff()
        print(winner_diff_test)
        entropy_test = model.compute_entropy()
        print(entropy_test)
    else:
        quantization_error_test = None
        winner_diff_test = None
        entropy_test = None
    
   
    return model

if __name__ == "__main__":
    # data_source = "../my_data.data" #"../my_data_act.data"
    #data_source = "../data_leyla2.data" #"../my_data_act.data"
    #data_source = "data/data/data_leyla.data"
    
    data_source = "data/data/leyla_filtered_data.data"
    name_for_pickle = input(f'Insert name for pickle:')


    """
    Training of arm SOM
    """

    np.set_printoptions(threshold=sys.maxsize)
    currentDirectory = os.getcwd()
    #print(currentDirectory)
    f = open(data_source)
    data = json.load(f)
    f.close()

    #print("\n{}\n".format(len(data)))

    left = np.array([i["leftHandPro"] for i in data])
    right = np.array([i["rightHandPro"] for i in data])

    left_hand = np.array(left[:, 5:])
    left_forearm = np.array(left[:, :5])

    right_hand = np.array(right[:, 5:])
    right_forearm = np.array(right[:, :5])

    """print("THIS WAS THE INPUT DATA TRAINING")
    print(left)

    print("THIS IS THE FOREARM-ONLY DATA INPUT")
    print(left_forearm)"""

    (row_dim, count) = right_forearm.shape
    #print(row_dim)
    #print(count)

    (count_whole, _) = right.shape
    #print(count_whole)
    separator = int(count_whole * 0.8)
    #print("separator is ", separator)

    #Training sets
    training_set_right_hand = np.array(right_hand[:separator, :]).T
    training_set_right_forearm = np.array(right_forearm[:separator, :]).T

    training_set_left_hand = np.array(left_hand[:separator, :]).T
    training_set_left_forearm = np.array(left_forearm[:separator, :]).T

    testing_set_right_hand = np.array(right_hand[separator:, :]).T
    testing_set_right_forearm = np.array(right_forearm[separator:, :]).T

    testing_set_left_hand = np.array(left_hand[separator:, :]).T
    testing_set_left_forearm = np.array(left_forearm[separator:, :]).T

    """print("this is TRAINING set (left fore)")
    print(training_set_left_forearm)
    print("This is TESTING set (left fore)")
    print(testing_set_left_forearm)"""

    # train & save SOM
    som_saver = SOMSaver()
    ROWS_HAND = 10 #8 musia byt aspon 4, 3 je prilis male
    COLS_HAND = 10 #8

    #skusit mensie rozmery siete, 3x3, atd , 8x8 je uz velke 
    ROWS_FORE = 10#9
    COLS_FORE = 10#12

    #### ---------- LEFT
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

    #### ---------- RIGHT

    right_som_forearm = train_new_som(ROWS_FORE, COLS_FORE, training_set_right_forearm)
    som_saver.save_som(som=right_som_forearm, name="right_forearm")

    right_som_hand = train_new_som(ROWS_HAND, COLS_HAND, training_set_right_hand)
    som_saver.save_som(som=right_som_hand, name="right_hand")


    """,
        "left_hand": {
            "som": left_som_hand,
            "quant_error_array": left_som_hand.quant_error_array,
            "entropy_array": left_som_hand.entropy_array,
            "winner_diff_array" : left_som_hand.win_array
        },
        "right_forearm": {
            "som": right_som_forearm,
            "quant_error_array": right_som_forearm.quant_error_array,
            "entropy_array": right_som_forearm.entropy_array,
            "winner_diff_array" : right_som_forearm.win_array
        },
        "right_hand": {
            "som": right_som_hand,
            "quant_error_array": right_som_hand.quant_error_array,
            "entropy_array": right_som_hand.entropy_array,
            "winner_diff_array" : right_som_hand.win_array
        }"""
# Save the SOMs and their quantization error arrays into a pickle file
    soms_and_errors = {
        "left_forearm": {
            "som": left_som_forearm,
            "quant_error_array": left_som_forearm.quant_error_array,
            "entropy_array": left_som_forearm.entropy_array,
            "winner_diff_array" : left_som_forearm.win_array
        },
        "left_hand": {
            "som": left_som_hand,
            "quant_error_array": left_som_hand.quant_error_array,
            "entropy_array": left_som_hand.entropy_array,
            "winner_diff_array" : left_som_hand.win_array
        },
        "right_forearm": {
            "som": right_som_forearm,
            "quant_error_array": right_som_forearm.quant_error_array,
            "entropy_array": right_som_forearm.entropy_array,
            "winner_diff_array" : right_som_forearm.win_array
        },
        "right_hand": {
            "som": right_som_hand,
            "quant_error_array": right_som_hand.quant_error_array,
            "entropy_array": right_som_hand.entropy_array,
            "winner_diff_array" : right_som_hand.win_array
        }
    }

    #left_som_hand.plot_winner_histogram(left_som_hand)

    with open(f"data/results/pickles/{name_for_pickle}.pickle", "wb") as f:
        pickle.dump(soms_and_errors, f)
