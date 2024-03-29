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
        "alpha_s": 0.02,
        "alpha_f": 0.01,
        "lambda_s": lambda_s,
        "lambda_f": 1,
        "eps": 50,#toto ovplyvnuje pocet epoch
        "trace_interval": 10
    }
    #model.train(inputsData, metric=metric, alpha_s=0.5, alpha_f=0.01, lambda_s=lambda_s,
    #            lambda_f=1, eps=50, trace_interval=10) #na toto dictionary
    model.train(inputsData, **training_parameters)


    """# Calculate metrics at the end of each epoch
    quantization_error_train = model.quant_err()
    winner_diff_train = model.winner_diff()
    entropy_train = model.compute_entropy()
    """
    # Calculate metrics for the testing set if available
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
    
    data_source = "data/data/data_matej_new.data"

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
    #right = np.array([i["rightHandPro"] for i in data])

    left_hand = np.array(left[:, 5:])
    left_forearm = np.array(left[:, :5])

    #right_hand = np.array(right[:, 5:])
    #right_forearm = np.array(right[:, :5])

    """print("THIS WAS THE INPUT DATA TRAINING")
    print(left)

    print("THIS IS THE FOREARM-ONLY DATA INPUT")
    print(left_forearm)"""

    (row_dim, count) = left_forearm.shape
    #print(row_dim)
    #print(count)

    (count_whole, _) = left.shape
    #print(count_whole)
    separator = int(count_whole * 0.8)
    #print("separator is ", separator)

    #Training sets
    #training_set_right_hand = np.array(right_hand[:separator, :]).T
    #training_set_right_forearm = np.array(right_forearm[:separator, :]).T

    training_set_left_hand = np.array(left_hand[:separator, :]).T
    training_set_left_forearm = np.array(left_forearm[:separator, :]).T

    #testing_set_right_hand = np.array(right_hand[separator:, :]).T
    #testing_set_right_forearm = np.array(right_forearm[separator:, :]).T

    testing_set_left_hand = np.array(left_hand[separator:, :]).T
    testing_set_left_forearm = np.array(left_forearm[separator:, :]).T

    """print("this is TRAINING set (left fore)")
    print(training_set_left_forearm)
    print("This is TESTING set (left fore)")
    print(testing_set_left_forearm)"""

    # train & save SOM
    som_saver = SOMSaver()
    ROWS_HAND = 4 #8 musia byt aspon 4, 3 je prilis male
    COLS_HAND = 4 #8

    #skusit mensie rozmery siete, 3x3, atd , 8x8 je uz velke 
    ROWS_FORE = 4#9
    COLS_FORE = 4#12

    #ukladat SOM_info na konci kde je quant error atd do pickle, kniznice
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

    #right_som_forearm = train_new_som(ROWS_FORE, COLS_FORE, training_set_right_forearm)
    #som_saver.save_som(som=right_som_forearm, name="right_forearm")

    #right_som_hand = train_new_som(ROWS_HAND, COLS_HAND, training_set_right_hand)
    #som_saver.save_som(som=right_som_hand, name="right_hand")

    # -------------------

    # ------ EVAL QUANTIZATION ERROR
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

    #error_RF_train = right_som_forearm.quant_err()
    #error_RF_test = right_som_forearm.quant_err(testing_set_right_forearm)
    #error_RH_train = right_som_hand.quant_err()
    #error_RH_test = right_som_hand.quant_err(testing_set_right_hand)

    #save info
    file_name = "SOM_trained_info_nove_data_04032024.txt"
    f = open(file_name, "w")
    f.write('Quantization error of left hand: {} [train], {} [test]'.format(error_LH_train, error_LH_test))
    f.write("\n")
    f.write('"Quantization error of left forearm: {} [train], {} [test]'.format(error_LF_train, error_LF_test))
    f.write("\n")
    f.write('Winner differentiation of left hand: {} [train], {} [test]'.format(dw_LH_train, dw_LH_test))
    f.write("\n")
    f.write('"Winner differentiation of left forearm: {} [train], {} [test]'.format(dw_LF_train, dw_LF_test))
    f.write("\n")
    f.write('"Enthropy of left forearm: {} [train], {} [test]'.format(entropy_LH_train, entropy_LH_test))
    f.write("\n")
    f.write('Enthropy of left hand: {} [train], {} [test]'.format(entropy_LF_train, entropy_LF_test))
    f.write("\n")
    
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

    #quantization error, winner diferentiation, enthropy v experimentoch
    #experimenty pre kazdu ruku zvlast
    #somka lava ruka, IBA LAVA RUKA, experiment kde sa meni rychlost ucenia, 5 hodnot budem skusat, budem skusat aj roznu startovu a roznu cielovu
    #vypis za kazdou epochou a tam vypisem kvantizacnu chybu, wd a e
    
