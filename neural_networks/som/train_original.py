import sys
import numpy as np
import json
import os

try:
    from .som import *
except Exception: #ImportError
    from som import *

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
        "inputsData": inputsData,
        "metric": metric,
        "alpha_s": 0.5,
        "alpha_f": 0.01,
        "lambda_s": lambda_s,
        "lambda_f": 1,
        "eps": 50,
        "trace_interval": 10
    }
    #model.train(inputsData, metric=metric, alpha_s=0.5, alpha_f=0.01, lambda_s=lambda_s,
    #            lambda_f=1, eps=50, trace_interval=10) #na toto dictionary
    model.train(**training_parameters)
    return model

if __name__ == "__main__":
    # data_source = "../my_data.data" #"../my_data_act.data"
    #data_source = "../data_leyla2.data" #"../my_data_act.data"
    data_source = "D:/DP_Budovanie_schemy_tela/data/data/data_leyla.data"
    """
    Training of arm SOM
    """
    np.set_printoptions(threshold=sys.maxsize)
    currentDirectory = os.getcwd()
    print(currentDirectory)
    f = open(data_source)
    data = json.load(f)
    f.close()

    print("\n{}\n".format(len(data)))

    left = np.array([i["leftHandPro"] for i in data])
    right = np.array([i["rightHandPro"] for i in data])

    left_hand = np.array(left[:, 5:])
    left_forearm = np.array(left[:, :5])

    right_hand = np.array(right[:, 5:])
    right_forearm = np.array(right[:, :5])

    print("THIS WAS THE INPUT DATA TRAINING")
    print(left)

    print("THIS IS THE FOREARM-ONLY DATA INPUT")
    print(left_forearm)

    (row_dim, count) = left_forearm.shape
    print(row_dim)
    print(count)

    (count_whole, _) = left.shape
    print(count_whole)
    separator = int(count_whole * 0.8)
    print("separator is ", separator)

    #Training sets
    training_set_right_hand = np.array(right_hand[:separator, :]).T
    training_set_right_forearm = np.array(right_forearm[:separator, :]).T

    training_set_left_hand = np.array(left_hand[:separator, :]).T
    training_set_left_forearm = np.array(left_forearm[:separator, :]).T

    testing_set_right_hand = np.array(right_hand[separator:, :]).T
    testing_set_right_forearm = np.array(right_forearm[separator:, :]).T

    testing_set_left_hand = np.array(left_hand[separator:, :]).T
    testing_set_left_forearm = np.array(left_forearm[separator:, :]).T

    print("this is TRAINING set (left fore)")
    print(training_set_left_forearm)
    print("This is TESTING set (left fore)")
    print(testing_set_left_forearm)

    # train & save SOM
    som_saver = SOMSaver()
    ROWS_HAND = 8
    COLS_HAND = 8

    ROWS_FORE = 9
    COLS_FORE = 12

    #### ---------- LEFT
    left_som_forearm = train_new_som(ROWS_FORE, COLS_FORE, training_set_left_forearm)
    som_saver.save_som(som=left_som_forearm, name="left_forearm")

    left_som_hand = train_new_som(ROWS_HAND, COLS_HAND, training_set_left_hand)
    som_saver.save_som(som=left_som_hand, name="left_hand")

    #### ---------- RIGHT

    right_som_forearm = train_new_som(ROWS_FORE, COLS_FORE, training_set_right_forearm)
    som_saver.save_som(som=right_som_forearm, name="right_forearm")

    right_som_hand = train_new_som(ROWS_HAND, COLS_HAND, training_set_right_hand)
    som_saver.save_som(som=right_som_hand, name="right_hand")

    # -------------------

    # ------ EVAL QUANTIZATION ERROR
    error_LF_train = left_som_forearm.quant_err()
    error_LF_test = left_som_forearm.quant_err(testing_set_left_forearm)
    error_LH_train = left_som_hand.quant_err()
    error_LH_test = left_som_hand.quant_err(testing_set_left_hand)

    #error_RF_train = right_som_forearm.quant_err()
    #error_RF_test = right_som_forearm.quant_err(testing_set_right_forearm)
    #error_RH_train = right_som_hand.quant_err()
    #error_RH_test = right_som_hand.quant_err(testing_set_right_hand)

    #save info
    file_name = "SOM_trained_info.txt"
    f = open(file_name, "w")
    f.write('Quantization error of left hand: {} [train], {} [test]'.format(error_LH_train, error_LH_test))
    f.write("\n")
    f.write('"Quantization error of left forearm: {} [train], {} [test]'.format(error_LF_train, error_LF_test))
    f.write("\n")
    #f.write('Quantization error of right hand: {} [train], {} [test]'.format(error_RH_train, error_RH_test))
    #f.write("\n")
    #f.write('"Quantization error of right forearm: {} [train], {} [test]'.format(error_RF_train, error_RF_test))
    #f.write("\n")
    #f.close()
    #quantization error, winner diferentiation, enthropy v experimentoch
    #experimenty pre kazdu ruku zvlast
    #somka lava ruka, IBA LAVA RUKA, experiment kde sa meni rychlost ucenia, 5 hodnot budem skusat, budem skusat aj roznu startovu a roznu cielovu
    #vypis za kazdou epochou a tam vypisem kvantizacnu chybu, wd a e
    #