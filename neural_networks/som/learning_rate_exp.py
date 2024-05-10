import numpy as np
from som import SOM, SOMLoader, SOMSaver
import os
import sys
import pickle
from train import train_new_som_lr
from experiment_util import create_experiment_sets, load_data
import matplotlib.pyplot as plt


def create_experiment(rows, cols, data_folder, som_data, learning_rates):
    soms_and_errors = {} 
    
    for lr in learning_rates:
        
        experiment_folder = os.path.join(data_folder, f'SOM_{rows}x{cols}_lr{lr}')
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)

        som = train_new_som_lr(rows, cols, som_data, lr) #toto uz nebude fungovat

        som_saver = SOMSaver()
        som_saver.save_som(som=som, name=f'lr_exp/SOM_{rows}x{cols}_lr{lr}')

        # Store SOM and quantization error in the dictionary
        som_name = f'SOM_{rows}x{cols}_lr{lr}'
        soms_and_errors[som_name] = {
            "som": som,
            "quant_error_array": som.quant_error_array,
            "learning_rate": lr,
            "entropy_array": som.entropy_array,
            "winner_diff_array" : som.win_array
        }

    return soms_and_errors

if __name__ == "__main__":
    
    cols = 8
    rows = 8
    data_folder = "./data/trained/som/lr_exp"  

    data_source = "data/data/leyla_filtered_data.data"
    np.set_printoptions(threshold=sys.maxsize)
    currentDirectory = os.getcwd()
    print(currentDirectory)
    data = load_data(data_source)

    separator = int(len(data) * 0.8)
    som_saver = SOMSaver()

    training_set_left_hand, training_set_left_forearm, testing_set_left_hand, testing_set_left_forearm = create_experiment_sets(data, separator)
    
    # Define learning rates to experiment with
    learning_rates = [0.1, 0.01, 0.001]
    alpha_s_values = [0.1, 0.5, 1.0]
    alpha_f_values = [0.001, 0.01, 0.1]

    lr_exp_data = create_experiment(rows, cols, data_folder, training_set_left_hand, learning_rates)

    # Save all SOMs and their quantization error arrays into a single pickle file
    with open(os.path.join(data_folder, f"soms_and_errors_lr_exp_{learning_rates}.pickle"), "wb") as f:
        pickle.dump(lr_exp_data, f)

    
