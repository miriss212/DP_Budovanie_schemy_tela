import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('neural_networks/som')
from som import SOM
import statistics
import histogram_for_QE_for_size_exp as se

#soms_and_errors_lr_exp_[0.1, 0.5, 1.0]_[0.001, 0.01, 0.1]


def compute_average_quantization_error_lr(data_folder, alpha_s_values, alpha_f_values):
    # Load the pickled data
    with open(os.path.join(data_folder, f"soms_and_errors_lr_exp_{alpha_s_values}_{alpha_f_values}.pickle"), "rb") as f:
        soms_and_errors = pickle.load(f)

    # Dictionary to store average quantization errors for each epoch of each SOM size
    average_quantization_errors = {}
    std_dev_quantization_errors = {}

    # Iterate over the loaded data and organize SOMs based on their parameters
    for som_id, som_data in soms_and_errors.items():
        rows_cols = som_id.split('_')[1].split('x')  # Extract rows and cols from SOM identifier
        print(rows_cols)
        rows, cols = rows_cols[0], rows_cols[1]

        quant_error_array = som_data["quant_error_array"]

        # Iterate over each epoch and calculate the average quantization error
        for epoch, q_error in enumerate(quant_error_array):
            if (rows, cols, epoch) not in average_quantization_errors:
                average_quantization_errors[(rows, cols, epoch)] = []
            average_quantization_errors[(rows, cols, epoch)].append(q_error)

    # Calculate the average quantization error for each epoch of each SOM size
    for key, value in average_quantization_errors.items():
        average_quantization_errors[key] = np.mean(value)
        std_dev_quantization_errors[key] = statistics.stdev(value)
    # Organize average quantization errors into arrays for each size combination
    avg_quantization_arrays = {}
    std_dev_quantization_arrays = {}

    for key, value in average_quantization_errors.items():
        size = (key[0], key[1])
        epoch = key[2]
        if size not in avg_quantization_arrays:
            max_epoch = max(average_quantization_errors.keys(), key=lambda x: x[2])[2]  # Get the maximum epoch
            avg_quantization_arrays[size] = np.zeros(max_epoch + 1)  # Initialize array with the maximum epoch
            std_dev_quantization_arrays[size] = np.zeros(max_epoch + 1)
        avg_quantization_arrays[size][epoch] = value
        std_dev_quantization_arrays[size][epoch] = std_dev_quantization_errors[key]

    return avg_quantization_arrays, std_dev_quantization_arrays



if __name__ == "__main__":
    data_folder = "./data/trained/som/lr_exp"  

    alpha_s_values = [0.1, 0.5, 1.0]
    alpha_f_values = [0.001, 0.01, 0.1]

    out_name = time.time()
    avg_quantization_arrays, std_dev_quantization_arrays = compute_average_quantization_error_lr(data_folder, alpha_s_values, alpha_f_values)
   
    num_epochs = 50
    se.create_stdev_table(std_dev_quantization_arrays, num_epochs, out_name)
    se.create_avg_error_table(avg_quantization_arrays, num_epochs, out_name)
   # Plotting example
    for size, errors in avg_quantization_arrays.items():
        plt.plot(range(len(errors)), errors, label=f"{size[0]}x{size[1]}")
    plt.xlabel("Epoch")
    plt.ylabel("Average Quantization Error")
    #plt.title("Average Quantization Error vs. Epoch")

    plt.grid(True)  
    plt.legend()

    plt.savefig(f"data/results/lr_exp_histo_{out_name}.pdf", format="pdf")
    plt.show()
