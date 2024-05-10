import numpy as np
from som import SOM, SOMLoader, SOMSaver
import os
import sys
import pickle
from train import train_new_som
from experiment_util import create_experiment_sets, load_data
import matplotlib.pyplot as plt

#Plotting each histogram separately
def plot_som_histograms(min_rows, max_rows, min_cols, max_cols, data_folder):
    pickle_file = os.path.join(data_folder, f"soms_and_errors_{min_rows}-{max_cols},{min_cols}-{max_cols}.pickle")

    with open(pickle_file, "rb") as f:
        soms_and_errors = pickle.load(f)

    num_rows = max_rows - min_rows + 1
    num_cols = max_cols - min_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axes = axes.flatten()

    # Iterate over each SOM in the dictionary and plot its winner histogram
    for idx, (som_name, som_data) in enumerate(soms_and_errors.items()):
        som = som_data["som"]
        quant_error_array = som_data["quant_error_array"]

        # Plot the histogram
        ax = axes[idx]
        ax.hist(quant_error_array, bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f"{som_name}")
        ax.set_xlabel("Quantization Error")
        ax.set_ylabel("Frequency")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_winning_neurons_histograms(min_rows, max_rows, min_cols, max_cols, data_folder, out_name):

    pickle_file = os.path.join(data_folder, f"soms_and_errors_{min_rows}-{max_cols},{min_cols}-{max_cols}.pickle")

    with open(pickle_file, "rb") as f:
        soms_and_errors = pickle.load(f)

    # Calculate the number of rows and columns in the grid
    num_rows = max_rows - min_rows + 1
    num_cols = max_cols - min_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 2.5 * num_rows))
    axes = axes.flatten()

    for idx, (som_name, som_data) in enumerate(soms_and_errors.items()):
        som = som_data["som"]
        winning_counts = som.winning_counts

        ax = axes[idx]
        ax.bar(range(len(winning_counts)), winning_counts.values())
        ax.set_title(f"Winner Histogram of {som_name}")
        ax.set_xlabel("Neurons")
        ax.set_ylabel("Number of Wins")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"winner_histogram_{out_name}.pdf", format="pdf")
    plt.show()


def create_experiment(min_rows, max_rows, min_cols, max_cols, data_folder, som_data):
    soms_and_errors = {} 
    for rows in range(min_rows, max_rows + 1):
        for cols in range(min_cols, max_cols + 1):
            # Create folder for the experiment
            experiment_folder = os.path.join(data_folder, f'SOM_{rows}x{cols}')
            if not os.path.exists(experiment_folder):
                os.makedirs(experiment_folder)

            som = train_new_som(rows, cols, som_data)

            som_saver = SOMSaver()
            som_saver.save_som(som=som, name=f'size_exp/SOM_{rows}x{cols}')

            # Store SOM and quantization error in the dictionary
            soms_and_errors[f'SOM_{rows}x{cols}'] = {
                "som": som,
                "quant_error_array": som.quant_error_array,
                "entropy_array": left_som_forearm.entropy_array,
                "winner_diff_array" : left_som_forearm.win_array
            }

    return soms_and_errors

if __name__ == "__main__":
    min_rows = 7
    max_rows = 11
    min_cols = 7
    max_cols = 11
    data_folder = "./data/trained/som/size_exp"  

    data_source = "data/data/leyla_filtered_data.data"
    np.set_printoptions(threshold=sys.maxsize)
    currentDirectory = os.getcwd()
    print(currentDirectory)
    data = load_data(data_source)

    separator = int(len(data) * 0.8)
    som_saver = SOMSaver()

    training_set_left_hand, training_set_left_forearm, testing_set_left_hand, testing_set_left_forearm = create_experiment_sets(data, separator)
    
    left_som_forearm = create_experiment(min_rows, max_rows, min_cols, max_cols, data_folder, training_set_left_hand)

    # Save all SOMs and their quantization error arrays into a single pickle file
    with open(os.path.join(data_folder, f"soms_and_errors_{min_rows}-{max_cols},{min_cols}-{max_cols}.pickle"), "wb") as f:
        pickle.dump(left_som_forearm, f)

    #This is the option for plotting each histogram into separate file
    #plot_som_histograms(min_rows, max_rows, min_cols, max_cols, data_folder)
    out_name = "miriam"
    plot_winning_neurons_histograms(min_rows, max_rows, min_cols, max_cols, data_folder, out_name)
    
