import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('neural_networks/som')
from som import SOM
import statistics


def compute_average_quantization_error(data_folder, min_rows, max_rows, min_cols, max_cols):

    with open(os.path.join(data_folder, f"soms_and_errors_{min_rows}-{max_cols},{min_cols}-{max_cols}.pickle"), "rb") as f:
        soms_and_errors = pickle.load(f)

    average_quantization_errors = {}
    std_dev_quantization_errors = {}

    # Iterate over the loaded data and organize SOMs based on their parameters
    for som_id, som_data in soms_and_errors.items():
        rows_cols = som_id.split('_')[1].split('x')  # Extract rows and cols from SOM identifier
        rows, cols = int(rows_cols[0]), int(rows_cols[1])

        quant_error_array = som_data["quant_error_array"]

        # Iterate over each epoch and calculate the average quantization error
        for epoch, q_error in enumerate(quant_error_array):
            if (rows, cols, epoch) not in average_quantization_errors:
                average_quantization_errors[(rows, cols, epoch)] = []
            average_quantization_errors[(rows, cols, epoch)].append(q_error)

    #Calculate standard deviation and average quantization error
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

def create_stdev_table(st_dev_quntization_arrays, num_epochs, output_name):
    # Generate LaTeX code for the table of standard deviation
    num_sizes = len(st_dev_quntization_arrays)
    
    table_std_dev_tex = "\\begin{table}[htbp]\n"
    table_std_dev_tex += "\\centering\n"
    table_std_dev_tex += "\\caption{Standard Deviation of Quantization Error for Each Epoch}\n"
    table_std_dev_tex += "\\begin{tabular}{|l|" + "|".join(["l"] * num_sizes) + "|}\n"
    table_std_dev_tex += "\\hline\n"
    table_std_dev_tex += "Ep & " + " & ".join([f"{size[0]}x{size[1]}" for size in st_dev_quntization_arrays.keys()]) + " \\\\\n"
    table_std_dev_tex += "\\hline\n"

    for epoch in range(num_epochs):
        table_std_dev_tex += f"Ep {epoch} & "
        for size, std_dev_errors in st_dev_quntization_arrays.items():
            table_std_dev_tex += f"{std_dev_errors[epoch]:.4f} & "
        table_std_dev_tex = table_std_dev_tex[:-2] + " \\\\\n"
    table_std_dev_tex += "\\hline\n"
    table_std_dev_tex += "\\end{tabular}\n"
    table_std_dev_tex += "\\end{table}"

    # Save the LaTeX code for standard deviation to a .tex file
    with open(f"data/results/std_dev_quantization_error_table_{output_name}.tex", "w") as file:
        file.write(table_std_dev_tex)

    print("LaTeX table code for standard deviation has been successfully saved to a file.")

def create_avg_error_table(avg_quantization_arrays, num_epochs, output_name):
     # Generate LaTeX code for the table
    num_sizes = len(avg_quantization_arrays)
    #num_epochs = 100
    table_tex = "\\begin{table}[htbp]\n"
    table_tex += "\\centering\n"
    table_tex += "\\caption{SOM Size and Error for Each Epoch}\n"
    table_tex += "\\begin{tabular}{|l|" + "|".join(["l"] * num_sizes) + "|}\n"
    table_tex += "\\hline\n"
    table_tex += "Ep & " + " & ".join([f"{size[0]}x{size[1]}" for size in avg_quantization_arrays.keys()]) + " \\\\\n"
    table_tex += "\\hline\n"

    for epoch in range(num_epochs):
        table_tex += f"Ep {epoch} & "
        for size, errors in avg_quantization_arrays.items():
            table_tex += f"{errors[epoch]:.4f} & "
        table_tex = table_tex[:-2] + " \\\\\n"
    table_tex += "\\hline\n"
    table_tex += "\\end{tabular}\n"
    table_tex += "\\end{table}"

    # Save the LaTeX code to a .tex file
    with open(f"data/results/quantization_error_table_{output_name}.tex", "w") as file:
        file.write(table_tex)

    print("LaTeX table code has been successfully saved to a file.")


if __name__ == "__main__":
    data_folder = "data/trained/som/size_exp"  
    min_rows = 8
    max_rows = 10
    min_cols = 8
    max_cols = 10
    out_name = time.time()
    avg_quantization_arrays, std_dev_quantization_arrays = compute_average_quantization_error(data_folder, min_rows, max_rows, min_cols, max_cols)
    
    create_stdev_table(std_dev_quantization_arrays, 100, out_name)
    create_avg_error_table(avg_quantization_arrays, 100, out_name)
    
    # Plotting example
    for size, errors in avg_quantization_arrays.items():
        plt.plot(range(len(errors)), errors, label=f"{size[0]}x{size[1]}")
    plt.xlabel("Epoch")
    plt.ylabel("Average Quantization Error")
    plt.grid(True)  
    plt.legend()
    result_path = "data/results/"  
    plt.savefig(f"data/results/size_exp_histo_{out_name}.pdf", format="pdf")
    plt.show()
