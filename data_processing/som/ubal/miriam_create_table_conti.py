import pickle
import numpy as np

# Load the results from the pickle file
with open('results_all_grid_hidden_layer_alphas.pkl', 'rb') as f:
    results_dict = pickle.load(f)

# Define the header for the LaTeX table
latex_table = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{|l|l|l|l|l|l|l|}\n\\hline\n"
latex_table += "Model & Hidden Layer Size & Learning Rate & Train Mean & Train Std & Test Mean & Test Std \\\\ \\hline\n"

# Iterate over the results and add rows to the LaTeX table
for key, results in results_dict.items():
    if isinstance(key, tuple):
        model_name, hidden_layer_size, learning_rate = key
        for result in results:
            train_mean = np.mean(result['train_accuracies'])
            train_std = np.std(result['train_accuracies'])
            test_mean = np.mean(result['test_accuracies'])
            test_std = np.std(result['test_accuracies'])
            latex_table += f"{model_name} & {hidden_layer_size} & {learning_rate} & "
            latex_table += f"{train_mean:.2f} & {train_std:.2f} & {test_mean:.2f} & {test_std:.2f} \\\\ \\hline\n"

# Complete the LaTeX table
latex_table += "\\end{tabular}\n\\caption{UBal Results}\n\\label{tab:ubal_results}\n\\end{table}"

# Print the LaTeX table
print(latex_table)
