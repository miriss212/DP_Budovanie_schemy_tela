import pickle

# Load the results from the pickle file
with open('results_all_grid_hidden_layer_alphas.pkl', 'rb') as f:
    results_dict = pickle.load(f)

# Open a LaTeX file to write the table
with open('results_table_all.tex', 'w') as latex_file:
    # Write the table header
    latex_file.write("\\begin{table}[htbp]\n")
    latex_file.write("\\centering\n")
    latex_file.write("\\begin{tabular}{|l|l|l|l|l|l|}\n")
    latex_file.write("\\hline\n")
    latex_file.write("Model & Epoch & Train Mean & Train Std & Test Mean & Test Std \\\\ \n")
    latex_file.write("\\hline\n")
    
    # Loop through each model in the results dictionary
    for model, epochs_data in results_dict.items():
        # Loop through each epoch data for the model
        for epoch, data in epochs_data.items():
            train_mean, train_std = data['train']
            test_mean, test_std = data['test']
            # Write a row of the table
            latex_file.write(f"{model} & {epoch} & {train_mean:.2f} & {train_std:.2f} & {test_mean:.2f} & {test_std:.2f} \\\\ \n")
    
    # Write the table footer
    latex_file.write("\\hline\n")
    latex_file.write("\\end{tabular}\n")
    latex_file.write("\\caption{Results summary}\n")
    latex_file.write("\\label{tab:results}\n")
    latex_file.write("\\end{table}\n")
