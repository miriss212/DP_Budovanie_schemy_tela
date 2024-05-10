import pickle

def load_results(file_path):
    with open(file_path, 'rb') as f:
        results_dict = pickle.load(f)
    return results_dict

def format_data(results_dict):
    # Format the data as needed for the LaTeX table
    # For example, extract relevant information like model names, alphas, hidden layers, and corresponding accuracies
    formatted_data = []
    for key, value in results_dict.items():
        model_name, alpha, hidden_layer = key
        for epoch, acc_data in value.items():
            train_acc, train_std = acc_data['train']
            test_acc, test_std = acc_data['test']
            train_acc = round(train_acc, 5)
            train_std = round(train_std, 5)
            test_acc = round(test_acc, 5)
            test_std = round(test_std, 5)
            formatted_data.append((model_name, alpha, hidden_layer, epoch, train_acc, train_std, test_acc, test_std))
    return formatted_data

def generate_latex_table(formatted_data):
    # Generate the LaTeX code for the table
    latex_code = "\\begin{table}[htbp]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n"
    latex_code += "\\hline\n"
    latex_code += "Model & Alpha & Hidden Layer & Epoch & Train Acc & Train Std & Test Acc & Test Std \\\\\n"
    latex_code += "\\hline\n"
    for row in formatted_data:
        latex_code += " & ".join(map(str, row)) + " \\\\\n"
    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\caption{Results}\n"
    latex_code += "\\label{tab:results}\n"
    latex_code += "\\end{table}\n"
    return latex_code

def write_latex_file(latex_code, file_path):
    # Write the LaTeX code to a file
    with open(file_path, 'w') as f:
        f.write(latex_code)

if __name__ == "__main__":
    # Load the results data
    results_dict = load_results('results_all_grid_hidden_layer_alphas.pkl')

    # Format the data
    formatted_data = format_data(results_dict)

    # Generate the LaTeX table
    latex_code = generate_latex_table(formatted_data)

    # Write LaTeX code to file
    write_latex_file(latex_code, 'data/results/results_table.tex')
