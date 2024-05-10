import pickle
import matplotlib.pyplot as plt
import statistics

def plot_results(results_dict):
    fig, ax = plt.subplots(len(results_dict), 1, figsize=(10, 6*len(results_dict)))
    for i, ((training_name, k), results) in enumerate(results_dict.items()):
        epochs = list(results['train'].keys())
        train_means = [mean for mean, _ in results['train'].values()]
        #train_stds = [std for _, std in results['train'].values()]
        test_means = [mean for mean, _ in results['test'].values()]
        #test_stds = [std for _, std in results['test'].values()]
        
        ax[i].errorbar(epochs, train_means, label="Train", fmt='-o')
        ax[i].errorbar(epochs, test_means, label="Test", fmt='-o')
        ax[i].set_title(f"Training and Testing Results for {training_name} with K={k}")
        ax[i].set_xlabel("Epochs")
        ax[i].set_ylabel("Accuracy (%)")
        ax[i].legend()
        ax[i].grid(True)

    plt.tight_layout()
    plt.savefig("exp_k_lf.pdf", format="pdf")
    plt.show()
def load_results(file_path):
    with open(file_path, "rb") as pickle_file:
        return pickle.load(pickle_file)

def results_to_latex(results_dict, file_path):
    latex_table = "\\begin{table}[h!]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{|c|c|c|c|}\n"
    latex_table += "\\hline\n"
    latex_table += "Training Name & K & Epoch & Train Accuracy (Mean, Std) & Test Accuracy (Mean, Std)\\\\\n"
    latex_table += "\\hline\n"
    
    for (training_name, k), results in results_dict.items():
        for epoch, epoch_results in results['train'].items():
            train_mean, train_std = epoch_results
            test_mean, test_std = results['test'][epoch]
            latex_table += f"{training_name} & {k} & {epoch} & ({train_mean:.2f}, {train_std:.2f}) & ({test_mean:.2f}, {test_std:.2f})\\\\\n"
    
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Training and Testing Results}\n"
    latex_table += "\\label{tab:results}\n"
    latex_table += "\\end{table}"

    # Write LaTeX table to a .tex file
    with open(file_path, "w") as f:
        f.write(latex_table)


def results_to_latex_new(results_dict, file_path):
    latex_table = "\\begin{table}[h!]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{|c|c|c|c|c|}\n"
    latex_table += "\\hline\n"
    latex_table += "K & Train Accuracy (Mean, Std) & Test Accuracy (Mean, Std)\\\\\n"
    latex_table += "\\hline\n"
    
    for k, (train_mean, test_mean, train_std, test_std) in results_dict.items():
        latex_table += f"{k} & ({train_mean:.2f} $\pm$ {train_std:.2f}) & ({test_mean:.2f} $\pm$ {test_std:.2f})\\\\\n"
    
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Training and Testing Results}\n"
    latex_table += "\\label{tab:results}\n"
    latex_table += "\\end{table}"

    # Write LaTeX table to a .tex file
    with open(file_path, "w") as f:
        f.write(latex_table)
def generate_latex_table(results_dict):
    latex_table = "\\begin{table}[h!]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{|c|c|c|c|c|}\n"
    latex_table += "\\hline\n"
    latex_table += "K & Training Accuracy (\%) & Training Std & Testing Accuracy (\%) & Testing Std\\\\\n"
    latex_table += "\\hline\n"

    for k, (train_acc, train_std, test_acc, test_std) in results_dict.items():
        latex_table += f"{k} & {train_acc:.2f} & {train_std:.2f} & {test_acc:.2f} & {test_std:.2f}\\\\\n"

    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Results}\n"
    latex_table += "\\label{tab:results}\n"
    latex_table += "\\end{table}"

    return latex_table


if __name__ == "__main__":
    #pickle_file_path = "results_lf.pkl"
    #loaded_results = load_results(pickle_file_path)
    #plot_results(loaded_results)
    #latex_file_path = "results_table_lf.tex"
    #results_to_latex_averages(loaded_results, latex_file_path)
    # Load pickled results
    pickle_file_path = "data/results/pickles/results_rh_1715126714.5953465.pkl"  # Replace with your pickle file path
    with open(pickle_file_path, "rb") as pickle_file:
        results_dict = pickle.load(pickle_file)

    # Create LaTeX table
    latex_file_path = "data/results/results_table_rh_100_eps_k16913.tex"  # Replace with your desired LaTeX file path
    #results_to_latex_new(results_dict, latex_file_path)
    tab = generate_latex_table(results_dict)
    print(tab)

    
