import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys
sys.path.append('neural_networks/som')
from som import SOM

def plot_histogram(name_for_pickle, output_pdf):
    with open(f'data/results/pickles/{name_for_pickle}.pickle', 'rb') as f:
        som_pickle = pickle.load(f)
        num_subplots = len(som_pickle)
        num_cols = 4  # Define the number of columns for the grid
        num_rows = -(-num_subplots // num_cols)  # Calculate the number of rows

        with PdfPages(output_pdf) as pdf:
            for som_name, som_data in som_pickle.items():
                som = som_data["som"]
                fig, ax = plt.subplots(figsize=(6, 4))
                som.plot_winner_histogram(title=f"Winner Histogram of {som_name}")
                pdf.savefig(fig)
                plt.close()

def plot_winning_neurons_histograms(name_of_pickle, num_rows, num_cols, out_name):

    with open(f'data/results/pickles/{name_of_pickle}.pickle', 'rb') as f:
        som_pickle = pickle.load(f)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2.5 * num_rows))
    axes = axes.flatten()

    for idx, (som_name, som_data) in enumerate(som_pickle.items()):
        som = som_data["som"]
        winning_counts = som.winning_counts

        ax = axes[idx]
        ax.bar(range(len(winning_counts)), winning_counts.values())
        ax.set_title(f"Winner Histogram of {som_name}")
        ax.set_xlabel("Neurons")
        ax.set_ylabel("Number of Wins")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"data/results/winner_histogram_{out_name}.pdf", format="pdf")
    plt.show()

pickle_name = "leyla_100_eps_all_parts"
output_pdf = "combined3"
plot_winning_neurons_histograms(pickle_name,2,2, output_pdf)
