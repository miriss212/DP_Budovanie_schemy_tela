import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
try:
    from .util import *
except Exception: # ImportError
    from util import *

from som import SOM 

def load_som_from_epoch(epoch):
    filename = f'som_epoch_{epoch}.pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            som = pickle.load(f)
        return som
    else:
        print(f"No data found for epoch {epoch}")
        return None

def load_all_soms(folder="."):
    soms = []
    for file in os.listdir(folder):
        if file.startswith("som_epoch_") and file.endswith(".pickle"):
            epoch = int(file.split("_")[2].split(".")[0])
            som = load_som_from_epoch(epoch)
            if som:
                soms.append((epoch, som))
    return soms

def plot_soms(soms):
    fig, axs = plt.subplots(len(soms), figsize=(8, 4 * len(soms)))

    for i, (epoch, som) in enumerate(soms):
        # Plotting code
        som.plot_map()
        axs[i].set_title(f"Epoch {epoch}", loc='center', pad=20)

    fig.suptitle('SOMs from Different Epochs', fontsize=16) 

    plt.tight_layout()
    plt.show()

def main():
    soms = load_all_soms()
    if soms:
        with open("metrics.txt", "w") as metrics_file:
            for epoch, som in soms:
                if som:
                    quantization_error = som.quant_err()
                    winner_diff = som.winner_diff()
                    entropy = som.compute_entropy()
                    metrics_file.write(f'Somka : {som}')
                    metrics_file.write(f"Epoch {epoch} \n")
                    metrics_file.write(f"Quantization Error: {quantization_error}\n")
                    metrics_file.write(f"Winner Differentiation: {winner_diff}\n")
                    metrics_file.write(f"Entropy: {entropy}\n\n")
        print("Metrics saved to metrics.txt")
        plot_soms(soms)
    else:
        print("No SOM data found.")


if __name__ == "__main__":
    main()
