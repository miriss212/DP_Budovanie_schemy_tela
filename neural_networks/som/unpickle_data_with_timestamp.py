"""import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time


def load_som_from_epoch(epoch):
    filename = f'som_epoch_{epoch}.pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            som = pickle.load(f)
        som.epoch = epoch  # Assign the epoch to the SOM object
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
    fig, axs = plt.subplots(len(soms), figsize=(10, 4 * len(soms)))

    for i, som in enumerate(soms):
        som.plot_map()
        axs[i].set_title(f"SOM from epoch {som.epoch}")

    plt.tight_layout()
    plt.show()

def main():
    timestamp = int(time.time())  # Get current timestamp in seconds
    soms = load_all_soms()
    if soms:
        with open("metrics.txt", "w") as metrics_file:
            for som in soms:
                quantization_error = som.quant_err()
                winner_diff = som.winner_diff()
                entropy = som.compute_entropy()
                metrics_file.write(f"Epoch {som.epoch}:\n")
                metrics_file.write(f"Quantization Error: {quantization_error}\n")
                metrics_file.write(f"Winner Differentiation: {winner_diff}\n")
                metrics_file.write(f"Entropy: {entropy}\n\n")
        print("Metrics saved to metrics.txt")
        plot_soms(soms)
    else:
        print("No SOM data found.")

if __name__ == "__main__":
    main()
"""