import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


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
    fig, axs = plt.subplots(len(soms), figsize=(10, 4 * len(soms)))

    for i, (epoch, som) in enumerate(soms):
        # Plotting code
        # Example: plot_grid_3d(inputs, som.weights, block=False)
        # You need to replace this with your actual plotting function
        axs[i].imshow(som.weights)  # Example plot, you may need to adjust this based on your SOM visualization method
        axs[i].set_title(f"Epoch {epoch}")

    plt.tight_layout()
    plt.show()

def main():
    soms = load_all_soms()
    if soms:
        plot_soms(soms)
    else:
        print("No SOM data found.")

if __name__ == "__main__":
    main()
