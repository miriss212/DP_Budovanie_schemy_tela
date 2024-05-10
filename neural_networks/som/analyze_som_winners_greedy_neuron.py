import numpy as np
import pickle
from train import train_new_som
from experiment_util import create_experiment_sets, load_data
import som
import os
import sys
sys.path.append('neural_networks/som')
from som import SOM, SOMSaver
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


def analyze_som(som, data):
    neuron_stats = {}

    # Iterate through all neurons in the SOM
    for row in range(som.n_rows):
        for col in range(som.n_cols):
            neuron_stats[(row, col)] = analyze_neuron(som, row, col, data)

    return neuron_stats

def analyze_neuron(som, row, col, data):
    # Get the inputs mapped to the specified neuron
    mapped_inputs = get_mapped_inputs(som, row, col, data)

    if mapped_inputs is not None:
        # Compute mean and standard deviation of the mapped inputs
        mean = np.mean(mapped_inputs, axis=0)
        std_dev = np.std(mapped_inputs, axis=0)
        return mapped_inputs, mean, std_dev
    else:
        return None, None, None

def get_mapped_inputs(som, row, col, data):
    mapped_inputs = []

    # Iterate through all data points and check if they are mapped to the specified neuron
    for i in range(data.shape[1]):
        input_vector = data[:, i]
        if som.winner(input_vector) == (row, col):
            mapped_inputs.append(input_vector)

    if mapped_inputs:
        return np.array(mapped_inputs)
    else:
        return None


rows = 10
cols = 10

data_source = "data/data/leyla_filtered_data.data"
np.set_printoptions(threshold=sys.maxsize)
currentDirectory = os.getcwd()
print(currentDirectory)
data = load_data(data_source)

separator = int(len(data) * 0.8)
som_saver = SOMSaver()

training_set_left_hand, training_set_left_forearm, testing_set_left_hand, testing_set_left_forearm = create_experiment_sets(data, separator)

left_forearm_som = train_new_som(rows, cols, training_set_left_forearm)
som_saver = SOMSaver()
som_saver.save_som(som=left_forearm_som, name=f'size_exp/SOM_{rows}x{cols}')
neuron_stats = analyze_som(left_forearm_som, training_set_left_forearm)

results = {
    'neuron_stats': neuron_stats,
    'rows': rows,
    'cols': cols
}

with open('neuron_stats.pickle', 'wb') as pickle_file:
    pickle.dump(results, pickle_file)


# Determine the maximum number of mapped inputs among all neurons
max_inputs = max(len(stats[0]) for stats in neuron_stats.values() if stats[0] is not None)

# Create a matrix to store the frequencies of mapped inputs for each neuron
heatmap_data = np.zeros((10, 10))

# Fill the heatmap matrix with the frequencies of mapped inputs for each neuron
for neuron, stats in neuron_stats.items():
    row, col = neuron
    if stats[0] is not None:
        num_inputs = len(stats[0])
        heatmap_data[row, col] = num_inputs / max_inputs

# Plot the heatmap
"""plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='viridis', square=True, cbar=True)"""
plt.figure(figsize=(12, 10))  # Increase the size of the display
heatmap = sns.heatmap(heatmap_data, cmap='viridis', square=True, cbar=False)  # Turn off automatic color bar

# Customize the color bar legend
color_map = plt.cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(vmin=0, vmax=max_inputs))
color_map.set_array([])  # Set empty array to avoid error
cbar = plt.colorbar(color_map)
cbar.set_label('Number of Wins')



plt.xlabel('Column Index', labelpad=10)  # Add padding to the labels
plt.ylabel('Row Index', labelpad=10)
plt.title('Neuron Activation Heatmap', pad=20)  # Add padding to the title
# Save the heatmap as a PDF file
plt.savefig('heatmap_new.pdf', format='pdf')
plt.show()

# Print the results
for neuron, stats in neuron_stats.items():
    row, col = neuron
    print("Neuron ({}, {}):".format(row, col))
    if stats[0] is not None:
        mapped_inputs, mean, std_dev = stats
        print("Mapped inputs:")
        print(len(mapped_inputs))
        print(mapped_inputs)
        print("Mean:")
        print(mean)
        print("Standard deviation:")
        print(std_dev)
    else:
        print("No inputs were mapped to this neuron.")
