import os
import pickle
import time
import matplotlib.pyplot as plt

# Define the maximum epoch
max_epoch = 50

# Load quantization error data from pickles
quantization_errors = []

# Assuming you have a function to load quantization error data from pickles
def load_som(folder="."):
    errors = []
    for file in os.listdir(folder):
        if file.startswith("som_epoch_") and file.endswith(".pickle"):
            with open(os.path.join(folder, file), 'rb') as f:
                error = pickle.load(f)
            errors.append(error)
    return errors

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
    return sorted(soms, key=lambda x: x[0])  # Sort SOMs by epoch

som = load_all_soms()



# Plot the performance metrics
plt.figure()

if som:

    for epoch, som in som:
        if som:
            quantization_error = som.quant_err()
            winner_diff = som.winner_diff()
            entropy = som.compute_entropy()
            
            quantization_errors.append(quantization_error)
            
    plt.plot(list(range(max_epoch)), quantization_errors, label="Quantization Error")
    plt.xlabel('Epoch')
    #plt.ylabel('Quantization Error')
    #plt.title('Quantization Error over Epochs')
    #plt.legend()
    plt.grid(True)
else:
    print("No quantization error data loaded.")

# Save and display the plot
result_path = "data/"  # Provide the path where you want to save the plot
plt.savefig("{}quantization_error_plot_{}.png".format(result_path, int(time.time())))
plt.show()
