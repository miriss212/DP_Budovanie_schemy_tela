import pickle
import sys
sys.path.append('neural_networks/som')
from som import SOM

def plot_histogram(name_for_pickle):

    with open(f'data/results/pickles/{name_for_pickle}.pickle', 'rb') as f:
            som_pickle = pickle.load(f)
            # Iterate over each SOM in the dictionary and plot its winner histogram
            for som_name, som_data in som_pickle.items():
                print(som_name)
                som = som_data["som"]
                som.plot_winner_histogram(title=f"Winner Histogram of {som_name}")

pickle_name = "leyla_100"
plot_histogram(pickle_name)