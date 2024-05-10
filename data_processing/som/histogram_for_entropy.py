import os
import pickle
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('neural_networks/som')
from som import SOM

# Nacitam SOMky a polia s entropiami  z  pickle
with open("data/results/pickles/leyla_100.pickle", "rb") as f:
    soms_and_errors = pickle.load(f)

# Vyplotim entropiu pre kazdu somku
plt.figure()

for name, data in soms_and_errors.items():
    som = data["som"]
    entropy_array = data["entropy_array"]
    plt.plot(entropy_array, label=name)

plt.xlabel('Data Index')
plt.ylabel('Enthropy')
plt.title('Enthropy for trained SOMs')
plt.legend()
plt.grid(True)

# Uloz a zobraz
result_path = "data/results/"  # sem ukladam
plt.savefig("{}entropy_plot_{}.png".format(result_path, int(time.time())))
plt.show()
