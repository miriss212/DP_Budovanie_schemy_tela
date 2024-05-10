import os
import pickle
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('neural_networks/som')
from som import SOM

# Nacitam SOMky a polia s winner differentiation z  pickle
with open("data/results/pickles/leyla_100_eps_all_parts.pickle", "rb") as f:
    soms_and_errors = pickle.load(f)

# Vyplotim winner differentiation pre kazdu somku
plt.figure()

for name, data in soms_and_errors.items():
    som = data["som"]
    winner_diff_array = data["winner_diff_array"]
    plt.plot(winner_diff_array, label=name)

plt.xlabel('Data Index')
plt.ylabel('Winner Differentiation')
#plt.title('Winner differentiation for trained SOMs')
plt.legend()
plt.grid(True)

# Save and display the plot
result_path = "data/results/"  # Provide the path where you want to save the plot
out_name = time.time()
#plt.savefig("{}winner_diff_plot_{}.png".format(result_path, int(time.time())))
plt.savefig(f"data/results/win_diff_histogram_{out_name}.pdf", format="pdf")
plt.show()
