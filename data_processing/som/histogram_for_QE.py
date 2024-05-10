import os
import pickle
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('neural_networks/som')
from som import SOM

data_folder = "data/results/pickles/"
# Nacitam SOMky a polia s quantizacnymi errormi z  pickle
with open(os.path.join(data_folder, "leyla_filtered_final_2500eps_1x0.1_10x10.pickle"), "rb") as f:
    soms_and_errors = pickle.load(f)

# Vyplotim quantizacny error pre kazdu somku
plt.figure()

#print(soms_and_errors.items())

for name, data in soms_and_errors.items():
    som = data["som"]
    quantization_error_array = data["quant_error_array"]
    #plt.plot(quantization_error_array, label=name)
    # Plot only every 10th value
    plt.plot(range(0, len(quantization_error_array), 10), 
             quantization_error_array[::10], 
             label=name)

plt.xlabel('Data Index')
plt.ylabel('Quantization Error')
#plt.title('Quantization Error for trained SOMs')
plt.legend()
plt.grid(True)

# Uloz a zobraz
result_path = "data/results/"  # sem ukladam
#plt.savefig("{}quantization_error_plot_{}.png".format(result_path, int(time.time())))
out_name = time.time()
plt.savefig(f"data/results/size_exp_histo_{out_name}.pdf", format="pdf")
plt.show()
