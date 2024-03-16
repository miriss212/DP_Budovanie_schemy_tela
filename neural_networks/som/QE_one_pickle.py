import os
import pickle
import time
import matplotlib.pyplot as plt

# Nacitam SOMky a polia s quantizacnymi errormi z  pickle
with open("soms_and_errors.pickle", "rb") as f:
    soms_and_errors = pickle.load(f)

# Vyplotim quantizacny error pre kazdu somku
plt.figure()

for name, data in soms_and_errors.items():
    som = data["som"]
    quantization_error_array = data["quant_error_array"]
    plt.plot(quantization_error_array, label=name)

plt.xlabel('Data Index')
plt.ylabel('Quantization Error')
plt.title('Quantization Error for trained SOMs')
plt.legend()
plt.grid(True)

# Uloz a zobraz
result_path = "data/" 
plt.savefig("{}quantization_error_plot_{}.png".format(result_path, int(time.time())))
plt.show()
