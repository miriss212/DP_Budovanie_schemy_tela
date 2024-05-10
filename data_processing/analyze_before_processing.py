import json
import sys
sys.path.append('D:/DP_Budovanie_schemy_tela')


def bin_data(data, num_bins):
    bin_ranges = []
    bin_size = 1.0 / num_bins
    for i in range(num_bins):
        bin_ranges.append((i * bin_size, (i + 1) * bin_size))

    binned_data = {}
    for key, values in data.items():
        binned_vector = []
        for value in values:
            for j, (lower, upper) in enumerate(bin_ranges):
                if lower <= value < upper:
                    binned_vector.append(j)
                    break
        binned_data[key] = binned_vector
    return binned_data


with open("D:/DP_Budovanie_schemy_tela/data/data/leyla_filtered_data.data", "r") as file:
    data = json.load(file)

num_bins = 10

binned_data = bin_data(data[0], num_bins)

print(binned_data)
