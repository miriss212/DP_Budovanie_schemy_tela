import csv
import json
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib import pyplot as plt
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


def taxel_touched(touch_data, index):
    values = set(touch_data[i] for i in range(index, index + 12))
    return 1 if len(values) > 1 else 0


def process_touch_data(touch_data):
    return [taxel_touched(touch_data, i) for i in range(0, len(touch_data), 12)]


def process_hand(hand):
    return hand[0:60] + hand[96:144]


def process_forearm(config):
    x = config[0:30 * 12]  # -2
    x = x[0:27 * 12] + x[28 * 12:]  # -1
    x = x[0:22 * 12] + x[24 * 12:]  # -2
    x = x[0:18 * 12] + x[21 * 12:]  # -3
    x = x[0:16 * 12] + x[17 * 12:]  # -1
    return x


filename = "data/data/leyla_data.rawdata"
out_name = "miriam"
# filename = "leyla_data_left-forearm.rawdata"
# out_name = "leyla"
# filename = "leyla_data2_left-forearm.rawdata"
# out_name = "leyla2"

with open(filename) as f:
    loaded_data = json.load(f)
print(len(loaded_data))
data_names = ["left-hand", "left-forearm", "right-hand", "right-forearm"]
data_display_names = {
    "left-hand": "left hand",
    "left-forearm": "left forearm",
    "right-hand": "right hand",
    "right-forearm": "right forearm"
}
processed_data = {}
for name in data_names:
    processed_data[name] = []
i = 0
for row in loaded_data:
    for name in data_names:
        data_item = [float(i) for i in row[name].split()]
        if name in ["left-forearm", "right-forearm"]:
            data_item = process_forearm(data_item)
        if name in ["left-hand", "right-hand"]:
            data_item = process_hand(data_item)
        data_item = process_touch_data(data_item)
        processed_data[name].append(data_item)
    i += 1
print(i)

# for idx,name in enumerate(data_names):
#     print(idx, " ", name)
#     data = np.array(processed_data[name])
#     print(np.argwhere(np.sum(data, axis=0) > 16).transpose().flatten())

# plt.style.use(['science','ieee'])
num_row = 2
num_col = 2
fig, axes = plt.subplots(num_row, num_col, figsize=(3 * num_col, 2.5 * num_row))
for idx,name in enumerate(data_names):
    print(idx," ",name)
    data = np.array(processed_data[name])
    sums = np.sum(data, axis=0)
    print(name,":",sums)
    # ax = axes[idx]
    ax = axes[idx // num_col, idx % num_col]
    ax.set_title(data_display_names[name])
    ax.bar(range(1,len(data[0])+1), sums)
    if "hand" in name:
        ax.set_xticks(range(1, len(data[0])+1))
    else:
        ax.set_xticks(range(1, len(data[0])+1, 2))
# joint_data = joint_data.transpose().flatten()
plt.tight_layout()
# plt.savefig("joints_right.svg", format="svg")
plt.savefig("touch_distrib_{}.pdf".format(out_name), format="pdf")
plt.savefig("touch_distrib_{}.png".format(out_name), format="png")
plt.show()