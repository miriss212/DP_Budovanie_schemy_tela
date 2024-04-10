import csv
import json
import numpy as np
import matplotlib.pyplot as plt

filename = "rh.baldata"
folder = "ubal_data/ubal_data_leyla/"
data_name = "right-hand"
out_name = "janka"

with open(folder + filename) as f:
    loaded_data = json.load(f)
print(len(loaded_data))
labeled = [(np.array(i['pos']), np.array(i["touch"])) for i in loaded_data]

data_propri = labeled[:,[0]]
data_touch = labeled[:,[1]]

sum_neuron_winner = np.sum(data_propri, axis=0)
print(sum_neuron_winner)

# num_row = 2
# num_col = 2
# fig, axes = plt.subplots(num_row, num_col, figsize=(3 * num_col, 2.5 * num_row))
# for idx in enumerate(data_propri):
#     print(idx," ",name)
#     data = np.array(processed_data[name])
#     sums = np.sum(data, axis=0)
#     print(name,":",sums)
#     # ax = axes[idx]
#     ax = axes[idx // num_col, idx % num_col]
#     ax.set_title(data_display_names[name])
#     ax.bar(range(1,len(data[0])+1), sums)
#     if "hand" in name:
#         ax.set_xticks(range(1, len(data[0])+1))
#     else:
#         ax.set_xticks(range(1, len(data[0])+1, 2))
# # joint_data = joint_data.transpose().flatten()
# plt.tight_layout()
# # plt.savefig("joints_right.svg", format="svg")
# plt.savefig("touch_distrib_{}.pdf".format(out_name), format="pdf")
# plt.savefig("touch_distrib_{}.png".format(out_name), format="png")
# plt.show()
