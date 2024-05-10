import csv
import json
import numpy as np
import matplotlib
# from matplotlib import pyplot as plt
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
import matplotlib.pyplot as plt


#joints_info_file = "icub_arm_joints.csv"
#joint_bin_step = 5
# filename = "leyla_data_left-forearm.rawdata"
# out_name = "leyla"
#filename = "my_data.rawdata"
joints_info_file = "data_processing/icub_arm_joints.csv"
joint_bin_step = 5
filename = "data/data/TEST_data_NEW.rawdata"
out_name = "matej"

with open(joints_info_file) as f1:
    reader = csv.DictReader(f1, delimiter=',')
    joint_info = list(reader)

with open(filename) as f:
    loaded_data = json.load(f)
# print(len(loaded_data))
right_pos_list = []
left_pos_list = []
print(len(loaded_data))
for row in loaded_data:
    left_pos_list.append([float(i) for i in row['left-pos'].split()])
    right_pos_list.append([float(i) for i in row['right-pos'].split()])
pos_data = np.array(left_pos_list + right_pos_list)
# pos_data = np.array(right_pos_list)
print(len(pos_data))

num_row = 4
num_col = 2

# plt.style.use(['science','ieee'])
fig, axes = plt.subplots(num_row, num_col, figsize=(3 * num_col, 2.5 * num_row))
for joint_no in range(7):
    joint_bins = np.arange(float(joint_info[joint_no]["min"]), float(joint_info[joint_no]["max"])+joint_bin_step, joint_bin_step)
    joint_data = pos_data[:, [joint_no]]
    joint_data = joint_data.transpose().flatten()
    ax = axes[joint_no // num_col, joint_no % num_col]
    ax.hist(joint_data, bins=joint_bins)
    ax.set_title(joint_info[joint_no]["shortname"])
axes[3,1].set_axis_off()
plt.tight_layout()
# plt.savefig("joints_right.svg", format="svg")
plt.savefig("joints_right_{}_{}deg_bin.png".format(out_name, joint_bin_step), format="png")
plt.savefig("joints_right_{}_{}deg_bin.pdf".format(out_name, joint_bin_step), format="pdf")
plt.show()