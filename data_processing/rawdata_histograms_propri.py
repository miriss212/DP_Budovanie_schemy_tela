import csv
import json
import numpy as np
import matplotlib.pyplot as plt


joints_info_file = "data_processing/icub_arm_joints.csv"
joint_bin_step = 5
filename = "TEST_data_NEW.rawdata"
out_name = "janka"
# filename = "data_leyla.rawdata"
# filename = "leyla_data_left-forearm.rawdata"
# out_name = "leyla"
# joint_min = -95
# joint_max = 270
# joint_bins = range(joint_min,joint_max,10)

with open(joints_info_file) as f1:
    reader = csv.DictReader(f1, delimiter=',')
    joint_info = list(reader)

with open(filename) as f:
    loaded_data = json.load(f)
# print(len(loaded_data))
right_pos_list = []
left_pos_list = []
touched = []
i = 0
for row in loaded_data:
    right_pos_list.append([float(i) for i in row['right-pos'].split()])
    left_pos_list.append([float(i) for i in row['left-pos'].split()])
    print([float(i) for i in row["left-forearm"].split()])
    touched.append(sum([float(i) for i in row["left-forearm"].split()]))
    #touched[i] += sum([float(i) for i in row["left-hand"].split()])
    #touched[i] += sum([float(i) for i in row["right-forearm"].split()])
    touched[i] += sum([float(i) for i in row["right-hand"].split()])
    i += 1
print(touched)
print(touched.count(0.0))
print(len(right_pos_list))
right_pos_data = np.array(right_pos_list)
left_pos_data = np.array(left_pos_list)

num_row = 4
num_col = 4

fig_right, axes_right = plt.subplots(num_row, num_col, figsize=(3 * num_col, 2.5 * num_row))
for joint_no in range(16):
    joint_bins = np.arange(float(joint_info[joint_no]["min"]), float(joint_info[joint_no]["max"])+joint_bin_step, joint_bin_step)
    joint_data = right_pos_data[:, [joint_no]]
    joint_data = joint_data.transpose().flatten()
    # if joint_no == 7:
    #     print(joint_data)
    #     print(joint_bins)
    ax = axes_right[joint_no // num_col, joint_no % num_col]
    ax.hist(joint_data, bins=joint_bins)
    ax.set_title(joint_info[joint_no]["shortname"])
plt.tight_layout()
# plt.savefig("joints_right.svg", format="svg")
plt.savefig("joints_right_{}_{}deg_bin.png".format(out_name, joint_bin_step), format="png")
plt.show()

fig_left, axes_left = plt.subplots(num_row, num_col, figsize=(3 * num_col, 2.5 * num_row))
for joint_no in range(16):
    joint_bins = np.arange(float(joint_info[joint_no]["min"]), float(joint_info[joint_no]["max"])+joint_bin_step, joint_bin_step)
    joint_data = left_pos_data[:, [joint_no]]
    joint_data = joint_data.transpose().flatten()
    ax = axes_left[joint_no // num_col, joint_no % num_col]
    ax.hist(joint_data, bins=joint_bins)
    ax.set_title(joint_info[joint_no]["shortname"])
plt.tight_layout()
# plt.savefig("joints_right.svg", format="svg")
plt.savefig("joints_left_{}_{}deg_bin.png".format(out_name, joint_bin_step), format="png")
plt.show()