import json
import time
import numpy as np
from scipy.spatial import cKDTree
import sys
sys.path.append('DP_Budovanie_schemy_tela')

def format_data(data):
    string = " ".join(str(number) for number in data)
    return string.strip()


start_time = time.time()

parent_dir = "data-icub-selftouch/20220628_leftForeArmRightHand_1_multifinger/"
data_files = {"left-pos": "joints_leftArm/data.log",
         "right-pos": "joints_rightArm/data.log",
         "left-forearm": "skin_tactile_comp_left_forearm/data.log",
         "right-hand": "skin_tactile_comp_right_hand/data.log",
         "right-forearm": "skin_tactile_comp_right_forearm/data.log",
         "left-hand": "skin_tactile_comp_left_hand/data.log"}
timestamp_idx = 2
pivot_name = "right-forearm"
dump_file = "matej_data_{}.rawdata".format(pivot_name)
pivot_data = []
pivot_times = []
final_data = []
file_sizes = {}

with open(parent_dir + data_files[pivot_name], "r") as limb_data:
    for entry in limb_data:
        lf_item = ([float(i) for i in entry.split()])
        pivot_data.append(lf_item)
        final_data.append({pivot_name: format_data(lf_item[timestamp_idx+1:])})
        pivot_times.append(lf_item[timestamp_idx])
    file_sizes[pivot_name] = len(pivot_data)
left_forearm_np = np.array(pivot_data)
# lf_times = np.array(lf_times)
# lf_times = left_forearm_np[:, timestamp_idx]

print("processed pivot ",pivot_name)
# data_all = {}
# visited = {}
for name, file_name in data_files.items():
    if name != pivot_name:
        with open(parent_dir + file_name, "r") as limb_data:
            # visited[name] = []
            print("processing ",name)
            current_data_list = []
            for entry in limb_data:
                current_data_list.append(([float(i) for i in entry.split()]))
            file_sizes[name] = len(current_data_list)
            curr_data = np.array(current_data_list)
            times = curr_data[:, timestamp_idx]
            for pivot_idx, item in enumerate(pivot_times):
                closest_idx = (np.abs(times - item)).argmin()
                # tree = cKDTree(np.c_[lf_times, times])
                final_data[pivot_idx].update({name: format_data(current_data_list[closest_idx][timestamp_idx + 1:])})
                # visited[name].append((current_data_list[closest_idx][0],current_data_list[closest_idx][1]))
            # Numpy Finished in 0:00:06
            # for i, lf_item in enumerate(left_forearm_data):
            #     min_i = -1
            #     min_diff = 1000
            #     for j, item in enumerate(current_data_list):
            #         if abs(item[timestamp_idx] - lf_item[timestamp_idx]) < min_diff:
            #             min_diff = abs(item[timestamp_idx] - lf_item[timestamp_idx])
            #             min_i = j
            #     # print(min_diff)
            #     print(min_i)
            #     final_data[i].update({name : current_data_list[min_i]})
            # LOOPS Finished in 0:06:12
        # data_all[name] = np.array(current_data_list)
# print(data_all)
#print(final_data)
print(len(final_data))
print(len(pivot_times))
print()
# print(visited)
print(file_sizes)

data_rawdata = json.dumps(final_data, sort_keys=True, indent=4)
f = open(dump_file, "w")
f.write(data_rawdata)
f.close()
print("Data exported successfully.")

end_time = time.time()
runtime = end_time - start_time
m, s = divmod(runtime, 60)
h, m = divmod(m, 60)
# print(s)
print('\nFinished in {:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s)))
