import json
import sys
import time
import numpy as np
from neural_networks.ubal.UBAL_numpy import UBAL, Sigmoid


def mean_squared_error(desired, estimate):
    return ((desired - estimate) ** 2).mean()


# load data
filename = "my_data.data"
f = open("../" + filename)
scaledData = json.load(f)  # [0:13]
f.close()
touch = np.array([i["touch"] for i in scaledData])
left = np.array([i["leftHandPro"] for i in scaledData])
right = np.array([i["rightHandPro"] for i in scaledData])
propri = np.hstack((left,right))
print(propri.shape)
touched = 0
touched_indeces = []
for i in range(len(touch)):
    if np.sum(touch[i]) > 0:
        touched = touched + 1
        touched_indeces.append(i)
print(touched)
# print(touched_indeces)
touch = touch[touched_indeces]
propri = propri[touched_indeces]
train_data_size = len(touch)
print(train_data_size)
# exit()

# net params
hidden_layer = 50
arch = [len(propri[0]), hidden_layer, len(touch[0])]
print(arch)
betas = [0.0, 1.0, 0.0]
gammasF = [float("nan"), 0.0, 0.0]
gammasB = [0.0, 0.0, float("nan")]
# betas = [1.0, 0.5, 0.0]
# gammasF = [float("nan"), 0.5, 0.0]
# gammasB = [0.0, 0.5, float("nan")]
learning_rate = 1.0
init_w_mean = 0.0
init_w_var = 1.0
max_epoch = 500
netcount = 1
sigmoid = Sigmoid()
act_fun_F = [sigmoid, sigmoid, sigmoid]
act_fun_B = [sigmoid, sigmoid, sigmoid]
results_mse = []
results_runtime = []

for n in range(netcount):
    epoch = 0
    runtime = {"start": time.time(), "end": time.time()}
    start_time = time.time()
    network = UBAL(arch, act_fun_F, act_fun_B, learning_rate, init_w_mean, init_w_var,
                   betas, gammasF, gammasB)
    epoch_mse_f = 0
    while epoch < max_epoch:
        indexer = np.random.permutation(train_data_size)
        mse_f_sum = 0
        mse_b_sum = 0
        succ_f = 0
        for i in indexer:
            input = np.transpose(propri[indexer[i:i + 1]])
            label = np.transpose(touch[indexer[i:i + 1]])
            act_FP, act_FE, act_BP, act_BE = network.activation(input, label)
            network.learning(act_FP, act_FE, act_BP, act_BE)
            mse_f_sum = mse_f_sum + mean_squared_error(label, act_FP[2])
            mse_b_sum = mse_b_sum + mean_squared_error(input, act_BP[0])
            # print(i)
            # print(np.transpose(label))
            # print(np.where(np.transpose(act_FP[2]) > 0.5, 1, 0))
            act_f_binarized = np.where(np.transpose(act_FP[2]) > 0.5, 1, 0)
            succ = (int)((np.sum(np.transpose(label) == act_f_binarized)) == len(act_f_binarized))
            succ_f = succ_f + succ / len(act_f_binarized)
        epoch_mse_f = mse_f_sum / len(indexer)
        epoch_mse_b = mse_b_sum / len(indexer)
        epoch_succ_f = succ_f / len(indexer)
        epoch += 1
        end_time = time.time()
        print("Ep: {}  mseF: {:.3f}% mseB: {:.3f}% succF: {:.3f}%".format(epoch + 1, epoch_mse_f, epoch_mse_b, epoch_succ_f))
        start_time = end_time
        runtime["end"] = end_time
    runtime_total = runtime["end"] - runtime["start"]
    # converged = (sum(success[-min_success_epochs:]) == min_success_epochs)
    # print("net {}: mse {}, {} epochs, runtime {:.1f} seconds".format(n, epoch_mse_f, epoch, runtime_total))
    results_runtime.append(runtime_total)
    results_mse.append(epoch_mse_f)
    # sys.stdout.write(".")

# print(results_mse)
# print(results_runtime)