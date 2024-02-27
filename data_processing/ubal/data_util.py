import json
import time
import numpy as np
import matplotlib.pyplot as plt


def filter_ambiguous_touch(labeld):
    res = []
    seen = {}
    for pos, touch in labeld:
        tmp = np.array(pos)
        # print(tmp)
        idx1 = np.argmax(tmp)
        tmp[idx1] = 0
        idx2 = np.argmax(tmp)
        key = idx1 * 1000 + idx2
        val = np.argmax(touch)

        if key not in seen:
            seen[key] = {val: {"p": pos, "t": touch, "s": 1}, "max": 1, "maxP": val}
        elif val in seen[key]:
            seen[key][val]["s"] += 1
            numb = seen[key][val]["s"]
            if numb > seen[key]["max"]:
                seen[key]["max"] = numb
                seen[key]["maxP"] = val
        else:
            seen[key][val] = {"p": pos, "t": touch, "s": 1}

    for key in seen.keys():
        obj = seen[key]
        obj = obj[obj["maxP"]]
        res.append((obj["p"], obj["t"]))
        #res.append((obj["p"], obj["t"]))
        for i in range(obj["s"]):
            # res.append((obj["p"], obj["t"]))
            pass

    return res


def analyze_data(data):
    seenP = {}
    duplicatesP = 0
    seenT = {}
    duplicatesT = 0
    for i in range(len(data)):
        proprio, touch = data[i]
        array = str(proprio)
        touch = np.argmax(touch)
        if array not in seenP:
            seenP[array] = True
        else:
            #already there..
            duplicatesP += 1

        if touch not in seenT:
            seenT[touch] = set()
            seenT[touch].add(array)
        else:
            #already there..
            seenT[touch].add(array)
            duplicatesT += 1
    # print("+======DATA ANALYSIS=======+")
    print("Samples unique proprio: {} / all: {}".format(len(data) - duplicatesP,len(data)))
    print("Number of UNIQUE samples: {0}".format(len(data) - duplicatesP))
    # print("[which is {0}%]".format(100 - ((duplicatesP / len(data))*100)))
    # print("---")
    print("Number of UNIQUE touches: {0}".format(len(data) - duplicatesT))
    print("Touch distribtuion",["T{}:{}".format(i,len(seenT[i])) for i in seenT])
    # for touchKey in seenT.keys():
    #     print("For touch #{0}, the amount of proprio configs are {1}".format(touchKey, len(seenT[touchKey])))


def split_data(data):
    data_x = []
    data_y = []
    for sample_x, sample_y in data:
        data_x.append(sample_x)
        data_y.append(sample_y)
    data_x_arr = np.array(data_x)
    data_y_arr = np.array(data_y)
    return data_x_arr,data_y_arr

def smallest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def kwta(distances, size, k=4):
    distances = np.reshape(distances, size)
    win_coords = smallest_indices(distances, k)
    result = np.zeros(size)
    result[win_coords] = 1.0
    # print(result)
    return result.flatten().tolist()


def kwta_conti(distances, size, k=12):
    distances = np.reshape(distances, size)
    win_coords = smallest_indices(distances, k)
    result = np.zeros(size)
    # print(distances[win_coords])
    # print(win_coords)
    coords_x, coords_y = win_coords
    winner = (coords_x[0], coords_y[0])
    rest = (coords_x[1:], coords_y[1:])
    result[winner] = 1.0
    result[rest] = 0.5
    # print(result)
    return result.flatten().tolist()


def kwta_per_bodypart(distances, k=4, continuous=False):
    parts = []
    parts.append(distances[0:64])  # left hand
    parts.append(distances[64:172])  # right hand
    parts.append(distances[172:236]) # left fore
    parts.append(distances[236:344])  # right fore
    size_hands = 8,8
    size_forearms = 9,12
    for i in range(len(parts)):
        if i % 2 == 0:
            if continuous:
                parts[i] = kwta_conti(parts[i], size_hands, k)
            else:
                parts[i] = kwta(parts[i], size_hands, k)
        else:
            if continuous:
                parts[i] = kwta_conti(parts[i], size_forearms, k)
            else:
                parts[i] = kwta(parts[i], size_forearms, k)
    return np.concatenate(parts)


def kwta_per_bodypart(distances, k=4, hand=(8,8), farm=(9,12), continuous=False):
    parts = []
    idx1 = 0
    idx2 = hand[0]**2
    # print(idx1,",",idx2)
    parts.append(distances[idx1:idx2])  # left hand
    idx1 = idx2
    idx2 += farm[0]*farm[1]
    # print(idx1,",",idx2)
    parts.append(distances[idx1:idx2])  # left fore
    idx1 = idx2
    idx2 += hand[0]**2
    # print(idx1,",",idx2)
    parts.append(distances[idx1:idx2]) # right hand
    idx1 = idx2
    idx2 += farm[0]*farm[1]
    # print(idx1,",",idx2)
    parts.append(distances[idx1:idx2])  # right fore
    for i in range(len(parts)):
        if i % 2 == 0:
            if continuous:
                parts[i] = kwta_conti(parts[i], hand, k)
            else:
                parts[i] = kwta(parts[i], hand, k)
        else:
            if continuous:
                parts[i] = kwta_conti(parts[i], farm, k)
            else:
                parts[i] = kwta(parts[i], farm, k)
    return np.concatenate(parts)


def normalize_and_invert(a):
    return 1.0 - (a - np.min(a))/np.ptp(a)


def winner_vector(distances, size):
    k=1
    distances = np.reshape(distances, size)
    win_coords = smallest_indices(distances, k)
    coords_x, coords_y = win_coords
    r = coords_x[0]
    c = coords_y[0]
    n_rows, n_cols = size
    # print(r," ",c)
    return toOneHot(r, c, n_rows, n_cols)


def toOneHot(x, y, maxI, maxJ):
    res = []
    for i in range(maxI):
        for j in range(maxJ):
            if x == i and y == j:
                res.append(1)
            else:
                res.append(0)
    return res


def binarize(arr, threshold=0.5):
    return np.array([0.0 if i < threshold else 1.0 for i in arr])

def visualize(data_pairs, no_samples=5, data_style="newdata", indeces=[]):
    num_row = no_samples
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col, num_row))
    for i in range(no_samples):
        index = np.random.randint(len(data_pairs))
        if len(indeces) == no_samples:
            index = indeces[i]
        print("Showing item {}".format(index))
        test_data_propri, test_data_touch = data_pairs[index]

        parts = []
        parts.append(test_data_propri[0:64])  # left hand
        parts.append(test_data_propri[64:172])  # right hand
        parts.append(test_data_propri[172:236])  # left fore
        parts.append(test_data_propri[236:344])  # right fore
        parts.append(test_data_touch)
        names = ["left hand", "left fore", "right hand", "right fore", "touch"]

        for idx, pattern in enumerate(parts):
            # print(pattern)
            sidex, sidey = 9, 12
            if len(pattern) == 64:
                sidex, sidey = 8, 8
            elif len(pattern) == 49:
                sidex, sidey = 7, 7
            pattern = pattern.reshape(sidex, sidey)
            ax = axes[i, idx]
            ax.imshow(pattern, cmap='viridis', interpolation='nearest')
            ax.set_title(names[idx])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # plt.imshow(pattern, cmap='gray')
    plt.tight_layout()
    fig_name = "figures/visualize_som_data_" + data_style
    plt.savefig(fig_name+".png", format="png")
    plt.savefig(fig_name+".svg", format="svg")
    plt.show()


def visualize_touch(data_pairs, data_name, num_col=5):
    _, data_touch = split_data(data_pairs)
    winners_touch = np.unique(np.argmax(data_touch, axis=1))
    num_row = int(len(winners_touch) / num_col)
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col, num_row))
    for i,w in enumerate(winners_touch):
        print("Showing item {}".format(i))
        pattern = np.zeros(len(data_touch[0]))
        pattern[w] = 1.0
        print(pattern)
        sidex, sidey = 9, 12
        if len(pattern) == 64:
            sidex, sidey = 8, 8
        elif len(pattern) == 49:
            sidex, sidey = 7, 7
        pattern = pattern.reshape(sidex, sidey)
        ax = axes[i]
        ax.imshow(pattern, cmap='viridis', interpolation='nearest')
        ax.set_title("Winner {}".format(w))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    # plt.
    fig_name = "visualize_mrf_som_winners_" + data_name
    plt.savefig(fig_name, format="png")
    plt.show()


def visualize_touch_all(data_pairs_all, data_names, num_col=5, num_row=7):
    fig, axes = plt.subplots(num_row, num_col, figsize=(10, 10))
    curr_row = 0
    for d,data_part in enumerate(data_pairs_all):
        _, data_touch = split_data(data_part)
        winners_touch = np.unique(np.argmax(data_touch, axis=1))
        curr_col = 0
        for i,w in enumerate(winners_touch):
            # print("Showing item {} {} d{}".format(i, data_names[d], d))
            pattern = np.zeros(len(data_touch[0]))
            pattern[w] = 1.0
            # print(pattern)
            sidex, sidey = 9, 12
            if len(pattern) == 64:
                sidex, sidey = 8, 8
            elif len(pattern) == 49:
                sidex, sidey = 7, 7
            pattern = pattern.reshape(sidex, sidey)
            ax = axes[curr_row, curr_col]
            ax.imshow(pattern, cmap='viridis', interpolation='nearest')
            ax.set_title("Winner {} {}".format(w, data_names[d].upper()))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            curr_col += 1
            if len(winners_touch) > num_col and i == (num_col - 1):
                curr_row += 1
                curr_col = 0
        curr_row += 1

    plt.tight_layout()
    # plt.
    fig_name = "visualize_mrf_som_winners_all"
    plt.savefig(fig_name, format="png")
    plt.show()


# trainingNames = ["lh","rh","lf","rf"]
# data_all = []
#
# for name in trainingNames:
#     with open("../ubal_data/my_data/" + name + ".baldata") as f:
#         data = json.load(f)
#     labeled = [(np.array(i['pos']), np.array(i["touch"])) for i in data]
#     data_all.append(labeled)
#     print(name.upper())
#     analyze_data(labeled)
#
# visualize_touch_all(data_all, trainingNames)

def visualize_errors(data_pairs, predictions, indeces, part_name='rh'):
    num_row = len(indeces)
    num_col = 8
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col, num_row))
    for i in range(num_row):
        index = indeces[i]
        print("Showing item {}".format(index))
        test_data_propri, test_data_touch = data_pairs[index]
        parts = []
        parts.append(test_data_propri[0:64])  # left hand
        parts.append(test_data_propri[64:172])  # right hand
        parts.append(test_data_propri[172:236])  # left fore
        parts.append(test_data_propri[236:344])  # right fore
        parts.append(test_data_touch)
        parts.append(predictions[index])
        names = ["LH", "LF", "RH", "RF", "touch real", "touch pred"]
        # cmaps = 4 * ['Purples'] + ['Greens','Reds']
        cmaps = 4 * ['viridis'] + ['plasma','magma']
        for idx, pattern in enumerate(parts):
            # print(pattern)
            sidex, sidey = 9, 12
            if len(pattern) == 64:
                sidex, sidey = 8, 8
            elif len(pattern) == 49:
                sidex, sidey = 7, 7
            pattern = pattern.reshape(sidex, sidey)
            ax = axes[i,idx]
            ax.imshow(pattern, cmap=cmaps[idx], interpolation='nearest')
            ax.set_title(names[idx])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # plt.imshow(pattern, cmap='gray')
    plt.tight_layout()
    # fig_name = "visualize_ubal_erros_" + part_name
    fig_name = "visualize_hand_arm_touch1"
    plt.savefig(fig_name+".png", format="png")
    plt.savefig(fig_name+".pdf", format="pdf")
    plt.show()


def visualize_backproj(data_pairs, no_samples=5, data_style="newdata", indeces=[]):
    num_row = no_samples
    num_col = 9
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col, num_row))
    for i in range(no_samples):
        index = np.random.randint(len(data_pairs))
        if len(indeces) == no_samples:
            index = indeces[i]
        print("Showing item {}".format(index))
        data_propri, data_touch, data_project = data_pairs[index]
        parts = []
        parts.append(data_propri[0:64])  # left hand
        parts.append(data_propri[64:172])  # right hand
        parts.append(data_propri[172:236])  # left fore
        parts.append(data_propri[236:344])  # right fore
        parts.append(data_touch)
        parts.append(data_project[0:64])  # left hand
        parts.append(data_project[64:172])  # right hand
        parts.append(data_project[172:236])  # left fore
        parts.append(data_project[236:344])  # right fore
        names = ["left hand", "left arm", "right hand", "right arm", "touch", "*left hand", "*left arm", "*right hand", "*right arm"]
        for idx, pattern in enumerate(parts):
            # print(idx)
            sidex, sidey = 9, 12
            if len(pattern) == 64:
                sidex, sidey = 8, 8
            elif len(pattern) == 49:
                sidex, sidey = 7, 7
            pattern = pattern.reshape(sidex, sidey)
            ax = axes[i, idx]
            ax.imshow(pattern, cmap='viridis', interpolation='nearest')
            ax.set_title(names[idx])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # plt.imshow(pattern, cmap='gray')
    plt.tight_layout()
    fig_name = "figures/visualize_som_data_" + data_style
    plt.savefig(fig_name+".png", format="png")
    plt.savefig(fig_name+".svg", format="svg")
    plt.show()


def note_end_time(start_time):
    end_time = time.time()
    runtime = end_time - start_time
    m, s = divmod(runtime, 60)
    h, m = divmod(m, 60)
    print(s)
    print('\nExperiment finished in {:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s)))