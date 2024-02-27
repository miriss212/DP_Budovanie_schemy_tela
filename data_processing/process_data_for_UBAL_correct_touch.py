
import os
from os import path
import numpy as np
import json

from neural_networks.som.som import SOM
from neural_networks.mrf.mrfsom import MRFSOM
from ubal import data_util as du


def loadSom(dim, rows, cols, folderName):
    model = SOM(dim, rows, cols)
    for i in range(rows):
        model.weights[i] = np.loadtxt(folderName+"/"+str(i)+".txt")
    return model


def loadMrfSom(dim, rows, cols, folderName):
    model = MRFSOM(dim, rows, cols)
    for i in range(rows):
        model.weights[i] = np.loadtxt(folderName+"/"+str(i)+".txt")
    return model


def toOneHot(x, y, maxI, maxJ):
    res =[]
    for i in range(maxI):
        for j in range(maxJ):
            if x == i and y == j:
                res.append(1)
            else:
                res.append(0)
    return res

def at_least_one_IsTouch(touchVectors):
    for v in touchVectors:
        if sum(v) != 0:
            return True

    return False

def load_process_save(dir, sub_folder, filename, leftHandSom, rightHandSom, leftForearmSom, rightForearmSom,
                      rightHandMrf, leftHandMrf, rightForearmMrf, leftForearmMrf):
    # load data
    f = open(dir + filename)
    scaledData = json.load(f)  # [0:13]
    f.close()
    touch = np.array([i["touch"] for i in scaledData])
    left = np.array([i["leftHandPro"] for i in scaledData])
    right = np.array([i["rightHandPro"] for i in scaledData])
    print("::::: Processing ",len(scaledData)," data items :::::")
    processedAll = []
    processedLh = []
    processedRh = []
    processedLf = []
    processedRf = []
    no_touch_items = 0
    all_touched_spots = {}
    for l, r, t in zip(left, right, touch):
        # no touch
        if sum(t) == 0:
            no_touch_items += 1
            continue
        # proprio
        leftH = l[5:]
        leftF = l[:5]
        rightH = r[5:]
        rightF = r[:5]
        act_propri_l_h = leftHandSom.distances(leftH)
        act_propri_l_f = leftForearmSom.distances(leftF)
        act_propri_r_h = rightHandSom.distances(rightH)
        act_propri_r_f = rightForearmSom.distances(rightF)
        lh = t[0:9] # left hand
        rh = t[32:41] # right hand
        lf = t[9:32] # left fore
        rf = t[41:64]# right fore
        # touch
        # lhWinner = np.random.normal(0,0.05,7*7).tolist()
        # rhWinner = np.random.normal(0,0.05,7*7).tolist()
        # lfWinner = np.random.normal(0,0.05,9*12).tolist()
        # rfWinner = np.random.normal(0,0.05,9*12).tolist()
        lhWinner = [0] * (7*7)
        rhWinner = [0] * (7*7)
        lfWinner = [0] * (9*12)
        rfWinner = [0] * (9*12)
        proprio_all = act_propri_l_h + act_propri_l_f + act_propri_r_h + act_propri_r_f
        lhWinner = leftHandMrf.winnerVector(lh)
        lfWinner = leftForearmMrf.winnerVector(lf)
        rhWinner = rightHandMrf.winnerVector(rh)
        rfWinner = rightForearmMrf.winnerVector(rf)

        touched_spots = ""
        if sum(lh) != 0:
            touched_spots += "lh"
        if sum(lf) != 0:
            touched_spots += "lf"
        if sum(rh) != 0:
            touched_spots += "rh"
        if sum(rf) != 0:
            touched_spots += "rf"

        if touched_spots == "lhlfrf":
            processedLh.append({"pos": proprio_all, "touch": lhWinner})
            processedRf.append({"pos": proprio_all, "touch": rfWinner})
            touched_spots = "lhrf"
            print("corrected")
        elif touched_spots == "lfrhrf":
            processedLf.append({"pos": proprio_all, "touch": lfWinner})
            processedRh.append({"pos": proprio_all, "touch": rhWinner})
            touched_spots = "lfrh"
            print("corrected")
        elif touched_spots == "lhlfrh":
            processedLh.append({"pos": proprio_all, "touch": lhWinner})
            processedRh.append({"pos": proprio_all, "touch": rhWinner})
            touched_spots = "lhrh"
            print("corrected")
        elif touched_spots == "lhrhrf":
            processedLh.append({"pos": proprio_all, "touch": lhWinner})
            processedRh.append({"pos": proprio_all, "touch": rhWinner})
            touched_spots = "lhrh"
            print("corrected")
        # elif touched_spots == "lhlfrf":
        #     print()
        else:
            if "lh" in touched_spots:
                processedLh.append({"pos": proprio_all, "touch": lhWinner})
            if "lf" in touched_spots:
                processedLf.append({"pos": proprio_all, "touch": lfWinner})
            if "rh" in touched_spots:
                processedRh.append({"pos": proprio_all, "touch": rhWinner})
            if "rf" in touched_spots:
                processedRf.append({"pos": proprio_all, "touch": rfWinner})

        if touched_spots in all_touched_spots:
            all_touched_spots[touched_spots] += 1
        else:
            all_touched_spots[touched_spots] = 1
        if touched_spots == "":
            # LH, LF, RH, RF
            # processedAll.append({"pos": proprio_all, "touch": lhWinner + lfWinner + rhWinner + rfWinner})
            # print(lhWinner + lfWinner + rhWinner + rfWinner)
            # exit()
        # else:
            print("ERROR: no touch - should not happen")
        if len(touched_spots) == 8:
            print("ERROR: all points touched")

    # exit()

    print("Processed data")
    print("No touch data: ",no_touch_items)
    print("Touch data: ",sum(all_touched_spots.values()))
    print(all_touched_spots)

    # save data
    if not os.path.exists('ubal_data'):
        os.mkdir('ubal_data')
        print("Created directory 'ubal_data' for file dumping.")
    if not os.path.exists('ubal_data/' + sub_folder):
        os.mkdir('ubal_data/' + sub_folder)
        print("Created directory 'ubal_data/{0}' for file dumping.".format(sub_folder))

    with open('ubal_data/' + sub_folder + '/lh.baldata', "w") as out:
        out.write(str(json.dumps(processedLh, indent=4)))

    with open('ubal_data/' + sub_folder + '/rh.baldata', "w") as out:
        out.write(str(json.dumps(processedRh, indent=4)))

    with open('ubal_data/' + sub_folder + '/lf.baldata', "w") as out:
        out.write(str(json.dumps(processedLf, indent=4)))

    with open('ubal_data/' + sub_folder + '/rf.baldata', "w") as out:
        out.write(str(json.dumps(processedRf, indent=4)))

    with open('ubal_data/' + sub_folder + '/all.baldata', "w") as out:
        out.write(str(json.dumps(processedAll, indent=4)))

    print("Files created.")
    return


if __name__ == "__main__":
    """
    Transform ../data.data to 4 baldata files containing transformed configurations and touches.
    """
    leftHandSom = loadSom(11, 8, 8, "./som/trained/left_hand")
    leftForearmSom = loadSom(5, 9, 12, "./som/trained/left_forearm")
    rightHandSom = loadSom(11, 8, 8, "./som/trained/right_hand")
    rightForearmSom = loadSom(5, 9, 12, "./som/trained/right_forearm")
    handMrf = loadMrfSom(9, 7, 7, "./mrf/trained/right-hand2")
    rightHandMrf = loadMrfSom(9, 7, 7, "./mrf/trained/right-hand2")
    leftHandMrf = loadMrfSom(9, 7, 7, "./mrf/trained/left-hand2")
    rightForearmMrf = loadMrfSom(23, 12, 9, "./mrf/trained/right-forearm2")
    leftForearmMrf = loadMrfSom(23, 12, 9, "./mrf/trained/left-forearm2")

    # default = "my_data.data"
    # sub_folder = "ubal_data_correct_touch"
    default = "data_leyla.data"
    sub_folder = "ubal_data_correct_touch_leyla"
    load_process_save("", sub_folder, default, leftHandSom=leftHandSom, rightHandSom=rightHandSom, leftForearmSom=leftForearmSom,
                      rightForearmSom=rightForearmSom, rightHandMrf=rightHandMrf, leftHandMrf=leftHandMrf,
                      rightForearmMrf=rightForearmMrf, leftForearmMrf=leftForearmMrf)
