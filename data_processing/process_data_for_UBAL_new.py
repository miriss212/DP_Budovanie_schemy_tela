
import os
from os import path
import numpy as np
import json
import sys
sys.path.append('neural_networks/som')
from som import SOM
sys.path.append('neural_networks/mrf')
from mrfsom import MRFSOM


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


def touched_index(touch_vector):
    for i,val in enumerate(touch_vector):
        if val == 1:
            return i
    return 0


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
        touched_spots = ""
        proprio_all = act_propri_l_h + act_propri_l_f + act_propri_r_h + act_propri_r_f
        if sum(lh) != 0:
            #printi tie sumy
            lhWinner = leftHandMrf.winnerVector(lh)
            processedLh.append({"pos": proprio_all, "touch": lhWinner})
            touched_spots += "lh"
        if sum(lf) != 0:
            lfWinner = leftForearmMrf.winnerVector(lf)
            processedLf.append({"pos": proprio_all, "touch": lfWinner})
            touched_spots += "lf"
        if sum(rh) != 0:
            rhWinner = rightHandMrf.winnerVector(rh)
            processedRh.append({"pos": proprio_all, "touch": rhWinner})
            touched_spots += "rh"
        if sum(rf) != 0:
            rfWinner = rightForearmMrf.winnerVector(rf)
            processedRf.append({"pos": proprio_all, "touch": rfWinner})
            touched_spots += "rf"
        if touched_spots in all_touched_spots:
            all_touched_spots[touched_spots] += 1
        else:
            all_touched_spots[touched_spots] = 1
        # if touched_spots == "lfrhrf":
        #     print(t)
        #     print("lh:",lh," lf:",lf," rh:",rh," rf:",rf)

        if touched_spots != "":
            # LH, LF, RH, RF
            processedAll.append({"pos": proprio_all, "touch": lhWinner + lfWinner + rhWinner + rfWinner})
            # print(lhWinner + lfWinner + rhWinner + rfWinner)
            # exit()
        else:
            print("no touch - should not happen")
        if len(touched_spots) == 8:
            print("ERROR : all points touched")

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
    leftHandSom = loadSom(11, 6, 6, "data/trained/som/left_hand")
    leftForearmSom = loadSom(5, 6, 6, "data/trained/som/left_forearm")
    rightHandSom = loadSom(11, 6, 6, "data/trained/som/right_hand")
    rightForearmSom = loadSom(5, 6, 6, "data/trained/som/right_forearm")
    #handMrf = loadMrfSom(9, 7, 7, "./mrf/trained/right-hand2")
    rightHandMrf = loadMrfSom(9, 6, 6, "data/trained/mrf/right_hand")
    leftHandMrf = loadMrfSom(9, 6, 6, "data/trained/mrf/left_hand")
    rightForearmMrf = loadMrfSom(23, 6, 6, "data/trained/mrf/right_forearm")
    leftForearmMrf = loadMrfSom(23, 6, 6, "data/trained/mrf/left_forearm")

    # default = "my_data.data"
    default = "data/data/data_leyla.data"
    # sub_folder = "my_data"
    sub_folder = "ubal_data_leyla"
    load_process_save("", sub_folder, default, leftHandSom=leftHandSom, rightHandSom=rightHandSom, leftForearmSom=leftForearmSom,
                      rightForearmSom=rightForearmSom, rightHandMrf=rightHandMrf, leftHandMrf=leftHandMrf,
                      rightForearmMrf=rightForearmMrf, leftForearmMrf=leftForearmMrf)
