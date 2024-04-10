
import os
from os import path
import numpy as np
import json
import sys
sys.path.append('D:/DP_Budovanie_schemy_tela')
from neural_networks.som.som import SOM
from neural_networks.mrf.mrfsom import MRFSOM


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

def load_process_save(dir, filename, leftHandSom, rightHandSom, leftForearmSom, rightForearmSom,
                      mrf_r_h, mrf_l_h, mrf_l_f, mrf_r_f, k=8):
    # load data
    f = open(dir + filename)
    scaledData = json.load(f)  # [0:13]
    f.close()
    touch = np.array([i["touch"] for i in scaledData])
    left = np.array([i["leftHandPro"] for i in scaledData])
    right = np.array([i["rightHandPro"] for i in scaledData])

    # process data
    processedLh = []
    processedRh = []
    processedLf = []
    processedRf = []
    processedAll = []
    for l, r, t in zip(left, right, touch):
        # no touch
        if sum(t) == 0:
            continue

        leftH = l[5:]
        leftF = l[:5]

        rightH = r[5:]
        rightF = r[:5]

        # SOM winners

        # left limb
        winnerL_H = leftHandSom.winnerVector(leftH)
        winnerL_F = leftForearmSom.winnerVector(leftF)
        k_winners_l_h = leftHandSom.k_winner_vector(leftH,1)
        k_winners_l_f = leftForearmSom.k_winner_vector(leftF,1)
        # k_winners_l_h = leftHandSom.k_winners_conti(leftH)
        # k_winners_l_f = leftForearmSom.k_winners_conti(leftF)
        actL_H = leftHandSom.distances(leftH)
        actL_F = leftForearmSom.distances(leftF)

        # right limb
        winnerR_H = rightHandSom.winnerVector(rightH)
        winnerR_F = rightForearmSom.winnerVector(rightF)
        k_winners_r_h = rightHandSom.k_winner_vector(rightH,1)
        k_winners_r_f = rightForearmSom.k_winner_vector(rightF,1)
        # k_winners_r_h = rightHandSom.k_winners_conti(rightH)
        # k_winners_r_f = rightForearmSom.k_winners_conti(rightF)
        actR_H = rightHandSom.distances(rightH)
        actR_F = rightForearmSom.distances(rightF)

        count = sum(winnerL_H) + sum(winnerL_F) + sum(winnerR_H) + sum(winnerR_F)
        if count != 4:
            print("INCOMPLETE PROPRIO info, discarding.")
            continue

        # left hand
        cropT = t[0:9]
        if sum(cropT) != 0:
            lh = mrf_l_h.winnerVector(cropT)
            if sum(lh) > 0:
                # processedLh.append({"pos": k_winners_l_h + k_winners_l_f + k_winners_r_h + k_winners_r_f, "touch": lh})
                processedLh.append({"pos": actL_H + actL_F + actR_H + actR_F, "touch": lh})
            else:
                print("ERROR: WAS TOUCH BUT NO WINNER [left hand]")

        # right hand
        cropT = t[32:41]
        if sum(cropT) != 0:
            rh = mrf_r_h.winnerVector(cropT)
            if sum(rh) > 0:
                # processedRh.append({"pos": k_winners_l_h + k_winners_l_f + k_winners_r_h + k_winners_r_f, "touch": rh})
                processedRh.append({"pos": actL_H + actL_F + actR_H + actR_F, "touch": rh})
            else:
                print("ERROR: WAS TOUCH BUT NO WINNER [Right hand]")

        # left fore
        cropT = t[9:32]
        if sum(cropT) != 0:
            lf = mrf_l_f.winnerVector(cropT)
            if sum(lf) > 0:
                # processedLf.append({"pos": k_winners_l_h + k_winners_l_f + k_winners_r_h + k_winners_r_f, "touch": lf})
                processedLf.append({"pos": actL_H + actL_F + actR_H + actR_F, "touch": lf})
            else:
                print("ERROR: WAS TOUCH BUT NO WINNER [Left  forearm]")

        # right fore
        cropT = t[41:64]
        if sum(cropT) != 0:
            rf = mrf_r_f.winnerVector(cropT)
            if sum(rf) > 0:
                # processedRf.append({"pos": k_winners_l_h + k_winners_l_f + k_winners_r_h + k_winners_r_f, "touch": rf})
                processedRf.append({"pos": actL_H + actL_F + actR_H + actR_F, "touch": rf})
            else:
                print("ERROR: WAS TOUCH BUT NO WINNER [Right forearm]")

        #======== ALL IN ONE ==========

        lh = t[0:9] # left hand
        rh = t[32:41] # right hand
        lf = t[9:32] # left fore
        rf = t[41:64]# right fore

        if at_least_one_IsTouch([lh, rh, lf, rf]):
            # rfWinner = forearmMrf.activation(rf)
            # lfWinner = forearmMrf.activation(lf)
            # lhWinner = handMrf.activation(lh)
            # rhWinner = handMrf.activation(rh)
            rfWinner = mrf_r_f.winnerVector(rf)
            lfWinner = mrf_l_f.winnerVector(lf)
            lhWinner = mrf_l_h.winnerVector(lh)
            rhWinner = mrf_r_h.winnerVector(rh)
            # print(rhWinner)
            # print("act rh ", np.shape(rhWinner))

            if at_least_one_IsTouch([rfWinner, lfWinner, lhWinner, rhWinner]):
                # processedAll.append({"pos": winnerL_H + winnerL_F + winnerR_H + winnerR_F, "touch": lhWinner + lfWinner + rhWinner + rfWinner})
                processedAll.append({"pos": actL_H + actL_F + actR_H + actR_F, "touch": lhWinner + lfWinner + rhWinner + rfWinner})

####################################################################
####################################################################
####################################################################

    # save data
    sub_folder = ""
    if '.' in filename:
        sub_folder = filename.split('.')[0]
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
    leftHandSom = loadSom(11, 4, 4, "./data/trained/left_hand")
    leftForearmSom = loadSom(5, 4, 4, "./data/trained/left_forearm")
    rightHandSom = loadSom(11, 8, 8, "./som/trained/right_hand")
    rightForearmSom = loadSom(5, 9, 12, "./som/trained/right_forearm")
    handMrf = loadMrfSom(9, 7, 7, "./mrf/trained/right-hand2")
    rightHandMrf = loadMrfSom(9, 7, 7, "./mrf/trained/right-hand")
    leftHandMrf = loadMrfSom(9, 7, 7, "./mrf/trained/left-hand")
    right_forearm_mrf = loadMrfSom(23, 12, 9, "./mrf/trained/right-forearm")
    left_forearm_mrf = loadMrfSom(23, 12, 9, "./mrf/trained/left-forearm")

    default = "my_data.data"
    load_process_save("", default, leftHandSom, rightHandSom, leftForearmSom, rightForearmSom, rightHandMrf, leftHandMrf, left_forearm_mrf, right_forearm_mrf)
    #load_process_save("", default, leftHandSom, leftForearmSom)
