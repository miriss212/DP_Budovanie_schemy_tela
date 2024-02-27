
import os
from os import path
import numpy as np
import json

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

def load_process_save(dir, filename, leftHandSom, rightHandSom, leftForearmSom, rightForearmSom, rightHandMrf, leftHandMrf, forearmMrf):
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
    processedRF_LH = []
    processedLF_RH = []
    processedAll = []
    for l, r, t in zip(left, right, touch):
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

        # right limb
        winnerR_H = rightHandSom.winnerVector(rightH)
        winnerR_F = rightForearmSom.winnerVector(rightF)

        count = sum(winnerL_H) + sum(winnerL_F) + sum(winnerR_H) + sum(winnerR_F)
        if count != 4:
            print("INCOMPLETE PROPRIO info, discarding.")
            continue

        # left hand
        cropT = t[0:9]
        if sum(cropT) != 0:
            lh = leftHandMrf.winnerVector(cropT)
            if sum(lh) > 0:
                processedLh.append({"pos": winnerL_H + winnerL_F + winnerR_H + winnerR_F, "touch": lh})
            else:
                print("ERROR: WAS TOUCH BUT NO WINNER [left hand]")

        # right hand
        cropT = t[32:41]
        if sum(cropT) != 0:
            rh = rightHandMrf.winnerVector(cropT)
            if sum(rh) > 0:
                processedRh.append({"pos": winnerL_H + winnerL_F + winnerR_H + winnerR_F, "touch": rh})
            else:
                print("ERROR: WAS TOUCH BUT NO WINNER [Right hand]")

        # left fore
        cropT = t[9:32]
        if sum(cropT) != 0:
            lf = forearmMrf.winnerVector(cropT)
            if sum(lf) > 0:
                processedLf.append({"pos": winnerL_H + winnerL_F + winnerR_H + winnerR_F, "touch": lf})
            else:
                print("ERROR: WAS TOUCH BUT NO WINNER [Left  forearm]")

        # right fore
        cropT = t[41:64]
        if sum(cropT) != 0:
            rf = forearmMrf.winnerVector(cropT)
            if sum(rf) > 0:
                processedRf.append({"pos": winnerL_H + winnerL_F + winnerR_H + winnerR_F, "touch": rf})
            else:
                print("ERROR: WAS TOUCH BUT NO WINNER [Right forearm]")


        #left fore + right hand
        cropT = t[9:32]
        if sum(cropT) != 0:
            lf = forearmMrf.winnerVector(cropT)
            cropT = t[32:41]
            if sum(cropT) != 0:
                rh = rightHandMrf.winnerVector(cropT)
                if sum(rh) != 0 and sum(lf) != 0:
                    processedLF_RH.append({"pos": winnerL_H + winnerL_F + winnerR_H + winnerR_F, "touch": lf+rh})

        #right forearm + left hand
        cropT = t[41:64]
        if sum(cropT) != 0:
            rf = forearmMrf.winnerVector(cropT)
            cropT = t[0:9]
            if sum(cropT) != 0:
                lh = leftHandMrf.winnerVector(cropT)
                if sum(rf) != 0 and sum(lh) != 0:
                    processedRF_LH.append({"pos": winnerL_H + winnerL_F + winnerR_H + winnerR_F, "touch": rf+lh})


        #======== ALL IN ONE ==========

        lh = t[0:9] # left hand
        rh = t[32:41] # right hand
        lf = t[9:32] # left fore
        rf = t[41:64]# right fore

        if at_least_one_IsTouch([lh, rh, lf, rf]):
            rfWinner = forearmMrf.winnerVector(rf)
            lfWinner = forearmMrf.winnerVector(lf)
            lhWinner = handMrf.winnerVector(lh)
            rhWinner = handMrf.winnerVector(rh)

            if at_least_one_IsTouch([rfWinner, lfWinner, lhWinner, rhWinner]):
                processedAll.append({"pos": winnerL_H + winnerL_F + winnerR_H + winnerR_F, "touch": lhWinner + lfWinner + rhWinner + rfWinner})

####################################################################
####################################################################
####################################################################

    # save data
    sub_folder = ""
    if '.' in filename:
        sub_folder = filename.split('.')[0]
    if not os.path.exists('bal_data'):
        os.mkdir('bal_data')
        print("Created directory 'bal_data' for file dumping.")
    if not os.path.exists('bal_data/' + sub_folder):
        os.mkdir('bal_data/' + sub_folder)
        print("Created directory 'bal_data/{0}' for file dumping.".format(sub_folder))

    with open('bal_data/' + sub_folder + '/lh.baldata', "w") as out:
        out.write(str(json.dumps(processedLh, indent=4)))

    with open('bal_data/' + sub_folder + '/rh.baldata', "w") as out:
        out.write(str(json.dumps(processedRh, indent=4)))

    with open('bal_data/' + sub_folder + '/lf.baldata', "w") as out:
        out.write(str(json.dumps(processedLf, indent=4)))

    with open('bal_data/' + sub_folder + '/rf.baldata', "w") as out:
        out.write(str(json.dumps(processedRf, indent=4)))

    with open('bal_data/' + sub_folder + '/lf_rh.baldata', "w") as out:
        out.write(str(json.dumps(processedLF_RH, indent=4)))

    with open('bal_data/' + sub_folder + '/rf_lh.baldata', "w") as out:
        out.write(str(json.dumps(processedRF_LH, indent=4)))

    with open('bal_data/' + sub_folder + '/all.baldata', "w") as out:
        out.write(str(json.dumps(processedAll, indent=4)))

    print("Files created.")

    return

#-----OLD------
    # load data
    f = open(dir + filename)
    rawData = json.load(f)  # [0:13]
    f.close()
    touch = np.array([i["touch"] for i in rawData])
    left = np.array([i["leftHandPro"] for i in rawData])
    right = np.array([i["rightHandPro"] for i in rawData])

    left_hand = np.array(left[:, 5:])
    left_forearm = np.array(left[:, :5])
    right_hand = np.array(right[:, 5:])
    right_forearm = np.array(right[:, :5])

    # process data
    processedLh = []
    processedRh = []
    processedLf = []
    processedRf = []
    processedAll = []
    processedLF_RH = []
    processedRF_LH = []
    for leftH, rightH, leftF, rightF, t in zip(left_hand, right_hand, left_forearm, right_forearm, touch):
        if sum(t) == 0:
            continue

        # SOM winners

        # left limb
        winnerL_H = leftHandSom.winnerVector(leftH)
        winnerL_F = leftForearmSom.winnerVector(leftF)

        # right limb
        winnerR_H = rightHandSom.winnerVector(rightH)
        winnerR_F = rightForearmSom.winnerVector(rightF)

        # separated touch vectors
        leftHandTouch = t[0:9]
        leftForearmTouch = t[9:32]
        rightHandTouch = t[32:41]
        rightForearmTouch = t[41:64]

        # MRF-SOM (touch) winners
        lh = handMrf.winnerVector(leftHandTouch)
        rh = handMrf.winnerVector(rightHandTouch)
        lf = forearmMrf.winnerVector(leftForearmTouch)
        rf = forearmMrf.winnerVector(rightForearmTouch)

        leftLimb = winnerL_H + winnerL_F
        rightLimb = winnerR_H + winnerR_F

        # left hand
        if sum(leftHandTouch) != 0:
            processedLh.append({"pos": leftLimb + rightLimb, "touch": lh})

        # right hand
        if sum(rightHandTouch) != 0:
            processedRh.append({"pos": leftLimb + rightLimb, "touch": rh})

        # left fore
        if sum(leftForearmTouch) != 0:
            processedLf.append({"pos": leftLimb + rightLimb, "touch": lf})

        # right fore
        if sum(rightForearmTouch) != 0:
            processedRf.append({"pos": leftLimb + rightLimb, "touch": rf})

        # everything
        processedAll.append({"pos":leftLimb + rightLimb, "touch":lh+lf+rh+rf})


    # save data
    sub_folder = ""
    if '.' in filename:
        sub_folder = filename.split('.')[0]
    if not os.path.exists('bal_data'):
        os.mkdir('bal_data')
        print("Created directory 'bal_data' for file dumping.")
    if not os.path.exists('bal_data/' + sub_folder):
        os.mkdir('bal_data/' + sub_folder)
        print("Created directory 'bal_data/{0}' for file dumping.".format(sub_folder))

    with open('bal_data/' + sub_folder + '/lh.baldata', "w") as out:
        out.write(str(json.dumps(processedLh, indent=4)))

    with open('bal_data/' + sub_folder + '/rh.baldata', "w") as out:
        out.write(str(json.dumps(processedRh, indent=4)))

    with open('bal_data/' + sub_folder + '/lf.baldata', "w") as out:
        out.write(str(json.dumps(processedLf, indent=4)))

    with open('bal_data/' + sub_folder + '/rf.baldata', "w") as out:
        out.write(str(json.dumps(processedRf, indent=4)))

    with open('bal_data/' + sub_folder + '/all.baldata', "w") as out:
        out.write(str(json.dumps(processedAll, indent=4)))

    print("Files created.")


if __name__ == "__main__":
    """
    Transform ../data.data to 4 baldata files containing transformed configurations and touches.
    """
    leftHandSom = loadSom(11, 8, 8, "./som/trained/left_hand")
    leftForearmSom = loadSom(5, 9, 12, "./som/trained/left_forearm")
    rightHandSom = loadSom(11, 8, 8, "./som/trained/right_hand")
    rightForearmSom = loadSom(5, 9, 12, "./som/trained/right_forearm")
    handMrf = loadMrfSom(9, 7, 7, "./mrf/trained/right-hand2")
    rightHandMrf = loadMrfSom(9, 7, 7, "./mrf/trained/right-hand")
    leftHandMrf = loadMrfSom(9, 7, 7, "./mrf/trained/left-hand")
    forearmMrf = loadMrfSom(23, 12, 9, "./mrf/trained/right-forearm2")

    Generate = False
    if Generate:
        dir = "collected_data/"
        file_names = ["0{0}.data".format(i) for i in range(1, 10)]
        file_names.append("10.data")
        for fileName in file_names:
            load_process_save(dir, fileName, leftHandSom, rightHandSom, leftForearmSom, rightForearmSom, handMrf, forearmMrf)
    else:
        default = "my_data.data"
        load_process_save("", default, leftHandSom, rightHandSom, leftForearmSom, rightForearmSom, rightHandMrf, leftHandMrf, forearmMrf)
