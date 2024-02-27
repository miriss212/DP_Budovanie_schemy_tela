import numpy as np
import os
import json

try:
    from .mrfsom import MRFSOM
except Exception: #ImportError
    from mrfsom import MRFSOM


def L_p(u, v, p, axis=0):
    return np.sum(np.abs(u-v)**p, axis=axis) ** (1/p)


def L_2(u, v, axis=0):
    return L_p(u, v, p=2, axis=axis)


def newSom(rows, cols, inputsData, mask):
    metric = L_2
    (dim, count) = inputsData.shape

    top_left = np.array((0, 0))
    bottom_right = np.array((rows - 1, cols - 1))
    lambda_s = metric(top_left, bottom_right) * 0.5

    model = MRFSOM(dim, rows, cols, inputs=None, mask=mask)
    model.train(inputsData, metric=metric, alpha_s=0.5, alpha_f=0.01, lambda_s=lambda_s,
                lambda_f=1, eps=500, trace=False, trace_interval=10)
    return model


def saveSom(som, name):
    try:
        os.mkdir("./trained/"+name+"/")
    except OSError:
        pass

    for i in range(som.weights.shape[0]):
        np.savetxt("./trained/"+name+"/"+str(i)+".txt", som.weights[i])


def genTouch(vecSize=9):
    res = []
    for i in range(0,vecSize):
        for j in range(i, vecSize):
            if abs(i-j) > 1:
                continue
            tmp = [0 for _ in range(vecSize)]
            tmp[i] = 1
            tmp[j] = 1
            if i == j:
                res.append(list(tmp))
            res.append(tmp)
    return np.array(res).T

def getRealTouchData(filename, limbType):
    # load data
    f = open(filename)
    scaledData = json.load(f)  # [0:13]
    f.close()
    touch = np.array([i["touch"] for i in scaledData])

    if limbType == "right-hand":
        tmp = np.array(touch[:, 32:41])
        right_hand_touch = np.array([touch for touch in tmp if sum(touch) > 0])
        return right_hand_touch
    elif limbType == "left-hand":
        tmp = np.array(touch[:, 0:9])
        left_hand_touch = np.array([touch for touch in tmp if sum(touch) > 0])
        return left_hand_touch
    else:
        raise Exception("Unknown limb type, specify either 'right-hand' or 'left-hand'")

def makeHandMask(row, col):
    fingers = [1 if i < 5 else 0 for i in range(9)]
    hand = [1 if i > 4 else 0 for i in range(9)]

    res = []
    for r in range(row):
        tmp = []
        for c in range(col):
            tmp.append(fingers if c < col/2+1 else hand)
        res.append(tmp)
    return np.array(res)

if __name__ == "__main__":
    """
    Training of hand MRF-SOM
    """
    Generate = True
    if Generate:
        inputs = genTouch()
    else:
        limbType = "right-hand"
        inputs = getRealTouchData("../my_data.data", limbType).T
    mask = makeHandMask(7, 7)
    hand = newSom(7, 7, inputs, mask)

    saveSom(hand, name=limbType)
