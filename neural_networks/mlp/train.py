import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
import random

try:
    from .mlp import Mlp
except Exception: #ImportError
    from mlp import Mlp


def process(labeld):
    res = []
    for data in labeld:
        #tocuh = 1 if sum(data["touch"]) != 0 else 0
        tmp = {
            "pos": data["proprio"],
            "touch": data["touch"]
        }
        res.append(tmp)
    return res


def compar(a, b):
    for i, j in zip(a, b):
        if i != j:
            return False
    return True


def per(size, ones):
    return list(filter(lambda x: sum(x) == ones, itertools.product([0, 1], repeat=size)))


def _cost(target, outputs):
    return np.sum((target - outputs) ** 2, axis=0)

if __name__ == "__main__":
    """
    Training MLP filter on ../my_data_act.data
    """
    test = True
    f = open("../my_data.data")
    collectedData = json.load(f)
    modifier = 0.5

    collectedData = process(collectedData)
    random.shuffle(collectedData)

    chunk = int(len(collectedData) * modifier)
    collectedData = collectedData[chunk:]

    separator = int(len(collectedData) * 0.8)

    trainX = np.array([i['pos'] for i in collectedData[0:separator]])
    trainY = np.array([i["touch"] for i in collectedData[0:separator]])
    testX = np.array([i['pos'] for i in collectedData[separator:]])
    testY = np.array([i["touch"] for i in collectedData[separator:]])

    inp = len(trainX[0])
    out = len(trainY[0])
    layers = [inp, int(inp*2.2), 50, out]
    eps = 1000
    res = []

    print(layers)
    model = Mlp(layers)
    res = model.train(trainX, trainY, iterations=eps)

    replot = plt.subplot(311)
    replot.set_title("mean sqr")
    replot.plot([i for i, _ in res])

    errplot = plt.subplot(313)
    errplot.set_title("classify error in %")
    errplot.set_ylim([-5, 105])
    errplot.plot([i for _, (i, _) in res])
    plt.show()

    fileName = "trained.mlp"
    model.save(fileName)
    model2 = Mlp(layers)
    model2.load(fileName)

    miss = 0
    re = 0
    for start, end in zip(testX, testY):
        prediction = model2.predict(np.array([start]))
        roundPrediction = np.array([0 if i < 0.5 else 1 for i in prediction])
        re += _cost(prediction, end)
        if np.all(roundPrediction == end):
            continue
        else:
            miss += 1
    print('test: miss', miss, "succ",  100 - (miss / max(len(testX), 1)) * 100, "%")
