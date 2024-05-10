import numpy as np
import os


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
        os.mkdir("/data/trained/mrf/"+name+"/")
    except OSError:
        pass

    for i in range(som.weights.shape[0]):
        np.savetxt("./data/trained/mrf/"+name+"/"+str(i)+".txt", som.weights[i])


def genTouch(vecSize=23):
    res = []
    for i in range(0,vecSize):
        for j in range(i, vecSize):
            tmp = [0 for _ in range(vecSize)]
            tmp[i] = 1
            tmp[j] = 1
            res.append(tmp)
            if i != j:
                res.append(list(tmp))
    return np.array(res).T


def makeHandMask(row, col):

    q1 = [1 if (i % 4 != 3) and i <= 14 else 0 for i in range(23)]
    q2 = [1 if (i % 4 != 0) and i <= 15 else 0 for i in range(23)]
    q3 = [1 if (i % 4 != 3) and i >= 11 else 0 for i in range(23)]
    q4 = [1 if (i % 4 != 0) and i >= 12 else 0 for i in range(23)]

    tmpc = (col / 2)
    tmpr = (row / 2)
    res = []
    for r in range(row):
        tmp = []
        for c in range(col):
            if r // tmpr == 0:
                if c // tmpc == 0:
                    tmp.append(q1)
                else:
                    tmp.append(q2)
            else:
                if c // tmpc == 0:
                    tmp.append(q3)
                else:
                    tmp.append(q4)

        res.append(tmp)
    return np.array(res)

if __name__ == "__main__":
    """
    Training of forearm MRF-SOM
    """
    inputs = genTouch()
    print(inputs.shape)
    mask = makeHandMask(6, 6)
    hand = newSom(6, 6, inputs, mask)

    saveSom(hand, name="left_forearm")
