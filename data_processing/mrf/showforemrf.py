import numpy as np
import matplotlib.pyplot as plt
import itertools


def loadSom(rows, cols, folderName):
    model = [0] * rows
    for i in range(rows):
        model[i] = np.loadtxt("./trained/"+folderName+"/"+str(i)+".txt")
    return np.array(model)

def showSom(som, name):
    somR, somC, _ = som.shape
    f, ax = plt.subplots(somR, somC)
    for r, c in itertools.product(range(somR), range(somC)):
        tmp = np.append(som[r][c], [0])
        heatMap = np.reshape(tmp, (6, 4))

        ax[r][c].set_yticklabels([])
        ax[r][c].set_xticklabels([])
        ax[r][c].set_xticks([])
        ax[r][c].set_yticks([])
        ax[r][c].imshow(heatMap)
    plt.suptitle(name)
    plt.subplots_adjust(left=0.09, bottom=0.01, right=0.34, top=0.95, wspace=0.20, hspace=0.10)
    plt.show()

if __name__ == "__main__":
    """
    Visualization of trained MRF-SOM for forearm
    """
    show_these = ["left-forearm2", "right-forearm2"]
    for mrfArm in show_these:
        som = loadSom(12, 9, mrfArm)
        showSom(som, mrfArm)
