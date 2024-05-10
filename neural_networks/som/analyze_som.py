
import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
import som

def analyze_som(code, som, data, touchData):
    somR, somC, _ = som.weights.shape
    heatmap = [[0 * j for j in range(somC)] for i in range(somR)]

    for sample, touch in zip(data, touchData):
        #print(sample)
        #print(sum(touch))
        if sum(touch) == 0: #if not sum(touch) == 0:
            continue
        #print("!")
        r, c = som.winner(sample)
        #print("winners are {0}[r] and {1}[c]".format(r,c))
        if heatmap[r][c] > 1000:
            continue
        heatmap[r][c] += 1

    print("****** Heatmap for {0} ******".format(code))
    print(heatmap)
    show_heatmap(code, som, heatmap)

def show_heatmap(code, som, heatmapa):
    plt.imshow(heatmapa, cmap='viridis')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    data_source = "data/data/leyla_filtered_data.data"
    f = open(data_source)
    data = json.load(f)
    f.close()

    left = np.array([i["leftHandPro"] for i in data])
    right = np.array([i["rightHandPro"] for i in data])
    touchData = np.array([i["touch"] for i in data])
 

    left_hand = np.array(left[:, 5:])
    left_forearm = np.array(left[:, :5])

    right_hand = np.array(right[:, 5:])
    right_forearm = np.array(right[:, :5])

    som_loader = som.SOMLoader()
    leftHandSom = som_loader.load_som(11, 8, 8, "data/trained/som/left_hand")
    leftForearmSom = som_loader.load_som(5, 8, 8, "data/trained/som/left_forearm")
    rightHandSom = som_loader.load_som(11, 8, 8, "data/trained/som/right_hand")
    rightForearmSom = som_loader.load_som(5, 8, 8, "data/trained/som/right_forearm")

    soms = {'lh': leftHandSom, 'lf': leftForearmSom, 'rh': rightHandSom, "rf": rightForearmSom}
    dataForSoms = {'lh': left_hand, 'lf': left_forearm, 'rh': right_hand, "rf": right_forearm}

    #soms = {'lh': leftHandSom, 'lf': leftForearmSom}
    #dataForSoms = {'lh': left_hand, 'lf': left_forearm}

    for somKey in soms.keys():
        analyze_som(somKey, soms[somKey], dataForSoms[somKey], touchData)

