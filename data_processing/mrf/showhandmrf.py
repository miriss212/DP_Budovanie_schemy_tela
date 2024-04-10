import numpy as np
import matplotlib.pyplot as plt
import itertools

handPos = [
    (1,1),
    (1,3),
    (1,5),
    (2,6),
    (3,0),
    (3,3),
    (3,4),
    (4,3),
    (4,4),
]

maps = \
    {
        0: [(0,1), (1,1)], #index finger
        1: [(0,3), (1,3)], #middle finger
        2: [(0,5), (1,5)], #ring finger
        3: [(2,6)], #pinky
        4: [(3,0)], #thumb
        5: [(3,3)], #Palm
        6: [(3,4)],#Palm
        7: [(4,3)],#Palm
        8: [(4,4)]#Palm

    }
right_hand_map = \
        {
            3: [(1,3), (2,3), (3,3), (4,3), (5,3)], # pinky
            2: [(1,6), (2,6), (3,6), (4,6), (5,6)], # ring finger
            1: [(1,8), (2,8), (3,8), (4,8), (5,8)], #middle finger
            0: [(1,10), (2,10), (3,10), (4,10), (5,10)], #index finger
            4: [(7,12), (8,12), (9,12), (10,12), (10,11), (11,11)], #thumb

            5: [(7,4), (7,5), (7,6), #Palm
                (8,4), (8,5), (8,6),
                (9,4), (9,5), (9,6),
                (10,4), (10,5), (10,6)],

            6: [(7,7), (7,8), (7,9),  #Palm
                (8,7), (8,8), (8,9),
                (9,7), (9,8), (9,9),
                (10,7),(10,8), (10,9)],

            7: [(11,4), (11,5), (11,6), #Palm
                (12,4), (12,5), (12,6),
                (13,4), (13,5), (13,6),
                (14,4), (14,5), (14,6)],

            8: [(11,7), (11,8), (11,9), #Palm
                (12,7), (12,8), (12,9),
                (13,7), (13,8), (13,9),
                (14,7), (14,8), (14,9)],
        }


left_hand_map = \
        {
            0: [(1,3), (2,3), (3,3), (4,3), (5,3)], #index finger
            1: [(1,6), (2,6), (3,6), (4,6), (5,6)], #middle finger
            2: [(1,8), (2,8), (3,8), (4,8), (5,8)], #ring finger
            3: [(1,10), (2,10), (3,10), (4,10), (5,10)], #pinky
            4: [(7,1), (8,1), (9,1), (10,1), (10,2), (11,2)], #thumb

            5: [(7,4), (7,5), (7,6), #Palm
                (8,4), (8,5), (8,6),
                (9,4), (9,5), (9,6),
                (10,4), (10,5), (10,6)],

            6: [(7,7), (7,8), (7,9),  #Palm
                (8,7), (8,8), (8,9),
                (9,7), (9,8), (9,9),
                (10,7),(10,8), (10,9)],

            7: [(11,4), (11,5), (11,6), #Palm
                (12,4), (12,5), (12,6),
                (13,4), (13,5), (13,6),
                (14,4), (14,5), (14,6)],

            8: [(11,7), (11,8), (11,9), #Palm
                (12,7), (12,8), (12,9),
                (13,7), (13,8), (13,9),
                (14,7), (14,8), (14,9)],
        }

def get_pixels(rightOrLeft, index):
    if rightOrLeft == "right":
        return right_hand_map[index]
    elif rightOrLeft == "left":
        return left_hand_map[index]
    else:
        raise Exception("Unknown hand type. Has to be either 'left', or 'right' [string]")

def loadSom(rows, cols, folderName):
    model = [0] * rows
    for i in range(rows):
        model[i] = np.loadtxt("./trained/"+folderName+"/"+str(i)+".txt")
    return np.array(model)


def showSom(som, name, rightOrLeft):
    somR, somC, _ = som.shape
    f, ax = plt.subplots(somC, somR)
    for r, c in itertools.product(range(somR), range(somC)):
        #14 cols, 16 rows
        heatMap = np.ones((16,13)) * -0.5
        weight = som[r][c]
        # hand
        for i in range(9):
            pixels = get_pixels(rightOrLeft, i)
            for pixel in pixels:
                x,y = pixel[0], pixel[1]
                heatMap[x][y] = np.floor(weight[i] * 10) / 10

        ax[r][c].set_yticklabels([])
        ax[r][c].set_xticklabels([])
        ax[r][c].set_xticks([])
        ax[r][c].set_yticks([])
        ax[r][c].imshow(heatMap)
    plt.suptitle(name)
    plt.subplots_adjust(left=0.12, bottom=0.12, right=0.45, top=0.88, wspace=0.04, hspace=0.04)
    plt.show()


#OLD, just bckup
def showSom123(som, name):
    somR, somC, _ = som.shape
    f, ax = plt.subplots(somC, somR)
    for r, c in itertools.product(range(somR), range(somC)):
        heatMap = np.ones((6,7)) * -0.5
        weight = som[r][c]
        # hand
        for i in range(9):
            pixels = maps[i]
            for pixel in pixels:
                x,y = pixel[0], pixel[1]
                heatMap[x][y] = np.floor(weight[i] * 10) / 10

        ax[r][c].set_yticklabels([])
        ax[r][c].set_xticklabels([])
        ax[r][c].imshow(heatMap)
    plt.suptitle(name)
    plt.show()

if __name__ == "__main__":
    """
    Visualization of trained MRF-SOM for hand
    """
    show_these = ["right-hand2", "left-hand2"]
    for mrfHand in show_these:
        som = loadSom(7, 7, mrfHand)
        rightOrLeft = str(mrfHand.split("-")[0])
        print(str(rightOrLeft[0]))
        showSom(som, mrfHand, rightOrLeft)
