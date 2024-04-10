import atexit
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import json

import sys
sys.path.append('neural_networks/som')
from som import SOMLoader



## plotting

palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']


def limits(values, gap=0.3):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0 - xg, x1 + xg))

#skusit to rozhybat, vahy su v textakoch
def plot_grid_2d(inputs, weights, i_x=0, i_y=1, s=60, block=True):
    plt.figure(1).canvas.mpl_connect('key_press_event', keypress)
    plt.clf()

    plt.scatter(inputs[i_x,:], inputs[i_y,:], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

    n_rows, n_cols, dim_in = weights.shape

    for r in range(n_rows):
        plt.plot(weights[r,:,i_x], weights[r,:,i_y], c=palette[0])

    for c in range(n_cols):
        plt.plot(weights[:,c,i_x], weights[:,c,i_y], c=palette[0])

    x_limits = limits(inputs[i_x,:]) 
    y_limits = limits(inputs[i_y,:]) 

    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.tight_layout()
    plt.show(block=block)

def plot_grid_2d_new(inputs, weights, i_x=0, i_y=1, s=60, block=True):
    plt.figure(1).canvas.mpl_connect('key_press_event', keypress)
    plt.clf()

    #plt.scatter(weights[:, 0], weights[:, 1], s=s, c=palette[-1], edgecolors=[0.4] * 3, alpha=0.5)
    plt.scatter(inputs[i_x, :], inputs[i_y, :], s=s, c=palette[-1], edgecolors=[0.4] * 3, alpha=0.5)
    
    print(weights.shape)
    for i in range(weights.shape[0]):
        plt.text(weights[i, 0], weights[i, 1], str(i), color='black', fontsize=10, ha='center', va='center')

    plt.scatter(inputs[:, 0], inputs[:, 1], s=s, c=palette[-1], edgecolors=[0.4] * 3, alpha=0.5)

    plt.xlim(limits(inputs[i_x, :]))
    plt.ylim(limits(inputs[i_y, :]))
    plt.tight_layout()
    plt.show(block=block)

def plot_grid_2d_sus(inputs, weights, i_x=0, i_y=1, s=60, block=True):
    plt.figure(1).canvas.mpl_connect('key_press_event', keypress)
    plt.clf()

    plt.scatter(inputs[:, i_x], inputs[:, i_y], s=s, c=palette[-1], edgecolors=[0.4] * 3, alpha=0.5)

    if len(weights.shape) == 3:
        n_rows, n_cols, _ = weights.shape
        for r in range(n_rows):
            plt.plot(weights[r, :, i_x], weights[r, :, i_y], c=palette[0])
        for c in range(n_cols):
            plt.plot(weights[:, c, i_x], weights[:, c, i_y], c=palette[0])
    elif len(weights.shape) == 2:
        plt.plot(weights[:, i_x], weights[:, i_y], c=palette[0])

    x_limits = limits(inputs[:, i_x])
    y_limits = limits(inputs[:, i_y])

    #print(x_limits)
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.tight_layout()
    plt.show(block=block)




def plot_grid_3d(inputs, i_x=0, i_y=1, i_z=2, s=60, block=True):
    fig = plt.figure(2)
    fig.canvas.mpl_connect('key_press_event', keypress)
    #plt.gcf().canvas.set_window_title('SOM neurons and inputs (3D)')

    if plot_grid_3d.ax is None:
        plot_grid_3d.ax = Axes3D(fig)

    ax = plot_grid_3d.ax
    ax.cla()

    #ax.scatter(inputs[i_x, :], inputs[i_y, :], inputs[i_z, :], s=s, c=palette[-1], edgecolors=[0.4] * 3, alpha=0.5)
    n_rows, n_cols = inputs.shape

    # Data for three-dimensional scattered points
    zdata = inputs[:, 2:3]
    xdata = inputs[:, 1:2]
    ydata = inputs[:, 0:1]
    ax.scatter3D(xdata, ydata, zdata)

plot_grid_3d.ax = None

## interactive drawing, very fragile....

wait = 0.0


def clear():
    plt.clf()


def ion():
    plt.ion()
    # time.sleep(wait)


def ioff():
    plt.ioff()


def redraw():
    plt.gcf().canvas.draw()
    plt.waitforbuttonpress(timeout=0.001)
    time.sleep(wait)


def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0)  # unclean exit, but exit() or sys.exit() won't work


## non-blocking figures still block at end

def finish():
    plt.show(block=True)  # block until all figures are closed


atexit.register(finish)


def analyze_data(inputs):

    #plot_grid_3d(inputs, block=False)
    som_loader = SOMLoader()
    leftHandSom = som_loader.load_som(11, 6, 6, "data/trained/som/left_hand")
    leftForearmSom = som_loader.load_som(5, 6, 6, "data/trained/som/left_forearm")
    plot_grid_2d(inputs, leftHandSom.weights , block=False)
    redraw()

    """# Load weights from files and plot 2D grids
    for i in range(6):
        weights_file = f"data/trained/som/left_forearm/{i}.txt"
        if os.path.exists(weights_file):
            weights = np.loadtxt(weights_file)
            plot_grid_2d(inputs, weights, block=False)
            redraw()"""


if __name__ == "__main__":

    AnalyzeTouchOnly = True
    data_source = "./data/data/leyla_filtered_data.data"
    f = open(data_source)
    data = json.load(f)
    f.close()



    left = np.array([i["leftHandPro"] for i in data])
    right = np.array([i["rightHandPro"] for i in data])
    touch = np.array([i["touch"] for i in data])

    if AnalyzeTouchOnly:
        leftNew = []
        for proprioL, touchData in zip(left, touch):
            if sum(touchData) == 0:
                continue
            leftNew.append(proprioL)

        rightNew = []
        for proprioR, touchData in zip(right, touch):
            if sum(touchData) == 0:
                continue
            rightNew.append(proprioR)

        left = np.array(leftNew)
        right = np.array(rightNew)

    left_hand = np.array(left[:, 5:])
    left_forearm = np.array(left[:, :5])

    right_hand = np.array(right[:, 5:])
    right_forearm = np.array(right[:, :5])

    analyze_data(left_forearm)




