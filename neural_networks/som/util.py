# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, 2017-2018

import atexit
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')  # or any other backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time


## plotting

palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))


def plot_grid_2d(inputs, weights, i_x=0, i_y=1, s=60, block=True):
    plt.figure(1).canvas.mpl_connect('key_press_event', keypress)
    plt.clf()

    #plt.gcf().canvas.set_window_title('SOM neurons and inputs (2D)')
    plt.scatter(inputs[i_x,:], inputs[i_y,:], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

    n_rows, n_cols, _ = weights.shape

    for r in range(n_rows):
        plt.plot(weights[r,:,i_x], weights[r,:,i_y], c=palette[0])

    for c in range(n_cols):
        plt.plot(weights[:,c,i_x], weights[:,c,i_y], c=palette[0])

    plt.xlim(limits(inputs[i_x,:]))
    plt.ylim(limits(inputs[i_y,:]))
    plt.tight_layout()
    plt.show(block=block)


def plot_grid_3d(inputs, weights, i_x=0, i_y=1, i_z=2, s=60, block=True):
    fig = plt.figure(2)
    fig.canvas.mpl_connect('key_press_event', keypress)
    #plt.gcf().canvas.set_window_title('SOM neurons and inputs (3D)')
    
    try:
        # If using a different backend, set the window title
        plt.gcf().canvas.set_window_title('SOM neurons and inputs (3D)')
    except Exception as e:
        print(f"Error setting window title: {e}")
    
    if plot_grid_3d.ax is None:
        plot_grid_3d.ax = Axes3D(fig)

    ax = plot_grid_3d.ax
    ax.cla()

    ax.scatter(inputs[i_x,:], inputs[i_y,:], inputs[i_z,:], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

    n_rows, n_cols, _ = weights.shape

    for r in range(n_rows):
        ax.plot(weights[r,:,i_x], weights[r,:,i_y], weights[r,:,i_z], c=palette[0])

    for c in range(n_cols):
        ax.plot(weights[:,c,i_x], weights[:,c,i_y], weights[:,c,i_z], c=palette[0])

    ax.set_xlim(limits(inputs[i_x,:]))
    ax.set_ylim(limits(inputs[i_y,:]))
    ax.set_zlim(limits(inputs[i_z,:]))
    plt.show(block=block)

plot_grid_3d.ax  = None


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
        os._exit(0) # unclean exit, but exit() or sys.exit() won't work


## non-blocking figures still block at end

def finish():
    plt.show(block=True) # block until all figures are closed


atexit.register(finish)
