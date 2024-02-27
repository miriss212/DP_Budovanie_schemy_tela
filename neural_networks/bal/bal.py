import math
import numpy as np
import matplotlib.pyplot as plt


@np.vectorize
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Bal:
    '''
        vanila BAL
    '''

    def __init__(self, inp, hid, out, alpha=0.25, f=sigmoid):
        scale = 1 / math.sqrt(inp + 1)

        self.H_f = np.random.normal(loc=0, scale=scale, size=(inp, hid))
        self.Y_b = np.random.normal(loc=0, scale=scale, size=(out, hid))

        self.Y_f = np.random.normal(loc=0, scale=scale, size=(hid, out))
        self.H_b = np.random.normal(loc=0, scale=scale, size=(hid, inp))

        self.alpha = alpha
        self.f = f
        self.g = f

    def load(self, H_f, Y_b, Y_f, H_b):
        self.H_f, self.Y_b, self.Y_f, self.H_b =  H_f, Y_b, Y_f, H_b

    def _forward(self, imput):
        a = self.f(np.dot(imput, self.H_f))
        b = self.f(np.dot(a, self.Y_f))
        return imput, a, b

    def forward(self, imput):
        return self._forward(imput)[2]

    def _backward(self, imput):
        a = self.f(np.dot(imput, self.Y_b))
        b = self.f(np.dot(a, self.H_b))
        return b, a, imput

    def backward(self, imput):
        return self._backward(imput)[2]

    def _update(self, W, before, forward, backward):
        dif = self.alpha * (backward - forward)
        for p in range(W.shape[0]):
            W[p] += before[p] * dif

    def _cost(self, target, outputs):
        return np.sum((target - outputs) ** 2, axis=0)

    def _draw(self, replot, REs, errplot, errT):
        replot.clear()
        replot.set_title("mean sqr")
        replot.plot(REs)

        errplot.clear()
        errplot.set_title("classify error in %")
        errplot.plot([i for i, _ in errT])
        errplot.set_ylim([-5, 105])
        plt.draw()
        # plt.show()
        plt.pause(1)

    def train(self, data, epoch=None, track=False):
        """"
            Training method
            data=[(X, Y)],
            split works correctly only if noTouch + touch data are used
            :return [(mean squared error, classification error[%], (predictT, predictNT, missClass) )]
        """
        if track:
            plt.ion()
            plt.show(block=False)
            replot = plt.subplot(311)
            errplot = plt.subplot(313)
            print(self.H_f.shape[0], self.H_f.shape[1], self.Y_f.shape[1])

        REs = []
        errT = []

        re = 100
        i = 0
        n = len(data)

        errSplit = []
        noTouch = data[0][1]

        while re/n > 0.0:
            if i == epoch:
                break

            re = 0

            # draw stats
            if i % 5 == 0:
                print("epoch", i)
                print("alpha", self.alpha)
                if track and i % 50 == 0 and i != 0:
                    self._draw(replot, REs, errplot, errT)

            # do epoch
            for start, end in np.random.permutation(data):

                s_f, h_f, e_f = self._forward(start)
                s_b, h_b, e_b = self._backward(end)

                # update smer dopredu
                self._update(self.H_f, s_f, h_f, h_b)
                self._update(self.Y_f, h_f, e_f, e_b)

                # update smer dozadu
                self._update(self.Y_b, e_b, h_b, h_f)
                self._update(self.H_b, h_b, s_b, s_f)

                # pripocitaj chybu
                re += self._cost(e_f, end)

            touched = 0
            miss = 0
            didntTouch = 0

            for start, end in data:
                prediction = self.forward(start)
                roundPrediction = np.array([0 if i < 0.5 else 1 for i in prediction])
                if np.all(roundPrediction == end):
                    continue

                if np.all(noTouch == end):
                    touched += 1
                    continue

                if np.all(roundPrediction == noTouch):
                    didntTouch += 1
                    continue

                miss += 1

            err = touched + didntTouch + miss
            errSplit.append(((touched / n) * 100, (didntTouch / n) * 100, (miss / n) * 100))
            errT.append(((err / n) * 100, err))
            REs.append(re / n)
            i += 1
        return list(zip(REs, errT, errSplit))
