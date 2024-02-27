import math
import numpy as np
import matplotlib.pyplot as plt
import os



@np.vectorize
def balSuportLearning(x):
    if x < 0.35:
        return max(0, x-0.15)
    if x > 0.65:
        return min(1, x+0.15)
    return x

@np.vectorize
def balBasicLearning(x):
    if x < 0.5:
        return 0
    else:
        return 1


@np.vectorize
def balFloor(x):
    if x < 0.35:
        return 0
    if x > 0.75:
        return 1
    return x


def identity(x):
    return x


@np.vectorize
def sigmoid(x, k=2):
    return 1 / (1 + math.exp(k*-x))


class MyBal:
    """
        Enhanced BAL
    """

    def __init__(self, inp, hid, out, testData=[], alpha=0.25, f=sigmoid, enhanceLearning=balSuportLearning):
        scale = 1 / math.sqrt(inp + 1)

        self.H_f = np.random.normal(loc=0, scale=scale, size=(inp, hid))
        self.Y_b = np.random.normal(loc=0, scale=scale, size=(out, hid))

        self.Y_f = np.random.normal(loc=0, scale=scale, size=(hid, out))
        self.H_b = np.random.normal(loc=0, scale=scale, size=(hid, inp))

        self.alpha = alpha
        self.f = f
        self.g = f
        self.enhance = enhanceLearning

        self.results = BalResults()
        self.testdata = testData

    def save(self, fileName):
        try:
            os.mkdir(fileName + "/")
        except OSError:
            pass

        np.savetxt(fileName + "/hf.txt", self.H_f)
        np.savetxt(fileName + "/yf.txt", self.Y_f)

        np.savetxt(fileName + "/hb.txt", self.H_b)
        np.savetxt(fileName + "/yb.txt", self.Y_b)

    def load(self, fileName):
        self.H_f = np.loadtxt(fileName + "/hf.txt")
        self.Y_b = np.loadtxt(fileName + "/yb.txt")

        self.Y_f = np.loadtxt(fileName + "/yf.txt")
        self.H_b = np.loadtxt(fileName + "/hb.txt")

    def _forward(self, imput):
        a = self.f(np.dot(imput, self.H_f))
        b = self.f(np.dot(a, self.Y_f))

        # zaokruhlovanie
        b = self.enhance(b)
        return imput, a, b

    def forward(self, imput):
        return self._forward(imput)[2]

    def forward_hiddenOutput(self, imput):
        return self._forward(imput)[1]

    def _backward(self, imput):
        a = self.f(np.dot(imput, self.Y_b))
        b = self.f(np.dot(a, self.H_b))

        # zaokruhlovanie
        b = self.enhance(b)
        return b, a, imput

    def backward(self, imput):
        return self._backward(imput)[0]

    def backward_hiddenOutput(self, imput):
        return self._backward(imput)[1]

    def _update(self, W, before, forward, backward):
        dif = self.alpha * (backward - forward)
        for p in range(W.shape[0]):
            W[p] += before[p] * dif

    def _cost(self, target, outputs):
        return np.sum((target - outputs) ** 2)

    def _draw(self, replot, REs, errplot, errTr):
        replot.clear()
        replot.set_title("mean sqr")
        replot.plot(REs[-1500:])

        errplot.clear()
        errplot.set_title("classification error in %")
        errplot.plot([i for i, _ in errTr][-1500:])
        errplot.set_ylim([-5, 105])
        plt.draw()
        # plt.show()
        plt.pause(1)

    def train(self, data, epoch=None, track=False):
        """
        Training method
        data=[(X, Y)]
        """
        if track:
            plt.ion()
            plt.show(block=False)
            replot = plt.subplot(311)
            errplot = plt.subplot(313)
            print(self.H_f.shape[0], self.H_f.shape[1], self.Y_f.shape[1])

        change_learningRate_coef = int((epoch // (self.alpha / 0.005)) * 1.5) #25 #
        print("Change of learning at every {0} Epoch".format(change_learningRate_coef))
        REs = []
        REs_back = []
        errTr = []
        errTs = []
        re = 100
        re_back = 100
        i = 0
        n = len(data)

        errSplit = []

        while re/n > 0.0:
            if i == epoch:
                break

            re = 0
            re_back = 0

            # stats
            # if i % 5 == 0:
                # print("epoch", i)
                # print("alpha", self.alpha)
            if i % change_learningRate_coef == 0 and i != 0:

                # decries learning rate
                self.alpha *= 0.95
                if self.alpha < 0.005:
                    self.alpha = 0.005
                self.alpha = max(self.alpha, 0.005)

                # draw
                if track and i % 50 == 0:
                    self._draw(replot, REs, errplot, errTr)

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
                re += self._cost(end, e_f)

                re_back += self._cost(start, s_b)

            differentNeuronPrediction = 0
            zeroVectorPrediction = 0
            for proprio, touchInfo in data:
                prediction = self.forward(proprio)
                roundPrediction = np.array([0 if i < 0.4 else 1 for i in prediction])

                if np.all(roundPrediction == touchInfo):
                    #is OK
                    continue
                elif sum(roundPrediction) == 0:
                    zeroVectorPrediction += 1
                else:
                    differentNeuronPrediction += 1

            # print("# of error where full-0 vector was predicted instead of touch: {0}".format(zeroVectorPrediction))
            # print("# of errors where different touch was predicted: {0}".format(differentNeuronPrediction))
            err = zeroVectorPrediction + differentNeuronPrediction
            err_ratio_train = (err / len(data)) * 100
            err_test,len_test = self.test()
            err_ratio_test = (err_test / len_test) * 100
            # print("Training: {}/{}\t acc: {:.3f}%".format(err, len(data), (err/len(data)*100)))
            errSplit.append([differentNeuronPrediction, differentNeuronPrediction, zeroVectorPrediction])
            # log
            if i % 10 == 0:
                print("Ep {} error train: {:.3f}%\ttest: {:.3f}%".format(i,err_ratio_train, err_ratio_test))

            errTr.append((err_ratio_train, err))
            errTs.append((err_ratio_test, err_test))
            REs.append(re / n)
            REs_back.append(re_back / n)
            # print("[FW] appended {0} ".format(re/n))
            # print("[BW] appended {0} ".format(re_back/n))
            i += 1
            # if err_ratio_test >= 80 and err_ratio_test <= 95 and i >= epoch//2:
            # if err_ratio_test < 20 and err_ratio_test >= 5 and i >= epoch//2:
            if err_ratio_test < 20 and i >= epoch//2:
                print(err_ratio_test)
                break

        if track:
            plt.ioff()
        print("BAL learning finished!")
        self.results.set_classification_errors(errTr,errTs)
        self.results.set_mse(REs)
        self.results.set_mse_back(REs_back)
        self.results.set_touchInfo(errSplit)
        return self.results

    def test(self):
        err = 0
        for X, y in self.testdata:
            prediction = self.forward(X)
            if not np.all(prediction == y):
                err += 1
        if len(self.testdata) != 0:
            return err,len(self.testdata)


class BalResults:
    def __init__(self):
        self.meanSquareErrorsBack = []
        self.meanSquareErrors = [] #forward
        self.touchInfo = []
        self.classificationErrorsTrain = []
        self.classificationErrorsTest = []

    def set_mse_back(self, mseList):
        self.meanSquareErrorsBack = mseList

    def set_mse(self, mseList):
        self.meanSquareErrors = mseList

    def set_classification_errors(self, allClassErrors, testallClassErrors):
        self.classificationErrorsTrain = allClassErrors
        self.classificationErrorsTest = testallClassErrors

    def set_touchInfo(self, touchErrors):
        self.touchInfo = []
        for touchError in touchErrors:
            self.touchInfo.append(TouchItem(touchError))

    def ClassificationErrorsValues(self):
        return [value for _, value in self.classificationErrorsTrain]

    def ClassificationErrorsPercentages(self):
        return [perc for perc, _ in self.classificationErrorsTrain]

    def ClassificationErrorsPercentagesTest(self):
        return [perc for perc, _ in self.classificationErrorsTest]

    def MeanSquareErrors(self):
        return self.meanSquareErrors

    def MeanSquareErrorsBack(self):
        return self.meanSquareErrorsBack

    def PredictionErrors(self):
        return self.touchInfo

class TouchItem(object):

    def __init__(self, item):
        self.predictedTouchButWasnt = item[0] #predicted 1 on a place where 0 should be
        self.predictedNoTouchButWas = item[1] #predicted 0 but should've been 1
        self.predictedZeroVector = item[2]

    def __str__(self):
        info = "====== touch info ======"
        string1 = "Predicted touch but wasn't: {0} \n".format(self.predictedTouchButWasnt)
        string2 = "Predicted no touch, but was: {0} \n".format(self.predictedNoTouchButWas)
        string3 = "At least one touch value missed: {0} \n".format(self.predictedZeroVector)
        return  info + string1 + string2 + string3
