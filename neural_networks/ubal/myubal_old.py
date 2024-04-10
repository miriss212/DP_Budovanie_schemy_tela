import math
import numpy as np
import matplotlib.pyplot as plt
import os

from neural_networks.ubal.UBAL_numpy import Sigmoid, SoftMax, UBAL

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


class MyUBal:

    def __init__(self, inp, hid, out, betas, gammasF, gammasB, alpha,
                 init_w_mean=0.0, init_w_variance=1.0, testData=[]):
        self.alpha = alpha
        self.testdata = testData
        self.results = BalResults()
        sigmoid = Sigmoid()
        softmax = SoftMax()
        act_fun_F = [sigmoid, sigmoid, softmax]
        # act_fun_F = [sigmoid, sigmoid, sigmoid]
        act_fun_B = [sigmoid, sigmoid, sigmoid]
        self.network = UBAL([inp, hid, out], act_fun_F, act_fun_B,
                       alpha, init_w_mean, init_w_variance,
                       betas, gammasF, gammasB)

    def mean_squared_error(self, desired, estimate):
        return ((desired - estimate)**2).mean()

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

        # change_learningRate_coef = int((epoch // (self.alpha / 0.005)) * 1.5) #25 #
        # change_learningRate_coef = 5
        # print("Change of learning at every {0} Epoch".format(change_learningRate_coef))
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

            acc_train = 0
            re = 0
            re_back = 0

            # stats
            # if i % 5 == 0:
                # print("epoch", i)
                # print("alpha", self.alpha)
            # if i % change_learningRate_coef == 0 and i != 0:
            #
            #     # decries learning rate
            #     self.alpha *= 0.95
            #     if self.alpha < 0.005:
            #         self.alpha = 0.005
            #     self.alpha = max(self.alpha, 0.005)
            #
            #     # draw
            #     if track and i % 50 == 0:
            #         self._draw(replot, REs, errplot, errTr)
            # do epoch
            for start, end in np.random.permutation(data):
                # print("inp",start)
                # print("targ",end)
                start = np.expand_dims(np.transpose(start), axis=1)
                end = np.expand_dims(np.transpose(end), axis=1)
                act_FP, act_FE, act_BP, act_BE = self.network.activation(start, end)
                self.network.learning(act_FP, act_FE, act_BP, act_BE)
                train_output_winners = np.argmax(act_FP[2], axis=0)
                train_target_winners = np.argmax(end, axis=0)
                acc_train += np.sum(train_output_winners == train_target_winners)
                re += self.mean_squared_error(start, act_BP[0])
                re_back += self.mean_squared_error(end, act_FP[2])

            differentNeuronPrediction = 0
            zeroVectorPrediction = 0
            for proprio, touchInfo in data:
                proprio = np.expand_dims(np.transpose(proprio), axis=1)
                prediction = self.network.activation_fp_last(proprio)
                prediction = balSuportLearning(prediction)
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
            err_test,len_test = self.test(self.network)
            err_ratio_test = (err_test / len_test) * 100
            # print("Training: {}/{}\t acc: {:.3f}%".format(err, len(data), (err/len(data)*100)))
            errSplit.append([differentNeuronPrediction, differentNeuronPrediction, zeroVectorPrediction])
            # log
            if i % 25 == 0:
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
            # if err_ratio_test < 5 and i >= epoch//2:
            #     print(err_ratio_test)
            #     break

        if track:
            plt.ioff()
        # print("BAL learning finished!")
        self.results.set_classification_errors(errTr,errTs)
        self.results.set_mse(REs)
        self.results.set_mse_back(REs_back)
        self.results.set_touchInfo(errSplit)
        return self.results, self.network

    def test(self, network):
        err = 0
        for X, y in self.testdata:
            X = np.expand_dims(np.transpose(X), axis=1)
            prediction = network.activation_fp_last(X)
            prediction = np.transpose(np.squeeze(prediction))
            # prediction = balSuportLearning(prediction)
            # print("pred",prediction)
            # print("target",y)
            # print("pred:",np.argmax(prediction)," act:",np.argmax(y))
            targ_winners = np.argmax(y, axis=0)
            pred_winners = np.argmax(prediction, axis=0)
            if not np.all(targ_winners == pred_winners):
            # if not np.all(prediction == y):
                err += 1
        # print("err:",err)
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
