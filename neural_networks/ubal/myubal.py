import math
import numpy as np
#from data_processing.ubal import data_util as du
#import data_processing.ubal.data_util as du
#from ...data_processing.ubal import data_util as du
import sys
sys.path.append('D:/DP_Budovanie_schemy_tela')
from data_processing.ubal import data_util as du


from UBAL_numpy import Sigmoid, SoftMax, UBAL


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

    def train_minimal(self, data, max_epoch=None):
        for e in range(max_epoch):
            matches_train = 0
            for X, y in np.random.permutation(data):
                X = np.expand_dims(np.transpose(X), axis=1)
                y = np.expand_dims(np.transpose(y), axis=1)
                act_FP, act_FE, act_BP, act_BE = self.network.activation(X, y)
                self.network.learning(act_FP, act_FE, act_BP, act_BE)
                train_output_winners = np.argmax(act_FP[2], axis=0)
                train_target_winners = np.argmax(y, axis=0)
                matches_train += np.sum(train_output_winners == train_target_winners)
            if e == 0 or (e+1) % 5 == 0:
                print("Ep {} acc train: {:.3f}%".format(e+1, (matches_train / len(data)) * 100))
        return self.network

    def train(self, data, max_epoch=None, track=False):
        for e in range(max_epoch):
            err_train = 0
            re = 0
            re_back = 0
            for X, y in np.random.permutation(data):
                # print("inp",X)
                # print("targ",y)
                # print()
                X = np.expand_dims(np.transpose(X), axis=1)
                y = np.expand_dims(np.transpose(y), axis=1)
                act_FP, act_FE, act_BP, act_BE = self.network.activation(X, y)
                self.network.learning(act_FP, act_FE, act_BP, act_BE)
                # train_output_winners = np.argmax(act_FP[2], axis=0)
                # train_target_winners = np.argmax(end, axis=0)
                # acc_train += np.sum(train_output_winners == train_target_winners)
                prediction_binary = du.binarize(act_FP[2])
                target = np.transpose(np.squeeze(y))
                # print(np.argwhere(prediction_binary == np.amax(prediction_binary)).flatten().tolist())
                # print(np.argwhere(target == np.amax(target)).flatten().tolist())
                if not np.all(prediction_binary == target):
                    err_train += 1
                re += self.mean_squared_error(X, act_BP[0])
                re_back += self.mean_squared_error(y, act_FP[2])
            test_acc,len_test = self.test()
            err_train_epoch = (err_train / len(data)) * 100
            err_test_epoch = (test_acc / len_test) * 100
            mse_train_epoch = (re / len(data)) * 100
            mse_train_back_epoch = (re_back / len(data)) * 100
            if e == 0 or (e+1) % 20 == 0:
                print("Ep {} err train: {:.3f}%\ttest: {:.3f}%\tMSE tr: {:.3f}\tback: {:.3f}".
                      format(e+1, err_train_epoch, err_test_epoch, mse_train_epoch, mse_train_back_epoch))
        return self.network

    def test(self):
        err = 0
        for X, y in self.testdata:
            X = np.expand_dims(np.transpose(X), axis=1)
            prediction = self.network.activation_fp_last(X)
            prediction = np.transpose(np.squeeze(prediction))
            prediction_binary = du.binarize(prediction)
            if not np.all(prediction_binary == y):
                err += 1
        # print("err:",err)
        if len(self.testdata) != 0:
            return err,len(self.testdata)
