import json
import random
import numpy as np
import math
import matplotlib.pyplot as plt

# symetricke vahy (len jedna matica), ucenie hebbovske,
#
# t.j. delta w_{ij} = \alpha x_i y_j.
# A potom aktivacie neuronov:
# y_j = sigmoid(sum w_{ji} x_i), x_i = sigmoid(sum w_{ij} y_j).
# Aktivity x_i a y_j \in {0,1}.



@np.vectorize
def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        print("FAILED on {}".format(-x))



def analyze_data(data):
    seenP = {}
    duplicatesP = 0
    seenT = {}
    duplicatesT = 0
    for i in range(len(data)):
        proprio, touch = data[i]
        array = str(proprio)
        touch = str(touch)
        if array not in seenP:
            seenP[array] = True
        else:
            # already there..
            duplicatesP += 1

        if touch not in seenT:
            seenT[touch] = set()
            seenT[touch].add(array)
        else:
            # already there..
            seenT[touch].add(array)
            duplicatesT += 1

    print("+======DATA ANALYSIS=======+")
    print("Number of samples: {0}".format(len(data)))
    print("Number of UNIQUE samples: {0}".format(len(data) - duplicatesP))
    print("[which is {0}%]".format(100 - ((duplicatesP / len(data)) * 100)))


class Hebbian:

    def __init__(self, inp, out, testdata, alpha=0.25, f=sigmoid):
        scale = 1 / (1 + math.exp(-inp+1))

        self.W = np.random.normal(loc=0, scale=0, size=(inp, out))
        self.W_back = np.random.normal(loc=0, scale=0, size=(out,inp))

        self.alpha = alpha
        self.f = f
        self.testdata = testdata

    def _forward(self, imput):
        a = self.f(np.dot(imput, self.W))
        return imput, a

    def forward(self, imput):
        return self._forward(imput)[1]

    def _backward(self, output):
        a = self.f(np.dot(output, self.W_back))
        return a, output

    def backward(self, imput):
        return self._backward(imput)[1]

    def _cost(self, target, outputs):
        return np.sum((target - outputs) ** 2)

    def _update(self, x, y):
        #the entire equation --> delta w_{ij} = \alpha (x_i y_j - w_{ij}*y_j^2)
        #outerProduct = np.outer(x, y) # x_i y_j
        #Δw_{ij} = η.out_j*in_i − η.outj.wij.outj
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                xiyj = x[i] * y[j]
                oja = self.W[i][j] * y[j]**2
                delta = self.alpha * (xiyj - oja)
                self.W[i][j] += delta

        # ---- this was working (learning happened), but not correct.
        #delta w_{j} = \alpha (x y_j - w_{j}*y_j^2)
        # oja = self.W.dot(y**2)
        # outerProduct = np.outer(x, y)
        # for j in range(self.W.shape[0]):
        #     self.W[j] += self.alpha * ( outerProduct[j] - oja[j])

    def _update_back(self, x, y):
        for i in range(self.W_back.shape[0]):
            for j in range(self.W_back.shape[1]):
                xiyj = x[i] * y[j]
                oja = self.W_back[i][j] * y[j]**2
                delta = self.alpha * (xiyj - oja)
                self.W_back[i][j] += delta

        # oja = self.W.dot(y ** 2)
        # outerProduct = np.outer(x, y)
        # for j in range(self.W.shape[0]):
        #     self.W[j] += self.alpha * (outerProduct[j] - oja[j])


    def train(self, data, epoch=300,):
        show_weights = True
        change_learningRate_coef = int((epoch // (self.alpha / 0.005)) * 2)  # int((epoch // (self.alpha / 0.005)) * 1.5)
        REs = []
        REs_back = []
        errT = []
        re = 100
        re_back = 100
        i = 0
        n = len(data)

        errSplit = []

        print("Training start: ALPHA changes after every --{}-- eps.".format(change_learningRate_coef))

        while re/n > 0.0:
            if i == epoch:
                break

            re = 0
            re_back = 0

            # stats
            if i % 5 == 0:
                print("Epoch: {}/{}".format(i, epoch))
                print("Alpha: ", self.alpha)
            if i % change_learningRate_coef == 0 and i != 0:
                # decries learning rate
                self.alpha *= 0.95
                if self.alpha < 0.005:
                    self.alpha = 0.005
                self.alpha = max(self.alpha, 0.005)

                if show_weights:
                    plt.imshow(self.W, cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    plt.draw()

                    plt.imshow(self.W_back, cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    plt.draw()

            # do epoch
            for start, end in np.random.permutation(data):
                # update smer dopredu
                self._update(start, end)

                # update smer dozadu
                self._update_back(end, start)

                povodne_f, out_f = self._forward(start)
                out_b, povodne_b = self._backward(end)

                # pripocitaj chybu
                re += self._cost(end, out_f,)
                re_back += self._cost(start, out_b)

            differentNeuronPrediction = 0
            zeroVectorPrediction = 0
            for proprio, touchInfo in data:
                prediction = self.forward(proprio)
                roundPrediction = np.array([0 if i < 0.4 else 1 for i in prediction])

                if np.all(roundPrediction == touchInfo):
                    # is OK
                    continue
                elif sum(roundPrediction) == 0:
                    zeroVectorPrediction += 1
                else:
                    differentNeuronPrediction += 1

            print('------')
            print(np.sum(self.W))
            print("MAX value is {0}".format(np.max(self.W)))
            print('------')
            print(np.sum(self.W_back))
            print("***")
            print("# of error where full-0 vector was predicted instead of touch: {0}".format(zeroVectorPrediction))
            print("# of errors where different touch was predicted: {0}".format(differentNeuronPrediction))
            err = zeroVectorPrediction + differentNeuronPrediction
            print("Current err value: {0}/{1}".format(err, len(data)))

            errT.append(((err / len(data)) * 100, err))
            REs.append(re / n)
            REs_back.append(re_back / n)
            print("[FW] appended {0} ".format(re/n))
            print("[BW] appended {0} ".format(re_back/n))
            i += 1


        print("HEBBIAN learning finished!")

        #Frobenian norm

        np.linalg.norm(self.W, ord='fro')

        #----------Train data
        err = 0
        for X, y in data:
            prediction = self.forward(X)
            roundPrediction = np.array([0 if i < 0.5 else 1 for i in prediction])
            if not np.all(roundPrediction == y):
                err += 1
        trainingAcc = 100 - ((err / len(data)) * 100)
        print("Training: succ", trainingAcc, "%")

        #----------Test data
        err = 0
        for X, y in self.testdata:
            prediction = self.forward(X)
            roundPrediction = np.array([0 if i < 0.5 else 1 for i in prediction])
            if not np.all(roundPrediction == y):
                err += 1

        testingSucc = -1
        if len(self.testdata) != 0:
            testingSucc = 100 - ((err / len(self.testdata)) * 100)
            print("Testing: succ", testingSucc, "%")

        return errT, REs, REs_back, errSplit

def get_xor_data(count=50):
    res = []

    one = [0, 1]
    two = [1, 0]
    three = [0, 0]
    four = [1, 1]

    for i in range(count):
        res.append( (np.array(one), np.array([1])) )
        res.append( (np.array(two), np.array([1])) )
        res.append( (np.array(three), np.array([0])) )
        res.append( (np.array(four), np.array([0])) )
    return res


def hebbian_train(inputDataFolder):

    suffix = ".baldata"
    #trainingName = "lh"
    #trainingName = "rh"
    trainingName = "lf"
    #trainingName = "rf"
    #trainingName = "all"

    #data loading
    with open(inputDataFolder + trainingName + suffix) as f:
        data = json.load(f)

    labeled = [(np.array(i['pos']), np.array(i["touch"])) for i in data]

    random.shuffle(labeled)

    #analyze_data(labeled)
    #REWPLACE with XOR data - for simplicity

    labeled = labeled[:50]
    #labeled = get_xor_data(200)

    #-----splitting train/test
    SplitData = True
    if not SplitData:
        line = len(labeled)
    else:
        line = int(len(labeled) * 0.7)
    trainig = labeled[0:line]
    test = labeled[line:]

    inp, out = len(labeled[0][0]), len(labeled[0][1])
    model = Hebbian(inp, out, test, alpha=0.5)
    errT, REs, REs_back, errSplit = model.train(trainig, epoch=100)

if __name__ == "__main__":
    """
    Training associators on transformed data in ./bal_data/[trainingName].baldata
    """
    np.set_printoptions(threshold=np.inf)
    hebbian_train("bal_data/my_data_act/") #), show_graph=True)

#LEFT - FORE
#90%, 70%
#87%, 67%
#90%, 83%
