import json, random
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('D:/DP_Budovanie_schemy_tela')
import data_processing.ubal.data_util as du
#from myubal import MyUBal, balSuportLearning
#from myubal_old import BalResults
from  myubal_old import MyUBal, balSuportLearning, BalResults
from sklearn.model_selection import train_test_split

def get_avg_mse_on_index(lists, index):
    vals = []
    for list in lists:
        print(list[index])
        vals.append(list[index][0]) #always [0], because the item in this list is a tuple; the first is the mean square error
        #replot.plot([i for i, _, _ in res]) <----------------------- this is the tuple ^
    print("AVG MSE for index {0} is : {1}".format(index, sum(vals) / len(vals)))
    return sum(vals) / len(vals)

def get_avg_ce_on_index(lists, index):
    vals = []
    for list in lists:
        vals.append(list[index][1])  # always [1], because the item in this list is the classification error of the result
        # errplot.plot([i for _, (i, _), _ in res] <----------------------- this  ^
    #print("AVG CE for index {0} is : {1}".format(index, sum(vals) / len(vals)))
    return sum(vals[0]) / len(vals)

def compute_avg_list(inputList, metricType): #list of lists.
    res = []
    if metricType == "MSE":
        all_mse = [balResult.MeanSquareErrors() for balResult in inputList]
        return list(map(lambda x: sum(x) / len(x), zip(*all_mse))) #this is a magic which does column avg. More info here: https://stackoverflow.com/questions/10919664/averaging-list-of-lists-python-column-wise
    elif metricType == "CE":
        all_ce = [balResult.ClassificationErrorsPercentages() for balResult in inputList]
        return list(map(lambda x: sum(x) / len(x), zip(*all_ce)))  # this is a magic which does column avg. More info here: https://stackoverflow.com/questions/10919664/averaging-list-of-lists-python-column-wise
    else:
        raise Exception("unknown metric type. can't compute :(")

def generateMockData():
    mockData = generate_vectors(500)
    return mockData

def generate_vectors(numberofVectors):
    result = []
    for i in range(numberofVectors):
        proprioMock, touchMock = generate_random_binary_vector(144, 4)
        result.append({"pos": proprioMock, "touch": touchMock})

    return result

def generate_random_binary_vector(length, numberOfOnes):
    proprioMock = [0 for i in range(length)]
    ones = [random.randint(0, length-1) for i in range(numberOfOnes)]
    for index in ones:
        proprioMock[index] = 1

    touchMock = [0 for i in range(length)]
    ones = [random.randint(0, length-1) for i in range(numberOfOnes)]
    for index in ones:
        touchMock[index] = 1

    return proprioMock, touchMock

def get_random_subset(data, count):
    used_indices = {}
    res = []
    for i in range(count):
        while True:
            index = random.randint(0, len(data)-1)
            if index not in used_indices:
                used_indices[index] = 1
                res.append(data[index])
                break
    return res


def train_bal(inputDataFolder, show_graph=False):
    suffix = ".baldata"
    trainingName = "lh"
    # trainingName = "rh"
    # trainingName = "lf" # THIS IS GOOD DONT RETRAIN
    #trainingName = "rf"  #
   #trainingName = "lf_rh"
    #trainingName = "all"  #

    SplitData = True
    finalTraining = True #whether this model should be saved
    mockDataTest = False
    saveModel = True
    # makeRandomBatch = False

    with open(inputDataFolder + trainingName + suffix) as f:
        data = json.load(f)
    data_len_orig = len(data)
    # if makeRandomBatch:
    #     data = get_random_subset(data, 300)

    labeled = [(np.array(i['pos']), np.array(i["touch"])) for i in data]

    # random.shuffle(labeled)
    #removePart = int(len(labeled) * 0.8)
    #labeled = labeled[0:removePart]
    # labeled = du.filterAmbiguousTouch(labeled)
    # print("Filtered out {} out of {}".format(len(labeled),data_len_orig))

    du.analyze_data(labeled)
    print(np.shape(labeled))
    training,test = train_test_split(labeled, test_size=0.2)
    print(np.shape(training))
    print(np.shape(test))

    # if not SplitData:
    #     line = len(labeled)
    # else:
    #     line = int(len(labeled) * 0.8)
    # training = labeled[0:line]
    # test = labeled[line:]

    print("\n::::: TRAIN data :::::")
    du.analyze_data(training)
    print("\n::::: TEST data :::::")
    du.analyze_data(test)

    # print(training[0])
    # exit()

    hiddenLayer = 100 # for one associator best is 350
    betas = [0.0, 1.0, 0.0]
    gammasF = [float("nan"), 1.0, 1.0]
    gammasB = [1.0, 1.0, float("nan")]
    # betas = [1.0, 1.0, 0.9]
    # gammasF = [float("nan"), 1.0, 1.0]
    # gammasB = [0.9, 1.0, float("nan")]
    alpha = 0.3
    inp, hid, out = len(labeled[0][0]), hiddenLayer, len(labeled[0][1])
    model = MyUBal(inp=inp, hid=hid, out=out, betas=betas, gammasF=gammasF, gammasB=gammasB, alpha=alpha, testData=test)
    print("inp:{} hid:{} out:{}".format(inp, hid, out))
    balResults = model.train(training, epoch=200)

    if show_graph:
        # plt.show()
        # replot = plt.subplot(311)
        # replot.set_title("mean squared error")
        # replot.plot(balResults.MeanSquareErrors())
        # plt.savefig("FINAL_{0}-mse".format(trainingName))
        # print("SAVED SUCC")
        # plt.show()
        #
        # replot_back = plt.subplot(312)
        # replot_back.set_title("mean squared error - backwards")
        # replot_back.plot(balResults.MeanSquareErrorsBack())
        # plt.savefig("FINAL_{0}-mse_back".format(trainingName))
        # print("SAVED SUCC")
        # plt.show()

        fig, errplot = plt.subplots()
        errplot.set_title("classification error [%]")
        errplot.set_ylim([0, 100])
        # print(balResults.ClassificationErrorsPercentages())
        epochs = list(range(len(balResults.ClassificationErrorsPercentages())))
        # print(epochs)
        errplot.plot(epochs,balResults.ClassificationErrorsPercentages())
        errplot.plot(epochs,balResults.ClassificationErrorsPercentagesTest())
        plt.savefig("FINAL_{0}-ce".format(trainingName))
        print("SAVED SUCC")
        plt.show()

        # errplot2 = plt.subplot(313)
        # errplot2.set_title("classification values [# of errors out of all data per EP]")
        # errplot2.plot(balResults.ClassificationErrorsValues())
        #plt.savefig("FINAL_{0}-ce".format(trainingName))
        #print("SAVED SUCC")
        #plt.show()

    err = 0
    for X, y in test:
        X = np.expand_dims(np.transpose(X), axis=1)
        prediction = model.network.activation_fp_last(X)
        prediction = np.transpose(np.squeeze(prediction))
        prediction = balSuportLearning(prediction)
        if not np.all(prediction == y):
            err += 1

    testingSucc = -1
    if len(test) != 0:
        testingSucc = 100 - ((err / len(test)) * 100)
        print("testing: succ", testingSucc, "%")

    # if finalTraining:
    #     if saveModel:
    #         model.save("./trained/" + trainingName)
    #         model = MyUBal(inp, hid, out, test)
    #         model.load("./trained/" + trainingName)

    err = 0
    for X, y in training:
        X = np.expand_dims(np.transpose(X), axis=1)
        prediction = model.network.activation_fp_last(X)
        prediction = np.transpose(np.squeeze(prediction))
        prediction = balSuportLearning(prediction)
        if not np.all(prediction == y):
            err += 1

    trainingAcc = 100 - ((err / max(1, len(training))) * 100)
    print("training succ: ", trainingAcc, "%")
    return balResults, trainingAcc, testingSucc


def run_multiple_bal_trainings():
    results = []
    testAccs = []
    trainAccs = []
    dir = "data/ubal/trained"
    dir_names = ["0{0}/".format(i) for i in range(1, 10)]
    dir_names.append("10/")
    #for directory in dir_names:
    for i in range(4):
        print("------ RUNNING FOR {0}".format(i))
        directory = dir + dir_names[i]
        res, trainAcc, testAcc = train_bal(directory)
        results.append(res)
        testAccs.append(testAcc)
        trainAccs.append(trainAcc)

    #compute averages

    avg_mse = compute_avg_list(results, "MSE")
    avg_classErrAcc = compute_avg_list(results, "CE")
    avg_testAcc = sum(testAccs) / len(testAccs)
    avg_trainAcc = sum(testAccs) / len(testAccs)

    # show the graph of x runs - MSE
    plt.show()
    replot = plt.subplot(311)
    replot.set_title("mean squared error")
    for res in results:
        replot.plot(res.MeanSquareErrors(), color="lightblue")
    replot.plot(avg_mse, color="orange")
    plt.savefig("LF-mse")
    print("SAVED SUCC")
    plt.show()

    #graph of x runs - CE
    errplot = plt.subplot(313)
    errplot.set_title("classification error [%]")
    errplot.set_ylim([0, 100])
    for res in results:
        errplot.plot(res.ClassificationErrorsPercentages(), color="lightblue")

    errplot.plot(avg_classErrAcc, color="orange")
    plt.savefig("LF-ce")
    print("SAVED SUCC")
    plt.show()

    #graph of x runs - training acc
    trainplot = plt.subplot(313)
    trainplot.set_title("Training accuracy [%]")
    trainplot.set_ylim([0, 100])
    trainplot.plot(trainAccs, color="lightblue")
    trainplot.plot(avg_trainAcc, color="orange")
    plt.savefig("LF-train")
    print("SAVED SUCC")
    plt.show()

    # graph of x runs - testing acc
    testplot = plt.subplot(313)
    testplot.set_title("Testing accuracy [%]")
    testplot.set_ylim([0, 100])
    testplot.plot(testAccs, color="lightblue")
    testplot.plot(avg_testAcc, color="orange")
    plt.savefig("LF-test")
    print("SAVED SUCC")
    plt.show()


if __name__ == "__main__":
    """
    Training associators on transformed data in ./bal_data/[trainingName].baldata
    """
    MultiRuns = False

    if MultiRuns:
        run_multiple_bal_trainings()
    else:
        #just one
        res, trainAcc, testAcc = train_bal("C:/Users/cidom/OneDrive/Dokumenty/mAIN1/dp_harvanova/dp_harvanova/src/ubal_data/ubal_data_leyla/", show_graph=True)
        # i = 1
        #for error in res.PredictionErrors():
        #    print("Error#{0} : ".format(i))
        #    print(error)

