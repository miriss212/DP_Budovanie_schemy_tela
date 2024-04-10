import json, random
import numpy as np
import matplotlib.pyplot as plt
import random

try:
    from .mybal import MyBal
    from .mybal import BalResults
except Exception:
    from mybal import MyBal
    from mybal import BalResults

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
            #already there..
            duplicatesP += 1

        if touch not in seenT:
            seenT[touch] = set()
            seenT[touch].add(array)
        else:
            #already there..
            seenT[touch].add(array)
            duplicatesT += 1

    print("+======DATA ANALYSIS=======+")
    print("Number of samples: {0}".format(len(data)))
    print("Number of UNIQUE samples: {0}".format(len(data) - duplicatesP))
    print("[which is {0}%]".format(100 - ((duplicatesP / len(data))*100)))
    print("---")
    print("Number of UNIQUE touches: {0}".format(len(data) - duplicatesT))
    i = 1
    for touchKey in seenT.keys():
        print("For touch #{0}, the amount of proprio configs are {1}".format(i, len(seenT[touchKey])))
        i += 1


def filterAmbiguousTouch(labeld):
    res = []
    seen = {}
    for pos, touch in labeld:
        tmp = np.array(pos)
        idx1 = np.argmax(tmp)
        tmp[idx1] = 0
        idx2 = np.argmax(tmp)
        key = idx1 * 1000 + idx2
        val = np.argmax(touch)

        if key not in seen:
            seen[key] = {val: {"p": pos, "t": touch, "s": 1}, "max": 1, "maxP": val}
        elif val in seen[key]:
            seen[key][val]["s"] += 1
            numb = seen[key][val]["s"]
            if numb > seen[key]["max"]:
                seen[key]["max"] = numb
                seen[key]["maxP"] = val
        else:
            seen[key][val] = {"p": pos, "t": touch, "s": 1}

    for key in seen.keys():
        obj = seen[key]
        obj = obj[obj["maxP"]]
        res.append((obj["p"], obj["t"]))
        #res.append((obj["p"], obj["t"]))
        for i in range(obj["s"]):
            # res.append((obj["p"], obj["t"]))
            pass

    return res

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
    #trainingName = "lf" # THIS IS GOOD DONT RETRAIN
    #trainingName = "rf"  #
   #trainingName = "lf_rh"
    #trainingName = "all"  #

    SplitData = True
    finalTraining = True #whether this model should be saved
    mockDataTest = False
    saveModel = True
    makeRandomBatch = False

    with open(inputDataFolder + trainingName + suffix) as f:
        data = json.load(f)

    if makeRandomBatch:
        data = get_random_subset(data, 300)

    labeled = [(np.array(i['pos']), np.array(i["touch"])) for i in data]

    random.shuffle(labeled)
    #removePart = int(len(labeled) * 0.8)
    #labeled = labeled[0:removePart]
    labeled = filterAmbiguousTouch(labeled)

    analyze_data(labeled)

    if not SplitData:
        line = len(labeled)
    else:
        line = int(len(labeled) * 0.7)
    trainig = labeled[0:line]
    test = labeled[line:]

    hiddenLayer = 350 # for one associator best is 350
    inp, hid, out = len(labeled[0][0]), hiddenLayer, len(labeled[0][1])
    model = MyBal(inp, hid, out, test, alpha=0.3)
    print("inp:{} hid:{} out:{}".format(inp, hid, out))
    balResults = model.train(trainig, epoch=500)

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
        prediction = model.forward(X)
        if not np.all(prediction == y):
            err += 1

    testingSucc = -1
    if len(test) != 0:
        testingSucc = 100 - ((err / len(test)) * 100)
        print("testing: succ", testingSucc, "%")

    if finalTraining:
        if saveModel:
            model.save("./trained/" + trainingName)
            model = MyBal(inp, hid, out, test)
            model.load("./trained/" + trainingName)

    err = 0
    for X, y in trainig:
        prediction = model.forward(X)
        if not np.all(prediction == y):
            err += 1

    trainingAcc = 100 - ((err / max(1, len(trainig))) * 100)
    print("training succ: ", trainingAcc, "%")
    return balResults, trainingAcc, testingSucc


def run_multiple_bal_trainings():
    results = []
    testAccs = []
    trainAccs = []
    dir = "../bal_data/data"
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
        res, trainAcc, testAcc = train_bal("../bal_data/my_data_act/", show_graph=True)
        i = 1
        #for error in res.PredictionErrors():
        #    print("Error#{0} : ".format(i))
        #    print(error)

