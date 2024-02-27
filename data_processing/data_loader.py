import json
import random
import numpy as np

""" Loads my_data_act.data and prepares the contents for further processing (i.e for MLP or BAL training/testing sets).

    Args:
    -file_path: file to load (assumes *.data format)
    -data_slip: Default is 80/20
    -shuffle: whether the loaded data should be shuffled randomly
    -ignoreHalfOfData: default = false. Used for debugging/faster training
"""
class DataLoader:

    def __init__(self, file_path="../my_data_act.data", data_split=0.8, shuffle=True, ignoreHalfOfData=False):
        f = open(file_path)
        collected_data = json.load(f)
        f.close()

        collected_data = self.__process(collected_data)
        if shuffle:
            random.shuffle(collected_data)

        if ignoreHalfOfData:
            ignoredAmount = len(collected_data)//2
            collected_data = collected_data[ignoredAmount:]

        separator = int(len(collected_data) * data_split)

        trainX = np.array([i['pos'] for i in collected_data[0:separator]])
        trainY = np.array([i["touch"] for i in collected_data[0:separator]])

        testX = np.array([i['pos'] for i in collected_data[separator:]])
        testY = np.array([i["touch"] for i in collected_data[separator:]])

        #store in class
        self.__collected_data = collected_data
        self.__trainingX = trainX
        self.__testingX = testX
        self.__trainingY = trainY
        self.__testingY = testY

    def __process(self, labeld):
        res = []
        for data in labeld:
            res.append(
                {
                    "pos": data["proprio"],
                    "touch": 1 if sum(data["touch"]) > 0 else 0
                })
        return res

    def training_X(self):
        return self.__trainingX

    def testing_X(self):
        return self.__testingX

    def training_Y(self):
        return self.__trainingY

    def testing_Y(self):
        return self.__testingY


