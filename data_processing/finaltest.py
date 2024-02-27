import os

import numpy as np
import json, random

# try:
#     from ./mlp import Mlp
#     from .mybal import MyBal
#     from .som.process_hand import loadSom, loadMrfSom
#     from .proces.processdata import process
# except Exception: #ImportError
#     from proces.processdata import process
#     from mlp.mlp import Mlp
#     from bal.mybal import MyBal
#     from som.som.som import SOM
#     from som.mrf.mrfsom import MRFSOM
#     from som.process_hand import loadSom, loadMrfSom
#
# from .mlp import Mlp
from neural_networks.bal.mybal import MyBal
from neural_networks.mlp.multi_layer_perceptron import MLPWrapper
from rawdata_parser import RawDataParser
from process_data_for_BAL import *


def filt(labeld):
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
            # res.append((pos, touch))
        elif val in seen[key]:
            seen[key][val]["s"] += 1
            numb = seen[key][val]["s"]
            if numb > seen[key]["max"]:
                seen[key]["max"] = numb
                seen[key]["maxP"] = val
            # res.append((pos, touch))
        else:
            seen[key][val] = {"p": pos, "t": touch, "s": 1}

    for key in seen.keys():
        obj = seen[key]
        obj = obj[obj["maxP"]]
        res.append((obj["p"], obj["t"]))
        for i in range(obj["s"]):
            #res.append((obj["p"], obj["t"]))
            pass

    return res



class ComplexModel:

    def __init__(self, filterModel, proprioLeftForearmModel, proprioLeftHandModel, proprioRightForearmModel, proprioRightHandModel,
                 touchHandLeftModel, touchHandRightModel, touchLeftForearmModel, touchRightForearmModel, leftHandAsoc, rightHandAsoc, leftForearmAsoc, rightForeArmAsoc):
        # filter
        self.filt = filterModel
        # arms SOM models
        self.l_f_Som = proprioLeftForearmModel
        self.l_h_Som = proprioLeftHandModel
        self.r_f_Som = proprioRightForearmModel
        self.r_h_Som = proprioRightHandModel
        # touch MRF models
        self.leftHandMRF = touchHandLeftModel
        self.rightHandMRF = touchHandRightModel
        self.forearmLeftMRF = touchLeftForearmModel
        self.forearmRightMRF = touchRightForearmModel
        # associators
        self.leftHandAsoc = leftHandAsoc
        self.rightHandAsoc = rightHandAsoc
        self.leftForearmAsoc = leftForearmAsoc
        self.rightForeArmAsoc = rightForeArmAsoc

        self.UsePerceptron = False


    def test(self, processedData):
        res = {}

        isTouch = sum(processedData["touch"]) != 0
        if self.UsePerceptron:
            predictTouch = self.filt.make_prediction(np.array([processedData["proprio"]]))
            if isTouch != predictTouch:
                res["filt"] = False
                res["ok"] = False
                if isTouch and not predictTouch:
                    res["errorType"] = "Was touch, but predicted NOT touch"
                else:
                    res["errorType"] = "Was NOT touch, but predicted that it was."
                return res
            else:
                #res["filt"] = True
                #res["ok"] = True
                res["info"] = "Was touch, predicted correctly." if isTouch else "Was NOT a touch, predicted correctly."
                if not isTouch:
                    return res

        left_hand = np.array(processedData["leftHandPro"][5:])
        left_forearm = np.array(processedData["leftHandPro"][:5])
        right_hand = np.array(processedData["rightHandPro"][5:])
        right_forearm = np.array(processedData["rightHandPro"][:5])

        proprioLeftFore = self.l_f_Som.winnerVector(left_forearm)
        proprioLeftHand = self.l_h_Som.winnerVector(left_hand)
        proprioRightFore = self.r_f_Som.winnerVector(right_forearm)
        proprioRightHand = self.r_h_Som.winnerVector(right_hand)

        ok = True

        leftLimb = proprioLeftHand + proprioLeftFore
        rightLimb = proprioRightHand + proprioRightFore

        # hands
        leftHandT  = processedData["leftTouch"][0:9]
        if sum(leftHandT) != 0:
            assocPrediction = self.leftHandAsoc.forward(leftLimb + rightLimb)
            res["leftHandAssoc"] = self.evaluate_prediction_match(self.leftHandMRF, leftHandT, assocPrediction)
            ok = ok and res["leftHandAssoc"]

        rightHandT = processedData["rightTouch"][0:9]
        if sum(rightHandT) != 0:
            assocPrediction = self.rightHandAsoc.forward(leftLimb + rightLimb)
            res["rightHandAssoc"] = self.evaluate_prediction_match(self.rightHandMRF, rightHandT, assocPrediction)
            ok = ok and res["rightHandAssoc"]

        # forearm
        leftForeT = processedData["leftTouch"][9:32]
        if sum(leftForeT) != 0:
            assocPrediction = self.leftForearmAsoc.forward(leftLimb + rightLimb)
            res["leftForeAssoc"] = self.evaluate_prediction_match(self.forearmLeftMRF, leftForeT, assocPrediction)
            ok = ok and res["leftForeAssoc"]

        rightForeT = processedData["rightTouch"][9:32]
        if sum(rightForeT) != 0:
            assocPrediction = self.rightForeArmAsoc.forward(leftLimb + rightLimb)
            res["rightForeAssoc"] = self.evaluate_prediction_match(self.forearmRightMRF, rightForeT, assocPrediction)
            ok = ok and res["rightForeAssoc"]


        res["ok"] = ok
        return res

    def evaluate_prediction_match(self, mrfSom, touchVector, assocPrediction):
        NeighborsCountAsCorrect = False
        if NeighborsCountAsCorrect:
            listOfOneHotVectors = mrfSom.getAllNeighborsOfWinnerVector(touchVector)
            indexBal = self.getIndexOfOne(assocPrediction)
            for i in range(len(listOfOneHotVectors)):

                indexMrf = self.getIndexOfOne(listOfOneHotVectors[i])
                diff = abs(indexBal - indexMrf)
                if diff <= 1:
                    return True
            return False

        touchVal = mrfSom.winnerVector(touchVector)
        return np.all(touchVal == assocPrediction)

    def getIndexOfOne(self, oneHotVector):
        for i in range(len(oneHotVector)):
            if oneHotVector[i] == 1:
                return i

        return -1


if __name__ == "__main__":
    """
    Final test
    """
    mlp = MLPWrapper(32, 70, 70)
    mlp.load_model("../mlp/trained_keras_model")

    leftHandSom = loadSom(11, 8, 8, "../som/trained/left_hand")
    leftForearmSom = loadSom(5, 9, 12, "../som/trained/left_forearm")
    rightHandSom = loadSom(11, 8, 8, "../som/trained/right_hand")
    rightForearmSom = loadSom(5, 9, 12, "../som/trained/right_forearm")

    rightHandMrf = loadMrfSom(9, 7, 7, "../mrf/trained/right-hand2")
    leftHandMrf = loadMrfSom(9, 7, 7, "../mrf/trained/left-hand2")
    rightForearmMrf = loadMrfSom(23, 12, 9, "../mrf/trained/right-forearm2")
    leftForearmMrf = loadMrfSom(23, 12, 9, "../mrf/trained/left-forearm2")

    lh = MyBal(216, 70, 49)
    lh.load("../bal/trained/lh")
    rh = MyBal(216, 70, 49)
    rh.load("../bal/trained/rh")
    lf = MyBal(216, 70, 108)
    lf.load("../bal/trained/lf")
    rf = MyBal(216, 70, 108)
    rf.load("../bal/trained/rf")


    finalModel = ComplexModel(
        filterModel=mlp,
        proprioLeftForearmModel=leftForearmSom,
        proprioLeftHandModel=leftHandSom,
        proprioRightForearmModel=rightForearmSom,
        proprioRightHandModel=rightHandSom,
        touchHandLeftModel=leftHandMrf,
        touchHandRightModel=rightHandMrf,
        touchLeftForearmModel=leftForearmMrf,
        touchRightForearmModel=rightForearmMrf,
        leftHandAsoc=lh,
        rightHandAsoc=rh,
        leftForearmAsoc=lf,
        rightForeArmAsoc=rf
    )

    if not os.path.exists('test_notouch.data') or not os.path.exists('test_touch.data'):
        processor = RawDataParser()
        processor.process_rawdata('test_notouch.rawdata', 'test_notouch.data')
        processor.process_rawdata('test_touch.rawdata', 'test_touch.data')

####### Different data sets for testing ###############

    #testingFile = "test_my_data.data" #a generated set of data, not the same one as used for training the models
    #testingFile = "test_notouch.data" #a collection of non-touch data only, to verify MLP accuracy
    #testingFile = "test_smaller_datadumper.data" #another set, generated / transformed from the data dumper
    testingFile = "test_T0_dumped.data" #generated simulteneously as the below, but was parsed through datadumper
    #testingFile = "test_T0_pythonScript.data" #worse accuracy, because fo incorrectly mapped proprio/touch pairs

#    with open('test_notouch.data') as f:
#        testData = json.load(f)
#    with open('test_touch.data') as f:
#        testData += json.load(f)
    with open(testingFile) as f:
        testData = json.load(f)

    ignored = 0

    errNT = []
    correct = []
    separator = int(len(testData)-1)#*0.3)
    testData = testData[:separator]
    for iCubData, i in zip(testData, range(separator)):
        res = finalModel.test(iCubData)
        if not res["ok"]:
            errNT.append((i, res))
        #if res["filt"] and res["ok"]:
        if res["ok"]:
            correct.append((i, "Prediction Correct"))

    totalMissedTouch = 0 #predicted no touch, but there was
    totalGhostTouch = 0 #predicted touch, but wasn;t there
    for i, bad in errNT:
        print("num.", i, "error", bad)
        if "errorType" in bad and "Was touch" in bad["errorType"]:
            totalMissedTouch += 1
        elif "errorType" in bad and "Was NOT touch" in bad["errorType"]:
            totalGhostTouch += 1

    print("========")

    for i, info in correct:
        print("num.", i, "info", info)

    print("========")
    print("Correct:", len(testData)-len(errNT), ", Incorrect:", len(errNT), ", succ", 100 - (len(errNT) / len(testData)) * 100, "%")
    print("Total Ghost touches: {}".format(totalGhostTouch))
    print("Total Missed touches: {}".format(totalMissedTouch))
