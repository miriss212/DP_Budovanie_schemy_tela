
#1. Take a random touch vector
    #1.1 : this will be done as : reading all .baldata files, generate a random int *i*. Then, take this (touch) vector *i*
    #from all these files.
    #1.2 : Load all trained BAL models (each for every limb part). Then, make a backward prediction for all associators.

#2  after getting 4 vector predictions, determine what SOM neuron they are, and then the specific proprio positions from
    #the SOM vectors.

import os
import random
from os import path
import numpy as np
import json
import move_robot.yarp as yarp
import time

yarp.Network.init()

from move_robot.iCubSim import iCubLimb
from data_processing.data_transform_func import *

from bal.mybal import MyBal
from som.som import SOM
from mrf.mrfsom import MRFSOM


class LimbPositionPredictor:
    def __init__(self):

        app = '/main'
        self.head = iCubLimb(app,'/icubSim/head')
        self.left_arm = iCubLimb(app,'/icubSim/left_arm')
        self.right_arm = iCubLimb(app,'/icubSim/right_arm')

        time.sleep(1)
        self.set_initial_arm_pos()
        time.sleep(1)

        self.lh_bal = MyBal(216, 70, 49)
        self.lh_bal.load("bal/trained/lh")
        self.rh_bal = MyBal(216, 70, 49)
        self.rh_bal.load("bal/trained/rh")
        self.lf_bal = MyBal(216, 70, 108)
        self.lf_bal.load("bal/trained/lf")
        self.rf_bal = MyBal(216, 70, 108)
        self.rf_bal.load("bal/trained/rf")
        self.lf_rh_bal = MyBal(216, 70, 49)
        #self.lf_rh_bal.load("bal/trained/lf_rh")


        self.leftHandSom = self.loadSom(11, 8, 8, "som/trained/left_hand")
        self.leftForearmSom = self.loadSom(5, 9, 12, "som/trained/left_forearm")
        self.rightHandSom = self.loadSom(11, 8, 8, "som/trained/right_hand")
        self.rightForearmSom = self.loadSom(5, 9, 12, "som/trained/right_forearm")

        self.leftHandMrf = self.loadMrfSom(9, 7, 7, "mrf/trained/left-hand2")
        self.rightHandMrf = self.loadMrfSom(9, 7, 7, "mrf/trained/right-hand2")
        self.rightForearmMrf = self.loadMrfSom(23, 12, 9, "mrf/trained/right-forearm2")
        self.leftForearmMrf = self.loadMrfSom(23, 12, 9, "mrf/trained/left-forearm2")

    def set_initial_arm_pos(self):
        LEFT_ARM_FIX = (-55, 15, 20, 45, 0, 0, 0, 59, 20, 20, 20, 10, 10, 10, 10, 10)
        self.right_arm.set(LEFT_ARM_FIX)
        self.left_arm.set(LEFT_ARM_FIX)

    def loadSom(self, dim, rows, cols, folderName):
        model = SOM(dim, rows, cols)
        for i in range(rows):
            model.weights[i] = np.loadtxt(folderName+"/"+str(i)+".txt")
        return model


    def loadMrfSom(self, dim, rows, cols, folderName):
        model = MRFSOM(dim, rows, cols)
        for i in range(rows):
            model.weights[i] = np.loadtxt(folderName+"/"+str(i)+".txt")
        return model

    def getRandomTouchSampleIndex(self, touch, whichLimbPart):
        number_of_samples = len(touch)
        while True:
            random_pos = random.randint(0, number_of_samples)

            touch_sample = touch[random_pos]
            if sum(touch_sample) == 0:
                continue
            else:
                leftHandTouch = touch_sample[0:9]
                rightHandTouch = touch_sample[32:41]
                leftForeTouch = touch_sample[9:32]
                rightForeTouch = touch_sample[41:64]

                desiredTouchedLimbPart = None
                if whichLimbPart == "lh":
                    desiredTouchedLimbPart = leftHandTouch
                elif whichLimbPart == "rh":
                    desiredTouchedLimbPart = rightHandTouch
                elif whichLimbPart == "rf":
                    desiredTouchedLimbPart = rightForeTouch
                elif whichLimbPart == "lf":
                    desiredTouchedLimbPart = leftForeTouch
                elif whichLimbPart == "lf-rh":
                    desiredTouchedLimbPart = leftForeTouch, rightHandTouch
                else:
                    raise Exception("[ERROR] Specify which limb part should have touched occured")

                if self.configurationIsTouch(desiredTouchedLimbPart):
                    return random_pos

    def configurationIsTouch(self, desiredTouchedLimbPart):
        if isinstance(desiredTouchedLimbPart, tuple):
            first, second = desiredTouchedLimbPart[0], desiredTouchedLimbPart[1]
            return sum(first) != 0 and sum(second) != 0

        return sum(desiredTouchedLimbPart) != 0

    def makeSpecificProprioPrediction(self, whereWeWantTouchOccuring, mrfWinnerVector):
        proprioPrediction = None
        if whereWeWantTouchOccuring == "lh":
            proprioPrediction = self.lh_bal.backward(mrfWinnerVector)
        elif whereWeWantTouchOccuring == "rh":
            proprioPrediction = self.rh_bal.backward(mrfWinnerVector)
        elif whereWeWantTouchOccuring == "rf":
            proprioPrediction = self.rf_bal.backward(mrfWinnerVector)
        elif whereWeWantTouchOccuring == "lf":
            proprioPrediction = self.lf_bal.backward(mrfWinnerVector)
        elif whereWeWantTouchOccuring == "lf-rh":
            proprioPrediction = self.lf_bal.backward(mrfWinnerVector[0]), self.rh_bal.backward(mrfWinnerVector[1])
        else:
            raise Exception("[ERROR]: unknown arg specified.")

        self.checkAllProprioPredictionsHaveFourOnes(proprioPrediction)

        if isinstance(proprioPrediction, tuple):
            self.getSOMWinnersAndSetPositionsToICub(proprioPrediction[0])
            time.sleep(5)
            self.set_initial_arm_pos()
            time.sleep(2)
            self.getSOMWinnersAndSetPositionsToICub(proprioPrediction[1])

            result = np.subtract(proprioPrediction[0], proprioPrediction[1])

            print('The difference in predicitons: {0}'.format(abs(sum(result))))
        else:
            self.getSOMWinnersAndSetPositionsToICub(proprioPrediction)


    def checkAllProprioPredictionsHaveFourOnes(self, proprioPrediction):
        if isinstance(proprioPrediction, tuple):
            self.checkAllProprioPredictionsHaveFourOnes(proprioPrediction[0])
            self.checkAllProprioPredictionsHaveFourOnes(proprioPrediction[1])
            return

        if sum(proprioPrediction) != 4:
            print("[WARNING] : Proprio prediction seems incorrect. Only {0}/4 ones present...".format(sum(proprioPrediction)))
            return False

        return True

    def getSOMWinnersAndSetPositionsToICub(self, proprioPrediction):
        # the whole prediction vector consists of one-hot-encoded limb parts:
        # For hand, the vector length is 64, 108 for forearm.
        # IMPORTANT: The order of the encodings is winnerL_H + winnerL_F + winnerR_H + winnerR_F (see `process_data_for_BAL.py`)
        oneHot_LH = proprioPrediction[0:64]
        oneHot_LF = proprioPrediction[64:172]
        oneHot_RH = proprioPrediction[172:236]
        oneHot_RF = proprioPrediction[236:345]

        som_winnerLH_Row, som_winnerLH_Col = self.leftHandSom.fromOneHot(oneHot_LH)
        som_winnerLF_Row, som_winnerLF_Col = self.leftForearmSom.fromOneHot(oneHot_LF)
        som_winnerRH_Row, som_winnerRH_Col = self.rightHandSom.fromOneHot(oneHot_RH)
        som_winnerRF_Row, som_winnerRF_Col = self.rightForearmSom.fromOneHot(oneHot_RF)

        mainPath = "som/trained/"
        self.show_prediction(mainPath + "left_hand", "hand", "left", som_winnerLH_Row, som_winnerLH_Col)
        self.show_prediction(mainPath + "left_forearm", "forearm", "left", som_winnerLF_Row, som_winnerLF_Col)
        self.show_prediction(mainPath + "right_hand", "hand", "right", som_winnerRH_Row, som_winnerRH_Col)
        self.show_prediction(mainPath + "right_forearm", "forearm", "right", som_winnerRF_Row, som_winnerRF_Col)

        leftPos = self.left_arm.get()
        rightPos = self.right_arm.get()

    def predict_proprio_vectors(self):

        data_path = "my_data.data"

        # load data
        f = open(data_path)
        scaledData = json.load(f)  # [0:13]
        f.close()
        touch = np.array([i["touch"] for i in scaledData])
        left = np.array([i["leftHandPro"] for i in scaledData])
        right = np.array([i["rightHandPro"] for i in scaledData])

        #----- Get a random sample, get the touch vectors, and transform them to the format
        #such vector would be in .baldata

        whereWeWantTouchOccuring = "lf-rh" #"rf"

        while(True):
            touchVectorIndex = self.getRandomTouchSampleIndex(touch, whereWeWantTouchOccuring)

            touch_sample = touch[touchVectorIndex]
            left_sample = left[touchVectorIndex]
            right_sample = right[touchVectorIndex]

            leftHandTouch = touch_sample[0:9]
            rightHandTouch = touch_sample[32:41]
            leftForeTouch = touch_sample[9:32]
            rightForeTouch = touch_sample[41:64]

            mrfWinner_LH = None
            mrfWinner_RH = None
            mrfWinner_LF = None
            mrfWinner_RF = None

            if sum(leftHandTouch) != 0:
                mrfWinner_LH = self.leftHandMrf.winnerVector(leftHandTouch)

            if sum(rightHandTouch) != 0:
                mrfWinner_RH = self.rightHandMrf.winnerVector(rightHandTouch)

            if sum(leftForeTouch) != 0:
                mrfWinner_LF = self.leftForearmMrf.winnerVector(leftForeTouch)

            if sum(rightForeTouch) != 0:
                mrfWinner_RF = self.rightForearmMrf.winnerVector(rightForeTouch)

            wantedTouchToMrfWinnerMap = \
                {"rh": mrfWinner_RH,
                 "rf": mrfWinner_RF,
                 "lh": mrfWinner_LH,
                 "lf": mrfWinner_LF,
                 "lf-rh": [mrfWinner_LF, mrfWinner_RH]}

            if self.allMrfWinnersAreNonZero(wantedTouchToMrfWinnerMap[whereWeWantTouchOccuring]):
                break

        self.makeSpecificProprioPrediction(whereWeWantTouchOccuring, wantedTouchToMrfWinnerMap[whereWeWantTouchOccuring])

    def allMrfWinnersAreNonZero(self, mrfWinners):
        if isinstance(mrfWinners, list):
            for winner in mrfWinners:
                if sum(winner) == 0:
                    return False

            return True
        else:
            return sum(mrfWinners) > 0

    def show_prediction(self, limb_file, limbType, rightOrLeft, winner_file, winner_column):
        scaler = LocalJointScaler() #FOR THE LOVE OF GOD, CHECK THIS!! IT HAS TO BE THE SAME AS IN RAWDATAPARSER !!

        neuron_file = open(limb_file + "/" + str(winner_file) + '.txt')
        whole_file = neuron_file.readlines()

        limb_vector = whole_file[winner_column]

        parsed = [float(i) for i in limb_vector.split()]
        rescaled = scaler.scale_vector_back(parsed, 0, len(parsed))
        #print("This is the read SOM vector: ")
        #print(parsed)

        #print("This is the RESCALED vector: ")
        #print(rescaled)
        print("========")

        # set the new joints; use then with screenshots
        if rightOrLeft == "left":
            values = self.left_arm.get()
            if limbType == "forearm" or limbType == "arm":
                self.left_arm.set(values, j0=rescaled[0], j1=rescaled[1], j2=rescaled[2],
                     j3=rescaled[3], j4=rescaled[4])

            elif limbType == "hand":
                self.left_arm.set(values, j5=rescaled[0], j6=rescaled[1], j7=rescaled[2],
                    j8=rescaled[3], j9=rescaled[4], j10=rescaled[5], j11=rescaled[6],
                    j12=rescaled[7], j13=rescaled[8], j14=rescaled[9], j15=rescaled[10])
        elif rightOrLeft == "right":
            values = self.right_arm.get()
            if limbType == "forearm" or limbType == "arm":
                self.right_arm.set(values, j0=rescaled[0], j1=rescaled[1], j2=rescaled[2],
                                  j3=rescaled[3], j4=rescaled[4])

            elif limbType == "hand":
                self.right_arm.set(values, j5=rescaled[0], j6=rescaled[1], j7=rescaled[2],
                                  j8=rescaled[3], j9=rescaled[4], j10=rescaled[5], j11=rescaled[6],
                                  j12=rescaled[7], j13=rescaled[8], j14=rescaled[9], j15=rescaled[10])
        else:
            raise Exception("ERROR: Invalid arm specified as arg. Use 'left' or 'right' only!")

        time.sleep(0.5)




LimbPositionPredictor().predict_proprio_vectors()