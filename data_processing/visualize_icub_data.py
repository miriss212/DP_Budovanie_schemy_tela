import yarp
import time
from iCubSim import iCubLimb
from data_collector import LimbDataCollector
import json
import numpy as np
import asyncio

yarp.Network.init()

class ICubBabblingSim:

    def __init__(self):
        self.raw_data_file = "my_data.rawdata"
        self.rescaled_data_file = "my_data.data"
        self.minutes_of_simulation = 10

        app = '/main'
        self.head = iCubLimb(app,'/icubSim/head')
        self.left_arm = iCubLimb(app,'/icubSim/left_arm')
        self.right_arm = iCubLimb(app,'/icubSim/right_arm')

        time.sleep(3)

        #init data collector
        self.data_collector = LimbDataCollector()

    def start_simulation(self, leftArm, rightArm, leftHandTouch, leftForeTouch, rightHandTouch, rightForeTouch):
        #goes through all saved proprio configurations of arms, from input
        i = 0
        for left, right, lHT, lFT, rHT, rFT in zip(leftArm, rightArm, leftHandTouch, leftForeTouch, rightHandTouch, rightForeTouch):
            vectorL = tuple(float(number) for number in left.split(' '))
            vectorR = tuple(float(number) for number in right.split(' '))
            leftForeTouch = tuple(float(number) for number in lFT.split(' '))
            rightForeTouch = tuple(float(number) for number in rFT.split(' '))
            leftHandTouch = tuple(float(number) for number in lHT.split(' '))
            rightHandTouch = tuple(float(number) for number in rHT.split(' '))
            if True: #sum(leftForeTouch) == 0:
                #print("setting pos")
                self.right_arm.set(vectorR)
                self.left_arm.set(vectorL)
                if sum(leftForeTouch) == 0 and sum(rightForeTouch) == 0 and sum(leftHandTouch) == 0 and sum(rightHandTouch) == 0:
                    print("NO touch seen now; {0}".format(i))
                else:
                    print("TOUCH right now!!")
                time.sleep(2.5)
                i+=1


if __name__ == "__main__":
    data_file = "data_martin.rawdata" #"../original src/python/data.rawdata"

    file = open(data_file)
    data = json.load(file)
    file.close()

    left = np.array([i["left-pos"] for i in data])
    right = np.array([i["right-pos"] for i in data])
    leftHT = np.array([i["left-hand"] for i in data])
    rightHT = np.array([i["right-hand"] for i in data])
    leftFT = np.array([i["left-forearm"] for i in data])
    rightFT = np.array([i["right-forearm"] for i in data])

    simulator = ICubBabblingSim()
    simulator.start_simulation(left, right, leftHT, leftFT, rightHT, rightFT)
