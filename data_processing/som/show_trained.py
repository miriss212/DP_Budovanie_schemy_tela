import move_robot.yarp as yarp
#import yarp
import time
import sys
import pyscreenshot as ImageGrab
import ctypes
import json
import numpy as np
import os
import cv2
from itertools import *

yarp.Network.init()
from move_robot.iCubSim import iCubLimb
from data_transform_func import *


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class LimbTuple:
    def __init__(self, minimum, maximum):
        self.min = minimum
        self.max = maximum

class SOMVisualizer:

    def __init__(self, neuronNum, rowNum, data_scaler):

        self.number_of_neurons = neuronNum
        self.number_of_snapshot_rows = rowNum
        self.data_scaler = data_scaler
        sys.path.append('../')

        app = '/main'
        self.head = iCubLimb(app,'/icubSim/head')
        self.left_arm = iCubLimb(app,'/icubSim/left_arm')
        self.right_arm = iCubLimb(app,'/icubSim/right_arm')

        time.sleep(1)
        self.set_initial_arm_pos()
        time.sleep(1)


    def set_initial_arm_pos(self):
        LEFT_ARM_FIX = (-55, 15, 20, 45, 0, 0, 0, 59, 20, 20, 20, 10, 10, 10, 10, 10)
        self.right_arm.set(LEFT_ARM_FIX)
        self.left_arm.set(LEFT_ARM_FIX)

    def create_screenshot(self, directory=".", img_name='default_pic.png'):
        # get monitor properties
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)

        Y_offset = -10
        image_width = screen_width // 4
        offset = screen_width // 40
        image_start = screen_height - ((screen_height // 3)*2) - Y_offset
        image_end = screen_height - (screen_height // 5)

        start_x = (screen_width // 2) - offset
        start_y = image_start
        end_x = (screen_width // 2) + image_width
        end_y = image_end

        # grab iCub screenshot: Assumes that iCubSim is fullscreen, in default position
        im = ImageGrab.grab(bbox=(start_x, start_y, end_x, end_y))  # X1,Y1,X2,Y2

        # save image file
        if not os.path.exists(directory):
            os.mkdir(directory)

        finalPath = directory + "/" + img_name
        im.save(finalPath)

    def screenshot_som_neurons(self, directory_name, limb_file, limbType, rightOrLeft):
        for i in range(self.number_of_neurons):
            if i % 2 != 0:
                continue
            neuron_file = open(limb_file + str(i) + '.txt')
            for j in range(self.number_of_snapshot_rows):
                if j % 2 != 0:
                    continue
                limb_vector = neuron_file.readline()
                parsed = [float(i) for i in limb_vector.split()]
                rescaled = self.data_scaler.scale_vector_back(parsed, 0, len(parsed))
                print("This is a read row number " + str(i) + ":")
                print(parsed)

                print("This is the RESCALED vector")
                print(rescaled)
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
                rowNumber = j
                if j > 0:
                    rowNumber = j-1
                self.create_screenshot(directory_name, "{0}{1}-{2}-{3}.jpg".format(rightOrLeft, limbType, i, rowNumber))
                self.set_initial_arm_pos()
                time.sleep(1)

        #now combine all into one big image
        final_name = "{0}-{1}_Final.jpg".format(rightOrLeft, limbType)
        self.make_final_screenshot(directory_name, final_name)


    def make_final_screenshot(self, directory, final_name):
        imgs = self.load_images_from_folder(directory)
        rows = list(chunks(imgs, self.number_of_snapshot_rows // 2))
        concatenated_horizontal_rows = []
        for row in rows:
            h_img = cv2.hconcat(row)
            concatenated_horizontal_rows.append(h_img)

        #finally, vertical concat
        finalImg = cv2.vconcat(concatenated_horizontal_rows)

        cv2.imshow('Final', finalImg)
        cv2.waitKey(0)
        cv2.imwrite(directory + "/" + final_name, finalImg)


    def load_images_from_folder(self, folder):
        images = []
        border_value = 5
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            img_with_border = cv2.copyMakeBorder(img,border_value, border_value, border_value, border_value, cv2.BORDER_CONSTANT, value=[255,255,255])
            if img is not None:
                images.append(img_with_border)
        return images


    def rescale_vector(self, limb_vector):
        rescaled_vector = []
        for i in range(len(limb_vector)):
            num = limb_vector[i]
            print("Num is: ", num)
            rescaled_value = num * 100 #inverted function of how rawdata_parser.py rescales the data
            print("Rescaled value is: ", rescaled_value)
            rescaled_vector.append(rescaled_value)

        return rescaled_vector


if __name__ == "__main__":
    scaler = LocalJointScaler() #FOR THE LOVE OF GOD, CHECK THIS!! IT HAS TO BE THE SAME AS IN RAWDATAPARSER !!
    NotOnlyFinalScrn = True

    leftOrRight = "left"
    limbType = "hand"
    arm_dir = "{0}{1}_New".format(leftOrRight, limbType)
    som_shape = [9, 12]
    if limbType == "hand":
        som_shape = [8,8]

    som_visualizer = SOMVisualizer(som_shape[0], som_shape[1], scaler) #8x8 for hand, 9x12 for forearm
    print("WAITING for 3 seconds to maximize iCub:")
    time.sleep(3)

    if NotOnlyFinalScrn:
        print("Done. Screenshots starting.")
        fpath = 'trained/{0}_{1}/'.format(leftOrRight, limbType)
        som_visualizer.screenshot_som_neurons(arm_dir, fpath, limbType, leftOrRight)
    else:
        som_visualizer.make_final_screenshot(arm_dir, "Right_Arm_Final.jpg")



