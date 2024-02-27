import yarp
import time
from iCubSim import iCubLimb
from data_processing.data_collector import LimbDataCollector
from data_processing.rawdata_parser import RawDataParser

yarp.Network.init()

LEFT_ARM_START = (-44.09800218089138, 25.727307501232396, 19.890250994191103, 54.299349576368854, -35.999966180984124,
                  -1.5556396279275911e-06, 1.1180395270159209e-05, 59.0000003457082, 20.0000001744945, 20.000000001098464,
                  20.000000189717543, 10.000000151989294, 10.000000122542103, 10.000000162112157, 10.000000129643745, 10.00000088336669)
RIGHT_ARM_START = (-38.84823692765845, 14.472066915711254, 67.86048363631082, 79.63945165111399, -62.99996375064009,
                   -3.250601010417533e-06, 0.6000077601790752, 49.80000022721013, 53.60000006693065, 31.50000008006696,
                   20.00000023748552, 13.499999942995807, 41.39999997161878, 43.19999998370278, 111.60000003415337, 134.99999975701846)

LEFT_ARM_FIX = (-55,12,20,45,0,0,0,59,20,20,20,10,10,10,10,10)

class ICubBabblingSim:

    def __init__(self):
        self.raw_data_file = "10.rawdata"
        self.rescaled_data_file = "10.data"
        self.minutes_of_simulation = 120

        app = '/main'
        self.head = iCubLimb(app,'/icubSim/head')
        self.left_arm = iCubLimb(app,'/icubSim/left_arm')
        self.right_arm = iCubLimb(app,'/icubSim/right_arm')

        time.sleep(1)

        #set arms
        self.right_arm.set(LEFT_ARM_FIX)
        self.left_arm.set(LEFT_ARM_FIX)

        time.sleep(0.4)

        #init data collector
        self.data_collector = LimbDataCollector()

    def start_simulation(self):
        duration_of_babbling = self.minutes_of_simulation * 60
        collect_data = True

        start_time = time.time() * 1000.0
        end_time = start_time + duration_of_babbling * 1000

        # loop and issue the babbling commands
        while time.time() * 1000.0 < end_time:
            t = time.time() * 1000.0 - start_time
            print("t = ", t, "/", duration_of_babbling)
            self.right_arm.simplifiedBabbling(t)
            time.sleep(1)

            if collect_data:
                self.data_collector.read_limb_touches(self.left_arm, self.right_arm)
                time.sleep(1)

        print("FINISHED babbling")
        if collect_data:
            self.data_collector.export_data(self.raw_data_file)
            print("Collected data exported, ", self.raw_data_file, " file was created in working directory.")

            parser = RawDataParser()
            parser.process_rawdata(self.raw_data_file, self.rescaled_data_file)
            print("Data parsed to .data format.")
            print("Finished successfully.")

if __name__ == "__main__":
    simulator = ICubBabblingSim()
    simulator.start_simulation()
