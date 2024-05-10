import json
import move_robot.limbTouchDetector as ltd

class LimbDataCollector:
    def __init__(self):
        self.collected_data = []
        self.touch_detector = ltd.LimbTouchDetector()
        self.previous_right_proprio = None
        self.previous_left_proprio = None

    def read_limb_touches(self, left_arm, right_arm):

        proprioLeft = left_arm.get()
        proprioRight = right_arm.get()

        left_hand =     self.touch_detector.read_touch_sensors('left-hand')
        left_forearm =  self.touch_detector.read_touch_sensors('left-forearm')
        right_hand =    self.touch_detector.read_touch_sensors('right-hand')
        right_forearm = self.touch_detector.read_touch_sensors('right-forearm')

        thereIsTouchOnLeft = sum(left_hand) > 0 or sum(left_forearm) > 0
        thereIsTouchOnRight = sum(right_hand) > 0 or sum(right_forearm) > 0

        if (thereIsTouchOnLeft and not thereIsTouchOnRight) or (not thereIsTouchOnLeft and thereIsTouchOnRight):
            # clear it!!
            hand_length = len(left_hand)
            forearm_length = len(left_forearm)
            left_hand =     [0.0 for i in range(hand_length)]
            left_forearm =  [0.0 for i in range(forearm_length)]
            right_hand =    [0.0 for i in range(hand_length)]
            right_forearm = [0.0 for i in range(forearm_length)]

            print("FALSE touch detected, removed.")
        else:
            if thereIsTouchOnRight and thereIsTouchOnLeft:
                print("TOUCH detected; stored ")
            else:
                print("Was NOT a touch.")

        batch = dict()
        batch['left-hand'] = self.__format_data(left_hand)
        #time.sleep(0.4) #TODO check if necessary
        batch['left-forearm'] = self.__format_data(left_forearm)
        # time.sleep(0.4)
        batch['right-hand'] = self.__format_data(right_hand)
        # time.sleep(0.4)
        batch['right-forearm'] = self.__format_data(right_forearm)
        # time.sleep(0.4)

        if self.previous_left_proprio is None and self.previous_right_proprio is None:
            #remember them
            self.previous_right_proprio = self.__format_data(proprioRight)
            self.previous_left_proprio = self.__format_data(proprioLeft)
        else:
            #current touch actually corresponds to previous config; overwrite this batch, and store it
            batch['left-pos'] = self.previous_left_proprio
            batch['right-pos'] = self.previous_right_proprio
            self.collected_data.append(batch)

            #update the previous batch to current.
            self.previous_left_proprio = self.__format_data(proprioLeft)
            self.previous_right_proprio = self.__format_data(proprioRight)

    def export_data(self, file_name):
        print("Exporting data.rawdata into .json ...")
        try:
            data_rawdata = json.dumps(self.collected_data, sort_keys=True, indent=4)
            f = open(file_name, "w")
            f.write(data_rawdata)
            f.close()
            print("Data exported successfully.")

        except:
            print("An error has occured while exporting data. No .json file was made. Exiting.")
        finally:
            self.__close_touch_ports()

    def __format_data(self, data):
        return " ".join(str(number) for number in data)

    def __close_touch_ports(self):
        print("Closing all touch ports..")
        self.touch_detector.close_touch_ports()
        print("Touch ports closed")
