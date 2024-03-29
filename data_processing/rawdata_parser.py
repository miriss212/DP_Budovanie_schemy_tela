import json

from data_transform_func import *


class RawDataParser:
    """
    Parser of the collected data from iCub sim (*.rawdata file) to a rescaled ( *.data file) format.

    Args:
        -scaleMetric: string; specify either 'global' or 'local'. If 'global', all joints are rescaled according to the
        lowest and highest joint value across *all* joints.
        If 'local', each joint is rescaled in its own bounds

    """
    def __init__(self, scale_metric='local'):

        if scale_metric == 'local':
            self.scaler = LocalJointScaler()
        elif scale_metric == 'global':
            self.scaler = SupremumInfinumScaler()
        elif scale_metric == "divBy100":
            self.scaler = DataScaler()
        else:
            raise Exception("ERROR: unknown scaler supplied as arg to RawDataParser() c'tor")


    def __convert_to_floats(self, string_data):
        return [float(i) for i in string_data.split()]

    def __rescale_arm_positions(self, batch):
        return self.scaler.rescale_vector(batch)


    # taxel is evaluated as touched if it has at least one non-zero value in the 12-value data batch
    def __taxel_touched(self, touch_data, index):
        values = set(touch_data[i] for i in range(index, index+12))
        return 1 if len(values) > 1 else 0


    def __process_touch_data(self, touch_data):
        return [self.__taxel_touched(touch_data, i) for i in range(0, len(touch_data), 12)]


    def __process_hand(self, hand):
        return hand[0:60] + hand[96:144]


    def __process_forearm(self, config):
        x = config[0:30*12]         # -2
        x = x[0:27*12] + x[28*12:]  # -1
        x = x[0:22*12] + x[24*12:]  # -2
        x = x[0:18*12] + x[21*12:]  # -3
        x = x[0:16*12] + x[17*12:]  # -1
        return x

    def __process_data_batch(self, batch):
        processed_batch = {}
        #process of rescaling, converting, and concating of .rawdata
        for key in ["left-pos", "right-pos", "left-hand", "right-hand", "left-forearm", "right-forearm"]:
            processed_batch[key] = self.__convert_to_floats(batch[key])

        for key in ["left-pos", "right-pos"]:
            processed_batch[key] = self.__rescale_arm_positions(processed_batch[key])

        for key in ["left-hand", "right-hand"]:
            processed_batch[key] = self.__process_hand(processed_batch[key])

        for key in ["left-forearm", "right-forearm"]:
            processed_batch[key] = self.__process_forearm(processed_batch[key])

        for key in ["left-hand", "right-hand", "left-forearm", "right-forearm"]:
            processed_batch[key] = self.__process_touch_data(processed_batch[key])

        right_arm_touch = processed_batch["right-hand"] + processed_batch["right-forearm"]
        left_arm_touch = processed_batch["left-hand"] + processed_batch["left-forearm"]

        #new format of .rawdata -> .data
        return [
            {
                "proprio": processed_batch["left-pos"] + processed_batch["right-pos"],
                "touch": left_arm_touch + right_arm_touch,
                "rightHandPro": processed_batch["right-pos"],
                "leftHandPro": processed_batch["left-pos"],
                "rightTouch": right_arm_touch,
                "leftTouch": left_arm_touch
            },
            # symmetrical
            # {
            #     "proprio": processed_batch["right-pos"] + processed_batch["left-pos"],
            #     "touch": right_arm_touch + left_arm_touch,
            #     "rightHandPro": processed_batch["left-pos"],
            #     "leftHandPro": processed_batch["right-pos"],
            #     "rightTouch": left_arm_touch,
            #     "leftTouch": right_arm_touch
            # }
        ]

    def process_rawdata(self, file_name, output_file_name):
        print("Rescaling raw data...")
        with open(file_name) as f:
            data = json.load(f)
            processed_rawdata = [conf for batch in data for conf in (self.__process_data_batch(batch))]

        f = open(output_file_name, "w")
        f.write(str(json.dumps(processed_rawdata, indent=4)))
        f.close()

        # with open(output_file_name) as out:
        #     out.write(str(json.dumps(processed_rawdata, indent=4)))

        print("Raw data rescaled. Output present in ", output_file_name, " file.")
        return processed_rawdata

# main
if __name__ == "__main__":
    """
    Process data from iCub
    file .rawdata to file .data

    """
    # original_file = "my_data.rawdata"
    # new_file = "my_data.data"
    #original_file = "leyla_data_left-forearm.rawdata"
    original_file = "TEST_data_NEW.rawdata"
    new_file = "data_matej_new.data"
    # original_file = "leyla_data2_left-forearm.rawdata"
    # new_file = "data_leyla2.data"

    processor = RawDataParser()
    result = processor.process_rawdata(original_file, new_file)

    print("data size", len(result))
