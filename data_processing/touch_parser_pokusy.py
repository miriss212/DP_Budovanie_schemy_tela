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

        return [
            {
                "proprio": processed_batch["left-pos"] + processed_batch["right-pos"],
                "touch": left_arm_touch + right_arm_touch,
                "rightHandPro": processed_batch["right-pos"],
                "leftHandPro": processed_batch["left-pos"],
                "rightTouch": right_arm_touch,
                "leftTouch": left_arm_touch
            }
        ]

    def process_rawdata_and_save_filtered(self, file_name, output_file_name):
        print("Rescaling raw data...")
        with open(file_name) as f:
            data = json.load(f)
            processed_rawdata = [conf for batch in data for conf in (self.__process_data_batch(batch)) if any(conf["touch"])]

        with open(output_file_name, "w") as f:
            json.dump(processed_rawdata, f, indent=4)

        print("Raw data rescaled and filtered. Output present in ", output_file_name, " file.")
        return processed_rawdata
    
def main():
    original_file = "data/data/TEST_data_NEW.rawdata"  # Replace with your file path
    output_file = "data/data/matej_filtered_data.data"  # Replace with your desired output file path

    parser = RawDataParser()
    parser.process_rawdata_and_save_filtered(original_file, output_file)

if __name__ == "__main__":
    main()