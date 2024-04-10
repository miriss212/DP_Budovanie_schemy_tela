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



def parse_touch_configurations(data):
    """
    Parse only the touch configurations from the provided data.

    Args:
    - data: list of dictionaries containing the data with keys like 'left-forearm', 'left-hand', etc.

    Returns:
    - touch_configurations: list of dictionaries containing only the touch configurations
    """
    touch_configurations = []
    for entry in data:
        touch_entry = {}
        for key, value in entry.items():
            #if key.endswith('-forearm') or key.endswith('-hand') or key.endswith('-pos'):
            touch_values = [1 if float(val) != 0 else 0 for val in value.split()]
            if any(touch_values):  # Check if any value is touched
                touch_entry[key] = touch_values
        if touch_entry:  # If there are any touched values in the entry
            touch_configurations.append(touch_entry)
    return touch_configurations

if __name__ == "__main__": #TOTO JE TA ZLA VERZIA
    original_file = "./data/data/my_data.rawdata"  # Replace with your file path
    output_file = "./data/data/janka_data_touch.data"  # Replace with desired output file path

    with open(original_file) as f:
        data = json.load(f)

    touch_configurations = parse_touch_configurations(data)

    with open(output_file, "w") as f:
        json.dump(touch_configurations, f, indent=4)

    print("Parsed touch data saved to:", output_file)