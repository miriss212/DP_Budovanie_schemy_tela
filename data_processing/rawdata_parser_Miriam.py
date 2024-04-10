import json

from data_transform_func import *


class RawDataParser:
    """
    Parser pro zpracování nasbíraných dat z iCub sim (*.rawdata soubor) do přeškálovaného formátu (*.data soubor).

    Args:
        -scaleMetric: string; určuje buď 'global' nebo 'local'. Pokud 'global', všechny klouby jsou přeškálovány podle
        nejnižší a nejvyšší hodnoty kloubu napříč *všemi* klouby.
        Pokud 'local', každý kloub je přeškálován ve svém vlastním rozsahu.

    """
    def __init__(self, scale_metric='local'):

        if scale_metric == 'local':
            self.scaler = LocalJointScaler()
        elif scale_metric == 'global':
            self.scaler = SupremumInfinumScaler()
        elif scale_metric == "divBy100":
            self.scaler = DataScaler()
        else:
            raise Exception("CHYBA: neznámý scaler poskytnutý jako argument konstruktoru RawDataParser()")


    def __convert_to_floats(self, string_data):
        return [float(i) for i in string_data.split()]

    def __rescale_arm_positions(self, batch):
        return self.scaler.rescale_vector(batch)


    # taxel je vyhodnocen jako dotknutý, pokud má alespoň jednu nenulovou hodnotu v datové dávce se 12 hodnotami
    def __taxel_touched(self, touch_data, index):
        values = set(touch_data[i] for i in range(index, index+12))
        return 1 if len(values) > 1 else 0


    def __process_touch_data(self, touch_data):
        return [self.__taxel_touched(touch_data, i) for i in range(0, len(touch_data), 12)]


    def __process_data_batch(self, batch):
        processed_batch = {}
        #proces přeškálování, konverze a spojování .rawdata
        for key in ["left-pos", "right-pos", "left-forearm", "right-hand"]:
            processed_batch[key] = self.__convert_to_floats(batch[key])

        for key in ["left-pos", "right-pos"]:
            processed_batch[key] = self.__rescale_arm_positions(processed_batch[key])

        for key in ["left-forearm", "right-hand"]:
            processed_batch[key] = self.__process_touch_data(processed_batch[key])

        right_arm_touch = processed_batch["right-hand"]
        left_arm_touch = processed_batch["left-forearm"]

        #nový formát .rawdata -> .data
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

    def process_rawdata(self, file_name, output_file_name):
        print("Přeškálování raw dat...")
        with open(file_name) as f:
            data = json.load(f)
            #processed_rawdata = [conf for batch in data for conf in (self.__process_data_batch(batch))]
            processed_rawdata = [conf for batch in data for conf in (self.__process_data_batch(batch)) if any(conf["touch"])]

        with open(output_file_name, "w") as f:
            json.dump(processed_rawdata, f, indent=4)

        print("Raw data přeškálována. Výstup je k dispozici v souboru ", output_file_name, ".")
        return processed_rawdata

# hlavní
if __name__ == "__main__":
    """
    Zpracování dat z iCub
    soubor .rawdata do souboru .data

    """
    original_file = "TEST_data_NEW.rawdata"
    new_file = "data/data/data_matej_touch.data"

    processor = RawDataParser()
    result = processor.process_rawdata(original_file, new_file)

    print("Velikost dat:", len(result))
