import json
from collections import Counter

import numpy as np


def __format_data(data):
    string = " ".join(str(number) for number in data)
    return string.strip()


def make_average(timestamp_dict, timestamp, batch_key, v1, v2):
    numbers_2darray = []

    number_array = [float(i) for i in v1.split(' ')]
    numbers_2darray.append(number_array)

    number_array = [float(i) for i in v2.split(' ')]
    numbers_2darray.append(number_array)

    finalValue = list(np.average(numbers_2darray, axis=0))
    stringFormat = " ".join(str(v) for v in finalValue)
    timestamp_dict[timestamp][batch_key] = stringFormat

    return timestamp_dict


def parse_file_and_append_to_dictionary(timestamp_dict, file_name, batch_key):
    with open(file_name, "r") as limbData:
        for entry in limbData:
            values = entry.split(' ')
            timestamp = round(float(values[2]), 3)
            if timestamp not in timestamp_dict:
                timestamp_dict[timestamp] = {}
            if batch_key not in timestamp_dict[timestamp]:
                timestamp_dict[timestamp][batch_key] = ""
                timestamp_dict[timestamp][batch_key] = __format_data(values[3:])  # ([0] is event ID, [1] and [2] are timestamps)
            else: #there is one already -> merge them
                v1 = timestamp_dict[timestamp][batch_key]
                v2 = __format_data(values[3:])
                timestamp_dict = make_average(timestamp_dict, timestamp, batch_key, v1, v2)


    print("FINISHED pre-process of {}".format(batch_key))
    print(".")
    return timestamp_dict


def get_only_complete_values_old(timestampDictionary):
    allValues = list(timestampDictionary.values())
    final = []

    for value in allValues:
        if len(value) == 6:
            final.append(value)
    return final


def merge_two_dicts(x, y):
    """Given two dictionaries, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def try_get_missing(keyList, sourceValue):
    result = {}
    for key,value in sourceValue.items():
        if key in keyList:
            result[key] = value
    return result

def get_only_complete_values(timestampDictionary):
    all_value_data = {'left-pos', 'right-pos', 'right-forearm', 'left-forearm', 'right-hand', 'left-hand'}
    final = []
    dictKeyList = list(timestampDictionary.keys())
    for i in range(len(dictKeyList)):
        key = dictKeyList[i]
        value = timestampDictionary[key]

        if len(value) >= 3:
            if len(value) == 6:
                #gr8, just remember it
                final.append(value)
            else:
                #try to complement missing keys with neighbors..
                missing_keys = all_value_data - value.keys()
                updated = {}
                if i > 0:
                    previousKey = dictKeyList[i-1]
                    previousValue = timestampDictionary[previousKey]
                    complement = try_get_missing(missing_keys, previousValue)
                    updated = merge_two_dicts(value, complement)
                    if len(updated) == 6:
                        #gr8, remember it - and continue since we have a complete 'batch'
                        final.append(updated)
                        continue
                    else:
                        #check might happen for nextValue in next if and merge; update the value
                        value = updated

                if i < len(dictKeyList)-1:
                    nextKey = dictKeyList[i+1]
                    nextValue = timestampDictionary[nextKey]
                    complement = try_get_missing(missing_keys, nextValue)
                    updated = merge_two_dicts(value, complement)
                    if len(updated) == 6:
                        # gr8, remember it - and continue since we have a complete 'batch'
                        final.append(updated)
                        continue
    print("printim dlzku kompletnych dat")
    print(len(final))
    return final


def get_only_4complete_values(timestampDictionary):
    all_value_data = {'left-pos', 'right-pos', 'left-forearm', 'right-hand'}
    final = []
    dictKeyList = list(timestampDictionary.keys())
    for i in range(len(dictKeyList)):
        key = dictKeyList[i]
        value = timestampDictionary[key]

        if len(value) >= 3:
            if len(value) >= 4:  # Zmena tu - aspoň 4 hodnoty, nie 6
                # gr8, just remember it
                final.append(value)
            else:
                # try to complement missing keys with neighbors..
                missing_keys = all_value_data - value.keys()
                updated = {}
                if i > 0:
                    previousKey = dictKeyList[i - 1]
                    previousValue = timestampDictionary[previousKey]
                    complement = try_get_missing(missing_keys, previousValue)
                    updated = merge_two_dicts(value, complement)
                    if len(updated) >= 4:  # Zmena tu - aspoň 4 hodnoty, nie 6
                        # gr8, remember it - and continue since we have a complete 'batch'
                        final.append(updated)
                        continue
                    else:
                        # check might happen for nextValue in next if and merge; update the value
                        value = updated

                if i < len(dictKeyList) - 1:
                    nextKey = dictKeyList[i + 1]
                    nextValue = timestampDictionary[nextKey]
                    complement = try_get_missing(missing_keys, nextValue)
                    updated = merge_two_dicts(value, complement)
                    if len(updated) >= 4:  # Zmena tu - aspoň 4 hodnoty, nie 6
                        # gr8, remember it - and continue since we have a complete 'batch'
                        final.append(updated)
                        continue
    print("printim dlzku kompletnych dat")
    print(len(final))
    return final



def get_values_with_completion(timestampDictionary):
    all_value_data = {'left-pos', 'right-pos', 'right-forearm', 'left-forearm', 'right-hand', 'left-hand'}
    final = []
    dictKeyList = list(timestampDictionary.keys())
    
    for i in range(len(dictKeyList)):
        key = dictKeyList[i]
        value = timestampDictionary[key]
        
        if len(value) >= 3:
            if len(value) == 6:
                # If the batch is complete, append it directly
                final.append(value)
            else:
                # If the batch is incomplete, try to complement missing keys with neighbors
                missing_keys = all_value_data - value.keys()
                updated = dict(value)  # Make a copy to avoid modifying the original dictionary
                
                # Try to find missing keys from previous timestamps
                if i > 0:
                    previousKey = dictKeyList[i-1]
                    previousValue = timestampDictionary[previousKey]
                    complement = try_get_missing(missing_keys, previousValue)
                    updated.update(complement)
                
                # Try to find missing keys from next timestamps
                if i < len(dictKeyList)-1:
                    nextKey = dictKeyList[i+1]
                    nextValue = timestampDictionary[nextKey]
                    complement = try_get_missing(missing_keys, nextValue)
                    updated.update(complement)
                
                # Append the updated batch whether it's complete or not
                final.append(updated)

    return final



def parse_all_datadumps():

    #This it the "main" dictionary: its keys are timestamps of the recorded joint/skin values.
    # The value for each key is another dictionary, which has values for specific limb/skin parts
    # (see how a batch is created in data_collector.py for more details)

    timestamp_dict = {}
    """
    parent_dir = "data/datadumper/"
    leftArm = parent_dir + "joints_leftArm/data.log"
    rightArm = parent_dir + "joints_rightArm/data.log"
    skin_rightFore = parent_dir + "skin_tactile_comp_right_forearm/data.log"
    skin_leftFore = parent_dir + "skin_tactile_comp_left_forearm/data.log"
    skin_rightHand = parent_dir + "skin_tactile_comp_right_hand/data.log"
    skin_leftHand = parent_dir + "skin_tactile_comp_left_hand/data.log"
    """

    parent_dir = "data-icub-selftouch/20220628_leftForeArmRightHand_3_singlefinger/"
    leftArm = parent_dir + "joints_leftArm/data.log"
    rightArm = parent_dir + "joints_rightArm/data.log"
    
    skin_leftFore = parent_dir + "skin_tactile_comp_left_forearm/data.log"
    skin_rightHand = parent_dir + "skin_tactile_comp_right_hand/data.log"

    #parent_dir = "datadumper_data/ --> teto som sem supla fejove od leyly"
    #skin_rightFore = parent_dir + "skin_tactile_comp_right_forearm/data.log" 
    #skin_leftHand = parent_dir + "skin_tactile_comp_left_hand/data.log" 
    
    #leftProprio
    timestamp_dict = parse_file_and_append_to_dictionary(timestamp_dict, leftArm, 'left-pos')
    timestamp_dict = parse_file_and_append_to_dictionary(timestamp_dict, rightArm, 'right-pos')
    #timestamp_dict = parse_file_and_append_to_dictionary(timestamp_dict, skin_rightFore, 'right-forearm')
    timestamp_dict = parse_file_and_append_to_dictionary(timestamp_dict, skin_leftFore, 'left-forearm')
    timestamp_dict = parse_file_and_append_to_dictionary(timestamp_dict, skin_rightHand, 'right-hand')
    #timestamp_dict = parse_file_and_append_to_dictionary(timestamp_dict, skin_leftHand, 'left-hand')
    #print(timestamp_dict)
    


    print("Datadumper ===> Rawdata Test run finished.")
    #file_name = "TEST_data.rawdata"
    file_name = "TEST_data_NEW.rawdata"
    print("Exporting {} into .json ...".format(file_name))


    """####### ne len complete data
    f = open("TEST.rawdata", "w")
    f.write(timestamp_dict)
    f.close()"""


    batch_values_only = get_only_complete_values(timestamp_dict)
    #print(len(batch_values_only))
    #print(batch_values_only[0])
    data_rawdata = json.dumps(list(batch_values_only), sort_keys=True, indent=4)

    """pokus_data = get_values_with_completion(timestamp_dict)
    new_raw = json.dumps(list(pokus_data), sort_keys=True, indent=4)"""

    pokus_data = get_only_4complete_values(timestamp_dict)
    new_raw = json.dumps(list(pokus_data), sort_keys=True, indent=4)

    """f = open(file_name, "w")
    f.write(data_rawdata)
    f.close()
    print("Data exported successfully.")"""

    f = open(file_name, "w")
    f.write(new_raw)
    f.close()
    print("Data exported successfully.")

    #batch = dict()
    #batch['left-hand'] = __format_data(left_hand)
    #batch['left-forearm'] = __format_data(left_forearm)
    #batch['right-hand'] = __format_data(right_hand)
    #batch['right-forearm'] = __format_data(right_forearm)

    #batch['left-pos'] = self.__format_data(proprioLeft)
    #batch['right-pos'] = self.__format_data(proprioRight)


if __name__ == "__main__":
    parse_all_datadumps()