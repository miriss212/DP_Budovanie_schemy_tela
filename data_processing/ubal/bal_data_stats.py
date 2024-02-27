import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
import data_util as du


def kwta_sanity_check_data_gen():
    with open("../ubal_data/my_data/" + trainingName + suffix) as f:
        data_generated = json.load(f)
    labeled = [(np.array(i['pos']), np.array(i["touch"])) for i in data_generated]
    with open("../bal_data/my_data/" + trainingName + suffix) as f:
        data_orig = json.load(f)
    labeled_orig = [(np.array(i['pos']), np.array(i["touch"])) for i in data_orig]
    du.analyze_data(labeled)
    du.analyze_data(labeled_orig)
    difs_propri = 0
    difs_touch = 0
    for i in range(len(data_generated)):
        # i = np.random.randint(len(labeled))
        # print("Showing item {}".format(i))
        test_data_propri, test_data_touch = labeled[i]
        test_data_propri_ori, test_data_touch_ori = labeled_orig[i]
        # print(test_data_touch)
        # print(test_data_touch_ori)
        # print(sum(test_data_touch == test_data_touch_ori))
        # print(test_data_propri)
        # print(test_data_propri_ori)
        # print(sum(test_data_propri == test_data_propri_ori))
        difs_propri += sum(test_data_propri == test_data_propri_ori) != len(test_data_propri)
        difs_touch += sum(test_data_touch == test_data_touch_ori) != len(test_data_touch)
    print("Differences in proprio: ",difs_propri)
    print("Differences in touch: ",difs_touch)


def kwta_sanity_check():
    with open("../ubal_data/my_data_act/" + trainingName + suffix) as f:
        data = json.load(f)
    k = 1
    # apply kwta per body part
    labeled = [(du.kwta_per_bodypart(np.array(i['pos']), k), np.array(i["touch"])) for i in data]
    with open("../bal_data/my_data/" + trainingName + suffix) as f:
        data_orig = json.load(f)
    labeled_orig = [(np.array(i['pos']), np.array(i["touch"])) for i in data_orig]
    du.analyze_data(labeled)
    du.analyze_data(labeled_orig)
    difs_propri = 0
    difs_touch = 0
    for i in range(len(data)):
        # i = np.random.randint(len(labeled))
        # print("Showing item {}".format(i))
        test_data_propri, test_data_touch = labeled[i]
        test_data_propri_ori, test_data_touch_ori = labeled_orig[i]
        # print(test_data_touch)
        # print(test_data_touch_ori)
        # print(sum(test_data_touch == test_data_touch_ori))
        # print(test_data_propri)
        # print(test_data_propri_ori)
        # print(sum(test_data_propri == test_data_propri_ori))
        difs_propri += sum(test_data_propri == test_data_propri_ori) != len(test_data_propri)
        difs_touch += sum(test_data_touch == test_data_touch_ori) != len(test_data_touch)
    print("Differences in proprio: ",difs_propri)
    print("Differences in touch: ",difs_touch)
    no_samples = 5
    # indeces = np.random.randint(len(labeled),size=no_samples)
    indeces = np.arange(0,no_samples)
    du.visualize(labeled,no_samples,"kwta_check_new",indeces)
    du.visualize(labeled_orig,no_samples,"kwta_check_orig",indeces)


suffix = ".baldata"
trainingName = "lh"
# trainingName = "rh"
# trainingName = "lf" # THIS IS GOOD DONT RETRAIN
#trainingName = "rf"  #
#trainingName = "lf_rh"
#trainingName = "all"  #

# kwta_sanity_check()
# kwta_sanity_check_data_gen()

# with open("../ubal_data/ubal_data_correct_touch/" + trainingName + ".baldata") as f:
# with open("../ubal_data/ubal_data_new/" + trainingName + ".baldata") as f:
with open("../ubal_data/ubal_data_leyla/" + trainingName + ".baldata") as f:
    data = json.load(f)
k = 13
# for k in range(20):
labeled = [(du.kwta_per_bodypart(np.array(i['pos']), k=k, continuous=True), np.array(i["touch"])) for i in data]
du.visualize(labeled,5,data_style="kwta-conti-{}".format(k))

# du.visualize_touch(labeled, "data_kwta")

# labeled = [(du.normalize_and_invert(np.array(i['pos'])), np.array(i["touch"])) for i in data]
# du.analyze_data(labeled)
# du.visualize(labeled,4,data_style="kwta-conti-16-RH")
# print()

# labeled = du.filter_ambiguous_touch(labeled)
# print("Filtered out {} out of {}".format(len(labeled),data_len_orig))
#
# du.analyze_data(labeled)
#
# print(np.shape(labeled))
# training,test = train_test_split(labeled, test_size=0.2)
# print(np.shape(training))
# print(np.shape(test))
#
# print("\n::::: TRAIN data :::::")
# du.analyze_data(training)
# print("\n::::: TEST data :::::")
# du.analyze_data(test)


# trainingNames = ["lh","rh","lf","rf"]
# data_all = []
#
# for name in trainingNames:
#     with open("../ubal_data/ubal_data_correct_touch/" + name + ".baldata") as f:
#         data = json.load(f)
#     labeled = [(np.array(i['pos']), np.array(i["touch"])) for i in data]
#     data_all.append(labeled)
#     print(name.upper())
#     du.analyze_data(labeled)
#
# du.visualize_touch_all(data_all, trainingNames)