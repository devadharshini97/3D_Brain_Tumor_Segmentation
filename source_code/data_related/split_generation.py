import random
import glob
import os
from tqdm import tqdm
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def compute_class_volumes(patient_folder_list):
    core_tumor = []
    peritumoral_tissue = []
    enhancing_tumor = []
    cube_size = 240 * 240 * 155
    for i in tqdm(patient_folder_list):
        patient_label_data_path = os.path.join(i, "BraTS2021_%05d_seg.nii.gz" % int(i.split("_")[-1]))
        patient_label_data = nib.load(patient_label_data_path).get_fdata()
        core_tumor.append((len(np.where(patient_label_data == 1)[0]) / cube_size) * 100)
        peritumoral_tissue.append((len(np.where(patient_label_data == 2)[0]) / cube_size) * 100)
        enhancing_tumor.append((len(np.where(patient_label_data == 4)[0]) / cube_size) * 100)
    return core_tumor, peritumoral_tissue, enhancing_tumor


def visualize_class_distributions(data):
    core_tumor, peritumoral_tissue, enhancing_tumor = data
    data_string = ["core_tumor", "peritumoral_tissue", "enhancing_tumor"]
    fig = plt.figure(figsize=(12, 12))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if i == 3:
            plt.bar(x=[0, 1, 2], height=[np.average(core_tumor), np.average(peritumoral_tissue), np.average(enhancing_tumor)])
            plt.title("Average volume of %s, %s, %s" % (data_string[0], data_string[1], data_string[2]))
        else:
            plt.hist(data[i])
            plt.title("Distribution of volume of %s" % data_string[i])
    return np.average(core_tumor), np.average(peritumoral_tissue), np.average(enhancing_tumor)


def compute_two_splits(data_list, split_ratio):
    continue_loop = True
    while continue_loop:
        list1 = random.sample(data_list, int(0.8 * len(data_list)))
        list2 = [i for i in data_list if i not in list1]
        list1_volumes = compute_class_volumes(list1)
        list2_volumes = compute_class_volumes(list2)
        list1_volumes_averages = [np.average(i) for i in list1_volumes]
        list2_volumes_averages = [np.average(i) for i in list2_volumes]
        overall_averages = [np.average(i + j) for i, j in zip(list1_volumes_averages, list2_volumes_averages)]
        list1_class_proportion = np.divide(list1_volumes_averages, np.sum(list1_volumes_averages))
        list2_class_proportion = np.divide(list2_volumes_averages, np.sum(list2_volumes_averages))
        overall_class_proportion = np.divide(overall_averages, np.sum(overall_averages))
        print("Current split data: ", list1_class_proportion, list2_class_proportion)
        if ((overall_class_proportion[0] - 0.01 <= list1_class_proportion[0] <= overall_class_proportion[0] + 0.01) and
           (overall_class_proportion[1] - 0.01 <= list1_class_proportion[1] <= overall_class_proportion[1] + 0.01) and
           (overall_class_proportion[2] - 0.01 <= list1_class_proportion[2] <= overall_class_proportion[2] + 0.01) and
           (overall_class_proportion[0] - 0.01 <= list2_class_proportion[0] <= overall_class_proportion[0] + 0.01) and
           (overall_class_proportion[1] - 0.01 <= list2_class_proportion[1] <= overall_class_proportion[1] + 0.01) and
           (overall_class_proportion[2] - 0.01 <= list2_class_proportion[2] <= overall_class_proportion[2] + 0.01)):
                return list1, list2


if __name__ == "__main1__":
    patient_data_folder = ""
    patient_data_list = sorted(glob.glob(os.path.join(patient_data_folder, "*")))
    train_split, test_split = compute_two_splits(patient_data_list, 0.8)
    training_final, validation = compute_two_splits(train_split, 0.8)
    train_final_volumes = compute_class_volumes(training_final)
    validation_volumes = compute_class_volumes(validation)
    test_volumes = compute_class_volumes(test_split)
    visualize_class_distributions(train_final_volumes)
    visualize_class_distributions(validation_volumes)
    visualize_class_distributions(test_volumes)


