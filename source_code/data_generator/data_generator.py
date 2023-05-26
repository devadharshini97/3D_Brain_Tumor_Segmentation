import torch
import nibabel as nib
import numpy as np
import os
import glob
from monai.transforms.spatial.array import Spacing
from monai.transforms.intensity.array import ScaleIntensity
from source_code.utilities.utils import text_file_reader


class BTSDataset(torch.utils.data.Dataset):
    def __init__(self, patient_data_list_path, src_folder=None, no_classes=4):
        if src_folder is not None:
          patient_data_list = sorted(text_file_reader(os.path.join(src_folder, patient_data_list_path)))
          patient_data_list = [os.path.join(src_folder, i) for i in patient_data_list]
        else:
          patient_data_list = sorted(text_file_reader(patient_data_list_path))
        self.patient_flair_scans_list = [glob.glob(os.path.join(i, "*_flair.nii.gz"))[0] for i in patient_data_list]
        self.patient_t1ce_scans_list = [glob.glob(os.path.join(i, "*_t1ce.nii.gz"))[0] for i in patient_data_list]
        self.patient_t2_scans_list = [glob.glob(os.path.join(i, "*_t2.nii.gz"))[0] for i in patient_data_list]
        self.patient_seg_scans_list = [glob.glob(os.path.join(i, "*_seg.nii.gz"))[0] for i in patient_data_list]
        self.normalizer = ScaleIntensity()
        self.no_classes = no_classes

    def __len__(self):
        return len(self.patient_flair_scans_list)

    def __getitem__(self, idx):
        t1ce = self.normalizer(torch.Tensor(np.asarray(nib.load(self.patient_t1ce_scans_list[idx]).get_fdata())[:, :, 5: -6]))
        t2 = self.normalizer(torch.Tensor(np.asarray(nib.load(self.patient_t2_scans_list[idx]).get_fdata())[:, :, 5: -6]))
        flair = self.normalizer(torch.Tensor(np.asarray(nib.load(self.patient_flair_scans_list[idx]).get_fdata())[:, :, 5: -6]))
        stacked = torch.stack([t1ce, t2, flair])
        # stacked = Spacing(pixdim=(1.875, 1.875, 1), mode="bilinear")(stacked)
        stacked = Spacing(pixdim=(2.51, 2.51, 1), mode="bilinear")(stacked)
        seg_lbl = np.asarray(nib.load(self.patient_seg_scans_list[idx]).get_fdata())[:, :, 5: -6]
        seg_lbl[seg_lbl == 4] = 3
        seg_lbl = Spacing(pixdim=(2.51, 2.51, 1), mode="nearest")(seg_lbl[np.newaxis, :, :, :])
        # seg_lbl = Spacing(pixdim=(1.875, 1.875, 1), mode="nearest")(seg_lbl[np.newaxis, :, :, :])
        return stacked.to(torch.float32), seg_lbl.to(torch.float32)


if __name__ == "__main1__":
    path = ""
    folder_list = sorted(glob.glob(os.path.join(path, "*")))
    folder_list = [i for i in folder_list if os.path.isdir(i)][: 20]
    sample_dataset = BTSDataset(folder_list)


if __name__ == "__main2__":
    import matplotlib.pyplot as plt
    import yaml

    config_path = "/content/drive/MyDrive/Personal/MS/Brain_Tumor_segmentaion3D/source_code/configs/overfitting_test.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    training_datagenerator = BTSDataset(**config["Training_Dataset"])
    ex_img, ex_lbl = training_datagenerator[0]
    ex_lbl_numpy = np.asarray(ex_lbl.numpy(), dtype=np.uint8)
    np.unique(ex_lbl_numpy, return_counts=True)
    plt.imshow(ex_lbl_numpy[0, :, :, 108] == 3)
    ex_img_numpy = np.asarray(ex_img.numpy(), dtype=np.float32)
    np.histogram(ex_img_numpy[2, :, :, 108])
    plt.imshow(ex_img_numpy[2, :, :, 108])
    plt.imshow(ex_img_numpy[1, :, :, 108])
    plt.imshow(ex_img_numpy[0, :, :, 108])
