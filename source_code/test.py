import nibabel as nib
import glob
import os
import torchvision.transforms as transforms
import torch
import random
from source_code.models.unetr import UNETR
from source_code.utilities.utils import text_file_writer


if __name__ == "__main1__":
    dataset_source_folder = "dataset/BRaTS2021_Training_Data"
    sample_size = 100
    save_path = "dataset/lists/sample_%d.txt" % sample_size
    patient_folders_list = sorted(os.listdir(dataset_source_folder))
    patient_folders_list = [os.path.join(dataset_source_folder, i) for i in patient_folders_list]
    sample_patient_folder_list = random.sample(patient_folders_list, sample_size)
    text_file_writer(save_path, sample_patient_folder_list)


if __name__ == "__main1__":
    model = UNETR(
            in_channels=3,
            out_channels=4,
            img_size=(144, 240, 240),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            conv_block=True,
            res_block=True,
            dropout_rate=0,
        )
    transformation = transforms.ToTensor()
    img_folder = "C:/Users/sumuk/OneDrive/Desktop/NCSU_related/Courses_and_stuff/Courses_and_stuff/NCSU_courses_and_books/ECE_542/FinalProj/Brain_Tumor_segmentaion3D/dataset/BRaTS2021_Training_Data/BraTS2021_00000"
    flair_path = glob.glob(os.path.join(img_folder, "*_flair.nii.gz"))[0]
    t1ce_path = glob.glob(os.path.join(img_folder, "*t1ce.nii.gz"))[0]
    t2_path = glob.glob(os.path.join(img_folder, "*_t2.nii.gz"))[0]
    flair = transformation(nib.load(flair_path).get_fdata()[:, :, 5: -6])
    t1ce = transformation(nib.load(t1ce_path).get_fdata()[:, :, 5: -6])
    t2 = transformation(nib.load(t2_path).get_fdata()[:, :, 5: -6])
    stacked = torch.stack([flair, t1ce, t2])
    stacked = stacked.unsqueeze(0)
    stacked = stacked.to(torch.float32)
    prediction = model(stacked)
    print("")


if __name__ == "__main2__":
    weights_path = "C:/Users/sumuk/OneDrive/Desktop/NCSU_related/Courses_and_stuff/Courses_and_stuff/NCSU_courses_and_books/ECE_542/FinalProj/model_weights/UNETR_model_best_acc.pth"
    model = UNETR(
            in_channels=3,
            out_channels=16,
            img_size=(144, 240, 240),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            conv_block=True,
            res_block=True,
            dropout_rate=0,
        )
    # model.load_state_dict(torch.load(weights_path))
    # model.eval()
    transformation = transforms.ToTensor()
    img_folder = "C:/Users/sumuk/OneDrive/Desktop/NCSU_related/Courses_and_stuff/Courses_and_stuff/NCSU_courses_and_books/ECE_542/FinalProj/Brain_Tumor_segmentaion3D_old/dataset/BRaTS2021_Training_Data/BraTS2021_00000"
    flair_path = glob.glob(os.path.join(img_folder, "*_flair.nii.gz"))[0]
    t1ce_path = glob.glob(os.path.join(img_folder, "*t1ce.nii.gz"))[0]
    t2_path = glob.glob(os.path.join(img_folder, "*_t2.nii.gz"))[0]
    flair = transformation(nib.load(flair_path).get_fdata()[:, :, 5: -6])
    t1ce = transformation(nib.load(t1ce_path).get_fdata()[:, :, 5: -6])
    t2 = transformation(nib.load(t2_path).get_fdata()[:, :, 5: -6])
    stacked = torch.stack([flair, t1ce, t2])
    stacked = stacked.unsqueeze(0)
    stacked = stacked.to(torch.float32)
    prediction = model(stacked)
    print("")


if __name__ == "__main2__":
    """
    Model prediction sanity check
    """
    import matplotlib.pyplot as plt
    import torch
    import yaml
    import numpy as np
    from ipywidgets import interact
    from source_code.data_generator.data_generator import BTSDataset


    class VisualizePredictionSanity:
        def __init__(self, img, lbl, model):
            with torch.no_grad():
                model.to(device)
                prediction = model(img.unsqueeze(0).cuda().to(device))
                prediction = prediction.squeeze()
                prediction = np.argmax(prediction, axis=0)
            self.scans = [lbl.squeeze(), prediction]
            self.fig = plt.figure(figsize=(1, 1));

        def visualize_brain_scans(self, idx):
            def create_display(layer):
                self.fig.add_subplot(2, 1, idx + 1)
                plt.imshow(scans_i[:, :, layer], cmap="BuPu");
                plt.axis('off')
                return layer

            scans_i = self.scans[idx]
            interact(create_display, layer=(0, scans_i.shape[2] - 1));

        def __call__(self, idx):
            self.visualize_brain_scans(idx)


    # config_path = "/content/drive/MyDrive/Personal/MS/Brain_Tumor_segmentaion3D/source_code/configs/overfitting_test.yaml"
    config_path = "/content/drive/MyDrive/ECE542/Brain_Tumor_segmentaion3D/source_code/configs/overfitting_test.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_weights = "/content/model_weights_18.pth"
    model = UNETR(**config["Model"])
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    training_datagenerator = BTSDataset(**config["Training_Dataset"])
    ex_img, ex_lbl = training_datagenerator[0]
    visualizer = VisualizePredictionSanity(ex_img, ex_lbl, model)
    visualizer(0)
    visualizer2 = VisualizePredictionSanity(ex_img, ex_lbl, model)
    visualizer2(1)
