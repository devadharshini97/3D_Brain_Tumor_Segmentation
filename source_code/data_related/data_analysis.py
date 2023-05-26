"""
# Data Description:

### Image types:
* Native T1-weighted (T1): This scan is obtained using a standard T1-weighted imaging sequence, which uses a short TR
  (repetition time) and a short TE (echo time) to provide high-resolution images of the brain tissue. This sequence
   highlights the differences in tissue types based on their contrast with the surrounding tissues.

* Post-contrast T1-weighted (T1Gd): This scan is obtained using a T1-weighted imaging sequence after the administration
  of a contrast agent such as Gadolinium. The contrast agent is injected intravenously and is taken up by cells with a
  disrupted blood-brain barrier, which is a common characteristic of brain tumors. This sequence highlights the regions
  of the brain with a disrupted blood-brain barrier, such as enhancing tumor regions.

* T2-weighted (T2): This scan is obtained using a T2-weighted imaging sequence, which uses a long TR and a long TE to
  provide a more detailed view of the brain tissue. This sequence highlights subtle differences in tissue types that
  are not visible on T1 scans.

* T2 Fluid Attenuated Inversion Recovery (T2-FLAIR): This scan is obtained using a T2-weighted imaging sequence that is
  modified to suppress the signal from cerebrospinal fluid (CSF). This is achieved by using an inversion recovery pulse
  before the T2-weighted acquisition. This sequence is useful for distinguishing between edema and other types of brain
  tissue because the CSF signal is suppressed.

### Segmentation Classes:
* label 0: No tumor
* label 1: necrotic tumor core (Visible in T2): This class represents the core of the tumor, which is composed of
  necrotic tissue and non-enhancing tumor cells.
* label 2: the peritumoral edematous/invaded tissue (Visible in flair):  This class represents the edema, or swelling,
  that occurs around the tumor due to the accumulation of fluid in the surrounding brain tissue.
* label 4: Gd-enhancing tumor (Needs to be converted to 3) (Visible in T1ce): This class represents the region of the
  tumor that enhances with the administration of contrast agent.
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from ipywidgets import interact
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.measurements import label, center_of_mass


class VisualizePatientData:
    def __init__(self, patient_data_folder, img_type_id):
        self.patient_data_list = sorted(glob.glob(os.path.join(patient_data_folder, "*")))
        self.image_types = ["flair", "seg", "t1", "t1ce", "t2"]
        self.cmap_list = ["gray", "BuPu", "gray", "gray", "gray"]
        self.i = img_type_id
        self.fig = plt.figure(figsize=(1, 1))

    def visualize_brain_scans(self, cube_path):
        def create_display(layer):
            self.fig.add_subplot(3, 2, self.i + 1)
            plt.imshow(self.scans[:, :, layer], cmap=self.cmap_list[self.i])
            plt.axis('off')
            return layer
        self.scans = np.asarray(nib.load(cube_path).get_fdata())
        interact(create_display, layer=(0, self.scans.shape[2] - 1))

    def __call__(self, idx):
        data_path = os.path.join(self.patient_data_list[idx],
                                 "BraTS20_Training_%03d_%s.nii" % (idx + 1, self.image_types[self.i]))
        self.visualize_brain_scans(data_path)


def centroid_volume_correlation(patient_data_list, label_id):
    centroids = []
    volumes = []
    for i in tqdm(range(len(patient_data_list) - 2)):
        if i == 354:
            patient_label_data_path = "/kaggle/input/brain-tumor-segmentation-in-mri-brats-2015/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/W39_1998.09.19_Segm.nii"
        else:
            patient_label_data_path = os.path.join(patient_data_list[i], "BraTS20_Training_%03d_seg.nii" % (i + 1))
        patient_label_data = nib.load(patient_label_data_path).get_fdata()
        centroid_i, volume_i = compute_centroid_volume_largest_component(patient_label_data, label_id)
        if volume_i is None:
            continue
        centroids.append(centroid_i)
        volumes.append(volume_i)
    return centroids, volumes


def compute_centroid_volume_largest_component(seg_labels, label_id):
    seg_labels_core = np.asarray(seg_labels == label_id, dtype=np.uint8)
    labels, num_labels = label(seg_labels_core)
    volumes = []
    for i in range(1, num_labels + 1):
        volume = np.sum(labels == i)
        volumes.append(volume)
    volumes = np.array(volumes)
    if len(volumes):
        largest_components_id = np.argmax(volumes, -1) + 1
        centroid_i = center_of_mass(seg_labels_core, labels=labels, index=largest_components_id)
        volume_i = volumes[largest_components_id - 1]
        return centroid_i, volume_i
    else:
        return None, None


if __name__ == "__main1__":
    """
    Basic Data Visualization
    """
    patient_data_folder = ""
    example_patient_id = 0
    visualizer = VisualizePatientData(patient_data_folder, 0)
    visualizer(example_patient_id)


if __name__ == "__main2__":
    """
    class centroids to volume correlation
    """
    label_id = 1
    patient_data_folder = ""
    centroids, volumes = centroid_volume_correlation(patient_data_folder, label_id)
    centroids = np.array(centroids)
    volumes = np.array(volumes)
    fig = plt.figure(figsize=(120, 120))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=volumes, c=volumes, cmap='BuPu')
    ax.set_xlabel('X Centroid', fontsize=100)
    ax.set_ylabel('Y Centroid', fontsize=100)
    ax.set_zlabel('Z Centroid', fontsize=100)
    ax.set_title('Correlation between Centroid and Volume', fontsize=100)
    plt.show()
