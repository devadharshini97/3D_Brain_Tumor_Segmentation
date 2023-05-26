from skimage.measure import marching_cubes 
import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import torch.cuda
from utils import instantiate_attribute
import yaml
from data_generator import BTSDataset
from unetr import UNETR
import meshio
import scipy.ndimage.measurements as mlabel
import trimesh
import open3d as o3d
import numpy as np

def save_3d(prediction, output_file=None):
    print(prediction.shape)
    verts, faces, _, _ = marching_cubes(prediction, 0)
    gray_prediction = prediction[prediction == 0]
    print(gray_prediction.shape)
    red_prediction = prediction[prediction == 1]
    print(red_prediction.shape)
    blue_prediction = prediction[prediction == 1]
    print(blue_prediction.shape)
    green_prediction = prediction[prediction == 1]
    #vert1, face1, _, _ = marching_cubes(prediction == 0,0)
    #print(vert1.shape)
    print(verts.shape)
    print(verts[0].shape)
    print(faces.shape)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    colors = np.empty(prediction.shape + (3,))
    colors[prediction == 0] = [0.5, 0.5, 0.5]  # gray
    colors[prediction == 1] = [0.984, 0.675, 0.596]  # light red
    colors[prediction == 2] = [0.478, 0.533, 0.8]  # light blue
    colors[prediction == 3] = [0.882, 1.0, 0.831] 
    print(colors.shape)
    vertsc, facesc, _, _ = marching_cubes(colors, 0)
    print(vertsc.shape)
    # gray_verts = verts[prediction == 0]
    # light_red_verts = verts[prediction == 1]
    # light_blue_verts = verts[prediction == 2]
    # light_green_verts = verts[prediction == 3]
    
    reshaped_colors = colors.reshape(-1,3)
    print(reshaped_colors.shape)
    print(reshaped_colors[0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(reshaped_colors)
    if output_file:
        o3d.io.write_triangle_mesh(output_file, mesh, write_vertex_colors=True, write_triangle_uvs= True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 128])
    ax.set_ylim([0, 128])
    ax.set_zlim([0, 144])
    ax.voxels(prediction, facecolors=colors)
    ax.set_aspect('equal')
    plt.show()
    # Create a visualizer object and add the mesh to it
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    cam.extrinsic = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 2],
                              [0, 0, 0, 1]])
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
    vis.run()
    vis.destroy_window()

device = "cuda" if torch.cuda.is_available() else "cpu"
config_path = "C:/Users/dayyapp/Desktop/Brain_Tumor_segmentaion3D-main/source_code/configs/unetr_fullscale_training_ncsu.yaml"
model_weights = "C:/Users/dayyapp/Desktop/Project/Train1_240/model_weights_95.pth"
with open(config_path, "r") as f:
        config = yaml.safe_load(f)
model = instantiate_attribute(config["Model"]["path"])(**config["Model"]["params"])
print("model initiated")
model.load_state_dict(torch.load(model_weights))  ##, map_location=torch.device('cpu')))
model.eval()
test_datagenerator = BTSDataset(**config["Test_Dataset"])
test_dataloader = torch.utils.data.DataLoader(test_datagenerator, batch_size=1,shuffle=True)
ex_img, ex_lbl = test_datagenerator[10]
print(ex_img.shape, ex_lbl.shape)
prediction = model(ex_img.unsqueeze(0))
print(prediction.shape)
prediction = np.argmax(prediction, 1)
prediction = prediction.squeeze()
print(np.unique(prediction, return_counts=True), np.unique(ex_lbl[0], return_counts=True))
#prediction[]
output_file = "C:/Users/dayyapp/Desktop/Project/240_Visualize_results/predicted2.obj"
save_3d(np.asarray(prediction, np.uint8),output_file = output_file)
save_3d(np.asarray(ex_lbl[0], np.uint8),output_file = "C:/Users/dayyapp/Desktop/Project/240_Visualize_results/label2.obj")