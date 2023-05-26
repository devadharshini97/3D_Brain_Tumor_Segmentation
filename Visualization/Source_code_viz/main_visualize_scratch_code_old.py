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

def save_3d(prediction, output_file=None):
    verts, faces, _, _ = marching_cubes(prediction, 0)
    # Create a mesh using trimesh
    vertex_colors = get_vertex_colors(prediction)
    print(vertex_colors.shape)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)
    
    # Save the mesh as an OBJ file
    if output_file:
        mesh.export(output_file)
    # Plot the mesh using matplotlib
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces, shade=True)
    # plt.show()
def get_vertex_colors(prediction):
    # Create vertex colors based on the prediction array
    vertex_colors = np.zeros((prediction.shape[0], prediction.shape[1], prediction.shape[2], 4), dtype=np.uint8)
    # Set the vertex colors based on the prediction labels
    vertex_colors[prediction == 0, :] = [128, 128, 128, 255]  # Gray for background
    vertex_colors[prediction == 1, :] = [251, 171, 152, 255]  # Red for tumor core
    vertex_colors[prediction == 2, :] = [122, 136, 204, 192]  # Blue for peritumoral edema
    vertex_colors[prediction == 3, :] = [225, 255, 212, 255]  # Green for enhancing tumor
    return vertex_colors

def plot_3d(prediction, output_file=None):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Set the limits of the axis
  ax.set_xlim([0, 128])
  ax.set_ylim([0, 128])
  ax.set_zlim([0, 144])

  #Create a 3D voxel grid and set the color based on the image intensity
  colors = np.empty(prediction.shape, dtype=object)
  colors[prediction == 0] = 'gray'
  colors[prediction == 1] = '#fbab98' #'red'
  colors[prediction == 2] = '#7A88CCC0' #'blue'
  colors[prediction == 3] = '#e1ffd4'  #'green'
  # ax.voxels(np.asarray(prediction * 0.4 + image[idx] * 0.6, np.int16), facecolors=colors)
  vertices, triangles, _, _ = marching_cubes(prediction, 0)
  mesh = meshio.Mesh(vertices=[vertices], cells=[("triangle", triangles)])
  meshio.write(output_file, mesh, file_format="stl")
  ax.voxels(prediction, facecolors=colors)
  ax.set_aspect('equal')
  plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
config_path = "C:/Users/dayyapp/Desktop/Brain_Tumor_segmentaion3D-main/source_code/configs/unetr_fullscale_training_ncsu.yaml"
model_weights = "C:/Users/dayyapp/Desktop/Project/Train1_240/model_weights_95.pth"
with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # model = UNETR(**config["Model"])
model = instantiate_attribute(config["Model"]["path"])(**config["Model"]["params"])
print("model initiated")
model.load_state_dict(torch.load(model_weights))  ##, map_location=torch.device('cpu')))
model.eval()
test_datagenerator = BTSDataset(**config["Test_Dataset"])
test_dataloader = torch.utils.data.DataLoader(test_datagenerator, batch_size=1,shuffle=True)
ex_img, ex_lbl = test_datagenerator[10]
print(ex_img.shape, ex_lbl.shape)
prediction = model(ex_img.unsqueeze(0))
prediction = np.argmax(prediction, 1)
prediction = prediction.squeeze()
print(np.unique(prediction, return_counts=True), np.unique(ex_lbl[0], return_counts=True))
output_file = "C:/Users/dayyapp/Desktop/Project/240_Visualize_results/predicted.obj"
save_3d(np.asarray(prediction, np.uint8),output_file = output_file)
save_3d(np.asarray(ex_lbl[0], np.uint8),output_file = "C:/Users/dayyapp/Desktop/Project/240_Visualize_results/label.obj")


# mesh = meshio.Mesh(np.transpose(prediction.nonzero()), {"hexahedron": prediction.nonzero()})
#   if output_file:
#         meshio.write(output_file, mesh, file_format="obj")