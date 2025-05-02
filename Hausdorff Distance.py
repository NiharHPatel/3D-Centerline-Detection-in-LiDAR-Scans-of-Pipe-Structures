import open3d as o3d
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import NearestNeighbors
#try
# Path to the ground truth (final registered centerline)
ground_truth_file = r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/final_registered_centerline.ply"

# Path to the detected centerline PLY file (another output you want to compare)
detected_centerline_file = r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_6_0006.500.ply"  # Change this for different comparisons

def load_ply_file(file_path):
    """ Load a PLY file and return its point cloud as a NumPy array. """
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def chamfer_distance(A, B):
    """ Computes Chamfer Distance between two point sets. """
    nn_A = NearestNeighbors(n_neighbors=1).fit(B)
    distances_A, _ = nn_A.kneighbors(A)
    
    nn_B = NearestNeighbors(n_neighbors=1).fit(A)
    distances_B, _ = nn_B.kneighbors(B)

    chamfer_dist = np.mean(distances_A) + np.mean(distances_B)
    return chamfer_dist

def hausdorff_distance(A, B):
    """ Computes Hausdorff Distance between two point sets. """
    return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])

# Load the point clouds
ground_truth_centerline = load_ply_file(ground_truth_file)
detected_centerline = load_ply_file(detected_centerline_file)

# Compute accuracy metrics
chamfer_dist = chamfer_distance(detected_centerline, ground_truth_centerline)
hausdorff_dist = hausdorff_distance(detected_centerline, ground_truth_centerline)

#  Display Results
print(f" Chamfer Distance: {chamfer_dist:.6f}")
print(f" Hausdorff Distance: {hausdorff_dist:.6f}")
