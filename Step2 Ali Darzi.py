import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

def preprocess_point_cloud(pcd, voxel_size=0.01):
    """ Downsample the point cloud to reduce data size and computational load. """
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def estimate_normals(pcd):
    """ Estimate normal vectors for the point cloud for better feature extraction. """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """ Remove statistical outliers from the point cloud. """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl

def apply_ransac_segmentation(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    """ Apply RANSAC to segment the largest planar structure in the point cloud. """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud

def apply_dbscan_clustering(pcd, eps=0.05, min_samples=10):
    """ Apply DBSCAN clustering to segment different objects in the point cloud. """
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    max_label = labels.max()
    print(f"DBSCAN identified {max_label + 1} clusters.")

    clustered_pcd = o3d.geometry.PointCloud()
    clustered_pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.random.rand(max_label + 1, 3)
    point_colors = np.array([colors[label] if label >= 0 else [0, 0, 0] for label in labels])
    clustered_pcd.colors = o3d.utility.Vector3dVector(point_colors)

    return clustered_pcd

# Load the PLY file
ply_file = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply"
pcd = o3d.io.read_point_cloud(ply_file)

# Apply preprocessing
pcd_downsampled = preprocess_point_cloud(pcd)
pcd_filtered = remove_outliers(pcd_downsampled)

# Apply RANSAC segmentation to remove planar surfaces (e.g., ground or walls)
inlier_cloud, outlier_cloud = apply_ransac_segmentation(pcd_filtered)

# Apply DBSCAN clustering to segment the remaining points
clustered_pcd = apply_dbscan_clustering(outlier_cloud)

# Visualize results
o3d.visualization.draw_geometries([clustered_pcd], window_name="DBSCAN Clustered Point Cloud")
