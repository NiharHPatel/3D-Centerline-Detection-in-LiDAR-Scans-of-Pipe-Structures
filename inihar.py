import open3d as o3d

# Correct file path with `.ply` extension
file_path = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply"  # Update as needed
pcd = o3d.io.read_point_cloud(file_path)

# Inspect the point cloud
print(f"Number of points: {len(pcd.points)}")
print(f"Bounding box: {pcd.get_axis_aligned_bounding_box()}")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

import open3d as o3d

def preprocess_point_cloud(pcd, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
    """
    Apply noise filtering to the point cloud.
    - Voxel downsampling to reduce points.
    - Statistical outlier removal to filter noise.
    """
    # Voxel downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Statistical outlier removal
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered_pcd = pcd_down.select_by_index(ind)

    return filtered_pcd

# Apply preprocessing to the point cloud
filtered_pcd = preprocess_point_cloud(pcd)

# Visualize filtered point cloud
o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def detect_boundary(pcd):
    """
    Detect the boundary of the pipe cross-section using convex hull on 2D projection.
    """
    # Convert to numpy array
    points = np.asarray(pcd.points)

    # Project onto X-Y plane (ignore Z for now)
    xy_points = points[:, :2]

    # Compute convex hull
    hull = ConvexHull(xy_points)

    # Visualize convex hull
    plt.figure(figsize=(6, 6))
    plt.scatter(xy_points[:, 0], xy_points[:, 1], s=1, label="Point Cloud")
    plt.plot(xy_points[hull.vertices, 0], xy_points[hull.vertices, 1], 'r-', label="Convex Hull")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Pipe Boundary Detection")
    plt.show()

    return hull, xy_points

# Detect boundary
hull, xy_points = detect_boundary(filtered_pcd)
def compute_geometric_center(hull, points):
    """
    Compute the geometric center (centroid) of the convex hull boundary.
    """
    # Extract boundary points using hull vertices
    boundary_points = points[hull.vertices]
    
    # Compute the centroid as the mean of boundary points
    centroid = np.mean(boundary_points, axis=0)
    
    print(f"Computed Centroid (X, Y): {centroid}")
    
    # Plot the centroid with the convex hull
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=1, label="Point Cloud")
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'r-', label="Convex Hull")
    plt.scatter(centroid[0], centroid[1], c='green', s=100, label="Geometric Center")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Geometric Center of Pipe Boundary")
    plt.show()
    
    return centroid

# Compute the geometric center for the convex hull
geometric_center = compute_geometric_center(hull, xy_points)

