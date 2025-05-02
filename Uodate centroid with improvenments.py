import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# Load the PLY file
file_path = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply"  # Update as needed
pcd = o3d.io.read_point_cloud(file_path)

# Preprocess the point cloud
def preprocess_point_cloud(pcd, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_down.select_by_index(ind)

filtered_pcd = preprocess_point_cloud(pcd)

# Slice the point cloud along the Z-axis
def slice_point_cloud(pcd, z_interval=0.05):
    points = np.asarray(pcd.points)
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    slices = []
    for z in np.arange(min_z, max_z, z_interval):
        mask = (points[:, 2] >= z) & (points[:, 2] < z + z_interval)
        slice_points = points[mask]
        if len(slice_points) > 0:
            slices.append(slice_points)
    return slices

z_slices = slice_point_cloud(filtered_pcd, z_interval=0.05)

# Compute centroids for each slice
slice_centroids = []
for i, slice_points in enumerate(z_slices):
    xy_points = slice_points[:, :2]
    if len(xy_points) >= 3:  # ConvexHull requires at least 3 points
        hull = ConvexHull(xy_points)
        boundary_points = xy_points[hull.vertices]
        centroid = np.mean(boundary_points, axis=0)
        slice_centroids.append([centroid[0], centroid[1], np.mean(slice_points[:, 2])])

slice_centroids = np.array(slice_centroids)

# Filter centroids with large deviations
distances = squareform(pdist(slice_centroids))
mean_distances = np.mean(distances, axis=1)
threshold = np.mean(mean_distances) + 2 * np.std(mean_distances)
filtered_centroids = slice_centroids[mean_distances < threshold]

# Fit a 3D spline through the filtered centroids
def fit_centerline(centroids):
    tck, u = splprep(centroids.T, s=500)  # Adjust smoothness parameter `s` if needed
    u_fine = np.linspace(0, 1, 500)
    return np.array(splev(u_fine, tck)).T

centerline = fit_centerline(filtered_centroids)

# Visualize the updated centerline
centerline_pcd = o3d.geometry.PointCloud()
centerline_pcd.points = o3d.utility.Vector3dVector(centerline)
centerline_pcd.paint_uniform_color([1, 0, 0])  # Red color for centerline

filtered_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Grey color for the pipe
o3d.visualization.draw_geometries([filtered_pcd, centerline_pcd], window_name="Improved 3D Centerline Visualization")

# Plot the centroids and centerline in Matplotlib
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(filtered_centroids[:, 0], filtered_centroids[:, 1], filtered_centroids[:, 2], c="green", label="Filtered Centroids")
ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], c="red", label="Fitted Centerline")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.title("3D Centerline and Centroids")
plt.show()
