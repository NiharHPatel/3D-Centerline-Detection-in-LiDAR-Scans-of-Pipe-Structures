import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from sklearn.cluster import KMeans

# List of PLY file paths
ply_files = [
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/2_0002.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/3_0000.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/4_0005.000.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/5_0015.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/6_0006.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/7_0029.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/8_0007.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/9_0000.000.ply"
]

### **Step 1: Preprocess Point Cloud**
def preprocess_point_cloud(pcd, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_down.select_by_index(ind)

### **Step 2: Slice and Compute Centroids**
def slice_point_cloud(pcd, z_interval=0.1):
    points = np.asarray(pcd.points)
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    slices = []
    for z in np.arange(min_z, max_z, z_interval):
        mask = (points[:, 2] >= z) & (points[:, 2] < z + z_interval)
        slice_points = points[mask]
        if len(slice_points) > 0:
            slices.append(slice_points)
    return slices

def compute_centroids(z_slices):
    centroids = []
    for slice_points in z_slices:
        xy_points = slice_points[:, :2]
        if len(xy_points) >= 3:
            hull = ConvexHull(xy_points)
            boundary_points = xy_points[hull.vertices]
            centroid = np.mean(boundary_points, axis=0)
            centroids.append([centroid[0], centroid[1], np.mean(slice_points[:, 2])])
    return np.array(centroids)

### **Step 3: Auto-Tune Clustering and Smoothing**
def auto_tune_parameters(centroids):
    kmeans = KMeans(n_clusters=min(5, len(centroids)//5)).fit(centroids)
    optimal_eps = np.mean(np.linalg.norm(centroids - kmeans.cluster_centers_[kmeans.labels_], axis=1))
    smooth_factor = np.clip(optimal_eps * 100, 5, 50)
    return optimal_eps, smooth_factor

### **Step 4: Fit B-Spline Centerline**
def fit_global_spline(centroids, degree=3, smooth_factor=10):
    centroids = np.array(centroids)
    tck, u = splprep(centroids.T, k=degree, s=smooth_factor)
    u_fine = np.linspace(0, 1, 500)
    return np.array(splev(u_fine, tck)).T

### **Step 5: Apply Color Mapping Based on Deviation**
def apply_color_based_on_deviation(pcd, centerline):
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points[:, None, :] - centerline[None, :, :], axis=2).min(axis=1)
    max_dev = np.percentile(distances, 98)
    colors = plt.cm.viridis(distances / max_dev)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

### **Step 6: Fill Gaps Using Poisson Surface Reconstruction**
def fill_pipeline_gaps(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    return mesh

# Process each PLY file
for file in ply_files:
    print(f"Processing: {file}")
    
    pcd = o3d.io.read_point_cloud(file)
    if len(pcd.points) == 0:
        print(f"Skipping empty file: {file}")
        continue

    # Preprocess and slice point cloud
    pcd_filtered = preprocess_point_cloud(pcd)
    z_slices = slice_point_cloud(pcd_filtered, z_interval=0.1)
    centroids = compute_centroids(z_slices)

    if len(centroids) < 3:
        print(f"Insufficient centroids for spline fitting in file: {file}")
        continue

    # Auto-tune parameters
    optimal_eps, smooth_factor = auto_tune_parameters(centroids)

    # Fit the centerline
    centerline = fit_global_spline(centroids, degree=3, smooth_factor=smooth_factor)

    # Compute deviation and apply color
    pcd_colored = apply_color_based_on_deviation(pcd_filtered, centerline)

    # Fill gaps using Poisson Reconstruction
    mesh = fill_pipeline_gaps(pcd_colored)

    # Visualize the reconstructed pipeline
    o3d.visualization.draw_geometries([mesh, pcd_colored], window_name=f"Reconstructed Pipeline - {file}")

    # Visualize Centerline
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="green", label="Centroids")
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], c="red", label="Fitted Centerline")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(f"Refined 3D Centerline for {file}")
    plt.show()

print("Processing complete.")
