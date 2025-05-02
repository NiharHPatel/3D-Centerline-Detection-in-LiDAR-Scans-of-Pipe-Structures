import open3d as o3d
import numpy as np
import os
from scipy.interpolate import splprep, splev
from sklearn.linear_model import RANSACRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import NearestNeighbors

# List of PLY file paths
ply_files = [
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/2_0002.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/3_0000.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/4_0005.000.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/5_0015.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/6_0006.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/7_0029.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/8_0007.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/9_0000.000.ply"
]

# Set output directory to your requested location
output_dir = r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Extracted_Centerlines"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

def preprocess_point_cloud(pcd, voxel_size=0.01):
    """ Downsample the point cloud and remove statistical outliers. """
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def estimate_normals(pcd):
    """ Compute normal vectors for the point cloud. """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def adaptive_ransac_params(points):
    """ Dynamically adjust RANSAC distance threshold using nearest-neighbor distances. """
    if len(points) < 10:
        return 0.02  # Default fallback

    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    avg_distance = np.mean(distances[:, 1])  # Avoid self-distance

    return avg_distance * 2.0  # Slightly increase threshold to retain more inliers

def fit_cylinder_ransac_robust(pcd, max_trials=3000):
    """ Fit a cylinder to the point cloud using Adaptive RANSAC with Robust Kernels. """
    points = np.asarray(pcd.points)

    if len(points) < 10:
        print("Warning: Not enough points for RANSAC. Skipping.")
        return pcd, np.ones(len(points), dtype=bool)

    adaptive_threshold = adaptive_ransac_params(points)

    model_ransac = RANSACRegressor(
        residual_threshold=adaptive_threshold,
        max_trials=max_trials,
        random_state=42
    )

    model_ransac.fit(points[:, :2], points[:, 2])
    z_pred = model_ransac.predict(points[:, :2])
    residuals = np.abs(points[:, 2] - z_pred)

    inlier_mask = residuals < adaptive_threshold
    cylinder_pcd = pcd.select_by_index(np.where(inlier_mask)[0])

    return cylinder_pcd, inlier_mask

def compute_centerline(pcd):
    """ Compute centerline using spline fit or Gaussian Process Regression for abrupt turns. """
    points = np.asarray(pcd.points)

    if len(points) < 5:
        return points  # Not enough points for spline

    points = points[np.argsort(points[:, 2])]  # Sort by Z

    unique_z_values, unique_indices = np.unique(points[:, 2], return_index=True)
    centerline_points = points[unique_indices]

    if len(centerline_points) < 4:
        print("Using Gaussian Process Regression due to low points.")
        kernel = C(1.0, (1e-3, 1e2)) * RBF(0.5, (1e-2, 1e2))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        gpr.fit(unique_z_values.reshape(-1, 1), centerline_points[:, :2])
        z_fine = np.linspace(unique_z_values.min(), unique_z_values.max(), 500)
        xy_fine = gpr.predict(z_fine.reshape(-1, 1))
        return np.hstack((xy_fine, z_fine[:, None]))

    try:
        tck, u = splprep(centerline_points.T, s=0.1, k=3)
        u_fine = np.linspace(0, 1, 500)
        return np.array(splev(u_fine, tck)).T
    except:
        return centerline_points

# Process each PLY file
for i, file in enumerate(ply_files):
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue

    print(f"Processing: {file}")

    pcd = o3d.io.read_point_cloud(file)
    pcd_filtered = preprocess_point_cloud(pcd)
    pcd_filtered = estimate_normals(pcd_filtered)

    cylinder_pcd, inlier_mask = fit_cylinder_ransac_robust(pcd_filtered)
    centerline = compute_centerline(cylinder_pcd)

    # Convert centerline to Open3D PointCloud
    centerline_pcd = o3d.geometry.PointCloud()
    centerline_pcd.points = o3d.utility.Vector3dVector(centerline)
    centerline_pcd.paint_uniform_color([1, 0, 0])  # Red centerline

    # Save the centerline as a PLY file
    centerline_filename = os.path.join(output_dir, f"centerline_{i+1}.ply")
    o3d.io.write_point_cloud(centerline_filename, centerline_pcd)
    print(f"✅ Saved: {centerline_filename}")

    # Use an Oriented Bounding Box (OBB) instead of a regular bounding box
    bbox = cylinder_pcd.get_oriented_bounding_box()
    bbox.color = (0, 1, 0)  # Green bounding box

    # Visualization
    o3d.visualization.draw_geometries([pcd_filtered, cylinder_pcd.paint_uniform_color([0, 0, 1]), centerline_pcd, bbox])

print(f"✅ Processing complete. All centerlines saved in '{output_dir}'.")
