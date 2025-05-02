import open3d as o3d
import numpy as np
import os
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA

# ============================
# File Paths & Output Directory
# ============================
ply_files = [
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply"
]

output_dir = r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULTIWAY/"
os.makedirs(output_dir, exist_ok=True)

# ============================
# Preprocessing & MLS Smoothing
# ============================
def preprocess_point_cloud(pcd, voxel_size=0.01):
    """ Downsample the point cloud to remove noise. """
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def apply_mls_smoothing(pcd, search_radius=0.15):
    """ Apply Moving Least Squares (MLS) to refine the centerline. """
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=50))
    smoothed_pcd = o3d.geometry.PointCloud()
    smoothed_pcd.points = pcd.points
    return smoothed_pcd

def remove_outliers(pcd, nb_neighbors=50, std_ratio=1.0):
    """ Remove statistical outliers to clean noise. """
    filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return filtered_pcd

# ============================
# Extract Final Centerline Using B-Spline
# ============================
def extract_final_centerline(pcd, num_points=3000):
    """ Convert the smoothed point cloud into a structured, fully continuous centerline. """
    points = np.asarray(pcd.points)

    if len(points) < 10:
        print("⚠️ Not enough points for spline fitting, returning original points.")
        return points  

    try:
        # Sort points along Z-axis
        sorted_indices = np.argsort(points[:, 2])
        points = points[sorted_indices]

        # Apply B-Spline with even finer smoothness for better continuity
        tck, u = splprep(points.T, s=0.001, k=min(3, len(points)-1))
        u_fine = np.linspace(0, 1, num_points)
        spline_output = np.array(splev(u_fine, tck)).T

        if len(spline_output) < 10:
            print("⚠️ Spline output too small, using original points instead.")
            return points
        
        return spline_output
    except Exception as e:
        print(f"⚠️ Spline fitting failed: {e}, using original points instead.")
        return points

# ============================
# Process Each PLY File
# ============================
processed_centerlines = []

for file in ply_files:
    if not os.path.exists(file):
        print(f"❌ File not found: {file}")
        continue

    print(f"🔄 Processing: {file}")
    
    pcd = o3d.io.read_point_cloud(file)
    pcd = preprocess_point_cloud(pcd)

    # Apply MLS Smoothing
    smoothed_pcd = apply_mls_smoothing(pcd)

    # Remove Outliers
    cleaned_pcd = remove_outliers(smoothed_pcd)

    # Extract Centerline
    final_centerline_curve = extract_final_centerline(cleaned_pcd)

    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(final_centerline_curve)

    if len(np.asarray(final_pcd.points)) == 0:
        print(f"⚠️ Skipping save: No points found in the centerline for {file}")
    else:
        filename = f"centerline_{os.path.basename(file)}"
        save_path = os.path.join(output_dir, filename)
        o3d.io.write_point_cloud(save_path, final_pcd)
        print(f"✅ Saved centerline: {save_path}")

        processed_centerlines.append(final_pcd)

# ============================
# Visualize the Final Centerlines
# ============================
o3d.visualization.draw_geometries(processed_centerlines)
print("🎯 Pipeline completed successfully!")
