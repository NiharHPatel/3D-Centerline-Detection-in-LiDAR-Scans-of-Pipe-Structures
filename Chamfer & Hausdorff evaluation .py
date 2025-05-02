import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
import os

# Set paths for original and fitted centerlines
raw_centerline_dir = r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Extracted_Centerlines"
smoothed_centerline_dir = r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Smoothed_Centerlines"

# List all PLY files
raw_files = sorted([os.path.join(raw_centerline_dir, f) for f in os.listdir(raw_centerline_dir) if f.endswith(".ply")])
smoothed_files = sorted([os.path.join(smoothed_centerline_dir, f) for f in os.listdir(smoothed_centerline_dir) if f.endswith("_smoothed.ply")])

# Function to compute Chamfer Distance
def chamfer_distance(A, B):
    """ Compute Chamfer Distance between two point clouds """
    dists_A_to_B = np.min(cdist(A, B), axis=1)  # Distance from A to nearest B
    dists_B_to_A = np.min(cdist(B, A), axis=1)  # Distance from B to nearest A
    return np.mean(dists_A_to_B) + np.mean(dists_B_to_A)

# Function to compute Hausdorff Distance
def hausdorff_distance(A, B):
    """ Compute Hausdorff Distance between two point clouds """
    dists_A_to_B = np.min(cdist(A, B), axis=1)
    dists_B_to_A = np.min(cdist(B, A), axis=1)
    return max(np.max(dists_A_to_B), np.max(dists_B_to_A))

# Process each centerline and compute distances
results = []
for raw_file, smoothed_file in zip(raw_files, smoothed_files):
    print(f"🔹 Processing: {raw_file} vs. {smoothed_file}")

    # Load raw and smoothed centerlines
    raw_pcd = o3d.io.read_point_cloud(raw_file)
    smoothed_pcd = o3d.io.read_point_cloud(smoothed_file)

    # Convert to NumPy arrays
    raw_points = np.asarray(raw_pcd.points)
    smoothed_points = np.asarray(smoothed_pcd.points)

    if len(raw_points) < 10 or len(smoothed_points) < 10:
        print(f"Skipping {raw_file}: Not enough points for evaluation.")
        continue

    # Compute Chamfer and Hausdorff Distances
    chamfer_dist = chamfer_distance(raw_points, smoothed_points)
    hausdorff_dist = hausdorff_distance(raw_points, smoothed_points)

    # Store results
    results.append((raw_file, chamfer_dist, hausdorff_dist))
    print(f" {raw_file}: Chamfer Distance = {chamfer_dist:.6f}, Hausdorff Distance = {hausdorff_dist:.6f}")

# Save results to a text file
output_path = os.path.join(smoothed_centerline_dir, "evaluation_results.txt")
with open(output_path, "w") as f:
    f.write("File Name, Chamfer Distance, Hausdorff Distance\n")
    for raw_file, chamfer_dist, hausdorff_dist in results:
        f.write(f"{os.path.basename(raw_file)}, {chamfer_dist:.6f}, {hausdorff_dist:.6f}\n")

print(f" Evaluation complete! Results saved to: {output_path}")
