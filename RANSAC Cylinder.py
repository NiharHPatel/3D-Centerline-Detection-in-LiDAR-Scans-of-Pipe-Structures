import open3d as o3d
import numpy as np
import os
from sklearn.cluster import DBSCAN

# List of PLY files
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

# 1️⃣ **Apply Statistical Outlier Removal (SOR)**
def apply_sor(pcd, nb_neighbors=30, std_ratio=1.5):
    """Removes scattered noise points using Statistical Outlier Removal (SOR)."""
    print("Applying Statistical Outlier Removal (SOR)...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

# 2️⃣ **Apply RANSAC Filtering to Remove Geometric Structures**
def apply_ransac_filtering(pcd, distance_threshold=0.02, num_iterations=2000):
    """Removes dominant geometric structures (e.g., planes) using RANSAC filtering."""
    print("Applying RANSAC Filtering to remove dominant geometric structures...")
    
    # Apply RANSAC to detect and remove planes (ground, walls, etc.)
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,ransac_n=3, num_iterations=num_iterations)
    
    # Keep only non-plane points, assuming the pipeline remains
    filtered_pcd = pcd.select_by_index(inliers, invert=True)
    
    return filtered_pcd

# 3️⃣ **Apply DBSCAN Clustering to Refine the Pipeline Structure**
def apply_dbscan(pcd, eps=0.1, min_samples=5):
    """Clusters the pipeline using DBSCAN and removes irrelevant points."""
    points = np.asarray(pcd.points)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    if len(set(labels)) <= 1:
        print("⚠ No significant clusters found.")
        return pcd  # Return original if clustering fails

    # Keep only the largest cluster (assumed to be the pipeline)
    largest_cluster = max(set(labels), key=list(labels).count)
    pipeline_points = points[labels == largest_cluster]

    # Convert back to Open3D point cloud
    pipeline_pcd = o3d.geometry.PointCloud()
    pipeline_pcd.points = o3d.utility.Vector3dVector(pipeline_points)
    return pipeline_pcd

# 4️⃣ **Processing Each File**
for file_path in ply_files:
    try:
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found -> {file_path}")
            continue

        print(f"📌 Processing file: {file_path}")

        # Load point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        
        if not pcd.has_points():
            print(f"⚠ Warning: {file_path} contains no valid points.")
            continue

        # Step 1: **Apply SOR** (No downsampling)
        pcd_filtered = apply_sor(pcd)

        # Step 2: **Apply RANSAC Filtering to remove geometric structures**
        pcd_ransac = apply_ransac_filtering(pcd_filtered)

        # Step 3: **Apply DBSCAN Clustering**
        pcd_pipeline = apply_dbscan(pcd_ransac)

        if not pcd_pipeline.has_points():
            print(f"⚠ No pipeline-like structure found in {file_path}.")
            continue

        # **Visualize the segmented pipeline**
        pcd_pipeline.paint_uniform_color([0.7, 0.7, 0.7])  # Grey for pipeline
        o3d.visualization.draw_geometries(
            [pcd_pipeline],
            window_name=f"RANSAC Filtering + DBSCAN Pipeline Extraction - {file_path}"
        )

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
