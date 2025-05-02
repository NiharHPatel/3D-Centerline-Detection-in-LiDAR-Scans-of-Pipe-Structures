import open3d as o3d
import numpy as np
import os
from sklearn.cluster import DBSCAN

# List of PLY files
ply_files = [
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/2_0002.500.ply"
]

def preprocess_point_cloud(pcd, voxel_size=0.01, nb_neighbors=30, std_ratio=1.5):
    """Preprocess the point cloud: Downsampling + Statistical Outlier Removal."""
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_down.select_by_index(ind)

def extract_pipeline_pca_dbscan(pcd, eps=0.1, min_samples=5):
    """Extracts pipeline points using PCA for orientation and DBSCAN for clustering."""
    points = np.asarray(pcd.points)

    # Step 1: Compute PCA to find the main axis
    mean = np.mean(points, axis=0)
    cov_matrix = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Select the primary axis (largest eigenvalue)
    primary_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Step 2: Project points onto the primary axis
    projected_points = np.dot(points - mean, primary_axis[:, None]) * primary_axis + mean

    # Step 3: Apply DBSCAN to keep pipeline points (increase `eps` for larger structures)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(projected_points)
    labels = clustering.labels_

    # Step 4: Keep largest cluster + original non-outlier points
    largest_cluster = max(set(labels), key=list(labels).count)
    pipeline_points = points[labels == largest_cluster]  # Use original points, not just projection

    # Convert back to Open3D point cloud
    pipeline_pcd = o3d.geometry.PointCloud()
    pipeline_pcd.points = o3d.utility.Vector3dVector(pipeline_points)
    return pipeline_pcd
# Processing each file
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

        # Step 1: Preprocess the point cloud
        pcd_filtered = preprocess_point_cloud(pcd)

        # Step 2: Extract the pipeline structure
        pcd_pipeline = extract_pipeline_pca_dbscan(pcd_filtered)

        if not pcd_pipeline.has_points():
            print(f"⚠ No pipeline-like structure found in {file_path}.")
            continue

        # Visualize the segmented pipeline
        pcd_pipeline.paint_uniform_color([0.7, 0.7, 0.7])  # Grey for pipeline
        o3d.visualization.draw_geometries(
            [pcd_pipeline],
            window_name=f"PCA + DBSCAN Pipeline Extraction - {file_path}"
        )

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")


