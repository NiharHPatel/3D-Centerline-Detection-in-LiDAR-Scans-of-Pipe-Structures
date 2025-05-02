import open3d as o3d
import numpy as np
import os

# List of centerline PLY files (Replace with actual paths)
centerline_files = [
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

def align_centerlines_with_icp(source, target):
    """Use ICP to align centerlines from different frames."""
    threshold = 0.02  # Distance threshold for ICP
    trans_init = np.eye(4)  # Initial transformation

    # Perform ICP Registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    print(f"ICP Fitness Score: {reg_p2p.fitness}")  # Closer to 1 = better alignment
    return reg_p2p.transformation

# Load and Validate Centerline Files
aligned_centerlines = []

if len(centerline_files) < 2:
    print("❌ Not enough centerline files for ICP registration.")
else:
    # Load the first file as the reference frame
    reference_pcd = o3d.io.read_point_cloud(centerline_files[0])

    if len(reference_pcd.points) == 0:
        print(f"❌ Skipping empty file: {centerline_files[0]}")
    else:
        aligned_centerlines.append(reference_pcd)

        # Align all subsequent files to the first one
        for i in range(1, len(centerline_files)):
            file_path = centerline_files[i]

            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue

            target_pcd = o3d.io.read_point_cloud(file_path)

            if len(target_pcd.points) == 0:
                print(f"❌ Skipping empty file: {file_path}")
                continue

            # Perform ICP alignment
            print(f"🔹 Aligning {file_path} to reference...")
            transformation = align_centerlines_with_icp(target_pcd, reference_pcd)

            # Apply transformation
            target_pcd.transform(transformation)

            # Visualize each file separately
            print(f"🔹 Displaying alignment for: {file_path}")
            o3d.visualization.draw_geometries([reference_pcd, target_pcd])

            # Store aligned point clouds separately
            aligned_centerlines.append(target_pcd)

        # **Final Visualization**: All aligned centerlines separately
        print("✅ Final Visualization: All Aligned Centerlines")
        for pcd in aligned_centerlines:
            o3d.visualization.draw_geometries([pcd])
