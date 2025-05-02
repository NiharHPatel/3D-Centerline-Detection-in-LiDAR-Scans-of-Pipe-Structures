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
    """ Use ICP to align centerlines from different frames. """
    threshold = 0.02  # Distance threshold for ICP
    trans_init = np.eye(4)  # Initial transformation

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return reg_p2p.transformation

# Load the first two centerlines
if len(centerline_files) < 2:
    print("Not enough centerline files for ICP registration.")
else:
    source_pcd = o3d.io.read_point_cloud(centerline_files[0])
    target_pcd = o3d.io.read_point_cloud(centerline_files[1])

    # Perform ICP registration
    transformation = align_centerlines_with_icp(source_pcd, target_pcd)

    # Apply transformation to align source with target
    source_pcd.transform(transformation)

    # Visualize aligned centerlines
    o3d.visualization.draw_geometries([source_pcd, target_pcd])
    print("Centerlines aligned successfully.")
