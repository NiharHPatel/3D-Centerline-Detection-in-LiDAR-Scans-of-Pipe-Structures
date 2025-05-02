import open3d as o3d

# 🔁 List your 9 .ply file paths here
ply_files = [
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Centerlines 2/Cent_1_0058.000.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Centerlines 2/Cent_2_0002.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Centerlines 2/Cent_3_0000.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Centerlines 2/Cent_4_0005.000.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Centerlines 2/Cent_5_0015.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Centerlines 2/Cent_5_0029.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Centerlines 2/Cent_6_0006.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Centerlines 2/Cent_8_0007.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Centerlines 2/Cent_10_0000.000.ply"
]

# Loop through and visualize each file
for i, file_path in enumerate(ply_files):
    print(f"\n📂 Loading file {i+1}/{len(ply_files)}: {file_path}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Print summary
    print(pcd)
    print("📌 Number of points:", len(pcd.points))
    
    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name=f"Frame {i+1}")
