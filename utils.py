import time
from fit_plane import *
from pathlib import Path
from helper_functions import *
from clustering import *


def run_clustering_benchmark(pcd, voxel_sizes=[0.01, 0.03]):
    results = {}

    for voxel_size in voxel_sizes:
        # Downsample
        pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Remove ground plane
        plane_model, best_inliers = fit_plane(pcd_sampled, confidence=0.85, inlier_threshold=0.05)
        scene_pcd = pcd_sampled.select_by_index(np.nonzero (best_inliers)[0], invert=True)
        points = np.asarray(scene_pcd.points, dtype=np.float32)

        n_points = len(points)
        print(f"\nPoints after downsampling (voxel_size={voxel_size}): {n_points}")

        # Test K-means
        start_time = time.time()
        centers, labels = kmeans(points, n_clusters=6, n_iterations=25, max_singlerun_iterations=100)
        kmeans_time = time.time() - start_time
        print(f"K-means time: {kmeans_time:.3f} seconds")

        # Test DBSCAN
        start_time = time.time()
        labels = dbscan(points, eps=0.05, min_samples=15)
        dbscan_time = time.time() - start_time
        print(f"DBSCAN time: {dbscan_time:.3f} seconds")


        results[voxel_size] = {
            'n_points': n_points,
            'kmeans_time': kmeans_time,
            'dbscan_time': dbscan_time,
        }

    return results


# In main:
if __name__ == '__main__':
    # Test on two different point clouds
    for idx in [0, 1]:
        print(f"\nTesting point cloud {idx}")
        current_path = Path(__file__).parent
        pcd = o3d.io.read_point_cloud(str (current_path.joinpath (f"pointclouds/image00{idx}.pcd")))

        # Remove zero points
        points_numpy = np.array(pcd.points)
        indices_origin_points = list(np.where(np.all(points_numpy == [0, 0, 0], axis=1))[0])
        pcd = pcd.select_by_index(indices_origin_points, invert=True)

        # Run benchmark
        results = run_clustering_benchmark(pcd)

        # Plot results
        voxel_sizes = list(results.keys())
        n_points = [results[vs]['n_points'] for vs in voxel_sizes]
        kmeans_times = [results[vs]['kmeans_time'] for vs in voxel_sizes]
        dbscan_times = [results[vs]['dbscan_time'] for vs in voxel_sizes]

        plt.figure(figsize=(10, 6))
        plt.plot(n_points, kmeans_times, 'o-', label='K-means')
        plt.plot(n_points, dbscan_times, 's-', label='DBSCAN')
        plt.xlabel('Number of points')
        plt.ylabel('Time (seconds)')
        plt.title(f'Clustering Runtime Analysis - Point Cloud {idx}')
        plt.legend()
        plt.grid(True)
        plt.show()