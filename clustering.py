#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find clusters of pointcloud

Author: FILL IN
MatrNr: FILL IN
"""

from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import distance
from scipy.stats import anderson
import matplotlib.pyplot as plt
from helper_functions import plot_clustering_results, silhouette_score


def kmeans(points: np.ndarray,
           n_clusters: int,
           n_iterations: int,
           max_singlerun_iterations: int,
           centers_in: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the k-means algorithm.

    :param points: The (down-sampled) points of the pointcloud to be clustered.
    :type points: np.ndarray with shape=(n_points, 3)

    :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
    :type n_clusters: int

    :param n_iterations: Number of time the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :param centers_in: Start centers of the k-means algorithm.  If centers_in = None, the centers are randomly sampled
        from input data for each iteration.
    :type centers_in: np.ndarray with shape = (n_clusters, 3) or None

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    # Write your own code here
    n_points = len(points)
    best_inertia = float('inf')
    best_centers = None
    best_labels = None

    for _ in range(n_iterations):
        if centers_in is not None:
            centers = centers_in.copy().astype(np.float32)
        else:
            random_indices = np.random.choice(n_points, n_clusters, replace=False)
            centers = points[random_indices].astype(np.float32)  # (k,3)

        labels = np.zeros(n_points, dtype=int)
        for _iter in range(max_singlerun_iterations):
            # Computing distance from each point to each center -> shape (n_points, n_clusters)
            dist_matrix = distance.cdist(points, centers, metric='euclidean')
            new_labels = np.argmin(dist_matrix, axis=1)

            # If labels don't change, we are converged
            if np.all(new_labels == labels):
                break
            labels = new_labels

            for k in range(n_clusters):
                cluster_points = points[labels == k]
                if len(cluster_points) > 0:
                    centers[k] = np.mean(cluster_points, axis=0)
                else:
                    random_idx = np.random.randint(n_points)
                    centers[k] = points[random_idx]

        # Compute inertia
        dist_matrix = distance.cdist(points, centers, metric='euclidean')
        assigned_distances = dist_matrix[np.arange(n_points), labels]
        inertia = np.sum(assigned_distances ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()

    return best_centers, best_labels


def iterative_kmeans(points: np.ndarray,
                     max_n_clusters: int,
                     n_iterations: int,
                     max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Applies the k-means algorithm multiple times and returns the best result in terms of silhouette score.

    This algorithm runs the k-means algorithm for all number of clusters until max_n_clusters. The silhouette score is
    calculated for each solution. The clusters with the highest silhouette score are returned.

    :param points: The (down-sampled) points of the pointcloud that should be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_n_clusters: The maximum number of clusters that is tested.
    :type max_n_clusters: int

    :param n_iterations: Number of time each k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of each k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    # Write your own code here
    best_sil_score = -1.0
    best_centers = None
    best_labels = None

    for k in range(2, max_n_clusters + 1):
        centers_k, labels_k = kmeans(points,
                                     n_clusters=k,
                                     n_iterations=n_iterations,
                                     max_singlerun_iterations=max_singlerun_iterations)

        sil_k = silhouette_score(points, centers_k, labels_k)
        if sil_k > best_sil_score:
            best_sil_score = sil_k
            best_centers = centers_k
            best_labels = labels_k

    return best_centers, best_labels


def gmeans(points: np.ndarray,
           tolerance: float,
           max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the g-means algorithm.

    The algorithm was proposed by Hamerly, Greg, and Charles Elkan. "Learning the k in k-means." Advances in neural
    information processing systems 16 (2003).

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    # Write your own code here
    n_points = len(points)
    centers = np.mean(points, axis=0, keepdims=True)  # shape=(1,3)

    while True:
        global_labels = np.zeros(n_points, dtype=int)
        for _ in range(max_singlerun_iterations):
            # searching nearest center for every point
            dist_matrix = distance.cdist(points, centers, 'euclidean')
            new_labels = np.argmin(dist_matrix, axis=1)

            if np.all(new_labels == global_labels):
                break
            global_labels = new_labels

            # recalculating centers
            for c_idx in range(len(centers)):
                cluster_points = points[global_labels == c_idx]
                if len(cluster_points) > 0:
                    centers[c_idx] = np.mean(cluster_points, axis=0)

        cluster_split = False
        new_centers_list = []

        for c_idx in range(len(centers)):
            cluster_points = points[global_labels == c_idx]
            if len(cluster_points) <= 1:
                new_centers_list.append(centers[c_idx])
                continue

            cov_mat = np.cov(cluster_points.T)
            eigvals, eigvecs = np.linalg.eig(cov_mat)
            max_eig_idx = np.argmax(eigvals)
            lamda = eigvals[max_eig_idx]
            s = eigvecs[:, max_eig_idx]
            s /= np.linalg.norm(s)

            c_i = centers[c_idx]
            c_i1 = c_i + s * (2 * lamda / np.pi)
            c_i2 = c_i - s * (2 * lamda / np.pi)

            child_centers = np.stack([c_i1, c_i2], axis=0).astype(np.float32)
            child_labels = np.zeros(len(cluster_points), dtype=int)

            for _local_iter in range(max_singlerun_iterations):
                dist_local = distance.cdist(cluster_points, child_centers, 'euclidean')
                new_child_labels = np.argmin(dist_local, axis=1)
                if np.all(new_child_labels == child_labels):
                    break
                child_labels = new_child_labels

                for kidx in range(2):
                    subset = cluster_points[child_labels == kidx]
                    if len(subset) > 0:
                        child_centers[kidx] = np.mean(subset, axis=0)

            c_i1, c_i2 = child_centers[0], child_centers[1]

            v = c_i1 - c_i2
            v_len_sq = np.dot(v, v)
            if v_len_sq < 1e-12:
                # degenerating direction => treating as no split
                new_centers_list.append(centers[c_idx])
                continue
            x_projected = np.dot(cluster_points - c_i2, v) / v_len_sq

            ad_stat, critical_values, _ = anderson(x_projected, dist='norm')
            # checking if data is gauus
            if ad_stat <= critical_values[-1] * tolerance:
                # keeping cluster
                new_centers_list.append(centers[c_idx])
            else:
                # spliting, cause not gaussigan
                new_centers_list.append(c_i1)
                new_centers_list.append(c_i2)
                cluster_split = True

        new_centers = np.stack(new_centers_list, axis=0)

        if not cluster_split:
            # exit if not more splits left
            centers = new_centers
            break
        else:
            centers = new_centers

    final_labels = np.zeros(n_points, dtype=int)
    for _ in range(max_singlerun_iterations):
        dist_matrix = distance.cdist(points, centers, 'euclidean')
        new_labels = np.argmin(dist_matrix, axis=1)
        if np.all(new_labels == final_labels):
            break
        final_labels = new_labels
        for c_idx in range(len(centers)):
            subset = points[final_labels == c_idx]
            if len(subset) > 0:
                centers[c_idx] = np.mean(subset, axis=0)

    return centers.astype(np.float32), final_labels


def dbscan(points: np.ndarray,
           eps: float = 0.05,
           min_samples: int = 10) -> np.ndarray:
    """ Find clusters in the provided data coming from a pointcloud using the DBSCAN algorithm.

    The algorithm was proposed in Ester, Martin, et al. "A density-based algorithm for discovering clusters in large
    spatial databases with noise." kdd. Vol. 96. No. 34. 1996.

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :type eps: float

    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core
        point. This includes the point itself.
    :type min_samples: float

    :return: Labels array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label -1 is assigned to points that are considered to be noise.
    :rtype: np.ndarray
    """
    ######################################################
    # Write your own code here
    n_points = len(points)
    labels = np.full(n_points, -2, dtype=int)  # -2 = unvisited
    cluster_id = 0

    for i in range(n_points):
        # If visited (either assigned to cluster or noise) - skip
        if labels[i] != -2:
            continue

        dist_i = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.where(dist_i <= eps)[0]

        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            cluster_id += 1
            labels[i] = cluster_id

            queue = list(neighbors)
            idx_ptr = 0
            while idx_ptr < len(queue):
                neighbor_idx = queue[idx_ptr]
                idx_ptr += 1

                # If neighbor is unvisited
                if labels[neighbor_idx] == -2:
                    labels[neighbor_idx] = cluster_id
                    dist_neighbor = np.linalg.norm(points - points[neighbor_idx], axis=1)
                    neighbors_of_neighbor = np.where(dist_neighbor <= eps)[0]

                    # If neighbor is also a core point, addint its neighbors to the queue
                    if len(neighbors_of_neighbor) >= min_samples:
                        queue.extend(neighbors_of_neighbor)

                elif labels[neighbor_idx] == -1:
                    # noise to border escalation
                    labels[neighbor_idx] = cluster_id

    # Replacing leftovers -2 (never visited) with -1 if any remain
    labels[labels == -2] = -1
    return labels