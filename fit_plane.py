#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fit Plane in pointcloud

Author: FILL IN
MatrNr: FILL IN
"""

from typing import Tuple


import numpy as np
import open3d as o3d
import math


def fit_plane(pcd: o3d.geometry.PointCloud,
              confidence: float,
              inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Find dominant plane in pointcloud with sample consensus.

    Detect a plane in the input pointcloud using sample consensus. The number of iterations is chosen adaptively.
    The concrete error function is given as an parameter.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from the plane to be considered an inlier (in meters)
    :type inlier_threshold: float

    :return: (best_plane, best_inliers)
        best_plane: Array with the coefficients of the plane equation ax+by+cz+d=0 (shape = (4,))
        best_inliers: Boolean array with the same length as pcd.points. Is True if the point at the index is an inlier
    :rtype: tuple (np.ndarray[a,b,c,d], np.ndarray)
    """
    ######################################################
    # Convert point cloud to NumPy array
    points = np.asarray(pcd.points)
    n_points = points.shape[0]

    best_plane = np.array([0., 0., 1., 0.])
    best_inliers = np.zeros(n_points, dtype=bool)
    max_inliers_count = 0

    outlier_ratio = 0.3
    p_inliers = 1.0 - outlier_ratio
    p_triplet = max(p_inliers ** 3, 1e-9)  # Prevent log(0)
    n_iter = int(np.clip (math.log (1 - confidence) / math.log (1 - p_triplet), 500, 5000))

    for _ in range(n_iter):
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]

        normal = np.cross(p2 - p1, p3 - p1)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue
        normal /= norm_len

        a, b, c = normal
        d = -np.dot(normal, p1)
        distances = np.abs(np.dot(points, normal) + d)


        inliers = distances < inlier_threshold
        inliers_count = np.sum(inliers)

        if inliers_count > max_inliers_count:
            max_inliers_count = inliers_count
            best_inliers = inliers.copy()
            best_plane = np.array([a, b, c, d])


    # This section refines the plane parameters, which have been obtained by the RANSEAC. It founds the most accurate plane.
    if max_inliers_count >= 3:
        inlier_points = points[best_inliers]
        A = np.hstack([inlier_points, np.ones((inlier_points.shape[0], 1))])
        b = np.zeros(inlier_points.shape[0])  # Zero vector for Ax + d = 0
        refined_plane, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        normal = refined_plane[:3]
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-9:
            refined_plane[:3] /= norm_len
            if refined_plane[2] < 0:
                refined_plane = -refined_plane
            best_plane = refined_plane
            distances = np.abs(np.dot(points, best_plane[:3]) + best_plane[3])
            best_inliers = distances < inlier_threshold

    return best_plane, best_inliers
