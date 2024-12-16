#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fit Plane in pointcloud

Author: FILL IN
MatrNr: FILL IN
"""

from typing import Tuple

import copy

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
    # Write your own code here
    points = np.asarray(pcd.points)
    n_points = points.shape[0]
    best_plane = np.array([0., 0., 1., 0.])
    best_inliers = np.full(points.shape[0], False)
    outlier_ratio = 0.3
    counter = 0
    p_inliers = 1.0 - outlier_ratio
    p_triplet = p_inliers ** 3
    if p_triplet < 1e-9:
        n_iter = 2000
    else:
        n_iter = math.log (1 - confidence) / math.log (1 - p_triplet)
        n_iter = int (np.clip (n_iter, 500, 5000))

    for _ in range(n_iter):
        idx = np.random.choice(points.shape[0], 3, replace = False)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
        normal = np.cross(p2-p1, p3-p1)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue
        normal /= norm_len
        a,b,c = normal
        d = -np.dot(normal, p1)
        plane_eq = np.array([a, b, c, d])
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        inliers = distances < inlier_threshold
        if np.sum(inliers) > counter:
            counter = np.sum(inliers)
            best_inliers = inliers.copy()
            best_plane = plane_eq.copy()
    if counter > 3:  # Need at least 3 points
        inlier_points = points[best_inliers]  # shape = (#inliers, 3)
        A = np.hstack ([inlier_points, np.ones ((inlier_points.shape[0], 1))])  # (n,4)
        b = np.zeros ((inlier_points.shape[0],))  # (n,)

        # Solve A * [a,b,c,d] = 0 in least-squares sense
        plane_params, residuals, rank, s = np.linalg.lstsq (A, b, rcond=None)
        a, b, c, d = plane_params

        # If c<0 => flip sign so plane normal has positive z component
        if c < 0:
            plane_params = -plane_params
            a, b, c, d = plane_params

        best_plane = plane_params
        # Recompute inliers after refinement
        distances = np.abs (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        best_inliers = distances < inlier_threshold
        counter = np.sum (best_inliers)

        # Return final plane & bool inlier mask
    best_plane = best_plane.astype (np.float64)
    return best_plane, best_inliers
