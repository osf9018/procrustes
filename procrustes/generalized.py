# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2024 The QC-Devs Community
#
# This file is part of Procrustes.
#
# Procrustes is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Procrustes is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Generalized Procrustes Module."""

from typing import List, Optional, Tuple

import numpy as np
import logging

from procrustes import orthogonal
from procrustes.utils import _check_arraytypes

__all__ = [
    "generalized",
]


def generalized(
    array_list: List[np.ndarray],
    ref: Optional[np.ndarray] = None,
    tol: float = 1.0e-7,
    n_iter: int = 200,
    min_iter: int = 0,
    check_finite: bool = True,
    lapack_driver: str = "gesvd",
    translate=False,
    scale=False
) -> Tuple[List[np.ndarray], float]:
    r"""Generalized Procrustes Analysis.

    Parameters
    ----------
    array_list : List
        The list of 2D-array which is going to be transformed.
    ref : ndarray, optional
        The reference array to initialize the first iteration. If None, the first array in
        `array_list` will be used.
    tol: float, optional
        Tolerance value to stop the iterations.
    n_iter: int, optional
        Number of total iterations.
    min_iter: minimal number of iterations.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.
    lapack_driver : {'gesvd', 'gesdd'}, optional
        Whether to use the more efficient divide-and-conquer approach ('gesdd') or the more robust
        general rectangular approach ('gesvd') to compute the singular-value decomposition with
        `scipy.linalg.svd`.

    Returns
    -------
    array_aligned : List
        A list of transformed arrays with generalized Procrustes analysis.
    new_distance_gpa: float
        The distance for matching all the transformed arrays with generalized Procrustes analysis.

    Notes
    -----
    Given a set of matrices, :math:`\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k` with
    :math:`k > 2`,  the objective is to minimize in order to superimpose pairs of matrices.

    .. math::
        \min \quad = \sum_{i<j}^{j} {\left\| \mathbf{A}_i \mathbf{T}_i  -
         \mathbf{A}_j \mathbf{T}_j \right\| }^2

    This function implements the Equation (20) and the corresponding algorithm in  Gower's paper.

    """
    # check input arrays
    _check_arraytypes(*array_list)
    # check finite
    if check_finite:
        array_list = [np.asarray_chkfinite(arr) for arr in array_list]

    # todo: translation and scaling
    if n_iter <= 0:
        raise ValueError("Number of iterations should be a positive number.")
    if ref is None:
        # the first array will be used to build the initial ref
        array_aligned = [array_list[0]] + [
            _orthogonal(arr, array_list[0], lapack_driver, translate, scale) for arr in array_list[1:]
        ]
        ref = np.mean(array_aligned, axis=0)
    else:
        array_aligned = [None] * len(array_list)
        ref = ref.copy()

    distance_gpa = np.inf
    for i in np.arange(n_iter):
        logging.info(f'-- distance_gpa: {diff_dist:.16f}')
        # align to ref
        array_aligned = [_orthogonal(arr, ref, lapack_driver, translate, scale) for arr in array_list]
        # the mean
        new_ref = np.mean(array_aligned, axis=0)
        # todo: double check if the error is defined in the right way
        # the error
        new_distance_gpa = np.square(ref - new_ref).sum()
        diff_dist = np.abs(new_distance_gpa - distance_gpa)
        logging.info(f'-- distance to mean: {diff_dist:.16f}')
        if distance_gpa != np.inf and diff_dist < tol and i >= (min_iter - 1):
            break
        distance_gpa = new_distance_gpa
    return array_aligned, new_distance_gpa


def _orthogonal(
        arr_a: np.ndarray,
        arr_b: np.ndarray,
        lapack_driver: str = "gesvd",
        translate=False,
        scale=False) -> np.ndarray:
    """Orthogonal Procrustes transformation and returns the transformed array."""
    res = orthogonal(
        arr_a, arr_b, translate=translate, scale=scale, unpad_col=False, unpad_row=False, lapack_driver=lapack_driver)
    return np.dot(res["new_a"], res["t"])
