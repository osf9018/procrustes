# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2022 The QC-Devs Community
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
"""Positive semidefinite Procrustes Module."""

import numpy as np
import scipy.linalg as lin
from scipy.optimize import minimize
from math import inf, sqrt
from typing import List

__all__ = ["woodgate"]


def permutation_matrix(arr: np.ndarray) -> np.ndarray:
    r"""
    Find required permutation matrix.

    Parameters
    ----------
    arr : np.ndarray
        The array :math:`A` such that :math:`v(A') = Pv(A)`.

    Returns
    -------
    np.ndarray
        The permutation matrix.
    """
    k = 0
    n = arr.shape[0]
    p = np.zeros((n**2, n**2))

    for i in range(n**2):
        if i % n == 0:
            j = k
            k += 1
            p[i, j] = 1
        else:
            j += n
            p[i, j] = 1
    return p


def no_update(error: List[int], threshold: int = 1e-5) -> bool:
    r"""
    Check if there has been any change in error,
    with new iteration.

    Parameters
    ----------
    error : List[int]
        The error list.

    threshold : int, optional
        The threshold below which we change isn't
        considered.

    Returns
    -------
    bool
        True if the error has reduced/changed.
    """
    return abs(error[-1] - error[-2]) < threshold


def is_pos_semi_def(arr: np.ndarray) -> bool:
    r"""
    Check if a matrix is positive semidefinite.

    Parameters
    ----------
    arr : np.ndarray
        The matrix to be checked.

    Returns
    -------
    bool
        True if the matrix is positive semidefinite.
    """
    return np.all(np.linalg.eigvals(arr) >= 0)


def make_positive(arr: np.ndarray) -> np.ndarray:
    r"""
    Re-construct a matrix by making all its negative eigenvalues zero.

    Parameters
    ----------
    arr : np.ndarray
        The matrix to be re-constructed.

    Returns
    -------
    np.ndarray
        The re-constructed matrix.
    """
    eigenvalues, unitary = np.linalg.eig(arr)
    eigenvalues_pos = [max(0, i) for i in eigenvalues]
    unitary_inv = np.linalg.inv(unitary)
    return unitary @ np.diag(eigenvalues_pos) @ unitary_inv


def find_gradient(e: np.ndarray, l: np.ndarray, g: np.ndarray) -> np.ndarray:
    r"""
    Find the gradient of the function f(E).

    Parameters
    ----------
    e : np.ndarray
        The input to the function f. This is E_i in the paper.

    l : np.ndarray
        A matrix to be used in the gradient computation.
        This is L(E_i) in the paper.

    g : np.ndarray
        This is the original G matrix obtained as input.

    Returns
    -------
    np.ndarray
        The required gradient. This is D_i in the paper.

    Notes
    -----
    The gradient is defined as :math:`D_i = \nabla_{E} f(E_i)` and it
    is constructed using two parts, namely, D1 and D2, which denote the
    top and bottom parts of the gradient matrix.

    Specifically, D1 denoyes the top :math:`s` rows of the gradient matrix,
    where, :math:`s` is the rank of the matrix :math:`E_i`. We, furthermore,
    define E1 as the first :math:`s` rows of E_i.

    .. math::
        D2 L(E_i) = 0
        (X + (I \otimes L(E_i))) v(D1) = (I \otimes L(E_i)) v(E1)

    In the following implementation, the variables d1 and d2 represent
    D1 and D2, respectively.

    References
    ----------
    Refer to equations (26) and (27) in [1] for exact deinitions of the
    several terms mentioned in this function.
    .. [1] Woodgate, K. G. (1996). "A new algorithm for positive
        semidefinite procrustes". Journal of the American Statistical
        Association, 93(453), 584-588.
    """
    n = e.shape[0]
    s = np.linalg.matrix_rank(e)
    v = lin.null_space(l.T).flatten()
    d2 = np.outer(v, v)

    p = permutation_matrix(e)
    identity_z = np.eye(
        (np.kron(e @ e.T, g @ g.T)).shape[0] // (e @ g @ g.T @ e.T).shape[0]
    )
    z = (
        np.kron(e @ g @ g.T, e.T) @ p
        + np.kron(e, g @ g.T @ e.T) @ p
        + np.kron(e @ g @ g.T @ e.T, identity_z)
        + np.kron(e @ e.T, g @ g.T)
    )

    x = z if s == n else z[: n * (n - s), : n * (n - s)]
    identity_x = np.eye(x.shape[0] // l.shape[0])
    flattened_d1 = (
        np.linalg.pinv(x + np.kron(identity_x, l))
        @ np.kron(identity_x, l)
        @ e[:s, :].flatten()
    )

    if s == n:
        d = flattened_d1.reshape(s, n)
    else:
        d = np.concatenate((flattened_d1, d2), axis=0)
    return d


def scale(e: np.ndarray, g: np.ndarray, q: np.ndarray) -> np.ndarray:
    r"""
    Find the scaling factor :math:`\hat{alpha}` and scale the
    matrix e.

    Parameters
    ----------
    e : np.ndarray
        This is the matrix to be scaled.

    g : np.ndarray
        This is the original G matrix obtained as input.

    q : np.ndarray
        This is the matrix Q in the paper.

    Returns
    -------
    np.ndarray
        The scaling factor. This is \hat{alpha} in the paper.
    """
    alpha = sqrt(
        max(1e-9, np.trace(e.T @ e @ q) / (2 * np.trace(e.T @ e @ e.T @ e @ g @ g.T)))
    )
    return alpha * e


def woodgate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""
    Woodgate's algorithm for positive semidefinite Procrustes.

    Parameters
    ----------
    a : np.ndarray
        The matrix to be transformed.
        This is relabelled to G as in the paper.
    b : np.ndarray
        The target matrix.
        This is relabellled to F as in the paper.

    Returns
    -------
    ProcrustesResult
        The result of the Procrustes transformation.

    Notes
    -----
    Given :math:`F, G \in R^{\ n\times m}`, the woodgate algorithm finds
    :math:`P \in S^{\ n\times n}_{\geq}` such that the following is true:

    .. math::
        \text{PSDP: } min_{P} \|F - PG\|

    Woodgate's algorithm takes a non-convex approach to the above problem.
    It finds solution to the following which serves as a subroutine to our
    original problem.

    .. math::
        \text{PSDP*: } min_{E \in R^{\ n\times n}} \|F - E'EG\|

    Now, since all local minimizers of PSDP* are also global minimizers, we
    have :math:`\hat{P} = \hat{E}'E` where :math:`\hat{E}` is any local
    minimizer of PSDP* and :math:`\hat{P}` is the required minimizer for
    our originak PSDP problem.

    The main algorithm is as follows:

    - :math:`E_0` is chosen randomly, :math:`i = 0`.
    - Compute :math:`L(E_i)`.
    - If :math:`L(E_i) \geq 0` then we stop and :math:`E_i` is the answer.
    - Compute :math:`D_i`.
    - Minimize :math:`f(E_i - w_i D_i)`.
    - :math:`E_{i + 1} = E_i - w_i_min D_i`
    - :math:`i = i + 1`, start from 2 again.


    References
    ----------
    .. [1] Woodgate, K. G. (1996). "A new algorithm for positive semidefinite
        procrustes". Journal of the American Statistical Association, 93(453),
        584-588.
    """

    # We define the matrices F, G and Q as in the paper.
    f = b
    g = a
    q = f @ g.T + g @ f.T

    # We define the functions L and f as in the paper.
    func_l = lambda arr: (arr.T @ arr @ g @ g.T) + (g @ g.T @ arr.T @ arr) - q
    func_f = lambda arr: (1 / 2) * (
        np.trace(f.T @ f)
        + np.trace(arr.T @ arr @ arr.T @ arr @ g @ g.T)
        - np.trace(arr.T @ arr @ q)
    )

    # Main part of the algorithm.
    i = 0
    n = f.shape[0]
    e = scale(e=np.eye(n), g=g, q=q)
    error = [inf]

    while True:
        l = func_l(e)
        error.append(np.linalg.norm(f - e.T @ e @ g))
        if is_pos_semi_def(l) or no_update(error):
            break

        l_pos = make_positive(l)
        d = find_gradient(e=e, l=l_pos, g=g)

        # Objective function which we want to minimize.
        func_obj = lambda w: func_f(e - w * d)
        w_min = minimize(func_obj, 1, bounds=((1e-9, None),)).x[0]
        e = scale(e=(e - w_min * d), g=g, q=q)
        i += 1

    # print(f"Woodgate's algorithm took {i} iterations.")
    # print(f"Error = {np.linalg.norm(f - e.T @ e @ g)}.")
    # print(f"Required P = {e.T @ e}")
    return e.T @ e, error[-1], i
