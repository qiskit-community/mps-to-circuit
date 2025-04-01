# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from scipy.sparse.linalg import eigs


def _right_env(u: np.ndarray, d: int, d_left: int) -> np.ndarray:
    """
    Calculates the right environment of a given translationally invariant MPS.
    Assumes u indices are (zero * vR, p * vL).

    Computes the right environment using the dominant eigenvalue of the transfer matrix.

    :param u: A unitary representing the translationally invariant MPS site.
    :param d: The physical dimension of the MPS site.
    :param d_left: The bond dimension of the MPS site.

    :return: A matrix representing the normalized right environment.
    """
    transfer_matrix = _construct_transfer_matrix(u, d, d_left)

    _, right_env = eigs(transfer_matrix, k=1, which="LM")
    right_env = right_env.reshape(d_left, d_left)

    norm = np.trace(right_env)
    return right_env / norm


def _construct_transfer_matrix(u: np.ndarray, d: int, d_left: int) -> np.ndarray:
    """
    Constructs the transfer tensor for a given translationally invariant MPS site.

    The transfer tensor encodes the transfer of quantum information through the MPS chain.

    :param u: A unitary representing the translationally invariant MPS site.
    :param d: The physical dimension of the MPS site.
    :param d_left: The bond dimension of the MPS site.

    :return: The constructed transfer tensor.
    """
    zero = np.zeros(d)
    zero[0] = 1

    u = u.reshape([d, d_left, d, d_left])
    transfer = np.einsum("i,ijkl,mnko,m->lojn", zero, u, np.conj(u), zero)
    return transfer.reshape(d_left * d_left, d_left * d_left)


def _env_unitary(right_env: np.ndarray) -> np.ndarray:
    """
    Convert full right environment R into single unitary form V.

    :param right_env: A 2D NumPy array representing the right environment.

    :return: A unitary matrix representing the single environment tensor V.
    """
    u, s, _ = np.linalg.svd(right_env, hermitian=True)
    s_sqrt = np.sqrt(np.diag(s))
    v = u @ s_sqrt
    return _vector_to_unitary(v.reshape(v.shape[0] * v.shape[1], 1))


def _env_unitary_cholesky(right_env: np.ndarray) -> np.ndarray:
    """
    Convert full right environment R into single unitary form V using the Cholesky decomposition.

    :param right_env: A 2D NumPy array representing the right environment.

    :return: A unitary matrix representing the single environment tensor V.
    """
    lower = np.linalg.cholesky(right_env)
    return _vector_to_unitary(lower.reshape(lower.shape[0] * lower.shape[0], 1))


def _vector_to_unitary(vector: np.ndarray) -> np.ndarray:
    """
    :param vector: Must be 2D array with 1 column.

    :return: A unitary matrix representing the input vector.
    """
    u, _, _ = np.linalg.svd(vector)
    assert np.allclose(vector[:, 0], u[:, 0])
    return u
