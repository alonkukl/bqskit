"""This module implements many helper functions to check types."""
from __future__ import annotations

import logging
import numbers
from collections.abc import Sequence
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sized

import numpy as np
from typing_extensions import TypeGuard


_logger = logging.getLogger(__name__)


def is_iterable(x: Any) -> TypeGuard[Iterable[Any]]:
    """Returns true if x is an iterable object."""
    try:
        iterator = iter(x)  # noqa: F841
        return True
    except TypeError:
        return False


def is_sized(x: Any) -> TypeGuard[Sized]:
    """Returns true if x is a sized object."""
    try:
        length = len(x)  # noqa: F841
        return True
    except TypeError:
        return False


def is_sequence(x: Any) -> TypeGuard[Sequence[Any]]:
    """Returns true if x is a sequence."""
    return isinstance(x, (Sequence, np.ndarray))


def is_mapping(x: Any) -> TypeGuard[Mapping[Any, Any]]:
    """Returns true if x is a mapping."""
    return isinstance(x, Mapping)


def is_numeric(x: Any) -> TypeGuard[numbers.Number]:
    """Return true if x is numeric."""
    return isinstance(x, numbers.Number) and not isinstance(x, bool)


def is_complex(x: Any) -> TypeGuard[complex]:
    """Return true if x is complex."""
    return isinstance(x, numbers.Complex)


def is_real_number(x: Any) -> TypeGuard[float]:
    """Return true if `x` is a real number."""
    return isinstance(x, numbers.Real)


def is_integer(x: Any) -> TypeGuard[int]:
    """Return true if x is an integer."""
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def is_bool(x: Any) -> TypeGuard[bool]:
    """Return true if x is an integer."""
    return isinstance(x, (bool, np.bool_))


def is_sequence_of_int(x: Any) -> TypeGuard[Sequence[int]]:
    """Return true if x is a sequence of integers."""
    return is_sequence(x) and all([is_integer(xi) for xi in x])


def is_valid_radixes(
    radixes: Sequence[int],
    num_qudits: int | None = None,
) -> bool:
    """
    Determines if the sequence of radixes are valid. Radixes must be integers
    greater than or equal to 2. If num_qudits is specified, then the length of
    radixes must be equal to num_qudits.

    Args:
        radixes (Sequence[int]): The radixes to check.

        num_qudits (int | None): The total number of qudits.
            Length of `radixes` should be equal to this. If None,
            don't check.

    Returns:
        (bool): True if the radixes are valid.
    """

    if not is_sequence(radixes):
        return False

    if not all([is_integer(qudit) for qudit in radixes]):
        fail_idx = [is_integer(qudit) for qudit in radixes].index(False)
        _logger.debug(
            'Radixes is not a tuple of ints, got: %s.' % type(
                radixes[fail_idx],
            ),
        )
        return False

    if not all([radix >= 2 for radix in radixes]):
        _logger.debug('Radixes invalid; radix indices must be >= 2.')
        return False

    if num_qudits is not None and len(radixes) != num_qudits:
        _logger.debug('Invalid number of radixes.')
        return False

    return True


def is_valid_coupling_graph(
    coupling_graph: Any,
    num_qudits: int | None = None,
) -> bool:
    """
    Checks if the coupling graph is valid.

    Args:
        coupling_graph (Any): The coupling graph to check.

        num_qudits (int | None): The total number of qudits. All qudits
            should be less than this. If None, don't check.

    Returns:
        (bool): Valid or not
    """

    if not is_iterable(coupling_graph):
        _logger.debug('Coupling graph is not iterable.')
        return False

    if len(list(coupling_graph)) == 0:
        return True

    if not all(isinstance(pair, tuple) for pair in coupling_graph):
        _logger.debug('Coupling graph is not a sequence of tuples.')
        return False

    if not all([len(pair) == 2 for pair in coupling_graph]) and not \
            all(len(triple) == 3 for triple in coupling_graph):
        _logger.debug('Coupling graph is not a sequence of pairs or triples.')
        return False

    if num_qudits is not None:
        if not (is_integer(num_qudits) and num_qudits > 0):
            _logger.debug('Invalid num_qudits in coupling graph check.')
            return False

        if not all(
            qudit < num_qudits
            for pair in coupling_graph
            for qudit in [pair[0], pair[1]]
        ):
            _logger.debug('Coupling graph has invalid qudits.')
            return False

    if not all([
        len(pair) == len(set(pair))
        for pair in coupling_graph
    ]):
        _logger.debug('Coupling graph has an invalid pair.')
        return False

    return True


def is_vector(V: np.ndarray) -> bool:
    """Return true if V is a vector."""

    if not isinstance(V, np.ndarray):
        _logger.debug('V is not an numpy array.')
        return False

    if len(V.shape) != 1:
        _logger.debug('V is not an 1-dimensional array.')
        return False

    if V.dtype.kind not in 'biufc':
        _logger.debug('V is not a numeric array.')
        return False

    return True


def is_matrix(M: np.ndarray) -> bool:
    """Checks if M is a matrix."""

    if not isinstance(M, np.ndarray):
        _logger.debug('M is not an numpy array.')
        return False

    if len(M.shape) != 2:
        _logger.debug('M is not an 2-dimensional array.')
        return False

    if M.dtype.kind not in 'biufc':
        _logger.debug('M is not a numeric array.')
        return False

    return True


def is_square_matrix(M: np.ndarray) -> bool:
    """Checks if M is a square matrix."""

    if not is_matrix(M):
        return False

    if M.shape[0] != M.shape[1]:
        return False

    return True


def is_unitary(U: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks if U is a unitary matrix."""

    if not is_square_matrix(U):
        return False

    X = U @ U.conj().T
    Y = U.conj().T @ U
    I = np.identity(X.shape[0])

    if not np.allclose(X, I, rtol=0, atol=tol):
        if _logger.isEnabledFor(logging.DEBUG):
            norm = np.linalg.norm(X - I)
            _logger.debug('Failed unitary condition, ||UU^d - I|| = %e' % norm)
        return False

    if not np.allclose(Y, I, rtol=0, atol=tol):
        if _logger.isEnabledFor(logging.DEBUG):
            norm = np.linalg.norm(Y - I)
            _logger.debug('Failed unitary condition, ||U^dU - I|| = %e' % norm)
        return False

    return True


def is_hermitian(H: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks if H is a hermitian matrix."""

    if not is_square_matrix(H):
        return False

    if not np.allclose(H, H.conj().T, rtol=0, atol=tol):
        if _logger.isEnabledFor(logging.DEBUG):
            norm = np.linalg.norm(H - H.conj().T)
            _logger.debug(
                'Failed hermitian condition, ||H - H^d|| = %e'
                % norm,
            )
        return False

    return True


def is_skew_hermitian(H: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks if H is a skew hermitian matrix."""

    if not is_square_matrix(H):
        return False

    if not np.allclose(-H, H.conj().T, rtol=0, atol=tol):
        if _logger.isEnabledFor(logging.DEBUG):
            norm = np.linalg.norm(-H - H.conj().T)
            _logger.debug(
                'Failed skew hermitian condition, ||H - H^d|| = %e'
                % norm,
            )
        return False

    return True


def is_permutation(P: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks if P is a permutation matrix."""

    if not is_unitary(P, tol):
        return False

    if not all(s == 1 for s in P.sum(0)):
        _logger.debug('Not all rows sum to 1.')
        return False

    if not all(s == 1 for s in P.sum(1)):
        _logger.debug('Not all columns sum to 1.')
        return False

    if not all(e == 1 or e == 0 for row in P for e in row):
        _logger.debug('Not all elements are 0 or 1.')
        return False

    return True
