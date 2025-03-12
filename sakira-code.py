"""
    The accompanying code for the paper "Shadow Art Kanji: Inverse Rendering Application"

    @codeauthor     Wil Louis Rothman
"""

import numpy as np
from scipy.optimize import linprog

# L.P. error tolerance
EPSILON = 0.239


def cartesian2D(a):
    return np.array(
        np.meshgrid(
            np.arange(a),
            np.arange(a)
        )
    ).T.reshape(-1, 2)

def cartesian3D(a):
    return np.array(
        np.meshgrid(
            np.arange(a),
            np.arange(a),
            np.arange(a)
        )
    ).T.reshape(-1, 3)

def lp_model(n, P, Q=None, R=None):
    """ 
        Runs the Linear Program model.

        @param n    side length
        @param P    bitmap
        @param Q    (optional) bitmap
        @param R    (optional) bitmap
        
        @author     Wil Louis Rothman
    """

    assert n > 0
    assert Q or not R

    # Construct b_ub
    b_ub = []
    for i, j, k in cartesian3D(n):
        b_ub.append(n * P[j, k] 
                    + EPSILON)
        if Q is not None:
            b_ub.append(n * Q[i, k] 
                        + EPSILON)
        if R is not None:
            b_ub.append(n * R[i, j] 
                        + EPSILON)
    
    b_ub = np.array(b_ub)

    # Construct A_ub
    A_ub = np.zeros((len(b_ub), n ** 3))
    
    ## Constraints for P
    index = 0
    for j, k in np.ndindex(n, n):
        for i in range(n):
            A_ub[index, i * n ** 2 
                 + j * n + k] = 1
        index += 1

    ## Constraints for Q
    if Q is not None:
        for i, k in np.ndindex(n, n):
            for j in range(n):
                A_ub[index, i * n ** 2 
                     + j * n + k] = 1
            index += 1

    ## Constraints for R
    if R is not None:
        for i, j in np.ndindex(n, n):
            for k in range(n):
                A_ub[index, i * n ** 2 
                     + j * n + k] = 1
            index += 1

    c = np.ones(n ** 3)
    bounds = [(0, 1)] * n ** 3
    
    result = linprog(c, A_ub=A_ub, 
                     b_ub=b_ub, 
                     bounds=bounds, 
                     method='highs')
    
    if result.success:
        return result.x.reshape((n, n, n))
    else:
        raise RuntimeError("LP failed.")


def direct_carving(n, P, Q=None, R=None):
    """ 
        Runs the Direct Carving model.

        @param n    side length
        @param P    bitmap
        @param Q    (optional) bitmap
        @param R    (optional) bitmap
        
        @author     Wil Louis Rothman
    """
    assert n > 0
    assert Q or not R

    # Initial setup for T as 1's vector
    T = np.ones((n, n, n))

    # Carve on plane P
    for j, k in cartesian2D(n):
        if P[j, k] == 0:
            for i in np.arange(n):
                T[i, j, k] = 0

    # Exit if Q and R do not exist
    if Q is None:
        return T

    # Carve on plane Q
    for i, k in cartesian2D(n):
        if Q[i, k] == 0:
            for j in np.arange(n):
                T[i, j, k] = 0

    # Exit if R does not exist
    if R is None:
        return T

    # Carve on R
    for i, j in cartesian2D(n):
        if R[i, j] == 0:
            for k in np.arange(n):
                T[i, j, k] = 0

    # Exit
    return T
