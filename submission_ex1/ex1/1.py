import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

   
    ix =cv2.Sobel(I,dx=1)
    iy =cv2.Sobel(I,dy=1)
    gx = cv2.GaussianBlur(ix,3,1,0)
    gy = cv2.GaussianBlur(iy,3,0,1)
    c = gx*gy
    R=a*b-c**2 - k*(A+B)**2

    return R,  Idx, Idy, A, B, C


def detect_corners(R: np.array, threshold: float = 0.5) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.
    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended) : pad the response image to facilitate vectorization (1 line)
    #np.pad(arr, pad width)
    R = np.pad(R,1)
    t = True
    for x in [(1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)]:
        r = np.roll(R,x)
        m = np.maximum(R,r)
        t = np.logical_and(t,m)

    m = R>th
    t = np.logical_and(t,m)
    return t



        


def detect_edges(R: np.array, edge_threshold: float = -0.01, epsilon=-.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant
    Returns:
        A boolean image with edge pixels set to True.
    """
    R = np.pad(R,1)
    s = R<=edge_threshold
    lx = np.minimum(np.roll(R,(1,0)),np.roll(R,(-1,0)))
    ly = np.minimum(np.roll(R,(0,1)),np.roll(R,(0,-1)))
    ror = np.logical_or(R<lx,R<ly)
    x_y = np.logical_and(s,ror)

    return x_y

