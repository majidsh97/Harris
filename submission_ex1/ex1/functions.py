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

    # Step 1: Compute dx and dy with cv2.Sobel. (2 Lines)
    # Sobel(I, depth, dx, dy)
    Idx = cv2.Sobel(I, cv2.CV_32F, 1, 0)
    Idy = cv2.Sobel(I, cv2.CV_32F, 0, 1)
    # Step 2: Ixx Iyy Ixy from Idx and Idy (3 Lines)
    Ixx = Idx**2
    Iyy = Idy**2
    Ixy = Idx * Idy

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur (5 Lines)
    sigma = 1
    A = cv2.GaussianBlur(Ixx, (3,3), sigma)
    B = cv2.GaussianBlur(Iyy, (3,3), sigma)
    C = cv2.GaussianBlur(Ixy, (3,3), sigma) 

    #Step 4:  Compute the harris response with the determinant and the trace of T (see announcement) (4 lines)
    trace = A + B
    det = (A*B) - (C*C)
    R = det - k * trace**2

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
    print(R.shape)
    padded = np.pad(R, 1)

    # Step 2 (recommended) : create one image for every offset in the 3x3 neighborhood (6 lines).
    offsets = np.array([[-1,-1] , [ -1,1] , [1,-1], [0, -1], [0, 1], [-1, 0], [1, 0], [1, 1]])
    images =  []
    for offset in offsets:
        image = np.roll(padded, tuple(offset), axis=(0,1))
        images.append(image)
    img = np.stack(images, axis=-1)
    # Step 3 (recommended) : compute the greatest neighbor of every pixel (1 line)
    img = img.max(axis=-1)
    img = img[ 1:-1, 1:-1 ]

    # Step 4 (recommended) : Compute a boolean image with only all key-points set to True (1 line)
    img = np.logical_and( R > threshold , R > img)
    # Step 5 (recommended) : Use np.nonzero to compute the locations of the key-points from the boolean image (1 line)
    x, y=  np.nonzero(img)
    # print(x_y)

    return (y,x)


def detect_edges(R: np.array, edge_threshold: float = -0.01, epsilon=-.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant
    Returns:
        A boolean image with edge pixels set to True.
    """

    # Step 1 (recommended) : pad the response image to facilitate vectorization (1 line)
    padded = np.pad(R, 1)

    # Step 2 (recommended) : Calculate significant response pixels (1 line)
    significant = padded <= edge_threshold 

    # Step 3 (recommended) : create two images with the smaller x-axis and y-axis neighbors respectively (2 lines).
    x_min =np.minimum(np.roll(padded, (0, -1)), np.roll(padded, (0, 1)))
    y_min = np.minimum(np.roll(padded, (-1, 0)), np.roll(padded, (1, 0)))

    # Step 4 (recommended) : Calculate pixels that are lower than either their x-axis or y-axis neighbors (1 line)
    img = np.logical_or(padded < x_min, padded < y_min)

    # Step 5 (recommended) : Calculate valid edge pixels by combining significant and axis_minimal pixels (1 line)
    x_y = np.logical_and(significant,img)
    x_y = x_y[1:-1, 1:-1]
    return x_y

