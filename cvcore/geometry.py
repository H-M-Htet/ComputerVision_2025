import cv2 as cv

# def apply_transform(frame):
#     rows,cols,_ = frame.shape
#     # Rotate 45 degrees around center
#     M = cv.getRotationMatrix2D((cols/2,rows/2),45,1)
#     return cv.warpAffine(frame,M,(cols,rows))

import cv2 as cv
import numpy as np

def apply_transform(frame, tx=0, ty=0, angle=0, scale=100):
    """
    Apply translation, rotation, and scaling to frame.
    tx, ty: translation in pixels
    angle: rotation angle in degrees
    scale: percentage (100 = same size)
    """
    h, w = frame.shape[:2]
    # Convert scale to factor
    scale_factor = scale / 100.0

    # Transformation matrix
    M = cv.getRotationMatrix2D((w//2, h//2), angle, scale_factor)
    M[0,2] += tx
    M[1,2] += ty

    return cv.warpAffine(frame, M, (w, h))

