import cv2 as cv

def adjust_bc(frame, alpha=1.2, beta=30):
    """
    Brightness & Contrast
    alpha = contrast factor
    beta = brightness shift
    """
    return cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
