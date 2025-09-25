import cv2 as cv

def gaussian_blur(frame, ksize=15, sigma=5):
    return cv.GaussianBlur(frame, (ksize,ksize), sigma)

def bilateral_blur(frame, d=9, sigmaColor=150, sigmaSpace=150):
    return cv.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
