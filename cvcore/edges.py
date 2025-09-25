import cv2 as cv

def detect_edges(frame, low=30, high=120):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, low, high)
    return cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
