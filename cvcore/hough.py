import cv2 as cv
import numpy as np

# def detect_lines(frame):
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     edges = cv.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv.HoughLines(edges, 1, np.pi/180, 150)
#     output = frame.copy()
#     if lines is not None:
#         for rho,theta in lines[:,0]:
#             a = np.cos(theta); b = np.sin(theta)
#             x0 = a*rho; y0 = b*rho
#             x1 = int(x0 + 1000*(-b)); y1 = int(y0 + 1000*(a))
#             x2 = int(x0 - 1000*(-b)); y2 = int(y0 - 1000*(a))
#             cv.line(output,(x1,y1),(x2,y2),(0,0,255),2)
#     return output


# def detect_lines(frame):
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     edges = cv.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv.HoughLinesP(edges, 1, np.pi/180, 90, minLineLength=50, maxLineGap=10)
#     output = frame.copy()
#     if lines is not None:
#         for x1,y1,x2,y2 in lines[:,0]:
#             cv.line(output, (x1,y1), (x2,y2), (0,0,255), 2)
#     return output


import cv2 as cv
import numpy as np

def detect_lines(frame,
                 canny_low=50, canny_high=150,
                 threshold=60, minLineLength=100, maxLineGap=40):     ###If lines look broken → increase maxLineGap.
                                                                      ##If too many small lines → increase minLineLength.
                                                                      ##If too few lines → lower threshold.
    """
    Detect straight line segments using Probabilistic Hough Transform.

    Parameters:
        frame (np.ndarray): Input BGR image (webcam frame).
        canny_low (int): Lower threshold for Canny.
        canny_high (int): Upper threshold for Canny.
        threshold (int): Votes needed to accept a line.
        minLineLength (int): Minimum length of a line.
        maxLineGap (int): Maximum gap between line segments to link them.

    Returns:
        np.ndarray: Frame with detected lines drawn.
    """
    # Preprocess
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 1)

    # Edge detection
    edges = cv.Canny(blur, canny_low, canny_high)

    # Hough transform (probabilistic)
    lines = cv.HoughLinesP(edges,
                           rho=1,
                           theta=np.pi/180,
                           threshold=threshold,
                           minLineLength=minLineLength,
                           maxLineGap=maxLineGap)

    # Draw results
    output = frame.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return output
