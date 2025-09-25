import cv2 as cv
import numpy as np

def show_hist(frame):
    """
    Show image + histogram side by side
    """
    # Make histogram canvas
    h = np.zeros((300, 256, 3), dtype=np.uint8)
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv.calcHist([frame], [i], None, [256], [0, 256])
        cv.normalize(hist, hist, 0, 300, cv.NORM_MINMAX)
        hist = hist.flatten()
        for x, y in enumerate(hist):
            cv.line(h, (x, 300), (x, 300 - int(y)),
                    (255 if col == 'b' else 0,
                     255 if col == 'g' else 0,
                     255 if col == 'r' else 0), 1)

    # Resize histogram to match frame height
    h_resized = cv.resize(h, (h.shape[1], frame.shape[0]))

    # Concatenate
    return cv.hconcat([frame, h_resized])
