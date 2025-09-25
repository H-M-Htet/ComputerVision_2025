import cv2 as cv
import numpy as np
import time

# --- Configuration ---
CHESSBOARD_SIZE = (9, 6)   # Number of inner corners
SQUARE_SIZE_MM = 25        # Real-world square size
TARGET_IMAGES = 20         # Number of snapshots to collect

# --- State ---
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM

objpoints = []   # 3D points
imgpoints = []   # 2D points
images_captured = 0
last_capture_time = time.time()
calibrated = False
mtx, dist = None, None


def run_calibration(frame):
    """Process a single frame for calibration mode."""
    global images_captured, last_capture_time, calibrated, mtx, dist

    show = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if not calibrated:
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            cv.drawChessboardCorners(show, CHESSBOARD_SIZE, corners, ret)

            # Capture at most every 2 sec
            if time.time() - last_capture_time > 2 and images_captured < TARGET_IMAGES:
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)

                images_captured += 1
                last_capture_time = time.time()
                print(f"📸 Captured calibration image {images_captured}/{TARGET_IMAGES}")

        # Status text
        cv.putText(show, f"Calibration: {images_captured}/{TARGET_IMAGES}",
                   (30, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Once enough snapshots, calibrate
        if images_captured >= TARGET_IMAGES:
            print("⚙️ Running final calibration...")
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)

            if ret:
                np.savez("calibration.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
                print("✅ Calibration complete, saved to calibration.npz")
                calibrated = True
                cv.putText(show, "Calibration Complete!", (30, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv.putText(show, "Calibration Complete", (30, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return show
