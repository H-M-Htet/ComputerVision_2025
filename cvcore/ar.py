# cvcore/ar.py
import cv2 as cv
import numpy as np
import os


def draw_text_lines(img, lines, start_y=30, line_height=30):
    """
    Draw multiple lines of text on an image.
    lines: list of (text, color) tuples
    """
    for i, (txt, col) in enumerate(lines):
        y = start_y + i * line_height
        cv.putText(img, txt, (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return img


# Load calibration
if os.path.exists("calibration.npz"):
    with np.load("calibration.npz") as X:
        mtx, dist = [X[i] for i in ("mtx", "dist")]
    print("✅ Calibration loaded.")
else:
    raise FileNotFoundError("calibration.npz not found. Run calibration first.")

# Load Trex model (from preprocess or .obj loader)
if os.path.exists("trex_preprocessed.npz"):
    data = np.load("trex_preprocessed.npz", allow_pickle=True)
    verts, faces = data["verts"], data["faces"]
    print(f"✅ Trex preprocessed model loaded: {len(verts)} verts, {len(faces)} faces")
else:
    raise FileNotFoundError("trex_preprocess.npz not found. Run preprocess first.")

# ArUco setup
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
aruco_params = cv.aruco.DetectorParameters()
aruco_detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)


# ------------------- Utility -------------------
def apply_transformations(verts, scale, rx, ry, rz):
    """Apply scaling + rotations to vertices."""
    verts = verts * scale
    rx, ry, rz = np.radians([rx, ry, rz])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return verts @ R.T


# ------------------- AR Overlay -------------------
def overlay_model(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is None:
        cv.putText(frame, "No ArUco marker detected", (30, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    # Estimate pose
    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
    if rvecs is None:
        cv.putText(frame, "Pose estimation failed", (20, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    rvec, tvec = rvecs[0], tvecs[0]

    # --- Read trackbars ---
    scale = cv.getTrackbarPos("Scale", "CV Playground") / 100.0
    rx = cv.getTrackbarPos("RotX", "CV Playground")
    ry = cv.getTrackbarPos("RotY", "CV Playground")
    rz = cv.getTrackbarPos("RotZ", "CV Playground")

    # --- Transform Trex vertices ---
    verts_t = apply_transformations(verts.copy(), scale, rx, ry, rz)

    # --- Project vertices ---
    img_pts, _ = cv.projectPoints(verts_t, rvecs[0], tvecs[0], mtx, dist)
    img_pts = np.int32(img_pts).reshape(-1, 2)

    # --- Draw model faces ---
    for f in faces:
        f = np.array(f, dtype=int).reshape(-1)
        if np.any(f >= len(img_pts)):
            continue
        pts = img_pts[f]
        cv.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

    cv.putText(frame, "AR: Trex model (use sliders)", (30, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame





