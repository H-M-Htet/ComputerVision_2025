# import cv2 as cv
# import numpy as np
# import os

# # Globals
# mtx, dist = None, None
# model_vertices = []
# model_faces = []
# scale = 3.0   # enlarge TREX

# def load_calibration():
#     global mtx, dist
#     if os.path.exists("calibration.npz"):
#         with np.load("calibration.npz") as X:
#             mtx, dist = [X[i] for i in ("mtx", "dist")]
#         print("✅ Calibration loaded")
#     else:
#         print("⚠️ calibration.npz not found, AR may not work")


# def load_obj(path):
#     """Load vertices & faces from .OBJ file"""
#     verts = []
#     faces = []
#     with open(path) as f:
#         for line in f:
#             if line.startswith("v "):   # vertex
#                 parts = line.strip().split()
#                 verts.append([float(p) for p in parts[1:4]])
#             elif line.startswith("f "): # face
#                 parts = line.strip().split()
#                 face = [int(p.split("/")[0]) - 1 for p in parts[1:4]]  # triangulate
#                 faces.append(face)
#     return np.array(verts, dtype=np.float32), faces


# # --- Load calibration + model once ---
# load_calibration()
# model_vertices, model_faces = load_obj("models/trex_model.obj")


# def overlay_model(frame):
#     """Main AR function: overlay TREX model"""
#     if mtx is None or dist is None:
#         cv.putText(frame, "⚠️ No calibration loaded", (30, 40),
#                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         return frame

#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # ArUco detection
#     aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
#     aruco_params = cv.aruco.DetectorParameters()
#     detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
#     corners, ids, _ = detector.detectMarkers(gray)

#     if ids is not None:
#         # Pose estimation
#         rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

#         # Project TREX vertices
#         verts = model_vertices * scale   # scale up TREX
#         imgpts, _ = cv.projectPoints(verts, rvecs[0], tvecs[0], mtx, dist)
#         imgpts = imgpts.reshape(-1, 2).astype(int)

#         # Draw faces
#         for face in model_faces:
#             pts = imgpts[face]
#             cv.fillConvexPoly(frame, pts, (0, 255, 0), lineType=cv.LINE_AA)
#             cv.polylines(frame, [pts], True, (0, 0, 0), 1, lineType=cv.LINE_AA)

#         cv.putText(frame, "🐊 TREX AR!", (30, 60),
#                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     else:
#         cv.putText(frame, "No marker detected", (30, 40),
#                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     return frame



##ligter_AR_CUBIC

# import cv2 as cv
# import numpy as np
# import os

# mtx, dist = None, None

# def load_calibration():
#     global mtx, dist
#     if os.path.exists("calibration.npz"):
#         with np.load("calibration.npz") as X:
#             mtx, dist = [X[i] for i in ("mtx", "dist")]
#         print("✅ Calibration loaded for AR")
#     else:
#         print("⚠️ calibration.npz not found")

# # Load calibration at import
# load_calibration()

# def overlay_model(frame):
#     """Test AR overlay: just draw a cube on ArUco marker"""
#     global mtx, dist
#     if mtx is None or dist is None:
#         cv.putText(frame, "No calibration loaded", (30, 40),
#                    cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#         return frame

#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # --- Detect ArUco ---
#     aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
#     params = cv.aruco.DetectorParameters()
#     detector = cv.aruco.ArucoDetector(aruco_dict, params)
#     corners, ids, _ = detector.detectMarkers(gray)

#     if ids is None or len(corners) == 0:
#         cv.putText(frame, "No marker detected", (30, 40),
#                    cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#         return frame

#     # --- Pose estimation ---
#     marker_size = 0.11  # 5 cm marker (adjust to your printed size in meters)
#     rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)

#     # --- Cube points ---

#     half = marker_size / 2
#     height = marker_size * 0.5  # cube height

#     axis = np.float32([
#             [-half,-half,0], [half,-half,0], [half,half,0], [-half,half,0],   # bottom face (on marker)
#             [-half,-half,-height], [half,-half,-height], [half,half,-height], [-half,half,-height]  # top face
#     ])

#     # axis = np.float32([
#     #     [0,0,0], [marker_size,0,0], [marker_size,marker_size,0], [0,marker_size,0],
#     #     [0,0,-marker_size*0.5], [marker_size,0,-marker_size*0.5],
#     #     [marker_size,marker_size,-marker_size*0.5], [0,marker_size,-marker_size*0.5]
#     # ])

#     imgpts, _ = cv.projectPoints(axis, rvecs[0], tvecs[0], mtx, dist)
#     imgpts = np.int32(imgpts).reshape(-1,2)

#     # --- Draw cube ---
#     # draw bottom face
#     cv.drawContours(frame, [imgpts[:4]], -1, (0,255,0), -3)
#     # draw pillars
#     for i,j in zip(range(4), range(4,8)):
#         cv.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255), 2)
#     # draw top face
#     cv.drawContours(frame, [imgpts[4:]], -1, (0,0,255), 3)

#     cv.putText(frame, "Cube overlay (test)", (30,70),
#                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     return frame


##combine_version
# import cv2 as cv
# import numpy as np
# import os

# # Globals
# mtx, dist = None, None
# model_vertices, model_faces = [], []
# scale = 0.02  # adjust for TREX size

# def load_calibration():
#     global mtx, dist
#     if os.path.exists("calibration.npz"):
#         with np.load("calibration.npz") as X:
#             mtx, dist = [X[i] for i in ("mtx", "dist")]
#         print("✅ Calibration loaded for AR")
#     else:
#         print("⚠️ calibration.npz not found")

# def load_obj(path):
#     """Load vertices & faces from OBJ file (triangular faces only)."""
#     verts, faces = [], []
#     with open(path) as f:
#         for line in f:
#             if line.startswith("v "):
#                 parts = line.strip().split()
#                 verts.append([float(p) for p in parts[1:4]])
#             elif line.startswith("f "):
#                 parts = line.strip().split()
#                 face = [int(p.split("/")[0]) - 1 for p in parts[1:]]  # indices
#                 faces.append(face)
#     return np.array(verts, dtype=np.float32), faces

# # --- Load calibration + model once ---
# load_calibration()
# if os.path.exists("models/trex_object.obj"):
#     model_vertices, model_faces = load_obj("models/trex_object.obj")
#     print(f"✅ Loaded TREX model with {len(model_vertices)} vertices, {len(model_faces)} faces")
# else:
#     print("⚠️ trex.obj not found in models/")

# def overlay_model(frame, use_cube=False):
#     """Overlay either cube (debug) or TREX .obj model."""
#     global mtx, dist, model_vertices, model_faces

#     if mtx is None or dist is None:
#         cv.putText(frame, "No calibration loaded", (30, 40),
#                    cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#         return frame

#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # --- Detect ArUco ---
#     aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
#     params = cv.aruco.DetectorParameters()
#     detector = cv.aruco.ArucoDetector(aruco_dict, params)
#     corners, ids, _ = detector.detectMarkers(gray)

#     if ids is None or len(corners) == 0:
#         cv.putText(frame, "No marker detected", (30, 40),
#                    cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#         return frame

#     # --- Pose estimation ---
#     marker_size = 0.10  # meters, match your printed marker
#     rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)

#     if use_cube or len(model_vertices) == 0:
#         # --- Cube fallback ---
#         half = marker_size / 2
#         height = marker_size * 0.5
#         axis = np.float32([
#             [-half,-half,0], [half,-half,0], [half,half,0], [-half,half,0],
#             [-half,-half,-height], [half,-half,-height], [half,half,-height], [-half,half,-height]
#         ])
#         imgpts, _ = cv.projectPoints(axis, rvecs[0], tvecs[0], mtx, dist)
#         imgpts = np.int32(imgpts).reshape(-1,2)
#         cv.drawContours(frame, [imgpts[:4]], -1, (0,255,0), -3)
#         for i,j in zip(range(4), range(4,8)):
#             cv.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255), 2)
#         cv.drawContours(frame, [imgpts[4:]], -1, (0,0,255), 3)
#         cv.putText(frame, "Cube overlay (debug)", (30,70),
#                    cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#         return frame

#     # --- Project TREX ---
#     verts = model_vertices * scale
#     imgpts, _ = cv.projectPoints(verts, rvecs[0], tvecs[0], mtx, dist)
#     imgpts = imgpts.reshape(-1, 2).astype(int)

#     for face in model_faces:
#         pts = imgpts[face]
#         cv.fillConvexPoly(frame, pts, (0,255,0), lineType=cv.LINE_AA)
#         cv.polylines(frame, [pts], True, (0,0,0), 1, lineType=cv.LINE_AA)

#     cv.putText(frame, "TREX overlay", (30,70),
#                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#     return frame


import cv2 as cv
import numpy as np
import os

# Load calibration
if os.path.exists("calibration.npz"):
    with np.load("calibration.npz") as X:
        mtx, dist = [X[i] for i in ("mtx", "dist")]
    print("✅ Loaded calibration.npz")
else:
    raise FileNotFoundError("calibration.npz not found – run calibration first.")

# Load preprocessed Trex
if os.path.exists("trex_preprocessed.npz"):
    data = np.load("trex_preprocessed.npz", allow_pickle=True)
    verts = data["verts"]

      # Ensure faces are int32 numpy arrays
    raw_faces = data["faces"]
    faces = [np.array(f, dtype=np.int32) for f in raw_faces]

    print(f"✅ Loaded Trex model: {verts.shape[0]} verts, {len(faces)} faces")
else:
    raise FileNotFoundError("trex_preprocessed.npz not found – run preprocess_obj.py first.")

# ArUco setup
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
aruco_params = cv.aruco.DetectorParameters()
aruco_detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)


def overlay_model(frame):
    """Overlay Trex model on ArUco marker."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is None:
        cv.putText(frame, "No ArUco marker", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    # Estimate pose of marker
    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
    if rvecs is None:
        cv.putText(frame, "Pose estimation failed", (20, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    rvec, tvec = rvecs[0], tvecs[0]

    # Project Trex vertices into image
    img_pts, _ = cv.projectPoints(verts, rvec, tvec, mtx, dist)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    # Draw faces
    for f in faces:
        pts = img_pts[f]
        cv.fillConvexPoly(frame, pts, (50, 200, 50), lineType=cv.LINE_AA)
        cv.polylines(frame, [pts], True, (0, 0, 0), 1, lineType=cv.LINE_AA)

    cv.putText(frame, "Trex AR Mode", (20, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


