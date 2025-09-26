##Verion_0.1

# import cv2 as cv
# import numpy as np

# snapshots = []      # store captured frames
# last_result = None  # keep last panorama to display if needed


# def multi_stitch(images):
#     """Stitch N images (>=2) using SIFT + chained homographies"""
#     sift = cv.SIFT_create()
#     bf = cv.BFMatcher()

#     base = images[0]
#     h, w = base.shape[:2]
#     pano_w = w * len(images)
#     pano_h = h * 2

#     # Start canvas with first image
#     result = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
#     result[0:h, 0:w] = base

#     # Features of first image (anchor)
#     prev_kp, prev_des = sift.detectAndCompute(cv.cvtColor(base, cv.COLOR_BGR2GRAY), None)
#     H_total = np.eye(3)  # identity

#     for i in range(1, len(images)):
#         img = images[i]
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         kp, des = sift.detectAndCompute(gray, None)

#         if prev_des is None or des is None:
#             print(f"⚠️ Skipping image {i+1}: no descriptors")
#             continue

#         # Match features (current vs previous)
#         matches = bf.knnMatch(prev_des, des, k=2)
#         good = [m for m, n in matches if m.distance < 0.75 * n.distance]
#         if len(good) < 4:
#             print(f"⚠️ Skipping image {i+1}: not enough good matches")
#             continue

#         src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

#         # Homography between current and previous
#         H, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
#         if H is None:
#             print(f"⚠️ Skipping image {i+1}: homography failed")
#             continue

#         # Update cumulative transform (chain)
#         H_total = H_total @ H

#         # Warp current image into base coordinates
#         result = cv.warpPerspective(img, H_total, (pano_w, pano_h),
#                                     dst=result, borderMode=cv.BORDER_TRANSPARENT)

#         # Update prev for next iteration
#         prev_kp, prev_des = kp, des

#     # Crop black borders
#     gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
#     coords = cv.findNonZero(gray)
#     if coords is not None:
#         x, y, w, h = cv.boundingRect(coords)
#         result = result[y:y+h, x:x+w]

#     return result



# def stitch(frame):
#     """
#     Panorama mode:
#     - 'c' to capture frames
#     - stitches when >= 2 frames are captured
#     - 'r' to reset
#     """

#     global snapshots, last_result

#     show = frame.copy()
#     cv.putText(show, "Press 'c'=capture  'r'=reset",
#                (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     # ---- Case 1: Not enough frames ----
#     if len(snapshots) < 2:
#         output = show

#     # ---- Case 2: Stitch 2 or more ----
#     else:
#         try:
#             result = multi_stitch(snapshots)
#             output = cv.resize(result, (frame.shape[1], frame.shape[0]))
#             cv.putText(output, f"✅ Panorama created ({len(snapshots)} frames)",
#                        (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             last_result = output
#         except Exception as e:
#             cv.putText(show, f"⚠️ Stitch error: {e}", (10, 80),
#                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             output = show

#     # ---- Always: add thumbnails at bottom ----
#     if snapshots:
#         target_h = frame.shape[0] // 4
#         thumbs = []
#         for snap in snapshots:
#             scale = target_h / snap.shape[0]
#             w = int(snap.shape[1] * scale)
#             thumb = cv.resize(snap, (w, target_h))
#             thumbs.append(thumb)

#         if thumbs:
#             strip = cv.hconcat(thumbs)

#             # Make canvas same width as output, center the strip
#             canvas = np.zeros((target_h, output.shape[1], 3), dtype=np.uint8)
#             if strip.shape[1] <= output.shape[1]:
#                 x_offset = (output.shape[1] - strip.shape[1]) // 2
#                 canvas[:, x_offset:x_offset+strip.shape[1]] = strip
#             else:
#                 canvas = cv.resize(strip, (output.shape[1], target_h))

#             output = cv.vconcat([output, canvas])

#     return output


# def capture_frame(frame):
#     """ Save a snapshot for panorama """
#     global snapshots
#     snapshots.append(frame.copy())
#     print(f"📸 Captured frame {len(snapshots)}")


# def reset_snapshots():
#     """ Reset snapshots for a new panorama """
#     global snapshots, last_result
#     snapshots = []
#     last_result = None
#     print("🔄 Snapshots reset")


##Version_0.2
import cv2 as cv
import numpy as np

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


# Maximum capture sequence
CAPTURE_ORDER = ["Center", "Left1", "Left2", "Right1", "Right2"]

snapshots = []      # store captured frames
last_result = None  # last panorama result


# ---------------------------
# Panorama core (SIFT + Homography)
# ---------------------------
def multi_stitch(images):
    """Stitch N images using SIFT + chained homographies."""
    sift = cv.SIFT_create()
    bf = cv.BFMatcher()

    base = images[0]
    h, w = base.shape[:2]
    pano_w = w * len(images)
    pano_h = h * 2

    # Start canvas with first image
    result = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
    result[0:h, 0:w] = base

    # Features of first image (anchor)
    prev_kp, prev_des = sift.detectAndCompute(
        cv.cvtColor(base, cv.COLOR_BGR2GRAY), None)
    H_total = np.eye(3)  # identity

    for i in range(1, len(images)):
        img = images[i]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        if prev_des is None or des is None:
            print(f"⚠️ Skipping image {i+1}: no descriptors")
            continue

        # Match features
        matches = bf.knnMatch(prev_des, des, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            print(f"⚠️ Skipping image {i+1}: not enough good matches")
            continue

        src_pts = np.float32(
            [prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Homography between current and previous
        H, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
        if H is None:
            print(f"⚠️ Skipping image {i+1}: homography failed")
            continue

        # Update cumulative transform
        H_total = H_total @ H

        # Warp current image into base coordinates
        result = cv.warpPerspective(img, H_total, (pano_w, pano_h),
                                    dst=result, borderMode=cv.BORDER_TRANSPARENT)

        # Update prev
        prev_kp, prev_des = kp, des

    # Crop black borders
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    coords = cv.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv.boundingRect(coords)
        result = result[y:y+h, x:x+w]

    return result


# ---------------------------
# Public API
# ---------------------------
def stitch(frame):
    """Panorama preview with guided capture (max 4 frames, clockwise only)."""
    global snapshots, last_result

    h, w = frame.shape[:2]
    show = frame.copy()

    # --- Draw vertical grid (rule-of-thirds) ---
    for i in range(1, 3):
        x = w * i // 3
        cv.line(show, (x, 0), (x, h), (0, 255, 0), 1)

    # --- Draw center cross ---
    cv.line(show, (w//2, 0), (w//2, h), (0, 255, 255), 2)
    cv.line(show, (0, h//2), (w, h//2), (0, 255, 255), 2)

    # --- Next capture guidance ---
    next_idx = len(snapshots)
    if next_idx < len(CAPTURE_ORDER):
        msg = f"Next capture: {CAPTURE_ORDER[next_idx]}"
    else:
        msg = "✅ All frames captured"
        
    # --- Instruction for capture order ---
    lines = [
        ("Rotate webcam CLOCKWISE only", (0,255,255)),
        ("Start Left -> Center -> Right (max 4 frames)", (0,255,0)),
        (msg, (255,0,0))   # msg = “Next capture: …”
    ]
    show = draw_text_lines(show, lines, start_y=90, line_height=30)

    # --- Stitch if ≥2 frames ---
    if len(snapshots) >= 2:
        try:
            result = multi_stitch(snapshots)
            scale = h / result.shape[0]
            out_w = int(result.shape[1] * scale)
            pano = cv.resize(result, (out_w, h))
            cv.putText(pano, f"Panorama ({len(snapshots)} frames)",
                       (10, 140), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            last_result = pano
            output = pano
        except Exception as e:
            output = draw_text_lines(show, [
                (f"⚠️ Stitch error: {e}", (0,0,255))
            ], start_y=220, line_height=30)
    else:
        output = show

    # --- Thumbnails strip ---
    if snapshots:
        target_h = h // 4
        thumbs = []
        for snap in snapshots:
            sc = target_h / snap.shape[0]
            tw = int(snap.shape[1] * sc)
            thumbs.append(cv.resize(snap, (tw, target_h)))
        strip = cv.hconcat(thumbs)
        canvas = np.zeros((target_h, output.shape[1], 3), dtype=np.uint8)
        if strip.shape[1] <= output.shape[1]:
            xoff = (output.shape[1] - strip.shape[1]) // 2
            canvas[:, xoff:xoff+strip.shape[1]] = strip
        else:
            canvas = cv.resize(strip, (output.shape[1], target_h))
        output = cv.vconcat([output, canvas])

    return output



def capture_frame(frame):
    """ Capture frame only if under max 4. """
    global snapshots
    if len(snapshots) < 4:
        snapshots.append(frame.copy())
        print(f"📸 Captured {CAPTURE_ORDER[len(snapshots)-1]} frame")
    else:
        print("⚠️ Max 4 frames already captured")


def reset_snapshots():
    """ Reset snapshots for a new panorama. """
    global snapshots, last_result
    snapshots = []
    last_result = None
    print("🔄 Snapshots reset")

