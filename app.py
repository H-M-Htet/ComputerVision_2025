# app.py
import cv2 as cv
import time

# # Mode names – each one will connect to a module in cvcore/
# MODES = [
#     "raw", "gray", "hsv",
#     "exposure", "hist",
#     "gaussian", "bilateral",
#     "canny", "hough",
#     "transform",
#     "panorama",
#     "calib", "ar"
# ]
# mode_idx = 0

# def main():
#     global mode_idx
#     cap = cv.VideoCapture(0)
#     if not cap.isOpened():
#         print("❌ Cannot open webcam")
#         return

#     window = "CV Playground"
#     cv.namedWindow(window, cv.WINDOW_NORMAL)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ Failed to grab frame")
#             break

#         mode = MODES[mode_idx]

#         # ---------- Mode logic ----------
#         if mode == "raw":
#             show = frame

#         elif mode == "gray":
#             show = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#             show = cv.cvtColor(show, cv.COLOR_GRAY2BGR)

#         elif mode == "hsv":
#             hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#             show = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

#         elif mode == "exposure":
#             from cvcore import exposure
#             show = exposure.adjust_bc(frame)

#         elif mode == "hist":
#             from cvcore import histogram
#             show = histogram.show_hist(frame)

#         elif mode == "gaussian":
#             from cvcore import filtering
#             show = filtering.gaussian_blur(frame)

#         elif mode == "bilateral":
#             from cvcore import filtering
#             show = filtering.bilateral_blur(frame)

#         elif mode == "canny":
#             from cvcore import edges
#             show = edges.detect_edges(frame)

#         elif mode == "hough":
#             from cvcore import hough
#             show = hough.detect_lines(frame)

#         elif mode == "transform":
#             from cvcore import geometry
#             show = geometry.apply_transform(frame)

#         elif mode == "panorama":
#             from cvcore import panorama
#             show = panorama.stitch(frame)

#         elif mode == "calib":
#             from cvcore import calib
#             show = calib.run_calibration(frame)

#         elif mode == "ar":
#             from cvcore import ar
#             show = ar.overlay_model(frame)

#         else:
#             show = frame
#         # ---------- End of modes ----------

#         cv.putText(show, f"Mode: {mode}   [ / ] switch   s=save   ESC=quit",
#                    (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#         cv.imshow(window, show)

#         last_key = None   # put this before your while loop

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             mode = MODES[mode_idx]
#             # ... your mode logic here ...

#             cv.putText(show, f"Mode: {mode}   n=next, p=prev, s=save, ESC=quit",
#                         (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#             cv.imshow(window, show)

#             key = cv.waitKey(30) & 0xFF

#             if key == 27 and last_key == 27:   # quit only on double ESC
#                 break
#             elif key == ord('n'):
#                 mode_idx = (mode_idx + 1) % len(MODES)
#             elif key == ord('p'):
#                 mode_idx = (mode_idx - 1) % len(MODES)
#             elif key == ord('s'):
#                 filename = f"output_{mode}_{int(time.time())}.png"
#                 cv.imwrite(filename, show)
#                 print("💾 Saved:", filename)

#             last_key = key



### Track Bar Version_0.1
# app.py
# import cv2 as cv
# import time

# # Mode names – each one will connect to a module in cvcore/
# MODES = [
#     "raw", "gray", "hsv",
#     "exposure", "hist",
#     "gaussian", "bilateral",
#     "canny", "hough",
#     "transform",
#     "panorama",
#     "calib", "ar"
# ]
# mode_idx = 0
# last_mode = None  # keep track of last mode to avoid re-creating sliders

# def nothing(x):  # dummy callback for trackbars
#     pass

# def setup_trackbars(mode, window):
#     """Create sliders depending on mode"""
#     if mode == "gaussian":
#         cv.createTrackbar("Kernel", window, 5, 30, nothing)   # odd numbers work best
#         cv.createTrackbar("Sigma", window, 1, 50, nothing)

#     elif mode == "bilateral":
#         cv.createTrackbar("d", window, 9, 20, nothing)
#         cv.createTrackbar("SigmaColor", window, 75, 200, nothing)
#         cv.createTrackbar("SigmaSpace", window, 75, 200, nothing)

#     elif mode == "canny":
#         cv.createTrackbar("Low", window, 50, 500, nothing)
#         cv.createTrackbar("High", window, 150, 500, nothing)

#     elif mode == "hough":
#         cv.createTrackbar("Threshold", window, 80, 300, nothing)
#         cv.createTrackbar("MinLen", window, 100, 300, nothing)
#         cv.createTrackbar("MaxGap", window, 20, 100, nothing)

#     elif mode == "transform":
#         cv.createTrackbar("Tx", window, 100, 200, nothing)    # -100..+100
#         cv.createTrackbar("Ty", window, 100, 200, nothing)
#         cv.createTrackbar("Angle", window, 180, 360, nothing) # -180..+180
#         cv.createTrackbar("Scale", window, 100, 200, nothing) # 100%..200%

# def main():
#     global mode_idx, last_mode
#     cap = cv.VideoCapture(0)
#     if not cap.isOpened():
#         print("❌ Cannot open webcam")
#         return

#     window = "CV Playground"
#     cv.namedWindow(window, cv.WINDOW_NORMAL)

#     last_key = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ Failed to grab frame")
#             break

#         mode = MODES[mode_idx]

#         # Setup trackbars if mode changed
#         if mode != last_mode:
#             cv.destroyAllWindows()
#             cv.namedWindow(window, cv.WINDOW_NORMAL)
#             setup_trackbars(mode, window)
#             last_mode = mode

#         # ---------- Mode logic ----------
#         if mode == "raw":
#             show = frame

#         elif mode == "gray":
#             show = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#             show = cv.cvtColor(show, cv.COLOR_GRAY2BGR)

#         elif mode == "hsv":
#             hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#             show = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

#         elif mode == "exposure":
#             import cvcore.exposure as exposure
#             show = exposure.adjust_bc(frame)

#         elif mode == "hist":
#             import cvcore.histogram as histogram
#             show = histogram.show_hist(frame)

#         elif mode == "gaussian":
#             import cvcore.filtering as filtering
#             ksize = cv.getTrackbarPos("Kernel", window) | 1   # force odd
#             sigma = cv.getTrackbarPos("Sigma", window)
#             show = filtering.gaussian_blur(frame, ksize, sigma)

#         elif mode == "bilateral":
#             import cvcore.filtering as filtering
#             d = cv.getTrackbarPos("d", window)
#             sc = cv.getTrackbarPos("SigmaColor", window)
#             ss = cv.getTrackbarPos("SigmaSpace", window)
#             show = filtering.bilateral_blur(frame, d, sc, ss)

#         elif mode == "canny":
#             import cvcore.edges as edges
#             low = cv.getTrackbarPos("Low", window)
#             high = cv.getTrackbarPos("High", window)
#             show = edges.detect_edges(frame, low, high)

#         elif mode == "hough":
#             import cvcore.hough as hough
#             thr = cv.getTrackbarPos("Threshold", window)
#             minL = cv.getTrackbarPos("MinLen", window)
#             maxG = cv.getTrackbarPos("MaxGap", window)
#             show = hough.detect_lines(frame, threshold=thr,
#                                       minLineLength=minL, maxLineGap=maxG)

#         elif mode == "transform":
#             import cvcore.geometry as geometry
#             tx = cv.getTrackbarPos("Tx", window) - 100
#             ty = cv.getTrackbarPos("Ty", window) - 100
#             angle = cv.getTrackbarPos("Angle", window) - 180
#             scale = cv.getTrackbarPos("Scale", window)
#             show = geometry.apply_transform(frame, tx, ty, angle, scale)

#         elif mode == "panorama":
#             import cvcore.panorama as panorama
#             show = panorama.stitch(frame)

#         elif mode == "calib":
#             import cvcore.calib as calib
#             show = calib.run_calibration(frame)

#         elif mode == "ar":
#             import cvcore.ar as ar
#             show = ar.overlay_model(frame)

#         else:
#             show = frame
#         # ---------- End of modes ----------

#         cv.putText(show, f"Mode: {mode}   n=next, p=prev, s=save, ESC=quit",
#                    (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#         cv.imshow(window, show)

#         key = cv.waitKey(30) & 0xFF

#         if key == 27 and last_key == 27:   # quit only on double ESC
#             break
#         elif key == ord('n'):
#             mode_idx = (mode_idx + 1) % len(MODES)
#         elif key == ord('p'):
#             mode_idx = (mode_idx - 1) % len(MODES)
#         elif key == ord('s'):
#             filename = f"output_{mode}_{int(time.time())}.png"
#             cv.imwrite(filename, show)
#             print("💾 Saved:", filename)

#         last_key = key

#     cap.release()
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()


##App_version_0.2
import cv2 as cv
import time
import numpy as np

def draw_text_lines(img, lines, start_y=30, line_height=30):
    """
    Draw multiple lines of text on an image.
    lines: list of (text, color) tuples
    start_y: Y position of first line
    line_height: spacing between lines
    """
    for i, (txt, col) in enumerate(lines):
        y = start_y + i * line_height
        cv.putText(img, txt, (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return img


# Mode names – each one will connect to a module in cvcore/
MODES = [
    "raw", "gray", "hsv",
    "exposure", "hist",
    "gaussian", "bilateral",
    "canny", "hough",
    "transform",
    "panorama",
    "calib", "ar"
]
mode_idx = 0
last_mode = None  # keep track of last mode to avoid re-creating sliders

def nothing(x):  # dummy callback for trackbars
    pass

def setup_trackbars(mode, window):
    """Create sliders depending on mode"""
    if mode == "gray":
        cv.createTrackbar("Thresh", window, 0, 255, nothing)
        cv.createTrackbar("Invert", window, 0, 1, nothing)
        
    elif mode == "hsv":
        cv.createTrackbar("Hmin", window, 0, 179, nothing)
        cv.createTrackbar("Hmax", window, 179, 179, nothing)
        cv.createTrackbar("Smin", window, 0, 255, nothing)
        cv.createTrackbar("Smax", window, 255, 255, nothing)
        cv.createTrackbar("Vmin", window, 0, 255, nothing)
        cv.createTrackbar("Vmax", window, 255, 255, nothing)

    elif mode == "exposure":
        cv.createTrackbar("Brightness", window, 50, 100, nothing)  # 0–100, default 50
        cv.createTrackbar("Contrast", window, 50, 100, nothing)    # 0–100, default 50


    elif mode == "gaussian":
        cv.createTrackbar("Kernel", window, 5, 30, nothing)   # odd numbers work best
        cv.createTrackbar("Sigma", window, 1, 50, nothing)

    elif mode == "bilateral":
        cv.createTrackbar("d", window, 9, 20, nothing)
        cv.createTrackbar("SigmaColor", window, 75, 200, nothing)
        cv.createTrackbar("SigmaSpace", window, 75, 200, nothing)

    elif mode == "canny":
        cv.createTrackbar("Low", window, 50, 500, nothing)
        cv.createTrackbar("High", window, 150, 500, nothing)

    elif mode == "hough":
        cv.createTrackbar("Threshold", window, 80, 300, nothing)
        cv.createTrackbar("MinLen", window, 100, 300, nothing)
        cv.createTrackbar("MaxGap", window, 20, 100, nothing)

    elif mode == "transform":
        cv.createTrackbar("Tx", window, 100, 200, nothing)    # -100..+100
        cv.createTrackbar("Ty", window, 100, 200, nothing)
        cv.createTrackbar("Angle", window, 180, 360, nothing) # -180..+180
        cv.createTrackbar("Scale", window, 100, 200, nothing) # 100%..200%

    # elif mode == "panorama":
    #     # overlap %, use_cyl(0/1), ratio*100, ransac px, bands
    #     cv.createTrackbar("Overlap%", window, 30, 60, nothing)
    #     cv.createTrackbar("CylWarp",  window, 1, 1, nothing)
    #     cv.createTrackbar("Ratiox100", window, 75, 95, nothing)
    #     cv.createTrackbar("RANSACpx", window, 4, 10, nothing)
    #     cv.createTrackbar("Bands", window, 5, 8, nothing)


    elif mode == "ar":
        cv.createTrackbar("Scale", window, 100, 200, nothing) 
        cv.createTrackbar("RotX", window, 0, 360, nothing)
        cv.createTrackbar("RotY", window, 0, 360, nothing)
        cv.createTrackbar("RotZ", window, 0, 360, nothing)


        

def main():
    global mode_idx, last_mode
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    window = "CV Playground"
    cv.namedWindow(window, cv.WINDOW_NORMAL)

    last_key = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        mode = MODES[mode_idx]

        # Setup trackbars if mode changed
        if mode != last_mode:
            cv.destroyAllWindows()
            cv.namedWindow(window, cv.WINDOW_NORMAL)

            # Reset panorama snapshots when entering pano mode
            if mode == "panorama":
                import cvcore.panorama as panorama
                panorama.reset_snapshots()
            
            setup_trackbars(mode, window)
            last_mode = mode

        # ---------- Mode logic ----------
        if mode == "raw":
            show = frame

        elif mode == "gray":
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            thresh_val = cv.getTrackbarPos("Thresh", window)
            invert = cv.getTrackbarPos("Invert", window)

            if thresh_val > 0:
                _, gray = cv.threshold(gray, thresh_val, 255, cv.THRESH_BINARY)

            if invert == 1:
                gray = cv.bitwise_not(gray)

            show = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

        elif mode == "hsv":
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # Trackbar values
            hmin = cv.getTrackbarPos("Hmin", window)
            smin = cv.getTrackbarPos("Smin", window)
            vmin = cv.getTrackbarPos("Vmin", window)
            hmax = cv.getTrackbarPos("Hmax", window)
            smax = cv.getTrackbarPos("Smax", window)
            vmax = cv.getTrackbarPos("Vmax", window)

            # Apply mask
            lower = (hmin, smin, vmin)
            upper = (hmax, smax, vmax)
            mask = cv.inRange(hsv, lower, upper)
            result = cv.bitwise_and(frame, frame, mask=mask)

            # Split channels
            h, s, v = cv.split(hsv)
            h_vis = cv.applyColorMap(h, cv.COLORMAP_HSV)   # colorful Hue
            s_vis = cv.cvtColor(s, cv.COLOR_GRAY2BGR)      # gray for Saturation
            v_vis = cv.cvtColor(v, cv.COLOR_GRAY2BGR)      # gray for Value

            # Stack channels horizontally
            channels_view = cv.hconcat([h_vis, s_vis, v_vis])

            # Match widths
            # result_resized = cv.resize(result, (channels_view.shape[1], frame.shape[0]))

            # Desired width = same as original frame
            w = frame.shape[1]
            h = frame.shape[0]

            # Resize each channel to 1/3 width of original frame
            h_vis_resized = cv.resize(h_vis, (w // 3, h // 3))
            s_vis_resized = cv.resize(s_vis, (w // 3, h // 3))
            v_vis_resized = cv.resize(v_vis, (w // 3, h // 3))
            # Original frame size
            H, W = frame.shape[:2]

            # Resize each channel to 1/3 width, keeping height ratio
            small_w = W // 3
            small_h = int(H * (small_w / W))  # keep aspect ratio

            h_vis_resized = cv.resize(h_vis, (small_w, small_h))
            s_vis_resized = cv.resize(s_vis, (small_w, small_h))
            v_vis_resized = cv.resize(v_vis, (small_w, small_h))

            # Concatenate bottom row
            channels_view = cv.hconcat([h_vis_resized, s_vis_resized, v_vis_resized])

            # Resize bottom row width to match result width
            channels_view = cv.resize(channels_view, (W, channels_view.shape[0]))

            # Stack top (full result) + bottom (channels)
            combined = cv.vconcat([result, channels_view])

            show = combined
            # channels_view = cv.hconcat([h_vis_resized, s_vis_resized, v_vis_resized])

            # # Resize result (top) to same width
            # result_resized = cv.resize(result, (w, h))

            # # Match height of channels row to ~1/3 of frame
            # channels_view = cv.resize(channels_view, (w, h // 3))

            # combined = cv.vconcat([result_resized, channels_view])

            # show = combined

        # elif mode == "hsv":
        #     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        #     hmin = cv.getTrackbarPos("Hmin", window)
        #     smin = cv.getTrackbarPos("Smin", window)
        #     vmin = cv.getTrackbarPos("Vmin", window)
        #     hmax = cv.getTrackbarPos("Hmax", window)
        #     smax = cv.getTrackbarPos("Smax", window)
        #     vmax = cv.getTrackbarPos("Vmax", window)

        #     lower = (hmin, smin, vmin)
        #     upper = (hmax, smax, vmax)
        #     mask = cv.inRange(hsv, lower, upper)
        #     result = cv.bitwise_and(frame, frame, mask=mask)

        #     show = result

        elif mode == "exposure":

            # Trackbar values
            b_val = cv.getTrackbarPos("Brightness", window) - 50  # range -50..+50
            c_val = cv.getTrackbarPos("Contrast", window) - 50    # range -50..+50

            # Apply adjustment
            frame_float = frame.astype(np.float32)
            alpha = 1.0 + (c_val / 50.0)   # contrast factor
            beta = b_val * 2               # brightness offset
            show = cv.convertScaleAbs(frame_float, alpha=alpha, beta=beta)

            # import cvcore.exposure as exposure
            # show = exposure.adjust_bc(frame)

        elif mode == "hist":
            import cvcore.histogram as histogram
            show = histogram.show_hist(frame)

        elif mode == "gaussian":
            import cvcore.filtering as filtering
            ksize = cv.getTrackbarPos("Kernel", window) | 1   # force odd
            sigma = cv.getTrackbarPos("Sigma", window)
            show = filtering.gaussian_blur(frame, ksize, sigma)

        elif mode == "bilateral":
            import cvcore.filtering as filtering
            d = cv.getTrackbarPos("d", window)
            sc = cv.getTrackbarPos("SigmaColor", window)
            ss = cv.getTrackbarPos("SigmaSpace", window)
            show = filtering.bilateral_blur(frame, d, sc, ss)

        elif mode == "canny":
            import cvcore.edges as edges
            low = cv.getTrackbarPos("Low", window)
            high = cv.getTrackbarPos("High", window)
            show = edges.detect_edges(frame, low, high)

        elif mode == "hough":
            import cvcore.hough as hough
            thr = cv.getTrackbarPos("Threshold", window)
            minL = cv.getTrackbarPos("MinLen", window)
            maxG = cv.getTrackbarPos("MaxGap", window)
            show = hough.detect_lines(frame, threshold=thr,
                                      minLineLength=minL, maxLineGap=maxG)

        elif mode == "transform":
            import cvcore.geometry as geometry
            tx = cv.getTrackbarPos("Tx", window) - 100
            ty = cv.getTrackbarPos("Ty", window) - 100
            angle = cv.getTrackbarPos("Angle", window) - 180
            scale = cv.getTrackbarPos("Scale", window)
            show = geometry.apply_transform(frame, tx, ty, angle, scale)

        elif mode == "panorama":
            import cvcore.panorama as panorama
            # # read UI params
            # ov  = cv.getTrackbarPos("Overlap%", window)
            # cyl = cv.getTrackbarPos("CylWarp", window)
            # rat = cv.getTrackbarPos("Ratiox100", window)
            # ran = cv.getTrackbarPos("RANSACpx", window)
            # bnd = cv.getTrackbarPos("Bands", window)
            # panorama.set_params(ov, cyl, rat, ran, bnd)

            show = panorama.stitch(frame)


        elif mode == "calib":
            import cvcore.calib as calib
            show = calib.run_calibration(frame)

        elif mode == "ar":
            import cvcore.ar as ar
            show = ar.overlay_model(frame)

        else:
            show = frame
        # ---------- End of modes ----------

        # cv.putText(show, f"Mode: {mode}   n=next, p=prev, s=save, ESC=quit",
        #            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        show = draw_text_lines(show, [
            (f"Mode: {mode}", (0,255,0)),
            ("n=next | p=prev | s=save | ESC=quit", (0,255,255))
        ], start_y=30, line_height=30)

        cv.imshow(window, show)

        key = cv.waitKey(30) & 0xFF

        if key == 27 and last_key == 27:   # quit only on double ESC
            break
        elif key == ord('n'):
            mode_idx = (mode_idx + 1) % len(MODES)
        elif key == ord('p'):
            mode_idx = (mode_idx - 1) % len(MODES)
        
        elif key == ord('c') and mode == "panorama":
            import cvcore.panorama as panorama
            panorama.capture_frame(frame)

        elif key == ord('r') and mode == "panorama":
            import cvcore.panorama as panorama
            panorama.reset_snapshots()

        elif key == ord('s'):
            filename = f"output_{mode}_{int(time.time())}.png"
            cv.imwrite(filename, show)
            print("💾 Saved:", filename)

        last_key = key

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
