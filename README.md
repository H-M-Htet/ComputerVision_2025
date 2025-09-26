# CV Playground Application

This is an **interactive computer vision test app** built with **Python + OpenCV**.  
It connects to your webcam and lets you try out different vision techniques in real time —  
from filters and edge detection to panorama stitching, camera calibration, and AR overlays.

---

## ✨ Features

- **Raw / Gray / HSV** views with live controls  
- **Filtering**: Gaussian blur, Bilateral filter  
- **Edges**: Canny edge detection, Hough line detection  
- **Geometric transforms**: Translate, Rotate, Scale with trackbars  
- **Exposure mode**: Adjust brightness & contrast with trackbars  
- **Histogram view**: See intensity distributions  
- **Panorama mode**:  
  - Capture up to **4 images** clockwise (Left → Center → Right)  
  - On-screen guides and thumbnails  
  - Automatic stitching into one panorama  
- **Calibration mode**:  
  - Detects a chessboard pattern (9×6 inner corners)  
  - Captures frames automatically every 2 seconds  
  - Runs full camera calibration once enough images are collected  
  - Saves calibration data to `calibration.npz`  
- **AR mode**:  
  - Detects ArUco markers  
  - Overlays 3D cube + axis on marker (if calibration available)  
  - On-screen instructions guide usage

---

## 🛠 Requirements

- Python 3.9+  
- OpenCV (with contrib modules)  
- NumPy  

Install with:

```bash
pip install opencv-contrib-python numpy
python app.py
🎮 Controls
Keyboard shortcuts:
n → next mode
p → previous mode
s → save current output image
ESC twice → quit
Mode-specific controls:
Panorama:
c → capture frame (follow on-screen guide, rotate clockwise)
r → reset panorama
Calibration:
Shows chessboard corners in real time
Captures automatically (1 frame every 2 seconds)
After enough frames, calibration runs and saves results
AR:
Point at ArUco marker
If calibration is loaded, cube will align correctly
Use on-screen sliders to adjust rotation/scale

CV_A1_app/
│── app.py               # main application (mode switching, UI)
│── README.md            # project documentation
│── cvcore/
│    ├── panorama.py     # panorama stitching
│    ├── calib.py        # camera calibration
│    ├── ar.py           # AR overlay with ArUco
│    ├── filtering.py    # gaussian + bilateral filters
│    ├── edges.py        # canny edge detection
│    ├── hough.py        # hough line transform
│    ├── histogram.py    # histogram visualization
│    └── geometry.py     # transforms (rotate, translate, scale)


#Example Workflow
Start in raw mode → press n to cycle through modes.
Go to exposure mode → use sliders to brighten/darken.
Try panorama mode → capture 3–4 frames clockwise and see them stitched.
Switch to calibration → show chessboard pattern, wait until it completes.
Finally, test AR mode with a printed ArUco marker.

✅ Notes
For calibration, print a 9×6 inner-corner chessboard (10×7 squares).
Panorama works best if you rotate smoothly clockwise.
AR requires a calibration file (calibration.npz) for best accuracy.