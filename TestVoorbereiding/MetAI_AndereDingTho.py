import numpy as np
import cv2
import glob

# --- CONFIGURATION ---
CHECKERBOARD = (13, 9)  # Internal corners (width, height)
SQUARE_SIZE = 20  # Size of a square in your preferred unit (e.g., 25mm)
IMAGES_PATH = 'phonecamera.jpg'
SAVE_FILE = "calibration_params.yaml"


def calibrate_and_save():
    # Prepare object points based on the real-world geometry of the board
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(IMAGES_PATH)

    if not images:
        print("No images found! Check your IMAGES_PATH.")
        return

    gray = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            # Refine corners for sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Perform the actual calibration
    print("Calibrating... please wait.")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        # Save results to YAML
        fs = cv2.FileStorage(SAVE_FILE, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", mtx)
        fs.write("distortion_coefficients", dist)
        fs.release()
        print(f"Calibration successful. Parameters saved to {SAVE_FILE}")

    return mtx, dist


def load_and_undistort(image_path, mtx=None, dist=None):
    # If mtx/dist aren't provided, load them from the file
    if mtx is None or dist is None:
        fs = cv2.FileStorage(SAVE_FILE, cv2.FILE_STORAGE_READ)
        mtx = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("distortion_coefficients").mat()
        fs.release()

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Get the optimal matrix to handle image borders
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort
    dst = cv2.undistort(img, mtx, dist, None, new_mtx)

    # Optional: Crop the result based on the Region of Interest (ROI)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    return img, dst


# --- EXECUTION ---
if __name__ == "__main__":
    # Step 1: Calibrate once and save
    mtx, dist = calibrate_and_save()

    # Step 2: Test it on a single image
    # Replace 'test_image.jpg' with a real file path
    try:
        original, corrected = load_and_undistort('phonecamera.jpg', mtx, dist)
        #cv2.imshow('Before (Distorted)', original)
        #cv2.imshow('After (Corrected)', corrected)

        cv2.namedWindow('Before (Distorted)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Before (Distorted)', 800, 600)  # Set display size
        cv2.imshow('Before (Distorted)', original)

        cv2.namedWindow('After (Corrected)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('After (Corrected)', 800, 600)  # Set display size
        cv2.imshow('After (Corrected)', original)


        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Could not run test image: {e}")