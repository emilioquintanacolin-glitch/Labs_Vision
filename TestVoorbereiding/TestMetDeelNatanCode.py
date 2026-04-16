import cv2
import numpy as np

# calirbreren en grootte in mm per pixel vinden
def calibrate_mm_pixel(img_path, board_dims, square_mm, save_path):


    image = cv2.imread(img_path,1)
    if image is None:
        print("Afbeelding kon niet geladen worden.")
        return None

    result_img = image.copy()
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, pts = cv2.findChessboardCorners(
        gray_img, (board_dims[0], board_dims[1])
    )

    if not found:
        print("Geen checkerboard gevonden.")
        return None

    # nauwkeurigere hoekpunten
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_pts = cv2.cornerSubPix(gray_img, pts, (11, 11), (-1, -1), term)

    # afstand tussen eerste twee punten
    pixel_distance = np.linalg.norm(refined_pts[0] - refined_pts[1])

    mm_per_pixel = square_mm / pixel_distance

    # visualisatie
    cv2.drawChessboardCorners(result_img, board_dims, refined_pts, found)
    cv2.putText(result_img, f"{mm_per_pixel:.4f} mm/px",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(save_path, result_img)

    cv2.imshow("Calibratie", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"mm/px ratio: {mm_per_pixel:.5f}")
    return mm_per_pixel


scale = calibrate_mm_pixel(
    'test.png',
    (7, 5),
    19,
    'test_annotated.jpg'
)