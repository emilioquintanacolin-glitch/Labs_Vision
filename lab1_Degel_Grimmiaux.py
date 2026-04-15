import cv2
import numpy as np

# calirbreren en grootte in mm per pixel vinden
def calibrate_mm_pixel(img_path, board_dims, square_mm, save_path):


    image = cv2.imread(img_path)
    if image is None:
        print("Afbeelding kon niet geladen worden.")
        return None

    result_img = image.copy()
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, pts = cv2.findChessboardCorners(
        gray_img, (board_dims[0] - 1, board_dims[1] - 1)
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
    'checkerboard_not_annotated_3.png',
    (9, 7),
    16.5,
    'calibration_annotated_3.png'
)

#coins bepalen op basis van diameter
def analyse_coins(img_path, save_path, scale):
    image = cv2.imread(img_path)
    if image is None:
        print("Afbeelding niet gevonden.")
        return

    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    #AANGEPAST: rechter 1/4 uitsluiten zodat namen niet meegerekend worden
    h, w = gray.shape
    roi_mask = np.zeros_like(gray)
    roi_mask[:, :int(0.75 * w)] = 255
    gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    # ------------------------------------------------------

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=400
    )

    total_value = 0

    if circles is not None:
        circles = np.uint16(np.round(circles))

        for c in circles[0]:
            x, y, r = c

            diameter_mm = 2 * r * scale

            # muntgroottes
            sizes = {
                0.05: 21.25,
                0.10: 19.75,
                0.20: 22.25,
                0.50: 24.25,
                1.00: 23.25,
                2.00: 25.75
            }

            tol = 0.35
            coin_val = 0
            draw_color = (0, 0, 255)

            for val, ref_size in sizes.items():
                if ref_size - tol <= diameter_mm <= ref_size + tol:
                    coin_val = val
                    total_value += val
                    draw_color = (0, 255, 0)
                    break

            cv2.circle(output, (x, y), r, draw_color, 3)
            cv2.circle(output, (x, y), 2, (255, 255, 0), 3)

            label = f"{diameter_mm:.2f} mm | {coin_val:.2f} EUR"
            cv2.putText(output, label, (x - 140, y + r + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)

        print(f"Aantal munten: {len(circles[0])}")
        print(f"Totaal: {total_value:.2f} EUR")

    else:
        print("Geen cirkels gevonden.")

    cv2.putText(output, f"Totaal: {total_value:.2f} EUR",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(save_path, output)

    cv2.imshow("Resultaat", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


analyse_coins(
    'coins_not_annotated_6.png',
    'coins_annotated_6.png',
    scale
)