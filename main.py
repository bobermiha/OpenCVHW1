import numpy as np
import cv2
import typing as tp

captured_video = cv2.VideoCapture("pens.mov")

red_lower_bound = np.array([0, 150, 40], np.uint8)
red_upper_bound = np.array([10, 255, 255], np.uint8)

red1_lower_bound = np.array([175, 150, 60], np.uint8)
red1_upped_bound = np.array([180, 255, 255], np.uint8)

blue_lower_bound = np.array([94, 50, 150], np.uint8)
blue_upper_bound = np.array([135, 255, 255], np.uint8)

def track_pens_in_captured_video(
        input_video: cv2.VideoCapture,
        first_lower_color_bound,
        first_upper_color_bound,
        red_first_lower_color_bound,
        red_first_upped_color_bound,
        red_second_lower_color_bound,
        red_second_upper_color_bound
):
    while input_video.isOpened():

        ret, captured_frame = captured_video.read()

        if not ret:
            print("Failed to capture frames from the video")
            break

        gauss_blurred_frame = cv2.GaussianBlur(captured_frame, (5, 5), 0)
        hsv_frame = cv2.cvtColor(gauss_blurred_frame, cv2.COLOR_BGR2HSV)

        #MARK masks
        blue_mask = cv2.inRange(hsv_frame, first_lower_color_bound, first_upper_color_bound)
        red_mask_lower_range = cv2.inRange(hsv_frame, red_first_lower_color_bound, red_first_upped_color_bound)
        red_mask_upper_range = cv2.inRange(hsv_frame, red_second_lower_color_bound, red_second_upper_color_bound)

        mask = red_mask_lower_range + red_mask_upper_range + blue_mask

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 400:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(captured_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.imshow("Captured Frame", captured_frame)
        cv2.imshow("Mask", mask)
        cv2.waitKey(28)

        if cv2.waitKey(1) == ord("q"):
            break

    captured_video.release()
    cv2.destroyAllWindows()

track_pens_in_captured_video(captured_video,
                             blue_lower_bound,
                             blue_upper_bound,
                             red_lower_bound,
                             red_upper_bound,
                             red1_lower_bound,
                             red1_upped_bound)