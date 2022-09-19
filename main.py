import numpy as np
import cv2
import typing as tp

captured_video = cv2.VideoCapture("pens.mov")

red_lower_bound = np.array([0, 50, 50], np.uint8)
red_upper_bound = np.array([10, 255, 255], np.uint8)

red1_lower_bound = np.array([170, 50, 50], np.uint8)
red1_upped_bound = np.array([180, 255, 255], np.uint8)

blue_lower_bound = np.array([94, 0, 150], np.uint8)
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

        captured_mask_for_first_object = cv2.inRange(hsv_frame, first_lower_color_bound, first_upper_color_bound)

        red_mask1 = cv2.inRange(hsv_frame, red_first_lower_color_bound, red_first_upped_color_bound)
        red_mask2 = cv2.inRange(hsv_frame, red_second_lower_color_bound, red_second_upper_color_bound)

        red_mask = red_mask1 + red_mask2

        first_contours, _ = cv2.findContours(captured_mask_for_first_object, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        second_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in first_contours:
            area = cv2.contourArea(contour)

            if area > 6000:
                cv2.drawContours(captured_frame, contour, -1, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(captured_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        for contour in second_contours:
            area = cv2.contourArea(contour)

            if area > 6000:
                cv2.drawContours(captured_frame, contour, -1, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(captured_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


        cv2.imshow("Captured Frame", captured_frame)
        # cv2.imshow("mask1", captured_mask_for_first_object)
        # cv2.imshow("mask2", red_mask)
        cv2.waitKey(25)

        if cv2.waitKey(1) == ord("q"):
            break

    captured_video.release()
    cv2.destroyAllWindows()

track_pens_in_captured_video(captured_video, blue_lower_bound, blue_upper_bound, red_lower_bound, red_upper_bound, red1_lower_bound, red1_upped_bound)