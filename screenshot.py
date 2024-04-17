import numpy as np
import pyautogui
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [4, 8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # accessing the landmarks by their position
            lm_list = []
            for lm in hand_landmark.landmark:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # array to hold true or false if finger is folded
            finger_fold_status = []
            for tip in finger_tips:
                # getting the landmark tip position and drawing blue circle
                x, y = lm_list[tip]
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                # writing condition to check if finger is folded i.e checking if finger tip starting value is smaller than finger starting position which is inner landmark. for index finger    
                # if finger folded changing color to green
                if lm_list[tip][0] < lm_list[tip - 1][0]:
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # checking if all fingers are folded
            if all(finger_fold_status):
                # Draw lines on the hand
                for connection in mp_hands.HAND_CONNECTIONS:
                    x_start, y_start = lm_list[connection[0]]
                    x_end, y_end = lm_list[connection[1]]
                    cv2.line(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

                # Take a screenshot and save it
                screenshot = pyautogui.screenshot()
                screenshot.save('screenshot.png')

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("hand tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
