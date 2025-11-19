import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # mirror like a selfie

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
