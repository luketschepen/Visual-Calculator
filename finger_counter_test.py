import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Finger landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

def count_fingers(hand_landmarks, handedness):
    fingers = 0

    # Thumb: depends on left vs right hand
    if handedness == "Right":
        if hand_landmarks.landmark[FINGER_TIPS[0]].x < hand_landmarks.landmark[FINGER_PIPS[0]].x:
            fingers += 1
    else:  # Left hand (mirrored)
        if hand_landmarks.landmark[FINGER_TIPS[0]].x > hand_landmarks.landmark[FINGER_PIPS[0]].x:
            fingers += 1

    # Other fingers: tip above pip means finger is extended
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers += 1

    return fingers


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers_up = count_fingers(hand_landmarks, label)

                cv2.putText(frame, f"{label}: {fingers_up}",
                            (10, 60 if label == "Left" else 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                print(label, ":", fingers_up)

        cv2.imshow("Finger Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
