import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

def count_fingers(hand_landmarks, handedness):
    fingers = 0

    # Thumb
    if handedness == "Right":
        if hand_landmarks.landmark[FINGER_TIPS[0]].x < hand_landmarks.landmark[FINGER_PIPS[0]].x:
            fingers += 1
    else:
        if hand_landmarks.landmark[FINGER_TIPS[0]].x > hand_landmarks.landmark[FINGER_PIPS[0]].x:
            fingers += 1

    # Four other fingers
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers += 1

    return fingers

def is_pinch(hand_landmarks, threshold=0.06):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    return distance < threshold

def is_thumb_zero(hand_landmarks, threshold=0.08):
    thumb_tip = hand_landmarks.landmark[4]
    clustered = 0
    for tip_id in FINGER_TIPS[1:]:
        finger_tip = hand_landmarks.landmark[tip_id]
        distance = ((thumb_tip.x - finger_tip.x)**2 + (thumb_tip.y - finger_tip.y)**2)**0.5
        if distance < threshold:
            clustered += 1
    return clustered == len(FINGER_TIPS[1:])

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2) as hands:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_count = 0
        right_count = 0
        pinch_detected = False
        zero_detected = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, hand_type in zip(results.multi_hand_landmarks,
                                     results.multi_handedness):
                label = hand_type.classification[0].label

                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                count = count_fingers(lm, label)
                if label == "Left":
                    left_count = count
                else:
                    right_count = count

                if is_pinch(lm):
                    pinch_detected = True

                if is_thumb_zero(lm):
                    zero_detected = True

        if zero_detected:
            display_text = "Number: zero"
        elif pinch_detected:
            display_text = "Number: pinch"
        else:
            number = left_count + right_count
            display_text = f"Number: {number}"

        cv2.putText(frame, display_text, (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        cv2.imshow("Number Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
