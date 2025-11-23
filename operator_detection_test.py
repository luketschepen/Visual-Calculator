import cv2
import mediapipe as mp

from gesture_logic import (
    is_pinch,
    is_thumbs_up,
    is_swipe_left,
    index_touch,
    is_slash
)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Persistence counters
hold_count = {
    "PLUS": 0,
    "MINUS": 0,
    "DIVIDE": 0,
    "EQUALS": 0,
    "CLEAR": 0
}


def detect_operator(lm_left, lm_right):
    """Return operator string or None."""
    global hold_count

    operator = None

    # --------------------------------------
    # PLUS (index fingers touch)
    # --------------------------------------
    if index_touch(lm_left, lm_right):
        hold_count["PLUS"] += 1
    else:
        hold_count["PLUS"] = 0

    # --------------------------------------
    # MINUS (left pinch)
    # --------------------------------------
    if lm_left and is_pinch(lm_left):
        hold_count["MINUS"] += 1
    else:
        hold_count["MINUS"] = 0

    # --------------------------------------
    # DIVIDE (right slash angle)
    # --------------------------------------
    if lm_right and is_slash(lm_right):
        hold_count["DIVIDE"] += 1
    else:
        hold_count["DIVIDE"] = 0

    # --------------------------------------
    # CLEAR (right swipe left)
    # no hold needed
    # --------------------------------------
    if is_swipe_left(lm_right):
        return "CLEAR"

    # --------------------------------------
    # EQUALS (thumbs up)
    # --------------------------------------
    if (lm_left and is_thumbs_up(lm_left)) or (lm_right and is_thumbs_up(lm_right)):
        hold_count["EQUALS"] += 1
    else:
        hold_count["EQUALS"] = 0

    # --------------------------------------
    # Threshold for stable detection
    # --------------------------------------
    for op, count in hold_count.items():
        if count > 6:  # ~0.1s hold
            return op

    return None


# ----------------------------------------------------------
# Main test program
# ----------------------------------------------------------

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        lm_left = None
        lm_right = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label  # "Left" or "Right"

                if label == "Left":
                    lm_left = lm
                else:
                    lm_right = lm

                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # detect operator
        operator = detect_operator(lm_left, lm_right)

        # draw result
        if operator:
            cv2.putText(frame, f"OPERATOR: {operator}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

        cv2.imshow("Operator Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
