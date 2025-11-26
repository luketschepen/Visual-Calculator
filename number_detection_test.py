import cv2
import mediapipe as mp
import math

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
    
def index_mid_touch(lm_left, lm_right, threshold=0.05):
    # Gesture for Multiplication: Cross your index fingers
    # Landmark 7 is the middle vertex of the index finger
    p1 = lm_left.landmark[7]
    p2 = lm_right.landmark[7]
    distance = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
    return distance < threshold

def index_tip_touch(lm_left, lm_right, threshold=0.05):
    # Gesture for Addition: Touch your index finger tips together
    # Landmark 8 is the tip of the index finger
    p1 = lm_left.landmark[8]
    p2 = lm_right.landmark[8]
    distance = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
    return distance < threshold

def thumbs_up(hand_landmarks, pinky_threshold=0.07, thumb_apart_threshold=0.1):
    pinky_tip = hand_landmarks.landmark[20]
    palm_pinky_base = hand_landmarks.landmark[17]
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]

    # Pinky curled: close distance between tip and base
    pinky_curled = ((pinky_tip.x - palm_pinky_base.x)**2 + (pinky_tip.y - palm_pinky_base.y)**2)**0.5 < pinky_threshold
    
    # Thumb extended up: tip and base apart, and tip above base (lower y)
    thumb_extended = ((thumb_tip.x - thumb_base.x)**2 + (thumb_tip.y - thumb_base.y)**2)**0.5 > thumb_apart_threshold
    thumb_up = thumb_tip.y < thumb_base.y
    
    return pinky_curled and thumb_extended and thumb_up

def thumbs_down(hand_landmarks, pinky_threshold=0.07, thumb_apart_threshold=0.1):
    pinky_tip = hand_landmarks.landmark[20]
    palm_pinky_base = hand_landmarks.landmark[17]
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]

    # Pinky curled: close distance between tip and base (same as thumbs up)
    pinky_curled = ((pinky_tip.x - palm_pinky_base.x)**2 + (pinky_tip.y - palm_pinky_base.y)**2)**0.5 < pinky_threshold
    
    # Thumb extended down: tip and base apart, and tip below base (higher y)
    thumb_extended = ((thumb_tip.x - thumb_base.x)**2 + (thumb_tip.y - thumb_base.y)**2)**0.5 > thumb_apart_threshold
    thumb_down = thumb_tip.y > thumb_base.y
    
    return pinky_curled and thumb_extended and thumb_down

def backspace(hand_landmarks, pinky_threshold=0.07, thumb_apart_threshold=0.1, threshold=0.08):
    pinky_tip = hand_landmarks.landmark[20]
    palm_pinky_base = hand_landmarks.landmark[17]
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    index_tip = hand_landmarks.landmark[8]
    index_base = hand_landmarks.landmark[5]

    # Pinky curled: close distance between tip and base
    pinky_curled = ((pinky_tip.x - palm_pinky_base.x)**2 + (pinky_tip.y - palm_pinky_base.y)**2)**0.5 < pinky_threshold
    
    # Thumb extended up: tip and base apart, and tip above base (lower y)
    thumb_extended = ((thumb_tip.x - thumb_base.x)**2 + (thumb_tip.y - thumb_base.y)**2)**0.5 > thumb_apart_threshold
    thumb_up = thumb_tip.y < thumb_base.y

    index_extended = ((index_tip.x - index_base.x)**2 + (index_tip.y - index_base.y)**2)**0.5 > threshold
    
    return pinky_curled and thumb_extended and thumb_up and index_extended

def backspace_upsidedown(hand_landmarks, pinky_threshold=0.07, thumb_apart_threshold=0.1, threshold=0.08):
    pinky_tip = hand_landmarks.landmark[20]
    palm_pinky_base = hand_landmarks.landmark[17]
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    index_tip = hand_landmarks.landmark[8]
    index_base = hand_landmarks.landmark[5]

    # Pinky curled: close distance between tip and base (same as thumbs up)
    pinky_curled = ((pinky_tip.x - palm_pinky_base.x)**2 + (pinky_tip.y - palm_pinky_base.y)**2)**0.5 < pinky_threshold
    
    # Thumb extended down: tip and base apart, and tip below base (higher y)
    thumb_extended = ((thumb_tip.x - thumb_base.x)**2 + (thumb_tip.y - thumb_base.y)**2)**0.5 > thumb_apart_threshold
    thumb_down = thumb_tip.y > thumb_base.y

    index_extended = ((index_tip.x - index_base.x)**2 + (index_tip.y - index_base.y)**2)**0.5 > threshold

    return pinky_curled and thumb_extended and thumb_down and index_extended

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
        index_mid_touch_detected = False
        index_tip_touch_detected = False
        thumbs_up_detected = False
        thumbs_down_detected = False
        backspace_detected = False
        backspace_upsidedown_detected = False

        lm_left = None
        lm_right = None # Pre-initialize left and right hand landmarks

        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, hand_type in zip(results.multi_hand_landmarks,
                                     results.multi_handedness):
                label = hand_type.classification[0].label

                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                count = count_fingers(lm, label)
                if label == "Left":
                    left_count = count
                    lm_left = lm
                else:
                    right_count = count
                    lm_right = lm

                if is_pinch(lm):
                    pinch_detected = True

                if is_thumb_zero(lm):
                    zero_detected = True

                if thumbs_up(lm):
                    thumbs_up_detected = True

                if thumbs_down(lm):
                    thumbs_down_detected = True

                if backspace(lm):
                    backspace_detected = True

                if backspace_upsidedown(lm):
                    backspace_upsidedown_detected = True

        if lm_left and lm_right:
            if index_mid_touch(lm_left, lm_right):
                index_mid_touch_detected = True
            if index_tip_touch(lm_left, lm_right):
                index_tip_touch_detected = True    
        if zero_detected:
            display_text = "Number: zero"
        elif pinch_detected:
            display_text = "Number: pinch"
        elif index_mid_touch_detected:
            display_text = "Number: multiply"
        elif index_tip_touch_detected:
            display_text = "Number: add"
        elif backspace_detected:
            display_text = "Number: backspace"
        elif backspace_upsidedown_detected:
            display_text = "Number: backspace upsidedown"
        elif thumbs_up_detected:
            display_text = "Number: thumbs up"
        elif thumbs_down_detected:
            display_text = "Number: thumbs down"  
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
