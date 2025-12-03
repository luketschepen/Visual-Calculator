import cv2
import mediapipe as mp
import math
import numpy as np
import time

# ==========================================
# 1. CALCULATOR BACKEND (LOGIC)
# ==========================================
class Calculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_input = ""
        self.expression = []
        self.result = ""
        self.last_op = ""
        self.error = False

    def input_digit(self, digit):
        # digit is a string '0'..'9'
        if self.result:
            # Start a new expression if user begins typing after result
            self.reset()
        if len(self.current_input) < 10:
            if digit == '0' and self.current_input == '0':
                return
            self.current_input += digit

    def input_op(self, op):
        # If we have a result, use it as the start of new math
        if self.result and not self.current_input:
            self.expression = [self.result, op]
            self.result = ""
            self.last_op = op
            self.error = False
            return

        # Standard operation entry
        if self.current_input:
            self.expression.append(self.current_input)
            self.expression.append(op)
            self.current_input = ""
            self.last_op = op
        elif self.expression and self.expression[-1] in "+-*/":
            # Overwrite last operator if user changes mind
            self.expression[-1] = op
            self.last_op = op

    def backspace(self):
        if self.current_input:
            self.current_input = self.current_input[:-1]

    def evaluate(self):
        if self.current_input:
            self.expression.append(self.current_input)

        if not self.expression:
            return

        try:
            full_expr = "".join(self.expression)
            val = eval(full_expr)

            # Pretty formatting
            if abs(val) > 99999999:
                self.result = "OVERFLOW"
                self.error = True
            elif int(val) == val:
                self.result = str(int(val))
            else:
                self.result = f"{val:.4f}".rstrip('0').rstrip('.')

            self.current_input = ""
            self.expression = []
            self.last_op = "="
            self.error = False
        except Exception:
            self.result = "ERROR"
            self.error = True
            self.current_input = ""
            self.expression = []
            self.last_op = ""

    def get_display_text(self):
        if self.error:
            return self.result
        if self.result:
            return self.result
        if self.current_input:
            return self.current_input
        if self.expression:
            # Show last operator if waiting for next number
            return self.last_op or "0"
        return "0"


# ==========================================
# 2. 14-SEGMENT DISPLAY RENDERER
# ==========================================
def draw_segment_display(img, text, x, y, scale=1.0):
    # Professional VFD Style Colors
    COLOR_BG = (20, 20, 20)      # Nearly black
    COLOR_OFF = (40, 45, 45)     # Faint grey-teal (ghost segments)
    COLOR_ON = (0, 255, 255)     # Bright Cyan
    COLOR_GLOW = (0, 100, 100)   # Dimmer Cyan for glow effect

    char_w = 40 * scale
    char_h = 70 * scale
    spacing = 15 * scale
    slant = 10 * scale
    thickness = max(2, int(2 * scale))

    # Background box
    box_w = int(len(text) * (char_w + spacing) + 40)
    box_h = int(char_h + 40)
    cv2.rectangle(img, (int(x), int(y)), (int(x + box_w), int(y + box_h)), COLOR_BG, -1)
    cv2.rectangle(img, (int(x), int(y)), (int(x + box_w), int(y + box_h)), (100, 100, 100), 2)

    def get_seg_coords(sx, sy):
        w, h = char_w, char_h
        return {
            'a':  ((sx + slant,        sy),         (sx + w + slant,       sy)),
            'b':  ((sx + w + slant,    sy),         (sx + w + slant/2,     sy + h/2)),
            'c':  ((sx + w + slant/2,  sy + h/2),   (sx + w,               sy + h)),
            'd':  ((sx,                sy + h),     (sx + w,               sy + h)),
            'e':  ((sx + slant/2,      sy + h/2),   (sx,                   sy + h)),
            'f':  ((sx + slant,        sy),         (sx + slant/2,         sy + h/2)),
            'g1': ((sx + slant/2,      sy + h/2),   (sx + w/2 + slant/2,   sy + h/2)),
            'g2': ((sx + w/2 + slant/2,sy + h/2),   (sx + w + slant/2,     sy + h/2)),
            'm':  ((sx + w/2 + slant,  sy),         (sx + w/2,             sy + h)),
            'h':  ((sx + slant,        sy),         (sx + w/2 + slant/2,   sy + h/2)),
            'i':  ((sx + w/2 + slant/2,sy + h/2),   (sx + w,               sy)),
            'j':  ((sx + w/2 + slant/2,sy + h/2),   (sx + w,               sy + h)),
            'k':  ((sx + w/2 + slant/2,sy + h/2),   (sx,                   sy + h)),
        }

    charmap = {
        '0': ['a','b','c','d','e','f'],
        '1': ['b','c'],
        '2': ['a','b','g1','g2','e','d'],
        '3': ['a','b','g1','g2','c','d'],
        '4': ['f','g1','g2','b','c'],
        '5': ['a','f','g1','g2','c','d'],
        '6': ['a','f','e','d','c','g1','g2'],
        '7': ['a','b','c'],
        '8': ['a','b','c','d','e','f','g1','g2'],
        '9': ['a','b','c','d','f','g1','g2'],
        '-': ['g1','g2'],
        '+': ['g1','g2','m'],
        '*': ['g1','g2','h','i','j','k'],
        '/': ['i','k'],           
        '=': ['g1','g2','d'],
        'E': ['a','f','g1','g2','e','d'],
        'R': ['a','f','b','g1','g2','e','j'],
        'O': ['a','b','c','d','e','f'],
        ' ': [],
    }

    start_text_x = x + 20
    start_text_y = y + 20

    all_keys = ['a','b','c','d','e','f','g1','g2','m','h','i','j','k']

    for i, ch in enumerate(text):
        cx = start_text_x + i * (char_w + spacing)
        segs = get_seg_coords(cx, start_text_y)

        # Ghost segments
        for k in all_keys:
            p1, p2 = segs[k]
            cv2.line(img, (int(p1[0]), int(p1[1])),
                          (int(p2[0]), int(p2[1])),
                          (40, 45, 45), thickness, cv2.LINE_AA)

        # Active segments
        active = charmap.get(ch, [])
        for k in active:
            p1, p2 = segs[k]
            cv2.line(img, (int(p1[0]), int(p1[1])),
                          (int(p2[0]), int(p2[1])),
                          (0, 100, 100), thickness + 4, cv2.LINE_AA)
            cv2.line(img, (int(p1[0]), int(p1[1])),
                          (int(p2[0]), int(p2[1])),
                          (0, 255, 255), thickness, cv2.LINE_AA)


# ==========================================
# 3. GESTURE RECOGNITION (YOUR WORKING VERSION)
# ==========================================
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
    p1 = lm_left.landmark[7]
    p2 = lm_right.landmark[7]
    distance = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
    return distance < threshold

def index_tip_touch(lm_left, lm_right, threshold=0.05):
    p1 = lm_left.landmark[8]
    p2 = lm_right.landmark[8]
    distance = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
    return distance < threshold

def thumbs_up(hand_landmarks, pinky_threshold=0.07, thumb_apart_threshold=0.1):
    pinky_tip = hand_landmarks.landmark[20]
    palm_pinky_base = hand_landmarks.landmark[17]
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]

    pinky_curled = ((pinky_tip.x - palm_pinky_base.x)**2 + (pinky_tip.y - palm_pinky_base.y)**2)**0.5 < pinky_threshold
    thumb_extended = ((thumb_tip.x - thumb_base.x)**2 + (thumb_tip.y - thumb_base.y)**2)**0.5 > thumb_apart_threshold
    thumb_up_flag = thumb_tip.y < thumb_base.y
    return pinky_curled and thumb_extended and thumb_up_flag

def thumbs_down(hand_landmarks, pinky_threshold=0.07, thumb_apart_threshold=0.1):
    pinky_tip = hand_landmarks.landmark[20]
    palm_pinky_base = hand_landmarks.landmark[17]
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]

    pinky_curled = ((pinky_tip.x - palm_pinky_base.x)**2 + (pinky_tip.y - palm_pinky_base.y)**2)**0.5 < pinky_threshold
    thumb_extended = ((thumb_tip.x - thumb_base.x)**2 + (thumb_tip.y - thumb_base.y)**2)**0.5 > thumb_apart_threshold
    thumb_down_flag = thumb_tip.y > thumb_base.y
    return pinky_curled and thumb_extended and thumb_down_flag

def backspace(hand_landmarks,
              pinky_threshold=0.07,
              thumb_apart_threshold=0.1):
    """
    Placeholder backspace gesture.
    Currently always returns False so it never triggers.
    Later you can replace this body with a real gesture detector
    without changing any other code.
    """
    return False

def backspace_upsidedown(hand_landmarks,
                          pinky_threshold=0.07,
                          thumb_apart_threshold=0.1):
    """
    Placeholder upside-down backspace / clear gesture.
    Currently always returns False so it never triggers.
    Later you can implement a real gesture here.
    """
    return False

# -----------------------------
# DIVISION (right-hand slash tilt)
# -----------------------------
def is_slash(lm_right):
    """Right-hand diagonal slash."""
    if lm_right is None:
        return False
    
    wrist = lm_right.landmark[0]
    mid = lm_right.landmark[9]

    dx = mid.x - wrist.x
    dy = mid.y - wrist.y

    angle = abs(math.degrees(math.atan2(dy, dx)))

    # approx slash angle
    return 110 < angle < 160


# ==========================================
# 4. MAIN APPLICATION WITH HOLD-TO-CONFIRM
# ==========================================
cap = cv2.VideoCapture(0)
calc = Calculator()

# Make the window resizable and set an initial size
cv2.namedWindow("Visual Calculator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Visual Calculator", 1920, 1080)  # or whatever you like

HOLD_DURATION = 2.0  # seconds for ANY input
current_stable_gesture = None
hold_start_time = 0.0
last_feedback = "READY"
progress = 0.0

with mp_hands.Hands(max_num_hands=2) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
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
        slash_detected = False

        lm_left = None
        lm_right = None

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

                # Division: right-hand slash
                if label == "Right" and is_slash(lm):
                    slash_detected = True

        if lm_left and lm_right:
            if index_mid_touch(lm_left, lm_right):
                index_mid_touch_detected = True
            if index_tip_touch(lm_left, lm_right):
                index_tip_touch_detected = True

        # How many hands are actually detected?
        num_hands_detected = int(lm_left is not None) + int(lm_right is not None)

        # Calculate current number from fingers (cap at 10 so 10 = "0")
        total_fingers = left_count + right_count
        if total_fingers > 10:
            total_fingers = 10

        # ----------------------------------
        # GESTURE → ACTION (single label)
        # ----------------------------------
        detected_action = None

        # Case 1: TWO HANDS visible
        if num_hands_detected == 2:
            # Two-hand operators only, never backspace/equals/sub/div here
            if index_tip_touch_detected:
                detected_action = "ADD"
            elif index_mid_touch_detected:
                detected_action = "MULT"
            else:
                # Number entry with two hands (1–10 fingers)
                if results.multi_hand_landmarks and total_fingers > 0:
                    detected_action = f"NUM_{total_fingers}"
                else:
                    detected_action = None

        # Case 2: ONE HAND visible
        elif num_hands_detected == 1:
            # One-hand commands
            if thumbs_down_detected or backspace_upsidedown_detected:
                detected_action = "CLEAR"
            elif backspace_detected:
                detected_action = "BACK"
            elif thumbs_up_detected:
                detected_action = "EQUAL"
            elif pinch_detected:
                detected_action = "SUB"
            elif slash_detected:
                detected_action = "DIV"
            else:
                # One-hand numeric input ONLY if at least 1 finger is up
                if total_fingers > 0:
                    detected_action = f"NUM_{total_fingers}"
                else:
                    detected_action = None

        # Case 3: ZERO hands
        else:
            detected_action = None

        # ----------------------------------
        # HOLD-TO-CONFIRM LOGIC
        # ----------------------------------
        now = time.time()
        if detected_action is None:
            # No stable gesture - reset
            current_stable_gesture = None
            hold_start_time = now
            progress = 0.0
        else:
            if detected_action == current_stable_gesture:
                duration = now - hold_start_time
                progress = min(1.0, duration / HOLD_DURATION)

                if duration >= HOLD_DURATION:
                    # CONFIRMED ACTION
                    if detected_action.startswith("NUM_"):
                        fingers_val = int(detected_action.split("_")[1])

                        # 10 fingers up → digit 0
                        if fingers_val == 10:
                            digit = '0'
                        else:
                            digit = str(fingers_val)

                        calc.input_digit(digit)
                        last_feedback = f"Typed {digit}"

                    elif detected_action == "ADD":
                        calc.input_op("+")
                        last_feedback = "Operator +"
                    elif detected_action == "MULT":
                        calc.input_op("*")
                        last_feedback = "Operator MULT"
                    elif detected_action == "SUB":
                        calc.input_op("-")
                        last_feedback = "Operator -"
                    elif detected_action == "DIV":
                        calc.input_op("/")
                        last_feedback = "Operator /"
                    elif detected_action == "EQUAL":
                        calc.evaluate()
                        last_feedback = "Evaluate ="
                    elif detected_action == "CLEAR":
                        calc.reset()
                        last_feedback = "Clear"
                    elif detected_action == "BACK":
                        calc.backspace()
                        last_feedback = "Backspace"

                    # Prevent re-triggering while holding the same gesture
                    current_stable_gesture = None
                    hold_start_time = now
                    progress = 0.0
            else:
                # New gesture detected, start counting
                current_stable_gesture = detected_action
                hold_start_time = now
                progress = 0.0

        # ----------------------------------
        # DRAW UI
        # ----------------------------------
        # Allow up to 12 display characters
        MAX_DISPLAY_CHARS = 12

        disp_text = calc.get_display_text()
        if len(disp_text) > MAX_DISPLAY_CHARS:
            disp_text = disp_text[:MAX_DISPLAY_CHARS]
        disp_text = disp_text.rjust(MAX_DISPLAY_CHARS, ' ')

        # Compute a scale so the display fits within the frame width
        # Base cell width at scale=1 is (char_w + spacing) = 40 + 15 = 55
        base_cell_width = 55
        base_box_width = MAX_DISPLAY_CHARS * base_cell_width + 40  # +40 padding used in draw_segment_display

        # Leave a bit of margin (e.g. 20px each side)
        available_width = w - 40
        scale = min(1.0, max(0.5, available_width / float(base_box_width)))

        # Draw starting a bit from the left edge
        draw_segment_display(frame, disp_text, 10, 40, scale=scale)


        # Status + debugging
        cv2.putText(frame, f"STATUS: {last_feedback}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if detected_action:
            label_clean = detected_action.replace("NUM_", "Fingers: ")
            text = f"Detecting: {label_clean}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            # Draw white background rectangle with padding
            cv2.rectangle(frame, (20 - 5, h - 30 - text_size[1] - 5), 
                          (20 + text_size[0] + 5, h - 30 + 5), (255, 255, 255), -1)
            cv2.putText(frame, text, (20, h - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

            # Progress bar
            if 0.0 < progress < 1.0:
                bar_w = 250
                bar_x = 20
                bar_y1 = h - 18
                bar_y2 = h - 10
                cv2.rectangle(frame, (bar_x, bar_y1),
                              (bar_x + int(bar_w * progress), bar_y2),
                              (0, 255, 255), -1)

        cv2.imshow("Visual Calculator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
