import math

# Track previous hand position for swipe detection
prev_positions = {
    "right_x": None
}

# -----------------------------
# Basic helper functions
# -----------------------------

def distance(a, b):
    return math.dist([a.x, a.y], [b.x, b.y])


# -----------------------------
# MINUS (left pinch)
# -----------------------------
def is_pinch(landmarks):
    """Thumb + index tip close."""
    thumb = landmarks.landmark[4]
    index = landmarks.landmark[8]
    return distance(thumb, index) < 0.05


# -----------------------------
# EQUALS (thumbs up)
# -----------------------------
def is_thumbs_up(landmarks):
    """Thumb up, other fingers curled."""
    thumb_up = landmarks.landmark[4].y < landmarks.landmark[3].y

    other_down = True
    # Index, middle, ring, pinky
    for tip, pip in zip([8,12,16,20], [6,10,14,18]):
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
            other_down = False
            break

    return thumb_up and other_down


# -----------------------------
# CLEAR (right-hand swipe left)
# -----------------------------
def is_swipe_left(lm_right):
    """Detect right-hand leftward swipe motion."""
    global prev_positions
    
    if lm_right is None:
        prev_positions["right_x"] = None
        return False

    wrist = lm_right.landmark[0].x
    prev_x = prev_positions["right_x"]
    prev_positions["right_x"] = wrist

    if prev_x is None:
        return False

    # negative Δx = moved left
    return wrist - prev_x < -0.03


# -----------------------------
# ADDITION (index fingers touch)
# -----------------------------
def index_touch(lm_left, lm_right):
    if lm_left is None or lm_right is None:
        return False

    idxL = lm_left.landmark[8]
    idxR = lm_right.landmark[8]

    return distance(idxL, idxR) < 0.05


# -----------------------------
# DIVISION (right-hand slash tilt)
# -----------------------------
def is_slash(lm_right):
    """Right-hand diagonal slash (~60° angle)."""
    if lm_right is None:
        return False
    
    wrist = lm_right.landmark[0]
    mid = lm_right.landmark[9]

    dx = mid.x - wrist.x
    dy = mid.y - wrist.y

    angle = abs(math.degrees(math.atan2(dy, dx)))

    # approx slash angle
    return 110 < angle < 160
