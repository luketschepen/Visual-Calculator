Visual Calculator — READ ME

Project Vision:
Create an intuitive, hands-free calculator controlled entirely through natural hand gestures using a laptop webcam. The system detects both hands simultaneously, interprets finger counts and dynamic gestures, and performs real-time arithmetic. The goal is to make calculator input feel seamless, visual, and accessible without relying on physical buttons or touch devices.

Core Technologies
MediaPipe Hands

Tracks two hands at once in real time.

Provides 21 landmark points per hand.

Detects finger up/down states and gesture patterns.

OpenCV

Streams the webcam.

Draws the calculator UI, current inputs, and results.

Overlays gesture visualizations, debug data, and indicators.

Python

Implements detection logic, gesture mapping, state machine, math parsing, and UI drawing.

Completed Design Decisions
1. Two-Hand Finger Counting for Digits (0–9)

Number input is calculated as:
digit = left_hand_fingers_up + right_hand_fingers_up

Examples:

3 = 1 finger left + 2 fingers right

8 = 4 + 4

0 = fists on both hands

Why: Simple, intuitive, instantly learnable by any calculator user.

2. Core Gesture Set (Operators + Actions)
Action	Gesture	Rationale
Add (+)	Both hands open (5+5) for >0.5s	High visibility / minimal false positives
Subtract (–)	Left-hand pinch (thumb + index)	“Taking away” motion
Multiply (×)	Both index fingers forming an “X” shape	Matches symbol conceptually
Divide (÷)	Right-hand pinch	Mirror of subtract gesture, easy to differentiate
Equals (=)	Thumbs-up	Natural confirmation gesture
Clear (C)	Both hands in fists	Universal reset motion

These gestures are chosen for clarity, distinctness, and easy reproducibility for first-time users.

3. Calculator Interaction State Machine

The system cycles through:

Waiting for Number

Waiting for Operator

Waiting for Second Number

Show Result

Resume input for chaining operations

This avoids ambiguity and provides predictable behavior.

4. UI Strategy

Use OpenCV to draw:

Current input

Stored value

Chosen operator

Result

Optional debug overlay (finger counts, detected gestures)

This keeps the project lightweight with no external GUI libraries.

Key Decisions Remaining
A. UI Design Finalization

Minimalistic black-and-white?

Color-coded input boxes?

Show a history panel or only current expression?

B. Gesture Confirmation

Decide whether to:

automatically accept gestures when detected,
or

require a brief “hold” time (0.3–0.5s) to avoid accidental operations.

C. Error Handling

What should happen if:

User performs an unrecognized gesture?

Two gestures appear simultaneously?

Hands temporarily leave the frame?

Possible options:

Ignore input

Display a warning overlay

Pause detection until hands return

D. Extended Functionality (future)

Decide which features to add later:

Decimal numbers

Parentheses / multi-step expressions

Backspace gesture

Scientific functions (sin, cos, sqrt, etc.)

Voice confirmation (“equals”, “clear”)

E. Visual Feedback / UX

Should the UI show:

Detected gesture thumbnails?

Left and right hand outlines?

Live number mapping (e.g., “Detected: 7”)?

Current Status

✔ Tech stack chosen
✔ Input method defined
✔ Gesture set defined
✔ Ready to implement MediaPipe + OpenCV pipeline
✔ Project structure established
✔ Development environment configured

Immediate Next Steps

Validate webcam feed & FPS using test_cam.py

Add hand tracking + finger counting

Implement two-hand finger summation

Add gesture classification

Link gestures to calculator state machine

Draw the real-time calculator UI
