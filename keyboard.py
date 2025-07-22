import cv2
import mediapipe as mp
from pynput.keyboard import Controller
import numpy as np
from pynput.keyboard import Key

keyboard = Controller()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Keyboard layout
keys_layout = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM< "),
]

# Settings
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
KEY_WIDTH, KEY_HEIGHT = 60, 60
start_x, start_y = 100, 500
typed_text = ""
cooldowns = {'Left': 0, 'Right': 0}

# Draw keyboard and return key zones
def draw_keyboard(img):
    key_zones = {}
    y = start_y
    for row in keys_layout:
        x = start_x
        for key in row:
            cv2.rectangle(img, (x, y), (x + KEY_WIDTH, y + KEY_HEIGHT), (200, 200, 200), -1)
            cv2.rectangle(img, (x, y), (x + KEY_WIDTH, y + KEY_HEIGHT), (50, 50, 50), 2)
            cv2.putText(img, key, (x + 15, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            key_zones[key] = (x, y, x + KEY_WIDTH, y + KEY_HEIGHT)
            x += KEY_WIDTH + 10
        y += KEY_HEIGHT + 10
    return key_zones

# Detect key press
def get_pressed_key(fx, fy, zones):
    for key, (x1, y1, x2, y2) in zones.items():
        if x1 <= fx <= x2 and y1 <= fy <= y2:
            return key
    return None

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    key_zones = draw_keyboard(frame)
    cv2.putText(frame, "Typed: " + typed_text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (handLms, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            label = handedness.classification[0].label  # "Left" or "Right"
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            index_tip = handLms.landmark[8]
            index_pip = handLms.landmark[6]

# Get fingertip coordinates
            fx, fy = int(index_tip.x * w), int(index_tip.y * h)

# Simulate "key press" gesture when finger is pressing down
            finger_depth = index_pip.y - index_tip.y  # > 0 when finger goes down

            cv2.circle(frame, (fx, fy), 12, (0, 0, 255), -1)

            if cooldowns.get(label, 0) == 0 and finger_depth > 0.05:
                pressed = get_pressed_key(fx, fy, key_zones)
                if pressed:
                    if pressed == "<":
                        typed_text = typed_text[:-1]
                        keyboard.press('key.backspace')
                        keyboard.release('key.backspace')
                    elif pressed == " ":
                        typed_text += " "
                        keyboard.press('key.space')
                        keyboard.release('key.space')
                    else:
                        typed_text += pressed
                        keyboard.press(pressed.lower())
                        keyboard.release(pressed.lower())
                    cooldowns[label] = 15

            cv2.circle(frame, (fx, fy), 12, (0, 0, 255), -1)

            if cooldowns.get(label, 0) == 0:
                pressed = get_pressed_key(fx, fy, key_zones)
                if pressed:
                    if pressed == "<":
                        typed_text = typed_text[:-1]
                        keyboard.press('key.backspace')
                        keyboard.release('key.backspace')
                    elif pressed == " ":
                        typed_text += " "
                        keyboard.press('key.space')
                        keyboard.release('key.space')
                    else:
                        typed_text += pressed
                        keyboard.press(pressed.lower())
                        keyboard.release(pressed.lower())
                    cooldowns[label] = 15

    # Decrease cooldown
    for hand in cooldowns:
        if cooldowns[hand] > 0:
            cooldowns[hand] -= 1

    cv2.imshow("2-Hand Virtual Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
