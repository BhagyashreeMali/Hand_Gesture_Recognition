"""
🖐️ Live Sign Language Recognition App
Detects ASL (American Sign Language) finger-spelling in real-time using webcam.
Uses the new Mediapipe Tasks API (0.10.30+).

Features:
- Letter stabilization (hold for 1s to accept)
- Word & sentence building
- Quick phrase detection
- Rich visual overlay UI
"""

import os
import sys
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.sign_language import recognize_asl_letter, check_quick_phrase

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
HOLD_THRESHOLD = 1.0
SPACE_THRESHOLD = 2.0
CLEAR_THRESHOLD = 4.0
MAX_SENTENCE_LEN = 100

# Colors (BGR)
COLOR_BG = (30, 30, 30)
COLOR_ACCENT = (234, 126, 102)
COLOR_SUCCESS = (0, 200, 100)
COLOR_TEXT = (255, 255, 255)
COLOR_DIMMED = (150, 150, 150)
COLOR_PURPLE = (200, 100, 255)
COLOR_GOLD = (0, 215, 255)

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]


class SignLanguageApp:
    def __init__(self):
        # Create HandLandmarker with new Tasks API
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found at: {MODEL_PATH}")
            print("   Downloading model...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("   ✅ Model downloaded!")
        
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # State
        self.current_letter = '?'
        self.letter_start_time = 0
        self.last_accepted_letter = ''
        self.sentence = ''
        self.word = ''
        self.last_hand_time = time.time()
        self.letter_history = []
        self.accepted_flash = 0
        self.fps_history = []

    def draw_hand_landmarks(self, frame, landmarks, w, h):
        """Draw hand landmarks and connections on frame."""
        points = [(int(lm[0]), int(lm[1])) for lm in landmarks]
        
        # Draw connections
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 255, 200), 2)
        
        # Draw landmark points
        for pt in points:
            cv2.circle(frame, pt, 4, (255, 0, 255), -1)
            cv2.circle(frame, pt, 6, (255, 0, 255), 1)

    def process_frame(self, frame):
        """Process a single frame and return the annotated result."""
        start_time = time.time()
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to mediapipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        
        detected_letter = '?'
        hand_detected = False
        
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_detected = True
            self.last_hand_time = time.time()
            
            hand_lms = result.hand_landmarks[0]
            
            # Extract pixel coordinates
            landmarks = []
            for lm in hand_lms:
                landmarks.append([int(lm.x * w), int(lm.y * h)])
            
            # Draw landmarks
            self.draw_hand_landmarks(frame, landmarks, w, h)
            
            # Recognize letter
            detected_letter = recognize_asl_letter(landmarks)
            
            # Draw bounding box
            x_coords = [l[0] for l in landmarks]
            y_coords = [l[1] for l in landmarks]
            x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
            y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)
            
            border_color = COLOR_SUCCESS if detected_letter != '?' else COLOR_ACCENT
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), border_color, 2)
            
            if detected_letter != '?':
                cv2.putText(frame, detected_letter, (x_min, y_min - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, border_color, 3)
        
        # Letter stabilization
        if detected_letter != '?' and detected_letter == self.current_letter:
            hold_time = time.time() - self.letter_start_time
            
            # Progress bar
            progress = min(hold_time / HOLD_THRESHOLD, 1.0)
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (w//2 - 100, 80), (w//2 - 100 + bar_width, 95), COLOR_SUCCESS, -1)
            cv2.rectangle(frame, (w//2 - 100, 80), (w//2 + 100, 95), COLOR_DIMMED, 2)
            
            if hold_time >= HOLD_THRESHOLD:
                if detected_letter == ' ':
                    if self.word:
                        self.sentence += self.word + ' '
                        self.word = ''
                else:
                    self.word += detected_letter
                
                self.last_accepted_letter = detected_letter
                self.letter_history.append(detected_letter)
                self.accepted_flash = 10
                self.letter_start_time = time.time()
        else:
            self.current_letter = detected_letter
            self.letter_start_time = time.time()
        
        # Auto-space
        if not hand_detected:
            no_hand_duration = time.time() - self.last_hand_time
            if no_hand_duration > SPACE_THRESHOLD and self.word:
                self.sentence += self.word + ' '
                self.word = ''
            if no_hand_duration > CLEAR_THRESHOLD and self.sentence:
                self.sentence = ''
                self.letter_history = []
        
        if len(self.sentence) > MAX_SENTENCE_LEN:
            self.sentence = self.sentence[-MAX_SENTENCE_LEN:]
        
        full_text = self.sentence + self.word
        phrase = check_quick_phrase(full_text)
        
        # Draw UI
        self._draw_ui(frame, detected_letter, hand_detected, phrase)
        
        # FPS
        fps = 1.0 / (time.time() - start_time + 1e-6)
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = np.mean(self.fps_history)
        cv2.putText(frame, f"FPS: {avg_fps:.0f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DIMMED, 1)
        
        return frame

    def _draw_ui(self, frame, detected_letter, hand_detected, phrase):
        """Draw the full UI overlay."""
        h, w, _ = frame.shape
        
        # Title Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (50, 30, 80), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "ASL Sign Language Recognition", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
        
        # Letter Display
        letter_display = detected_letter if detected_letter != '?' else '-'
        letter_color = COLOR_SUCCESS if detected_letter != '?' else COLOR_DIMMED
        if self.accepted_flash > 0:
            letter_color = COLOR_GOLD
            self.accepted_flash -= 1
        
        cv2.rectangle(frame, (w - 90, 60), (w - 10, 140), letter_color, 2)
        cv2.putText(frame, letter_display, (w - 75, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, letter_color, 3)
        cv2.putText(frame, "Detected", (w - 85, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_DIMMED, 1)
        
        # Bottom Panel
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 120), (w, h), COLOR_BG, -1)
        cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
        
        current_text = self.sentence + self.word
        if current_text:
            cv2.putText(frame, "Text:", (15, h - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIMMED, 1)
            display_text = current_text + ('|' if int(time.time() * 2) % 2 == 0 else '')
            cv2.putText(frame, display_text, (15, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)
        else:
            cv2.putText(frame, "Show ASL letters to build words...", (15, h - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DIMMED, 1)
        
        if phrase:
            cv2.putText(frame, phrase, (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GOLD, 2)
        
        history_text = ''.join(self.letter_history[-15:])
        cv2.putText(frame, f"History: {history_text}", (w - 280, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIMMED, 1)
        
        # Instructions
        instructions = [
            "Controls:", "Q - Quit", "C - Clear text",
            "S - Screenshot", "SPACE - Add space", "",
            "Hold letter 1s", "to accept it"
        ]
        for i, line in enumerate(instructions):
            color = COLOR_PURPLE if i == 0 else COLOR_DIMMED
            cv2.putText(frame, line, (10, 180 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Status
        status = "Hand Detected" if hand_detected else "No Hand"
        status_color = COLOR_SUCCESS if hand_detected else COLOR_ACCENT
        cv2.circle(frame, (25, 65), 8, status_color, -1)
        cv2.putText(frame, status, (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    def run(self):
        """Main loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot open webcam.")
            print("Please ensure your webcam is connected and accessible.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=" * 60)
        print("🖐️  ASL SIGN LANGUAGE RECOGNITION - LIVE")
        print("=" * 60)
        print()
        print("  Show ASL hand signs to the camera to spell words!")
        print("  Hold a letter for 1 second to accept it.")
        print()
        print("  Controls:")
        print("    Q     - Quit")
        print("    C     - Clear text")
        print("    S     - Save screenshot")
        print("    SPACE - Insert space manually")
        print()
        print("  Supported: A B C D E F G H I K L O R U V W X Y + SPACE")
        print("=" * 60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame)
            cv2.imshow("ASL Sign Language Recognition", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.sentence = ''
                self.word = ''
                self.letter_history = []
                print("Text cleared!")
            elif key == ord('s'):
                filename = f"sign_language_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord(' '):
                if self.word:
                    self.sentence += self.word + ' '
                    self.word = ''
        
        cap.release()
        cv2.destroyAllWindows()
        
        final_text = self.sentence + self.word
        if final_text:
            print(f"\n📝 Final text: {final_text}")


def main():
    app = SignLanguageApp()
    app.run()

if __name__ == "__main__":
    main()
