"""
ASL (American Sign Language) Finger-Spelling Recognition
Uses Mediapipe hand landmarks to detect ASL alphabet letters in real-time.
Maps 21 hand landmarks to letter classifications using geometric features.
"""

import numpy as np
import math

# ============================================================
# ASL Landmark-Based Letter Detection
# ============================================================

def get_finger_states(landmarks):
    """
    Determine which fingers are up/down based on landmark positions.
    Returns a list of 5 booleans: [thumb, index, middle, ring, pinky]
    
    Landmarks:
    0: Wrist
    1-4: Thumb (1=CMC, 2=MCP, 3=IP, 4=TIP)
    5-8: Index (5=MCP, 6=PIP, 7=DIP, 8=TIP)
    9-12: Middle (9=MCP, 10=PIP, 11=DIP, 12=TIP)
    13-16: Ring (13=MCP, 14=PIP, 15=DIP, 16=TIP)
    17-20: Pinky (17=MCP, 18=PIP, 19=DIP, 20=TIP)
    """
    fingers = []
    
    # Thumb: compare x position (depends on handedness)
    # If thumb tip is to the left of thumb IP joint (for right hand)
    if landmarks[4][0] < landmarks[3][0]:
        fingers.append(True)  # Thumb is open
    else:
        fingers.append(False)
    
    # Other fingers: tip should be above PIP joint (lower y = higher on screen)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip][1] < landmarks[pip][1]:
            fingers.append(True)
        else:
            fingers.append(False)
    
    return fingers

def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def angle(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 * mag2 == 0:
        return 0
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))

def recognize_asl_letter(landmarks):
    """
    Recognize ASL finger-spelling letters based on hand landmark positions.
    Uses finger states and geometric relationships for classification.
    
    Args:
        landmarks: List of 21 [x, y] coordinates from Mediapipe
    
    Returns:
        Detected letter (A-Z) or '?' if unknown
    """
    if len(landmarks) < 21:
        return '?'
    
    fingers = get_finger_states(landmarks)
    thumb, index, middle, ring, pinky = fingers
    
    # Calculate useful distances
    thumb_index_dist = distance(landmarks[4], landmarks[8])
    index_middle_dist = distance(landmarks[8], landmarks[12])
    middle_ring_dist = distance(landmarks[12], landmarks[16])
    wrist_middle_dist = distance(landmarks[0], landmarks[12])
    thumb_pinky_dist = distance(landmarks[4], landmarks[20])
    index_tip_to_thumb_tip = distance(landmarks[8], landmarks[4])
    
    # Normalize distances by hand size (wrist to middle MCP)
    hand_size = distance(landmarks[0], landmarks[9])
    if hand_size == 0:
        return '?'
    
    norm_thumb_index = thumb_index_dist / hand_size
    norm_index_middle = index_middle_dist / hand_size
    
    # ============================================================
    # ASL Letter Recognition Rules
    # ============================================================
    
    # A: Fist with thumb to the side
    if not index and not middle and not ring and not pinky and thumb:
        return 'A'
    
    # B: All fingers up, thumb across palm
    if not thumb and index and middle and ring and pinky:
        return 'B'
    
    # C: Curved hand (all fingers partially open, forming C shape)
    if thumb and index and not middle and not ring and not pinky:
        if norm_thumb_index > 0.3 and norm_thumb_index < 0.8:
            return 'C'
    
    # D: Index up, others closed touching thumb
    if not thumb and index and not middle and not ring and not pinky:
        return 'D'
    
    # E: All fingers curled down, thumb across
    if not thumb and not index and not middle and not ring and not pinky:
        return 'E'
    
    # F: OK sign with index and thumb, other fingers up
    if thumb and not index and middle and ring and pinky:
        return 'F'
    
    # G: Index pointing sideways, thumb out
    if thumb and index and not middle and not ring and not pinky:
        if abs(landmarks[8][1] - landmarks[5][1]) < hand_size * 0.3:
            return 'G'
    
    # H: Index and middle pointing sideways
    if not thumb and index and middle and not ring and not pinky:
        if abs(landmarks[8][1] - landmarks[5][1]) < hand_size * 0.3:
            return 'H'
    
    # I: Pinky up, others closed
    if not thumb and not index and not middle and not ring and pinky:
        return 'I'
    
    # K: Index and middle up spread, thumb between
    if thumb and index and middle and not ring and not pinky:
        if norm_index_middle > 0.3:
            return 'K'
    
    # L: L-shape with thumb and index
    if thumb and index and not middle and not ring and not pinky:
        if norm_thumb_index > 0.8:
            return 'L'
    
    # O: All fingers touching thumb forming O
    if thumb and index and not middle and not ring and not pinky:
        if norm_thumb_index < 0.3:
            return 'O'
    
    # R: Index and middle crossed
    if not thumb and index and middle and not ring and not pinky:
        if norm_index_middle < 0.15:
            return 'R'
    
    # U: Index and middle up together
    if not thumb and index and middle and not ring and not pinky:
        if norm_index_middle < 0.3:
            return 'U'
    
    # V: Index and middle up spread (peace sign)
    if not thumb and index and middle and not ring and not pinky:
        if norm_index_middle > 0.3:
            return 'V'
    
    # W: Index, middle, ring up spread
    if not thumb and index and middle and ring and not pinky:
        return 'W'
    
    # X: Index hooked
    if not thumb and index and not middle and not ring and not pinky:
        idx_angle = angle(landmarks[5], landmarks[6], landmarks[8])
        if idx_angle < 120:
            return 'X'
    
    # Y: Thumb and pinky out, others closed
    if thumb and not index and not middle and not ring and pinky:
        return 'Y'
    
    # Default: Open hand = 5 / SPACE
    if thumb and index and middle and ring and pinky:
        return ' '  # Space gesture (open palm)
    
    return '?'


# Gesture-to-word quick phrases
QUICK_PHRASES = {
    'HELLO': '👋 Hello!',
    'THANKS': '🙏 Thank you!',
    'YES': '✅ Yes',
    'NO': '❌ No',
    'HELP': '🆘 Help!',
    'PLEASE': '🙏 Please',
    'SORRY': '😔 Sorry',
    'LOVE': '❤️ Love',
    'OK': '👌 OK',
    'HI': '👋 Hi!',
    'BYE': '👋 Goodbye!',
    'STOP': '🛑 Stop',
    'GO': '🟢 Go',
    'GOOD': '👍 Good',
    'BAD': '👎 Bad',
    'WATER': '💧 Water',
    'FOOD': '🍽️ Food',
    'HOME': '🏠 Home',
}

def check_quick_phrase(text):
    """Check if the accumulated text matches a quick phrase."""
    text_upper = text.strip().upper()
    return QUICK_PHRASES.get(text_upper, None)
