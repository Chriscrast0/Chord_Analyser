import cv2
import mediapipe as mp
import numpy as np

def rescale_frame(frame, percent=100):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def detect_chord(hand_positions):
    if len(hand_positions) < 21:  # We need all 21 hand landmarks
        return "No Hand Detected"
    
    # Get finger positions
    thumb_tip = hand_positions[4]
    index_tip = hand_positions[8]
    middle_tip = hand_positions[12]
    ring_tip = hand_positions[16]
    pinky_tip = hand_positions[20]
    
    # Get finger base positions
    index_base = hand_positions[5]
    middle_base = hand_positions[9]
    ring_base = hand_positions[13]
    pinky_base = hand_positions[17]
    
    # Detect finger positions (up or down)
    fingers_up = []
    fingers_up.append(thumb_tip[1] < hand_positions[3][1])  # Thumb
    fingers_up.append(index_tip[1] < index_base[1])  # Index
    fingers_up.append(middle_tip[1] < middle_base[1])  # Middle
    fingers_up.append(ring_tip[1] < ring_base[1])  # Ring
    fingers_up.append(pinky_tip[1] < pinky_base[1])  # Pinky
    
    # Chord detection logic
    if fingers_up == [True, True, True, True, True]:  # All fingers up
        return "Chord: A"
    elif fingers_up == [False, True, True, False, False]:  # Index and middle up
        return "Chord: C"
    elif fingers_up == [False, True, True, True, True]:  # All fingers except thumb up
        return "Chord: F"
    elif fingers_up == [False, True, True, True, False]:  # Index, middle, ring up
        return "Chord: G"
    else:
        return "No Chord Detected"

def draw_guides(image):
    height, width = image.shape[:2]
    
    # Draw center guide
    cv2.line(image, (width//2, 0), (width//2, height), (0, 255, 0), 1)
    
    # Draw hand placement box
    box_width = width // 3
    box_height = height // 2
    x1 = width//2 - box_width//2
    y1 = height//2 - box_height//2
    x2 = x1 + box_width
    y2 = y1 + box_height
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add instructions
    cv2.putText(image, "Place hand in box", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image, "Press 'q' to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return image

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set up drawing specs
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)

    print("Starting chord recognition...")
    print("Available chords: A, C, F, G")
    print("Press 'q' to quit")

    while True:
        ret, image = cap.read()
        if not ret:
            break

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        image_orig = image.copy()
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image_rgb)
        
        # Draw guides
        image = draw_guides(image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec)
                
                # Get hand position data
                hand_positions = []
                for landmark in hand_landmarks.landmark:
                    hand_positions.append((landmark.x, landmark.y, landmark.z))
                
                # Detect chord
                chord = detect_chord(hand_positions)
                cv2.putText(image, chord, (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Chord Recognition', image)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 