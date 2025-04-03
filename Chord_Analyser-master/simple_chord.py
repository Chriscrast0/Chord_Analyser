import cv2
import mediapipe as mp
import numpy as np

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("Starting chord recognition...")
    print("Press 'q' to quit")

    while True:
        ret, image = cap.read()
        if not ret:
            break

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image_rgb)
        
        # Draw a box in the center
        height, width = image.shape[:2]
        box_width = width // 3
        box_height = height // 2
        x1 = width//2 - box_width//2
        y1 = height//2 - box_height//2
        x2 = x1 + box_width
        y2 = y1 + box_height
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS)
                
                # Get finger positions
                fingers = []
                for landmark in hand_landmarks.landmark:
                    fingers.append((landmark.x, landmark.y, landmark.z))
                
                if len(fingers) >= 21:  # We have all hand landmarks
                    # Get finger tips and bases
                    thumb_tip = fingers[4]
                    index_tip = fingers[8]
                    middle_tip = fingers[12]
                    ring_tip = fingers[16]
                    pinky_tip = fingers[20]
                    
                    # Get finger bases
                    index_base = fingers[5]
                    middle_base = fingers[9]
                    ring_base = fingers[13]
                    pinky_base = fingers[17]
                    
                    # Check if fingers are up
                    fingers_up = []
                    fingers_up.append(thumb_tip[1] < fingers[3][1])  # Thumb
                    fingers_up.append(index_tip[1] < index_base[1])  # Index
                    fingers_up.append(middle_tip[1] < middle_base[1])  # Middle
                    fingers_up.append(ring_tip[1] < ring_base[1])  # Ring
                    fingers_up.append(pinky_tip[1] < pinky_base[1])  # Pinky
                    
                    # Detect C chord (index and middle fingers up)
                    if fingers_up == [False, True, True, False, False]:
                        cv2.putText(image, "C Chord Detected!", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "No C Chord", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the image
        cv2.imshow('C Chord Recognition', image)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 