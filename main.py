import cv2
import math
import mediapipe as mp
  
#Build Keypoints using MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    results = model.process(image)                 # Make prediction
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 
    
def claw(coord1, coord2):
    return math.sqrt(math.pow(coord1.x - coord2.x, 2) + math.pow(coord1.y - coord2.y, 2) * 1.0)

def calculate_movement(coordinate):
    print("Work on this")

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)

            if results.right_hand_landmarks != None:
#                 print("THUMB_TIP")
#                 print(results.right_hand_landmarks.landmark[4])
#                 print("INDEX_FINGER_TIP")
#                 print(results.right_hand_landmarks.landmark[8])
                distance = claw(results.right_hand_landmarks.landmark[4], results.right_hand_landmarks.landmark[8])
                print("distance = %.6f"%distance)
                print("WRIST")
                print(results.right_hand_landmarks.landmark[0])
                calculate_movement(results.right_hand_landmarks.landmark[0])
                    
            draw_styled_landmarks(image, results)
    
            cv2.imshow('OpenCV Feed', image)
    
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()