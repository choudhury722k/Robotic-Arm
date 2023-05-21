import cv2
import math
import mediapipe as mp
  
mp_holistic = mp.solutions.holistic     
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    results = model.process(image)                 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 

def draw_styled_landmarks(image, results):  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)

            if results.right_hand_landmarks != None:
                coord1 = results.right_hand_landmarks.landmark[4]
                coord2 = results.right_hand_landmarks.landmark[8]
                distance = math.sqrt(math.pow(coord1.x - coord2.x, 2) + math.pow(coord1.y - coord2.y, 2) * 1.0)
                if distance > 0.1:
                    cv2.putText(image,
                                "Distance = %.2f (Open)" % distance, 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                else:
                    cv2.putText(image, 
                                "Distance = %.2f (Close)" % distance, 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                cv2.putText(image, 
                            "Wrist_x = %.2f" % results.right_hand_landmarks.landmark[0].x, 
                            (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                cv2.putText(image, 
                            "Wrist_y = %.2f" % results.right_hand_landmarks.landmark[0].y, 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                    
            draw_styled_landmarks(image, results)
    
            cv2.imshow('OpenCV Feed', image)
    
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()