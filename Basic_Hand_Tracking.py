#easy goal
import mediapipe as mp
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands( static_image_mode = True,max_num_hands = 1,min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)


def main():
    pTime = 0
    cTime = 0
    while True:
        ret, img = cap.read()
        if ret:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(imgRGB)
            
            imgHeight = img.shape[0]
            imgWidth = img.shape[1]
            blank = np.zeros((imgHeight,imgWidth,3) , dtype= np.uint8)

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                    mpDraw.draw_landmarks(blank, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)

            blank = cv2.flip(blank,1)
            img = cv2.flip(img,1)
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('Blank',blank)
            cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('q'):
            break
    
    

if __name__ == "__main__":
    main()
