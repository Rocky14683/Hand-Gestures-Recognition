#medium goal
from operator import lt
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import pygame
import os
import uuid


# def get_label(index, hand, result):
#     output = None
#     for idx, classification in enumerate(result.multi_hand)


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands( static_image_mode =  True,max_num_hands = 1,min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_video = mpHands.Hands(static_image_mode = False,max_num_hands = 1,min_detection_confidence =0.5 , min_tracking_confidence= 0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)
pTime = 0
cTime = 0
    
    
    
def detectHandsLandmarks(image, hands, draw = True, display = True):
    output = image.copy()
    
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    
    if result.multi_hand_landmarks and draw:
        for hand_lm in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(image = output,landmark_list=hand_lm, connections = mpHands.HAND_CONNECTIONS,
                                  landmark_drawing_spec = handLmsStyle, connection_drawing_spec = handConStyle)
            
    if display:
        
        plt.figure(figsize = [15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
        plt.subplot(122);plt.imshow(output[:,:,::-1]);plt.title("Output");plt.axis('off')
    else:
        return output, result
    
   
    
def countFinger(image, result, draw = True, display =True):
    height, width, _ = image.shape
    
    output_image = image.copy()
    
    finger_tips_ids = [mpHands.HandLandmark.INDEX_FINGER_TIP, mpHands.HandLandmark.MIDDLE_FINGER_TIP,
                   mpHands.HandLandmark.RING_FINGER_TIP, mpHands.HandLandmark.PINKY_TIP]
    fingers_status = {'RIGHT_THUMB':False, 'RIGHT_INDEX':False, 'RIGHT_MIDDLE': False, 'RIGHT_RING':False, 'RIGHT_PINKY':False
                      ,'LEFT_THUMB':False, 'LEFT_INDEX':False, 'LEFT_MIDDLE': False, 'LEFT_RING':False, 'LEFT_PINKY':False}  
    finger_counts = {'RIGHT':0, 'LEFT':0}
    
    for hand_index, hand_info in enumerate(result.multi_handedness):
        
        hand_label = hand_info.classification[0].label
        
        hand_landmarks = result.multi_hand_landmarks[hand_index]
        for tip_index in finger_tips_ids:
            
            finger_name = tip_index.name.split("_")[0]
            
            if(hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index -2].y):
                fingers_status[hand_label.upper()+"_"+finger_name] = True
                finger_counts[hand_label.upper()]+=1
                
        thumb_tip_x =hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP-1].x

        if(hand_label =='Right' and (thumb_tip_x < thumb_mcp_x))or(hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
            fingers_status[hand_label.upper()+"_THUMB"] = True
            
            finger_counts[hand_label.upper()]+=1
            
    if draw: 
        cv2.putText(output_image, ("Current Detect Hand:"),(10,25),cv2.FONT_HERSHEY_COMPLEX, 1, (20,255,155), 2)  
        cv2.putText(output_image,hand_label+str(sum(finger_counts.values())),(width//2-250,240),cv2.FONT_HERSHEY_SIMPLEX,8.9, (20,255,155), 10, 10)
        
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.tutle("Output Image");plt.axis('off')
        
    else:
        return output_image, fingers_status, finger_counts
    


def recognizeGestures(image, fingers_status,count, draw = True, display =True):
    output_image = image.copy()
    hand_label = ['RIGHT','LEFT']
    hand_gesture = {'RIGHT':"UNKNOWN", 'LEFT': "UNKNOWN"}
    
    for hand_index, hand_label in enumerate(hand_label):
        color = (0, 0, 255)
        
        if count[hand_label] == 0:
            hand_gesture[hand_label] = "Stone"
            color = (0,255,0)
        
        elif count[hand_label] == 2 and fingers_status[hand_label+'_MIDDLE'] and fingers_status[hand_label+'_INDEX']:
            hand_gesture[hand_label] = "Scissors"
            color = (0,255,0)
        elif count[hand_label] == 5:
            hand_gesture[hand_label] = "Paper"
            color = (0,255,0)
        elif count[hand_label] == 1 and fingers_status[hand_label+'_MIDDLE']:
            hand_gesture[hand_label] = "Middle Finger"
            color = (0,255,0)
        else:
            hand_gesture[hand_label] = "UNKNOWN"
            color = (0,255,0)
        
        if draw:
            cv2.putText(output_image, hand_label + ': '+hand_gesture[hand_label],(10,(hand_index+1)*60),cv2.FONT_HERSHEY_PLAIN,4,color,5)
        
        if display:
            plt.figure(figsize=[10,10])
            plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off')
        else:
            return output_image, hand_gesture
            
        
        
def alwaysWIN(gesture):
    if gesture == "UNKNOWN":
        return 'none'
    elif gesture == "Middle Finger":
        return 'middle finger'
    elif gesture == "Paper":
        return 'scissor'
    elif gesture == "Scissors":
        return 'stone'
    elif gesture == "Stone":
        return 'paper'        
        
                
    
def main():
  while True:
      ret, img = cap.read()

      if not ret:
          continue

      img = cv2.flip(img,1)

      img, result = detectHandsLandmarks(img,hands_video, display = False)

      if result.multi_hand_landmarks:

          img, fingers_status, count = countFinger(img, result,draw= False ,display = False)
          img, gesture = recognizeGestures(img, fingers_status, count, draw = True, display=False)
          winnerGesture = alwaysWIN(gesture)

      cv2.imshow('Finger Counter', img)

      if cv2.waitKey(1) == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()
  
  if __name__ == "__main__":
    main()

