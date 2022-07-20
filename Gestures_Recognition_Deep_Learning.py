#difficult goal
import cv2
from cv2 import KeyPoint
from cv2 import destroyAllWindows
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from pyparsing import null_debug_action
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import *
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.models import load_model




mpHands = mp.solutions.hands
hands = mpHands.Hands( static_image_mode = True,max_num_hands = 1,min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_video = mpHands.Hands(static_image_mode = False,max_num_hands = 1,min_detection_confidence =0.5 , min_tracking_confidence= 0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=3)
handConStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
cap = cv2.VideoCapture(0)
###################

def detectHandsLandmarks(image, hands, draw = True, display = True, gestureToColor = handConStyle):
    output = image.copy()
    
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    
    if result.multi_hand_landmarks and draw:
        for hand_lm in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(image = output,landmark_list=hand_lm, connections = mpHands.HAND_CONNECTIONS,
                                  landmark_drawing_spec = handLmsStyle, connection_drawing_spec = gestureToColor)
            
    if display:
        
        plt.figure(figsize = [15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
        plt.subplot(122);plt.imshow(output[:,:,::-1]);plt.title("Output");plt.axis('off')
    else:
        return output, result

#####
handpoints = []
def extract_keypoints(result):
    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            # for id, lm in enumerate(res.landmark):
            #     handpoints = np.array([[lm.x,lm.y,lm.z]id]).flatten()
            handpoints = np.array([[lm.x,lm.y,lm.z] for _,lm in enumerate(res.landmark)]).flatten()
    else:#ok
        handpoints = np.zeros(21*3)    
    return np.concatenate([handpoints])
    
#######################

####################
#####
#Setup Folders for collection
####################
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['paper','scissors','stone'])
no_sequences = 40 #number of sequences
sequence_length = 3 #frames


def setup():#setup file and path AND start capturing video * sequence_length frame
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
#START capturing sequence_length frame per npy file (3(action)*40(sequence)*3(frame))
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                #read frame
                ret, frame = cap.read()
                frame = cv2.flip(frame,1)
                #draw landmarks and return results
                image, result = detectHandsLandmarks(frame, hands_video,draw=True,display=False)
                
                    
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                    cv2.putText(image, 'Collection frame for {} Video Number {}'.format(action, sequence),(15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255,),1,cv2.LINE_AA)
                    cv2.imshow('Camera Capturing',image)
                    cv2.waitKey(1000)
                else:
                    cv2.putText(image, 'Collection frame for {} Video Number {}'.format(action, sequence),(15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255,),1,cv2.LINE_AA)
                    cv2.imshow('Camera Capturing',image)
                    
                keyPoint = extract_keypoints(result)
                npy_path = os.path.join(DATA_PATH, action,str(sequence),str(frame_num)) #Open a specific file
                np.save(npy_path, keyPoint) #save the coordinate array into the file
                print("Done Sorting: ", action ,str(sequence),str(frame_num))
                  
                  
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
        cv2.putText(image,'Change gesture to {}'.format(action),(15,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255,),1,cv2.LINE_AA)
        cv2.waitKey(2000)
        
    cap.release()
    cv2.destroyAllWindows()
            
    return keyPoint


def createLabel():
    label_map = {label:num for num,label in enumerate(actions)}
    #print(label_map)
    sequences, labels =[], []
    for action in actions:
        for sequence in range(no_sequences):
            window =[]
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence),"{}.npy".format(frame_num)))
                window.append(res)
                #print(str(action)+" "+str(frame_num)+" "+str(sequence))
            sequences.append(window)
            labels.append(label_map[action])
    #print(np.array(sequences).shape)    
    #print(np.array(labels).shape)
    X = np.array(sequences)
    #print(X.shape)
    y = to_categorical(labels).astype(int)
    print(y)
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size= 0.05)
    #print(X_train.shape)
    return X_train, X_test, y_train,y_test


def NeturalNet():
    X_train, X_test, y_train,y_test = createLabel()
    log_dir = os.path.join('Logs')
    callback = TensorBoard(log_dir=log_dir)
    model = Sequential()
    model.add(LSTM(64,return_sequences=True,activation='relu', input_shape = (sequence_length ,63)))
    model.add(LSTM(128, return_sequences=True,activation='relu'))
    model.add(LSTM(64, return_sequences=False,activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0],activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=150 , callbacks =[callback])
    
    print(model.summary())
    model.save('action.h5')
    
    return


def winner(input):
    if input == 'scissors':
        return 'stone'
    elif input == 'stone':
        return 'paper'
    elif input == 'paper':
        return 'scissors'
    else:
        return 
    
colors = [(124,255,234),(210,0,223), (16,117,255)]    
def prob_frame(res, actions, input_frame, colors):
    output = input_frame.copy()
    for num,prob in enumerate(res):
        cv2.rectangle(output, (0,60+num*40), (int(prob*100),90+num*40), colors[num],-1)
        cv2.putText(output, actions[num],(0,85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        if prob*100 >= 50:
            cv2.putText(output, f"{int(prob*100)}%",(int(prob*100)+50,85+num*40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
    
    return output



def realTimeDec(display = False):
    dectTimeC = 0
    dectTimeP = 0
    cTime = 0
    pTime = 0
    model = load_model('action.h5')
    print(model.summary())
    sequence = []
    predictions = []
    threshold = 0.97
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        dectTimeP = time.time()
        image, result = detectHandsLandmarks(frame,hands_video,False,False)
        keypoint = extract_keypoints(result)
        sequence.append(keypoint)
        sequence = sequence[-sequence_length:]
        cv2.flip(image,1) 
        if display == False:
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                if np.unique(predictions[-1:][0] == np.argmax(res)):
                    if res[np.argmax(res)] > threshold:
                        current_gesture = (actions[np.argmax(res)])
                    else:
                        current_gesture = 'None'
            dectTime = dectTimeP - dectTimeC
            dectTimeC = dectTimeP
            try:
                win = winner(current_gesture)
                print(f"{current_gesture} vs. {win} , {dectTime}")
            except:
                pass
        #---------------------------------------------
        else:
            pTime = time.time()
            
            
            
            cv2.rectangle(image,(0,0),(640,40),(255,0,16),-1)
            cv2.rectangle(image,(640,0),(1280,40),(0,255,0),-1)
            
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                #print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                image = prob_frame(res, actions,image, colors)
                
                if np.unique(predictions[-1:])[0] == np.argmax(res):
                    dectTime = dectTimeP - dectTimeC
                    dectTimeC = dectTimeP
                    if res[np.argmax(res)] > threshold:
                        current_gesture = (actions[np.argmax(res)])
                        cv2.putText(image,f"You: {current_gesture}", (3,30),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                        win = str(winner(current_gesture))
                        cv2.putText(image,f"Robot: {win}",(643,30),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                        num = np.argmax(res)
                        image, result = detectHandsLandmarks(image,hands_video,True,False, mpDraw.DrawingSpec(color = colors[num], thickness=5))
                    else:
                        cv2.putText(image,"None", (3,30),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        image, result = detectHandsLandmarks(image,hands_video,True,False)
                        current_gesture = 'None'
                    
                else:
                    cv2.putText(image,"None", (3,30),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    current_gesture = 'None'
            try:
                print(f"{dectTime} - {current_gesture}")
            except:
                pass
            cTime = time.time()
            fps = 1/(cTime - pTime)
            pTime = cTime        
            cv2.rectangle(image, (1090,580),(1230,640),(255,255,255),-1)
            cv2.putText(image, f"FPS:{int(fps)}",(1100,620),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),3)
            
            #--------------------------------- 
  
        cv2.imshow('Real Time Detect',image)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows  
    return

def debug():
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        image, result = detectHandsLandmarks(frame,hands_video,True,False)
    
        # for handlms in result.multi_hand_landmarks:
        #     for _, lm in enumerate(handlms.landmark):
        #         print(lm.x)
        print(extract_keypoints(result))
        
        
        
        cv2.imshow('Finger Counter', image)
    
        if cv2.waitKey(1) == ord('q'):
            break  
    return

    
def main():
    # setup()
    if os.path.exists('action.h5')==False:
        NeturalNet()
    realTimeDec(False)
    #createLabel()
    #NeturalNet()
    #createLabel()
    
    
    return






    
if __name__== "__main__":
    main()
    
        
            


