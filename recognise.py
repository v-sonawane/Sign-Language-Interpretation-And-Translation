import cv2
import numpy as np
from keras.models import load_model
import math
import copy
import pyttsx3
    


model = load_model('D:\\VSCode\\transferlearning2.h5') 

#Mapping the classes with gestures

gestures={0:'1',1:'2',2:'3',3:'4',4:'5',5:'6',6:'7',7:'8',8:'9',9:'A',10:'B',11:'C',12:'D',13:'E',14:'F',15:'G',16:'H',17:'I',18:'J',19:'K',20:'L',21:'M',22:'N',23:'O',24:'P',
        25:'Q',26:'R',27:'S',28:'T',29:'U',30:'V',31:'W',32:'X',33:'Y',34:'Z'}

def predict(gesture):   #Method for predicting the gesture
    img = cv2.resize(gesture, (200,200))
    img = img.reshape(-1,200,200,3)
    img = img/255.0
    prd = model.predict(img)
    index = prd.argmax()    #Selecting Best Estimate
    return gestures[index]

capture= cv2.VideoCapture(0)
rval, frame = capture.read()
pred_text = ''
count_frames = 0
flag = False

while True:
    
    if frame is not None: 
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize( frame, (400,400) )
        cv2.rectangle(frame, (300,300), (100,100), (0,255,0), 2) #Defining ROI
        
        #The process for thresholding the desired input frame and contour detection
        crop_img = frame[100:300, 100:300]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (7, 7), 0)
        thresh = cv2.threshold(blur,210,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    area_index= i

            result = contours[area_index]
            hull = cv2.convexHull(result) 

            track_hand= np.zeros(crop_img.shape, np.uint8)    #Creates black frame for displaying the detected contours
            cv2.drawContours(track_hand, [result], 0, (0, 255, 0), 2)
            cv2.drawContours(track_hand, [hull], 0, (0, 0, 255), 3)
        cv2.imshow('output', track_hand)
        blackboard = np.zeros(frame.shape, dtype=np.uint8)
        
        if flag == True:
            wait=0
            pred_text = predict(crop_img)
            count_frames = 0
            cv2.putText(blackboard, pred_text, (100, 180), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (255, 255, 255))
            wait+=1
            if wait==1:
                engine = pyttsx3.init()
                engine.say(pred_text)
                engine.setProperty('rate',120)  
                engine.setProperty('volume',0.9) 
                wait=0
            engine.runAndWait()

        result= np.hstack((frame, blackboard)) #Concatening both frames
        
        cv2.imshow("Frame", result)
        cv2.imshow("Thresholded", thresh)
        
        
    rval, frame = capture.read()
    keypress = cv2.waitKey(1)
    if keypress == ord('l') or keypress==ord('L'):  #Press C/c for enabling translation mode
        flag = True
 
    if keypress == ord('q') or keypress==ord('Q'):    #Press q to exit
        break

capture.release()
cv2.destroyAllWindows()