
import cv2
import numpy as np
from keras.models import load_model
import math
import copy
import pyttsx3
    


model = load_model('D:\\VSCode\\new_model6.h5')




gestures = {
    1:'A',
    2:'B',
    3:'C',
    4:'D',
    5:'E',
    6:'F',
    7:'G',
    8:'H',
    9:'I',
    10:'K',
    11:'L',
    12:'M',
    13:'N',
    14:'O',
    15:'P',
    16:'Q',
    17:'R',
    18:'S',
    19:'T',
    20:'U',
    21:'V',
    22:'W',
    23:'X',
    24:'Y',
}


# In[10]:


def predict(gesture):
    img = cv2.resize(gesture, (50,50))
    img = img.reshape(1,50,50,1)
    img = img/255.0
    prd = model.predict(img)
    index = prd.argmax()
    return gestures[index]


# In[15]:


vc = cv2.VideoCapture(0)
rval, frame = vc.read()
old_text = ''
pred_text = ''
count_frames = 0
total_str = ''
flag = False


# In[16]:


while True:
    
    if frame is not None: 
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize( frame, (400,400) )
        alpha_layer=frame.copy()

        cv2.rectangle(frame, (300,300), (100,100), (0,255,0), 2)
        
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
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)

            drawing = np.zeros(crop_img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        cv2.imshow('output', drawing)
        blackboard = np.zeros(frame.shape, dtype=np.uint8)
        cv2.putText(blackboard, "Predicted text - ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        if count_frames > 20 and pred_text != "":
            total_str += pred_text
            count_frames = 0
        wait=0
        if flag == True:
            old_text = pred_text
            pred_text = predict(thresh)
        
            if old_text == pred_text:
                count_frames += 1
            else:
                count_frames = 0
            cv2.putText(blackboard, pred_text, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
            wait+=1
            if wait==1:

                engine = pyttsx3.init()
                engine.say(pred_text)
                
                engine.setProperty('rate',120)  
                engine.setProperty('volume',0.9) 
                wait=0
            engine.runAndWait()
        res = np.hstack((frame, blackboard))
        
        cv2.imshow("image", res)
        cv2.imshow("hand", thresh)
        
        
    rval, frame = vc.read()
    keypress = cv2.waitKey(1)
    if keypress == 
    ord('c') or keypress==ord('C'):
        flag = True
    if keypress == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


# In[17]:


vc.release()