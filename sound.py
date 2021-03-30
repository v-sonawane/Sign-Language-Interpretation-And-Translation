import time
from playsound import playsound
import pyttsx3
time.sleep(20)
while True:
    file=open("testing_file.txt","r")
    s=file.read()
    print(s)#[len(s)-2])
#    d="sounds/"+s.lower()+".mp3"
#    playsound(d)
    engine = pyttsx3.init()
    engine.say(s)       
    engine.setProperty('rate',120)  
    engine.setProperty('volume',0.9)
    engine.runAndWait()
    file.close()