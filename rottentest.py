import cv2
from time import sleep
from PIL import Image
import numpy as np
import keras as k
from keras.models import load_model
import serial
import tensorflow.keras.models as k1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input

video_capture = cv2.VideoCapture(0)
arduino = serial.Serial(port='COM6', baudrate=9600, timeout=.1)
model = load_model('C:/Users/udit2/OneDrive/Desktop/ees/rotten.h5')
model2 = load_model('FV.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

file = 'C:/Users/udit2/OneDrive/Desktop/ees/test.png'
process_this_frame = 0

while True:
    
    ret, frame = video_capture.read()
    
    # Resize
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    #Convert BGR to RGB    
    rgb_small_frame = small_frame[:, :, ::-1]

    # Process once per 30 frames
    if process_this_frame%30==0:

        cv2.imwrite(file, small_frame)
        img = k.preprocessing.image.load_img(file, target_size=(224, 224, 3))
        img = k.preprocessing.image.img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, [0])
        answer = model2.predict(img)
        y_class = answer.argmax(axis=-1)
        print(y_class)
        y = " ".join(str(x) for x in y_class)
        y = int(y)
        print(labels[y])
        process_this_frame = 0 #prevent overflow
        #Only predict if a fruit is found
        if(y==0 or y==21):
            print(labels[y])
            #image = cv2.imread(file)
           # image = cv2.resize(image, (100, 100))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Preprocess the image
            #image = image / 255.0
            #image = np.expand_dims(img, axis=0)
            image = image_utils.load_img(file, target_size=(100, 100))
            image = image_utils.img_to_array(image)
            image = image.reshape(1,100,100,3)
            image = preprocess_input(image)
            result = model.predict(image)
            #Send to Arduino
            if result[0][0] > 0.5:
                print("rotten")
                arduino.write(bytes('1','utf-8'))
                sleep(0.05)
                y = arduino.readline()
                print(y)
            else:
                print("not rotten")
                arduino.write(bytes('2', 'utf-8'))
             
                        
    process_this_frame +=1

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
