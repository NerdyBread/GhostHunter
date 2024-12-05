# This is spaghetti code I made for fun, don't judge me

import cv2 as cv
import numpy as np
from random import randint, choice

skull_path = r".\skull.png"
ghost_path = r".\ghost.jpg"
skull_img = cv.imread(skull_path)
ghost_img = cv.imread(ghost_path)

images = [skull_img, ghost_img]
box_img = np.ones((100,430,3),dtype='uint8')*255

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Can't open camera")
    exit()

count = 0
show_ghost = int()

alpha = 1 # Tranparency of the image

while True:
    if show_ghost != 2:
        show_ghost = randint(1, 180)
        chosen_img = choice(images)
        x = randint(10, 300)
        y = randint(10, 300)
    else:
        count += 1
        if alpha >= 3/150:
            alpha -= 3/150
        if count == 150:
            show_ghost = 0
            count = 0
            alpha = 1
    # Get current frame and make sure it's read correctly
    ret, frame = cap.read()
    if not ret:
        print("Can't get frame. Exiting ...")
        break
    background = cv.flip(frame, 1)

    # Define x and y here if you want the image to shake
    #x = randint(10, 200)
    #y = randint(10, 200)
    
    if show_ghost == 2:
        added_image = cv.addWeighted(background[x:x+200,y:y+200,:],alpha,chosen_img[0:200,0:200,:],1-alpha,0)
        background[x:x+200,y:y+200] = added_image
        
    # Detect faces, also slows frame rate a lot
    '''face_cascade = cv.CascadeClassifier(classifier_path)
    # Convert into grayscale
    gray = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(background, (x, y), (x+w, y+h), (255, 0, 0), 2)'''
    font = cv.FONT_HERSHEY_SIMPLEX
    temp = alpha * 100
    if temp < 50:
        distance_to_fifty = 50 - temp
        result = temp + (distance_to_fifty * 2)
    else:
        distance_to_fifty = temp - 50
        result = temp - (distance_to_fifty * 2)

    
    box = cv.addWeighted(background[10:80,10:390,:],0,box_img[0:70,0:380,:],1,0)
    # Change the region with the result
    background[10:80,10:390] = box
    cv.putText(background, f"Ghost detection: {round(result, 2)}%",(15,55), font, 1,(0, 255, 0),2,cv.LINE_AA)
    # heatmap = cv.applyColorMap(background, cv.COLORMAP_HOT) (If you want the image to be like a heatmap thing)
    grayscale = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    cv.imshow('Ghost Hunter', grayscale)
    k = cv.waitKey(10)
    # Press q to break
    if k == ord('q'):
        break
    '''# press a to increase alpha by 0.1
    if k == ord('a'):
        alpha +=0.1
        if alpha >=1.0:
            alpha = 1.0
    # press d to decrease alpha by 0.1
    elif k== ord('d'):
        alpha -= 0.1
        if alpha <=0.0:
            alpha = 0.0'''
    # Press q to quit
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
