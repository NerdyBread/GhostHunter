import cv2 as cv
import numpy as np
from random import randint, choice

skull_path = r"C:\Users\afondacaro\ghostHunter\skull.png"
ghost_path = r"C:\Users\afondacaro\ghostHunter\ghost.jpg"
skull_img = cv.imread(skull_path)
ghost_img = cv.imread(ghost_path)

images = [skull_img, ghost_img]

classifier_path = r"C:\Users\afondacaro\ghostHunter\faceClassifier.xml"

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

count = 0
show_ghost = int()

alpha = 1

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
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    background = cv.flip(frame, 1)

    # Define x and y here if you want the image to shake
    #x = randint(10, 200)
    #y = randint(10, 200)
    
    if show_ghost == 2:
        added_image = cv.addWeighted(background[x:x+200,y:y+200,:],alpha,chosen_img[0:200,0:200,:],1-alpha,0)
        # Change the region with the result
        background[x:x+200,y:y+200] = added_image
    # For displaying current value of alpha(weights)
    '''face_cascade = cv.CascadeClassifier(classifier_path)
    # Convert into grayscale
    gray = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(background, (x, y), (x+w, y+h), (255, 0, 0), 2)'''
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(background,f'alpha:{alpha}',(10,30), font, 1,(255,255,255),2,cv.LINE_AA)
    # heatmap = cv.applyColorMap(background, cv.COLORMAP_HOT)
    grayscale = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    cv.imshow('a', grayscale)
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
    '''gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', gray)'''
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
