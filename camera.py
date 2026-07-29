import cv2 as cv
import numpy as np

#default camera index 0
cam = cv.VideoCapture(0)
cv.CAP_DSHOW #direct show, forces CV to open Microsoft DirectShow backend vidoe capture on windows, fixes startup time
#parameters for camera window dimension
width = 1280
height = 960
dim = (width, height) #wanted dimensions for window

if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    #capture frame-by-frame
    ret, frame = cam.read()
    #if frame is read correctly, ret = True
    if not ret:
        print("Can't read frame. Exit")
        break

    #resize camera display window
    frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA) 

    #takes in frame(which is numPy array of pixels) and resize to the wanted dimensions with a new array. 
    # Interpolations is necessary to resize because it's resampling the image onto a new grid which needs more/less pixels depending on the need
    
    #display captured frame
    cv.imshow('Camera', frame)

    #press 'q' to exit loop = stop displaying camera
    if cv.waitKey(1) == ord('q'):
        break
#release capture 
cam.release()
cv.destroyAllWindows()

    
