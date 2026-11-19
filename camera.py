import cv2 as cv
import numpy as np

# cv.CAP_DSHOW #direct show, forces CV to open Microsoft DirectShow backend vidoe capture on windows, fixes startup time
#default camera index 0
cam = cv.VideoCapture(0, cv.CAP_DSHOW)

#parameters for camera window dimension
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

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

    # #resize camera display window
    frame = cv.resize(frame, (1280, 960), interpolation=cv.INTER_AREA)

    #takes in frame(which is numPy array of pixels) and resize to the wanted dimensions with a new array. 
    # Interpolations is necessary to resize because it's resampling the image onto a new grid which needs more/less pixels depending on the need
    
    #display captured frame
    cv.imshow('Camera', frame)

    #press 'q' to exit loop = stop displaying camera
    if cv.waitKey(1) & 0xFF == ord('1'):
        break

#release capture 
cam.release()
cv.destroyAllWindows()

    
