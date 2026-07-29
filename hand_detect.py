import cv2 as cv
import numpy as np
import mediapipe as mp
import time 
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
#build and configure tasks
class landmarker_and_result():
    def __init__(self):
        self.result = None #change this when want to display results
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.createLandmarker()
#create landmarker instance with the live stream mode
    def createLandmarker(self):
        #callback function
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result
    
        options = mp.tasks.vision.HandLandmarkerOptions( 
            base_options = mp.tasks.BaseOptions(model_asset_path = MODEL_PATH), # path to model
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
            num_hands = 2, # track both hands
            min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
            min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
            min_tracking_confidence = 0.3, # lower than value to get predictions more often
            result_callback = update_result)
    
    #init landmarker
        self.landmarker = self.landmarker.create_from_options(options)

    #convert image frame into image datatype that MediaPipe expects
    def detect_async(self, frame):
        #convert np to mp image
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
        #detects landmarks
        self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
    #function to close landmarker
    def close(self):
        #close landmarker
        self.landmarker.close() 

def draw_landmarks_on_image(frame, detection_result):
    #if no result or no landmarks, return image as is
    if detection_result is None or not detection_result.hand_landmarks:
        return frame
    
    #define landmarks connections
    HAND_CONNECTIONS = [
        #thumb 
        (0, 1), (1, 2), (2, 3), (3, 4),
        #index
        (0, 5), (5, 6), (6, 7), (7, 8),
        #middle
        (0, 9), (9, 10), (10, 11), (11,12),
        #ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        #palm(knuckles across)
        (5, 9), (9, 13), (13, 17),
        #pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        #ring base to wrist
        (0, 5), (5, 9)
    ]

    for hand_landmarks in detection_result.hand_landmarks:
        #get hand dimension once per hand
        h, w, _ = frame.shape

        #draw connections(skeleton)
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]

            #landmarks are normalized (0.0 to 1.0), scale to pixel coords
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            #parameters (image, start, end, color, thickness)
            cv.line(frame, start_point, end_point, (255, 255, 255), 5)

        #draw dot on each of the 21 landmarks
        for idx, landmark in enumerate(hand_landmarks):
            cx, cy = int(landmark.x * w), int(landmark.y * h)

            #fingertips = 4, 8, 12, 16, 20
            if idx in [4, 8, 12, 16, 20]:
                #parameters (image, center point(x, y), thickness, color, thickness(-1 means filled circle))
                cv.circle(frame, (cx, cy), 8, (0, 255, 0), -1) # green
            else: #else it joints
                cv.circle(frame, (cx, cy), 8, (0, 0, 255), -1) #red
   
    return frame
  

def main():
    #access video
    cam = cv.VideoCapture(0)

    if not cam.isOpened():
        print("Can't access camera")
        exit()

    #intialize capture loop
    hand_landmarker = landmarker_and_result()
    #if able to open camera
    while True:
        #read frame by frame
        success, frame = cam.read()
        #if not success, then can't capture
        if not success:
            print("Can't read frame")
            break
        #mirror the displayed video
        frame = cv.flip(frame, 1)

        hand_landmarker.detect_async(frame)
        #draw landmarks if available 
        frame = draw_landmarks_on_image(frame, hand_landmarker.result)

        #display video output
        cv.imshow('Video', frame)
        #press q to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    #release frame and destroy all windows
    cam.release()
    cv.destroyAllWindows()
    hand_landmarker.close()

if __name__ == "__main__":
    main()