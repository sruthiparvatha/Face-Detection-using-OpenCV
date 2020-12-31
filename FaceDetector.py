import cv2
from random import randrange

# Load some pre-trained data on frontal face from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to read faces from
# img = cv2.imread('hope.jpg')
# cv2.imshow('hope', img)
# To capture video from webcam
webcam = cv2.VideoCapture(0)

# Since it is video need to iterate forever to capture all frames one by one
while True:
    successful_frame_read, frame= webcam.read()
    # Need to convert each frame to grayscale in order to pass to haar cascade classifier
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and return the coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # To draw rectangle around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
    cv2.imshow('Face Detector Application', frame)
    # Wait till keypressed or 1ms whichever is earlier before going to the next iteration, to read the next frame
    key = cv2.waitKey(1)
    # Press Q to exit the application
    if key =='q' or key =='Q':
        break

# Release the video capture
webcam.release()

print("Code completed")