from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from datetime import datetime
import csv

# Sound the alarm using mpg123 audio player


def alarm():
    global alarmOnOf1
    global alarmOnOf2
    global writingFeedback

    if alarmOnOf1:
        s = 'mpg123 -q alert.mp3'
        os.system(s)

    if alarmOnOf2:
        writingFeedback = True
        s = 'mpg123 -q alert.mp3'
        os.system(s)
        writingFeedback = False


# helper function to calculate the eye aspect ratio for an eye
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


# calculate eye aspect ratio of both eyes and find the average
def EAR(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eyeAspectRatio(leftEye)
    rightEAR = eyeAspectRatio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

# calculate mouth aspect ratio


def lipDistance(shape):
    upperLip = shape[50:53]
    upperLip = np.concatenate((upperLip, shape[61:64]))

    lowerLip = shape[56:59]
    lowerLip = np.concatenate((lowerLip, shape[65:68]))

    top_mean = np.mean(upperLip, axis=0)
    low_mean = np.mean(lowerLip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


# support for the use of external camera
argParser = argparse.ArgumentParser()
argParser.add_argument("-w", "--webcam", type=int, default=0,
                       help="index of webcam on system")
args = vars(argParser.parse_args())

# welcome message
print('Welcome to our drowsiness and Yawning Detector built to prevent accidents')
print('And also store the same data for future reference and improvement')

# Feedback - To store the data of each driver
s = input('Enter driver\'s name:')


# opening file with driver's name in the driver folder
file = open('/home/rishit/Desktop/Accident_Prevention_Using_Driver_Mood-main/Driver/'+s+'.txt', 'a')
# record = open('/home/yash/Documents/IT204/Record.csv', 'a')
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
dt_string = now.strftime("%d/%m/%Y")
file.write("\n\nRecord of " + dt_string + "\nStarting Time " +
           current_time + "\n----------------------\n")

print('Starting the Live Stream')

# Standard Values
eyeThresh = 0.3
eyeThreshFrames = 30
yawnThresh = 20

# flag and counter variables
alarmOnOf1 = False
alarmOnOf2 = False
COUNTER = 0
yawn_freq = 0
drowsy_freq = 0
writingFeedback = False
# Face Detector
detector = dlib.get_frontal_face_detector()
# Face Predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Opening Webcam and starting the stream
vs = VideoStream(src=args["webcam"]).start()

# to add delay for opening of webcam and starting the live video stream
time.sleep(1.0)


while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects all the faces and makes an array of them
    rects = detector(gray, 0)

    # checks every face
    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = EAR(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lipDistance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < eyeThresh:
            COUNTER += 1

            if COUNTER >= eyeThreshFrames:
                if alarmOnOf1 == False:
                    alarmOnOf1 = True
                    t = Thread(target=alarm, args="")
                    t.deamon = True
                    t.start()
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    file.write("Drowsy at "+current_time+"\n")
                    drowsy_freq += 1
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarmOnOf1 = False

        if (distance > yawnThresh):

            if alarmOnOf2 == False and writingFeedback == False:
                alarmOnOf2 = True
                t = Thread(target=alarm, args="")
                t.deamon = True
                t.start()
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                file.write("Yawn at "+current_time+"\n")
                yawn_freq += 1
            cv2.putText(frame, "Yawn Alert", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            alarmOnOf2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    # Stop the code when user presses q (imp : q should be pressed while the webcam live stream is in focus and not the terminal)
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break;
with open('/home/rishit/Desktop/Accident_Prevention_Using_Driver_Mood-main/Record.csv', 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow([s, str(drowsy_freq), str(yawn_freq)])
file.close()
cv2.destroyAllWindows()
vs.stop()

