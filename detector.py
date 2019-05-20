import cv2
import time
import dlib
import imutils
import playsound
import numpy as np
from threading import Thread
from imutils import face_utils
from scipy.spatial import distance as dist
from imutils.video import VideoStream, FileVideoStream

SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"


class Detector:
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold for to set off the
    # alarm
    EYE_AR_THRESH = 0.17
    EYE_AR_CONSEC_FRAMES = 60

    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    COUNTER = 0
    ALARM_ON = False

    def __init__(self, kwargs):
        self.args = kwargs

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        print("[INFO] loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (self.l_start, self.l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.r_start, self.r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # start the video stream thread
        print("[INFO] starting video stream thread...")

        try:
            video_src = int(self.args["webcam"])
            print('[INFO] Using webcam...')
            self.vs = VideoStream(src=video_src).start()
        except:
            print('[INFO] Using video file...')
            self.vs = FileVideoStream(path=self.args["webcam"]).start()

        time.sleep(1.0)

    def __del__(self):
        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.vs.stop()
        print('Cleaned up!')

    @staticmethod
    def sound_alarm(path):
        # play an alarm sound
        # Linux
        playsound.playsound(path)
        # MacOS
        # playsound.playsound(path, False)

    @staticmethod
    def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        a = dist.euclidean(eye[1], eye[5])
        b = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        c = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (a + b) / (2.0 * c)

        # return the eye aspect ratio
        return ear

    def run(self):
        # loop over frames from the video stream
        while True:
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            # channels)
            frame = self.vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                left_eye = shape[self.l_start:self.l_end]
                right_eye = shape[self.r_start:self.r_end]
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)

                # average the eye aspect ratio together for both eyes
                ear = (left_ear + right_ear) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                # left_eyeHull = cv2.convexHull(left_eye)
                # right_eyeHull = cv2.convexHull(right_eye)
                # cv2.drawContours(frame, [left_eyeHull], -1, (0, 255, 0), 1)
                # cv2.drawContours(frame, [right_eyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1

                    # if the eyes were closed for a sufficient number of
                    # then sound the alarm
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        # if the alarm is not on, turn it on
                        if not self.ALARM_ON:
                            self.ALARM_ON = True

                            # check to see if an alarm file was supplied,
                            # and if so, start a thread to have the alarm
                            # sound played in the background
                            if self.args["alarm"] != "no":
                                t = Thread(target=self.sound_alarm,
                                           args=(self.args["alarm"],))
                                t.daemon = True
                                t.start()

                        # draw an alarm on the frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                    self.COUNTER = 0
                    self.ALARM_ON = False

                # draw the computed eye aspect ratio on the frame to help
                # with debugging and setting the correct eye aspect ratio
                # thresholds and frame counters
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # show the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
