import argparse

from detector import Detector

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--shape-predictor", required=True,
    #                 help="path to facial landmark predictor")
    ap.add_argument("-a", "--alarm", type=str, default="alarm.wav",
                    help="path alarm .WAV file")
    ap.add_argument("-v", "--webcam", type=str, default='0',
                    help="index of webcam on system or path to video file")
    args = vars(ap.parse_args())

    detector = Detector(args)
    detector.run()
