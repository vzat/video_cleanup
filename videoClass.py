import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

class Video:
    def __init__(self, videoPath):
        inVid = cv2.VideoCapture(videoPath)

        # Refernce - https://docs.opencv.org/3.2.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704
        self.fourcc = int(inVid.get(cv2.CAP_PROP_FOURCC))
        self.fps = inVid.get(cv2.CAP_PROP_FPS)
        self.widthVid = int(inVid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.heightVid = int(inVid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.rawFrames = []
        validFrame, frame = inVid.read()
        while validFrame:
            self.rawFrames.append(frame)
            validFrame, frame = inVid.read()
        inVid.release()

        self.frames = self.rawFrames

inputPath = 'videos/'
inputFile = inputPath + 'Zorro.mp4'
video = Video(inputFile)
frame = video.frames[99]
print video.fps
cv2.imshow('Frame', frame)
cv2.waitKey(0)
