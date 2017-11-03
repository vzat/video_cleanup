import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

class Video:
    def __init__(self, file):
        self.path = self.getFilePath(file)
        self.extension = self.getFileExtension(file)

        inVid = cv2.VideoCapture(file)

        # Refernce - https://docs.opencv.org/3.2.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704
        # Get video metadata
        self.fourcc = int(inVid.get(cv2.CAP_PROP_FOURCC))
        self.fps = inVid.get(cv2.CAP_PROP_FPS)
        self.widthVid = int(inVid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.heightVid = int(inVid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read Frames
        self.rawFrames = []
        validFrame, frame = inVid.read()
        while validFrame:
            self.rawFrames.append(frame)
            validFrame, frame = inVid.read()
        inVid.release()

        self.frames = self.rawFrames

    def display(self, compare = False):
        for rawFrame, frame in zip(self.rawFrames, self.frames):
            if compare:
                newFrame = np.hstack((frame, rawFrame))
                cv2.imshow('Video', newFrame)
            else:
                cv2.imshow('Video', frame)
            key = cv2.waitKey(int(round(self.fps)))

            if key == ord('q'):
                break

    def getFilePath(self, filename):
        slashPos = filename.rfind('/')
        filePath = ''

        # For Windows
        if slashPos == -1:
            slashPos = filename.rfind('\\')

        if slashPos != -1:
            filePath = filename[: slashPos + 1]

        return filePath

    def getFileExtension(self, filename):
        dotPos = filename.rfind('.')
        extension = '.mp4'

        if dotPos != -1 and dotPos < len(filename) - 1:
            extension = filename[dotPos :]

        return extension

    def write(self, name):
        filename = self.path + name + self.extension
        outVid = cv2.VideoWriter(filename, self.fourcc, self.fps, (self.widthVid, self.heightVid))

        for frame in self.frames:
            outVid.write(frame)

        outVid.release()

    def denoise(self, temporalWindowSize = 1):
        newFrames = []
        # for frameNo in xrange(temporalWindowSize / 2 + 95, len(self.frames) - temporalWindowSize / 2):
        #     print frameNo
        #     # newFrame = cv2.fastNlMeansDenoisingColoredMulti(srcImgs = self.frames, imgToDenoiseIndex = frameNo, temporalWindowSize = temporalWindowSize)
        #     newFrame = cv2.fastNlMeansDenoising(self.frames[frameNo])
        #     newFrames.append(newFrame)
        #     comp = np.hstack((newFrame, self.frames[frameNo]))
        #     cv2.imshow('Denoise', comp)
        #     cv2.waitKey(0)

        for frame in self.frames:
            newFrame = cv2.fastNlMeansDenoising(src = frame, h = 3, templateWindowSize = 7, searchWindowSize = 7)
            newFrames.append(newFrame)

        self.frames = newFrames


inputPath = 'videos/'
inputFile = inputPath + 'Zorro.mp4'

video = Video(inputFile)
video.denoise()
video.display(compare = True)

# frame = video.frames[99]

# # CLARHE normalisation
# # Reference: https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c
# # and: https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE
# yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
# y = yuv[:, :, 0]
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# newY = clahe.apply(y)
# yuv[:, :, 0] = newY
# newFrame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
# comparedFrames = np.hstack((newFrame, frame))
# cv2.imshow('Video', comparedFrames)

# video.write('newZorro')
# video.display()
# frame = video.frames[99]
# print video.fps
# cv2.imshow('Frame', frame)
cv2.waitKey(0)
