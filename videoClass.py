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

    def denoise(self, temporalWindowSize = 3):
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

    def getYUV(self):
        yuvFrames = []
        yFrames = []
        for frame in self.frames:
            newFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuvFrames.append(newFrame)
            yFrames.append(newFrame[:, :, 0])

        return (yuvFrames, yFrames)

    def getBGR(self, yuvFrames):
        bgrFrames = []
        for frame in yuvFrames:
            newFrame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            bgrFrames.append(newFrame)

        return bgrFrames

    def eqHist(self, frame, minI, maxI):
        # Constrast Stretching
        return 255 * ((frame - minI) / (maxI - minI))

    def normalise(self, windowSize = 5):
        (yuvFrames, yFrames) = self.getYUV()

        for frameNo, frame in enumerate(yFrames):
            startFrame = frameNo - windowSize / 2
            endFrame = frameNo + windowSize / 2

            # Make sure frames are within bounds
            if startFrame < 0:
                startFrame = 0
            if endFrame >= len(yFrames):
                endFrame = len(yFrames) - 1

            windowFrames = yFrames[startFrame:endFrame]

            # Find the mean of the frames in the window
            meanFrames = []
            for wFrame in windowFrames:
                mean = np.mean(wFrame)
                meanFrames.append(mean)

            # Find closest frame to the mean of the window
            meanWindow = np.mean(meanFrames)
            difMean = abs(meanWindow - meanFrames[0])
            bestFrameNo = 0
            for meanFrameNo, meanFrame in enumerate(meanFrames):
                if abs(meanWindow - meanFrame) < difMean:
                    difMean = abs(meanWindow - meanFrame)
                    bestFrameNo = meanFrameNo

            bestFrame = yFrames[bestFrameNo]
            (minVal, maxVal, _, _) = cv2.minMaxLoc(bestFrame)

            newY = self.eqHist(frame, minVal, maxVal)
            yuvFrames[frameNo][:, :, 0] = newY

        self.frames = self.getBGR(yuvFrames)

    def sharpen(self):
        newFrames = []
        for frame in self.frames:
            # Reference: http://web.pdx.edu/~jduh/courses/Archive/geog481w07/Students/Ludwig_ImageConvolution.pdf
            # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype = float)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype = float)
            frame = cv2.filter2D(frame, ddepth = -1, kernel = kernel)
            newFrames.append(frame)

        self.frames = newFrames

    def gammaCorrection(self):
        # Reference: http://www.normankoren.com/makingfineprints1A.html#Gammabox
        newFrames = []
        for frame in self.frames:
            gamma = 2
            normalisedFrame = frame / 255.0
            enhancedFrame = normalisedFrame ** (1 / gamma)
            newFrame = enhancedFrame * 255
            newFrames.append(newFrame)
        self.frames = newFrames

inputPath = 'videos/'
inputFile = inputPath + 'Zorro.mp4'

video = Video(inputFile)
# print 'Denoising'
# video.denoise()
# print 'Normalising'
# video.normalise()
# video.display(compare = True)

def eqHist(img):
    (height, width) = img.shape[:2]
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    luminance = yuv[:, :, 0]
    (minI, maxI, _, _) = cv2.minMaxLoc(luminance)

    newLuminance = luminance.copy()
    for x in range(width):
        for y in range(height):
            newLuminance[y, x] = 255.0 * float(luminance[y, x] - minI) / float(maxI - minI)
            print luminance[y, x], newLuminance[y, x]

    yuv[:, :, 0] = newLuminance
    cv2.imshow('new', newLuminance)
    cv2.waitKey(0)

    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

frame = video.frames[50]
# eqHist(frame)
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# newFrame = cv2.equalizeHist(frame)
# newFrame = frame.copy()
# newFrame = cv2.normalize(frame, newFrame, 0.0, 255.0, cv2.NORM_MINMAX)
# values = newFrame.ravel()
# hist = plt.hist(x = values, bins = 256, range = [0, 256])
# plt.show(hist)

# video.sharpen()
video.gammaCorrection()
video.display(compare = True)
# cv2.imshow('Frame', frame)

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
