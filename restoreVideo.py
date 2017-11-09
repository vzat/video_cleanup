###############################
#
#   (c) Vlad Zat 2017
#   Student No: C14714071
#   Course: DT228
#   Date: 10-11-2017
#
#	Title: Video Restoration
#
#   Introduction:
#   Restore videos by firstly performing some cleaning on each individual frame
#   such as contrast stretching, denoising and sharpening. After that temporal
#   processing is used to stabilise the image to minimise the jittering.

import numpy as np
import cv2

class Video:
    def __init__(self, file):
        print 'Reading video'
        (self.path, self.originalName, self.extension) = self.getFileDetails(file)

        inVid = cv2.VideoCapture(file)

        # Refernce - https://docs.opencv.org/3.2.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704
        # Get video metadata
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = inVid.get(cv2.CAP_PROP_FPS)
        self.width = int(inVid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(inVid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read Frames
        self.rawFrames = []
        validFrame, frame = inVid.read()
        while validFrame:
            self.rawFrames.append(frame)
            validFrame, frame = inVid.read()
        inVid.release()

        self.frames = self.rawFrames[:]

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

    def getFileDetails(self, filename):
        slashPos = filename.rfind('/')
        filePath = '/'

        # Get file path
        # For Windows
        if slashPos == -1:
            slashPos = filename.rfind('\\')

        if slashPos != -1:
            filePath = filename[: slashPos + 1]

        # Get Extension
        dotPos = filename.rfind('.')
        extension = '.avi'

        if dotPos != -1 and dotPos < len(filename) - 1:
            extension = filename[dotPos :]

        # Get video name
        name = 'video'
        if slashPos != -1 and dotPos != -1:
            name = filename[slashPos + 1 : dotPos]

        return (filePath, name, extension)

    def write(self, name):
        print 'Writing Video'
        # .avi with the XVID codec to make videos writable on Windows
        filename = self.path + name + '.avi'
        outVid = cv2.VideoWriter(filename, self.fourcc, self.fps, (self.width, self.height))

        for frame in self.frames:
            outVid.write(frame)

        outVid.release()

    def stretchHist(self):
        # Constrast Stretching
        print 'Stretching Contrast'
        for frameNo, frame in enumerate(self.frames):
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y = yuv[:, :, 0]
            (minI, maxI, _, _) = cv2.minMaxLoc(y)
            yuv[:, :, 0] = 255 * ((y - minI) / (maxI - minI))
            self.frames[frameNo] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def normalise(self):
        clahe = cv2.createCLAHE(clipLimit=1.1)
        for frameNo, frame in enumerate(self.frames):
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            luminance = yuv[:, :, 0]
            yuv[:, :, 0] = clahe.apply(luminance)
            self.frames[frameNo] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def enhanceDetail(self):
        print 'Enhancing Details and Denoising'
        for frameNo, frame in enumerate(self.frames):
            bFrame = cv2.medianBlur(frame, 9)

            # Extract most important edges to be used as details
            gFrame = cv2.cvtColor(bFrame, cv2.COLOR_BGR2GRAY)
            threshold, _ = cv2.threshold(src = gFrame, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cannyFrame = cv2.Canny(image = gFrame, threshold1 = 0.5 * threshold, threshold2 = threshold)

            # Make the details more pronounced
            shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            cannyFrame = cv2.dilate(cannyFrame, shape)
            rCannyFrame = cv2.bitwise_not(cannyFrame)

            # Blur background to denoise it
            bg = cv2.bitwise_and(bFrame, bFrame, mask = rCannyFrame)

            # Reference: https://bohr.wlu.ca/hfan/cp467/12/notes/cp467_12_lecture6_sharpening.pdf
            # Sharpen details
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype = float)
            sharpenedFrame = cv2.filter2D(frame, ddepth = -1, kernel = kernel)
            sharpenedFrame = cv2.GaussianBlur(sharpenedFrame, (5, 5), 0)

            # Extract only the details from the frame
            roi = cv2.bitwise_and(sharpenedFrame, sharpenedFrame, mask = cannyFrame)

            self.frames[frameNo] = cv2.bitwise_or(bg, roi)

    def stabilise(self):
        # Reference: https://docs.opencv.org/2.4/modules/imgproc/doc/motion_analysis_and_object_tracking.html#phasecorrelate
        print 'Stabilising Video'
        for frameNo in range(len(self.frames) - 1):
            firstFrame = self.frames[frameNo]
            secondFrame = self.frames[frameNo + 1]

            gFirstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            gSecondFrame = cv2.cvtColor(secondFrame, cv2.COLOR_BGR2GRAY)
            gFirstFrame = cv2.medianBlur(gFirstFrame, 9)
            gSecondFrame = cv2.medianBlur(gSecondFrame, 9)

            threshold, _ = cv2.threshold(src = gFirstFrame, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            gFirstFrame = cv2.Canny(image = gFirstFrame, threshold1 = 0.5 * threshold, threshold2 = threshold)

            threshold, _ = cv2.threshold(src = gSecondFrame, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            gSecondFrame = cv2.Canny(image = gSecondFrame, threshold1 = 0.5 * threshold, threshold2 = threshold)

            # Convert frames to CV_32FC1
            gFirstFrame32 = np.float32(gFirstFrame)
            gSecondFrame32 = np.float32(gSecondFrame)

            # Find the translation between the two frames
            (xDif, yDif), _ = cv2.phaseCorrelate(src1 = gSecondFrame32, src2 = gFirstFrame32)

            # Reference: https://docs.opencv.org/3.2.0/da/d6e/tutorial_py_geometric_transformations.html
            translationMatrix = np.float32([[1, 0, xDif], [0, 1, yDif]])
            newFrame = cv2.warpAffine(secondFrame, translationMatrix, (self.width, self.height))

            self.frames[frameNo + 1] = newFrame

# TODO: Replace with easygui
inputPath = 'videos/'
inputFile = inputPath + 'Zorro.mp4'

video = Video(inputFile)

video.stretchHist()
video.stabilise()
video.enhanceDetail()

video.write('Restored_' + video.originalName)
