import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image

class Video:
    def __init__(self, file):
        self.path = self.getFilePath(file)
        self.extension = self.getFileExtension(file)

        inVid = cv2.VideoCapture(file)

        # Refernce - https://docs.opencv.org/3.2.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704
        # Get video metadata
        self.fourcc = int(inVid.get(cv2.CAP_PROP_FOURCC))
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
        outVid = cv2.VideoWriter(filename, self.fourcc, self.fps, (self.width, self.height))

        for frame in self.frames:
            outVid.write(frame)

        outVid.release()

    def eqHist(self):
        for frameNo, frame in enumerate(self.frames):
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y = yuv[:, :, 0]
            newY = cv2.equalizeHist(y)
            yuv[:, :, 0] = newY
            self.frames[frameNo] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def stretchHist(self):
        # Constrast Stretching
        for frameNo, frame in enumerate(self.frames):
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y = yuv[:, :, 0]
            (minI, maxI, _, _) = cv2.minMaxLoc(y)
            yuv[:, :, 0] = 255 * ((y - minI) / (maxI - minI))
            self.frames[frameNo] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def normalise(self):
        clahe = cv2.createCLAHE(clipLimit=2.0)
        for frameNo, frame in enumerate(self.frames):
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            luminance = yuv[:, :, 0]
            yuv[:, :, 0] = clahe.apply(luminance)
            self.frames[frameNo] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def sharpen(self):
        for frameNo, frame in enumerate(self.frames):
            # Reference: http://web.pdx.edu/~jduh/courses/Archive/geog481w07/Students/Ludwig_ImageConvolution.pdf
            # or: https://bohr.wlu.ca/hfan/cp467/12/notes/cp467_12_lecture6_sharpening.pdf
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype = float)
            # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype = float)
            frame = cv2.filter2D(frame, ddepth = -1, kernel = kernel)
            # frame = cv2.medianBlur(frame, 3)
            self.frames[frameNo] = frame

    # Not working that well
    def newDenoise(self):
        for frameNo in range(len(self.frames) - 1):
            firstFrame = self.frames[frameNo]
            secondFrame = self.frames[frameNo + 1]

            gFirstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            bFirstFrame = cv2.medianBlur(gFirstFrame, 9)

            difs = gFirstFrame - bFirstFrame

            _, contours, _ = cv2.findContours(image = difs.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

            # contours = sorted(contours, key = lambda contour:cv2.contourArea(contour))

            mask = np.zeros((self.height, self.width, 1), np.uint8)
            for contour in contours:
                area = cv2.contourArea(contour)

                if area < 10:
                    cv2.drawContours(mask, contour, -1, 255, 2)

            # cv2.imshow('Mask', mask)
            # cv2.waitKey(0)

            rmask = cv2.bitwise_not(mask)

            firstROI = cv2.bitwise_and(firstFrame, firstFrame, mask = rmask)
            secondROI = cv2.bitwise_and(secondFrame, secondFrame, mask = mask)

            self.frames[frameNo] = cv2.bitwise_or(firstROI, secondROI)

    def superResolution(self):
        for frameNo in range(len(self.frames) - 2):
            frame1 = self.frames[frameNo]
            frame2 = self.frames[frameNo + 1]
            frame3 = self.frames[frameNo + 2]

            gFrame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gFrame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gFrame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

            iFrame = np.zeros((self.height, self.width, 1), np.uint8)
            for x in range(self.width):
                for y in range(self.height):
                    minPixel = min(gFrame1[y, x], gFrame3[y, x])
                    iFrame[y, x] = minPixel + (abs(gFrame1[y, x] - gFrame3[y, x]) / 2)

            # comparedFrames = np.hstack((iFrame, gFrame2))
            newFrame = cv2.subtract(gFrame2, iFrame)

            cv2.imshow('Comp', newFrame)
            cv2.waitKey(0)

    def newSuperResolution(self):
        sr = superres.superResolution()

    def testLab(self):
        for frameNo, frame in enumerate(self.frames):
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            l = lab[:, :, 0]
            (minI, maxI, _, _) = cv2.minMaxLoc(l)
            lab[:, :, 0] = 255 * ((l - minI) / (maxI - minI))
            self.frames[frameNo] = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    def removeArtifacts(self):
        for frameNo in range(len(self.frames) - 1):
            firstFrame = self.frames[frameNo]
            secondFrame = self.frames[frameNo + 1]

            gFirstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            _, thImg = cv2.threshold(src = gFirstFrame, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            _, contours, _ = cv2.findContours(image = thImg.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

            mask = np.zeros((self.height, self.width, 1), np.uint8)
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(points = contour)
                # if abs(w - h) < 50:
                cv2.drawContours(mask, contour, -1, 255, 20)

            rmask = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(firstFrame, firstFrame, mask = rmask)
            roi = cv2.bitwise_and(secondFrame, secondFrame, mask = mask)

            newFrame = cv2.bitwise_or(bg, roi)
            self.frames[frameNo] = newFrame.copy()

            # cv2.imshow('bg', newFrame)
            # cv2.waitKey()

    def newRemoveArtifacts(self):
        for frameNo in range(len(self.frames) - 2):
            frame1 = self.frames[frameNo]
            frame2 = self.frames[frameNo + 1]
            frame3 = self.frames[frameNo + 2]

            gFrame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gFrame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gFrame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

            dif1 = gFrame2 - gFrame1
            dif2 = gFrame2 - gFrame3

            dif1[dif1 > 127] = 255
            dif1[dif1 < 128] = 0

            dif2[dif2 > 127] = 255
            dif2[dif2 < 128] = 0

            diffs = cv2.bitwise_and(dif1, dif2)

            cv2.imshow('Diffs', diffs)
            # cv2.imshow('Dif1', dif1)
            # cv2.imshow('Dif2', dif2)
            cv2.waitKey(0)

    def blobDetector(self):
        blobDetector = cv2.SimpleBlobDetector_create()

        for frameNo, frame in enumerate(self.frames):
            gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            threshold, _ = cv2.threshold(src = gFrame, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            gFrame = cv2.Canny(image = gFrame.copy(), threshold1 = 0.5 * threshold, threshold2 = threshold)

            shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gFrame = cv2.morphologyEx(gFrame, cv2.MORPH_CLOSE, shape)

            keyPoints = blobDetector.detect(gFrame)
            print len(keyPoints)

            frame = cv2.drawKeypoints(frame, keyPoints, -1, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.imshow('Frame', frame)
            cv2.waitKey(0)

    def testSharpen(self):
        for frameNo, frame in enumerate(self.frames):
            gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            oFrame = gFrame.copy()
            gFrame = cv2.medianBlur(gFrame, 9)

            scale = 0.5
            downScaledFrame = cv2.resize(src = gFrame, dsize = (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_AREA)

            threshold, _ = cv2.threshold(src = downScaledFrame.copy(), thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cannyFrame = cv2.Canny(image = downScaledFrame.copy(), threshold1 = 0.5 * threshold, threshold2 = threshold)

            upScaledCanny = cv2.resize(src = cannyFrame, dsize = (0, 0), fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_CUBIC)
            rCanny = cv2.bitwise_not(upScaledCanny)


            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype = float)
            sFrame = cv2.filter2D(oFrame, ddepth = -1, kernel = kernel)

            bg = cv2.bitwise_and(oFrame, oFrame, mask = rCanny)
            roi = cv2.bitwise_and(sFrame, sFrame, mask = upScaledCanny)

            newFrame = cv2.bitwise_or(bg, roi)

            cv2.imshow('Downscaled Frame', newFrame)
            cv2.waitKey(0)

    def foregroundMask(self):
        bsMOG = cv2.bgsegm.createBackgroundSubtractorMOG()
        # bsMOG2 = cv2.createBackgroundSubtractorMOG2()
        # bsGMG = cv2.bgsegm.createBackgroundSubtractorGMG()
        for frame in self.frames:
            mask = bsMOG.apply(frame)
            cv2.imshow('Mask', mask)
            cv2.waitKey(0)

    def getScenes(self):
        for frameNo in range(len(self.frames) - 1):
            firstFrame = self.frames[frameNo]
            secondFrame = self.frames[frameNo + 1]

            gFirstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            gSecondFrame = cv2.cvtColor(secondFrame, cv2.COLOR_BGR2GRAY)

            mse = np.mean((gFirstFrame - gSecondFrame) ** 2)

            if mse > 50:
                cv2.imshow('Frame', gFirstFrame)
                cv2.waitKey(0)


    def stabilise(self):
        # Reference: https://docs.opencv.org/2.4/modules/imgproc/doc/motion_analysis_and_object_tracking.html#phasecorrelate
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

            # Convert to CV_32FC1
            gFirstFrame32 = np.float32(gFirstFrame)
            gSecondFrame32 = np.float32(gSecondFrame)

            (xDif, yDif), _ = cv2.phaseCorrelate(src1 = gSecondFrame32, src2 = gFirstFrame32)

            # Reset frame position is the scene changes
            # if abs(np.std(gFirstFrame) - np.std(gSecondFrame)) > 5:
            #     xDif = 0
            #     yDif = 0

            # Reference: https://docs.opencv.org/3.2.0/da/d6e/tutorial_py_geometric_transformations.html
            translationMatrix = np.float32([[1, 0, xDif], [0, 1, yDif]])
            newFrame = cv2.warpAffine(secondFrame, translationMatrix, (self.width, self.height))

            self.frames[frameNo + 1] = newFrame

            if frameNo % 100 == 0:
                print int(float(frameNo) / len(self.frames) * 100.0), '%'

    def newNewDenoise(self):
        for frameNo, frame in enumerate(self.frames):
            gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            scale = 0.25
            downscaledFrame = cv2.resize(src = gFrame, dsize = (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_AREA)

            newFrame = cv2.fastNlMeansDenoising(downscaledFrame)
            mask = newFrame - downscaledFrame

            upscaledMask = cv2.resize(src = mask, dsize = (self.width, self.height), interpolation = cv2.INTER_CUBIC)
            shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            upscaledMask = cv2.erode(upscaledMask, shape)

            rmask = cv2.bitwise_not(upscaledMask)

            secondFrame = cv2.cvtColor(self.frames[frameNo + 1], cv2.COLOR_BGR2GRAY)
            bg = cv2.bitwise_and(gFrame, gFrame, mask = rmask)
            roi = cv2.bitwise_and(secondFrame, secondFrame, mask = upscaledMask)

            denoisedFrame = cv2.bitwise_or(bg, roi)

            cv2.imshow('Denoise', denoisedFrame)
            cv2.waitKey(0)


inputPath = 'videos/'
inputFile = inputPath + 'Zorro.mp4'

video = Video(inputFile)

# video.stretchHist();
# video.sharpen()
# video.normalise()
# video.stabilise()
# video.newDenoise()
video.newNewDenoise()

# frame = video.frames[107]

# gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# bFrame = cv2.medianBlur(gFrame, 9)
# bFrame = gFrame.copy()
#
# threshold, thImg = cv2.threshold(src = bFrame, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cannyImg = cv2.Canny(image = bFrame, threshold1 = 0.5 * threshold, threshold2 = threshold)
#
# cv2.imshow('Threshold', thImg)
# cv2.waitKey(0)
# cv2.imshow('Canny', cannyImg)


video.display(compare = True)
cv2.waitKey(0)
