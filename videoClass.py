import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
# import easygui

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
        outVid = cv2.VideoWriter(filename, self.fourcc, self.fps, (self.width, self.height))

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
            # yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # y = yuv[:, :, 0]
            # newY = cv2.fastNlMeansDenoising(src = y, h = 3, templateWindowSize = 7, searchWindowSize = 21)
            # yuv[:, :, 0] = newY
            # newFrame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            newFrame = cv2.fastNlMeansDenoisingColored(src = frame, h = 3, templateWindowSize = 5, searchWindowSize = 5)
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
            frame = cv2.medianBlur(frame, 3)
            newFrames.append(frame)

        self.frames = newFrames

    def removeArtifacts(self):
        # WIP
        newFrames = [self.frames[0]]
        for frameNo in range(len(self.frames) - 1):
            firstFrame = self.frames[frameNo]
            secondFrame = self.frames[frameNo + 1]

            gFirstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            gSecondFrame = cv2.cvtColor(secondFrame, cv2.COLOR_BGR2GRAY)
            gFirstFrame = cv2.bilateralFilter(gFirstFrame, 9, 75, 75)
            gSecondFrame = cv2.bilateralFilter(gSecondFrame, 9, 75, 75)

            # diffs = cv2.absdiff(gFirstFrame, gSecondFrame)
            diffs = gSecondFrame - gFirstFrame
            diffs[diffs < 127] = 0
            # print np.mean(diffs)

            # INPAINT_NS or INPAINT_TELEA
            if np.mean(diffs) < 10:
                newFrame = cv2.inpaint(src = secondFrame, inpaintMask = diffs, inpaintRadius = 3, flags = cv2.INPAINT_NS)
            else:
                newFrame = secondFrame.copy()

            newFrames.append(newFrame)

        self.frames = newFrames

    def getMatches(self, img1, img2):
        orb = cv2.ORB_create()
        kp1, desc1 = orb.detectAndCompute(img1, None)
        kp2, desc2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key = lambda match:match.distance)

        matchedCoordinates = []
        for match in matches:
            keyPoint1 = kp1[match.queryIdx]
            keyPoint2 = kp2[match.trainIdx]

            currentMatch = {
                'pt1': {
                    'x': keyPoint1.pt[0],
                    'y': keyPoint1.pt[1]
                },
                'pt2': {
                    'x': keyPoint2.pt[0],
                    'y': keyPoint2.pt[1]
                },
                'distance': match.distance
            }

            matchedCoordinates.append(currentMatch)

        return matchedCoordinates

    def stabilise(self):
        for frameNo in range(len(self.frames) - 1):
            firstFrame = self.frames[frameNo]
            secondFrame = self.frames[frameNo + 1]

            gFirstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            gSecondFrame = cv2.cvtColor(secondFrame, cv2.COLOR_BGR2GRAY)
            gFirstFrame = cv2.bilateralFilter(gFirstFrame, 9, 75, 75)
            gSecondFrame = cv2.bilateralFilter(gSecondFrame, 9, 75, 75)

            matches = self.getMatches(gFirstFrame, gSecondFrame)
            # matches = self.getMatches(firstFrame, secondFrame)
            # match = matches[0]
            # print cv2.estimateRigidTransform(firstFrame, secondFrame, False)[0]

            # TODO Maybe use the average of the first 5 - 10 matches??
            # xDif = int(round(match['pt1']['x'] - match['pt2']['x']))
            # yDif = int(round(match['pt1']['y'] - match['pt2']['y']))

            noMatches = 10.0
            totalX = totalY = 0
            for matchIndex in range(int(noMatches)):
                if matchIndex < len(matches):
                    match = matches[matchIndex]
                    totalX += int(round(match['pt1']['x'] - match['pt2']['x']))
                    totalY += int(round(match['pt1']['y'] - match['pt2']['y']))

            xDif = int(round(totalX / noMatches))
            yDif = int(round(totalY / noMatches))

            if abs(np.std(gFirstFrame) - np.std(gSecondFrame)) > 5:
                xDif = 0
                yDif = 0

            if abs(xDif) > 20 or abs(yDif) > 20:
                xDif = 0
                yDif = 0

            # Image has shifted
            if xDif != 0 or yDif != 0:
                newFrame = np.zeros((self.height, self.width, 3), np.uint8)

                if xDif < 0:
                    startOldX = - xDif
                    endOldX = self.width
                    startNewX = 0
                    endNewX = self.width + xDif
                else:
                    startOldX = 0
                    endOldX = self.width - xDif
                    startNewX = xDif
                    endNewX = self.width

                if yDif < 0:
                    startOldY = - yDif
                    endOldY = self.height
                    startNewY = 0
                    endNewY = self.height + yDif
                else:
                    startOldY = 0
                    endOldY = self.height - yDif
                    startNewY = yDif
                    endNewY = self.height

                print xDif, yDif
                newFrame[startNewY : endNewY, startNewX : endNewX] = secondFrame[startOldY : endOldY, startOldX : endOldX]
            else:
                newFrame = secondFrame.copy()

            self.frames[frameNo + 1] = newFrame

            if frameNo % 100 == 0:
                print int(float(frameNo) / len(self.frames) * 100.0), '%'


    # def getAvgMovement(self, corners):
    #     avgX = 0
    #     avgY = 0
    #     for corner in corners:
    #         avgX += corner[0][0]
    #         avgY += corner[0][1]
    #     avgX /= len(corners)
    #     avgY /= len(corners)
    #
    #     return (avgX, avgY)
    #
    # def getMovement(self, corners, newCorners):
    #     (avgX, avgY) = self.getAvgMovement(corners)
    #     (newAvgX, newAvgY) = self.getAvgMovement(newCorners)
    #
    #     return (int(round(avgX - newAvgX)), int(round(avgY - newAvgY)))

    # def opticalFlow(self):
    #     frame = self.frames[0]
    #     gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     corners = cv2.goodFeaturesToTrack(gFrame, maxCorners = 100, qualityLevel = 0.1, minDistance = 10)
    #     for frameNo in range(1, len(self.frames) - 1):
    #         prevFrame = frame
    #         frame = self.frames[frameNo]
    #
    #         gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #         if frameNo % 5 == 0:
    #             newCorners = cv2.goodFeaturesToTrack(gFrame, maxCorners = 100, qualityLevel = 0.1, minDistance = 10)
    #             (xDif, yDif) = self.getMovement(corners, newCorners)
    #             corners = newCorners
    #         else:
    #             newCorners = corners
    #             newCorners, _, _ = cv2.calcOpticalFlowPyrLK(prevImg = prevFrame, nextImg = frame, prevPts = corners, nextPts = newCorners)
    #             (xDif, yDif) = self.getMovement(corners, newCorners)
    #             corners = newCorners
    #
    #         # Image has shifted
    #         if xDif != 0 or yDif != 0:
    #             newFrame = np.zeros((self.height, self.width, 3), np.uint8)
    #
    #             if xDif < 0:
    #                 startOldX = - xDif
    #                 endOldX = self.width
    #                 startNewX = 0
    #                 endNewX = self.width + xDif
    #             else:
    #                 startOldX = 0
    #                 endOldX = self.width - xDif
    #                 startNewX = xDif
    #                 endNewX = self.width
    #
    #             if yDif < 0:
    #                 startOldY = - yDif
    #                 endOldY = self.height
    #                 startNewY = 0
    #                 endNewY = self.height + yDif
    #             else:
    #                 startOldY = 0
    #                 endOldY = self.height - yDif
    #                 startNewY = yDif
    #                 endNewY = self.height
    #
    #             print xDif, yDif
    #             newFrame[startNewY : endNewY, startNewX : endNewX] = frame[startOldY : endOldY, startOldX : endOldX]
    #         else:
    #             newFrame = frame.copy()
    #
    #         self.frames[frameNo] = newFrame

            # for corner in corners:
            #     x = corner[0][0]
            #     y = corner[0][1]
            #     cv2.circle(frame, (x, y), 3, 255, -1)


            # gFirstFrame = cv2.medianBlur(gFirstFrame, 5)
            # gSecondFrame = cv2.medianBlur(gFirstFrame, 5)

            # threshold, _ = cv2.threshold(src = gFirstFrame, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # cannyImg = cv2.Canny(image = gFirstFrame, threshold1 = 0.5 * threshold, threshold2 = threshold)
            #
            # _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)
            #
            # newFrame = np.zeros((self.height, self.width, 1), np.uint8)
            # for contour in contours:
            #     # hull = cv2.convexHull(contour)
            #     arcPercentage = 0.1
            #     epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
            #     corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
            #     # if cv2.contourArea(hull) < 3:
            #     cv2.drawContours(newFrame, corners, -1, 255, 3)


            # for corner in corners:
            #     x = corner[0][0]
            #     y = corner[0][1]
            #     cv2.circle(firstFrame, (x, y), 3, 255, -1)

            # cv2.imshow('Image', frame)
            # cv2.waitKey(0)


    # def smartSharpen(self):
    #     # Reference: https://www.gimp.org/tutorials/Smart_Sharpening/
    #     newFrames = []
    #     for frame in self.frames:
    #         gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         gFrame = cv2.bilateralFilter(gFrame, 9, 75, 75)
    #
    #         # Find edges
    #         threshold, _ = cv2.threshold(src = gFrame, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #         cannyImg = cv2.Canny(image = gFrame, threshold1 = 0.5 * threshold, threshold2 = threshold)
    #
    #         # Make the edges thicker
    #         # shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #         # cannyImg = cv2.morphologyEx(cannyImg, cv2.MORPH_CLOSE, shape)
    #
    #         rCannyImg = cv2.bitwise_not(cannyImg)
    #         roi = cv2.bitwise_and(frame, frame, mask = cannyImg)
    #         bg = cv2.bitwise_and(frame, frame, mask = rCannyImg)
    #
    #         # Sharpen edges
    #         # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype = float)
    #         # roi = cv2.filter2D(roi, ddepth = -1, kernel = kernel)
    #
    #         # Combine images
    #         # newFrame = cv2.addWeighted(frame, 0.3, roi, 0.7, 0)
    #         newFrame = cv2.bitwise_or(bg, roi)
    #
    #         newFrames.append(newFrame)
    #         # cv2.imshow('dsa', newFrams)
    #         # cv2.waitKey(0)
    #     self.frames = newFrames

    # def removeGraininess(self):
    #     newFrames = []
    #     for frame in self.frames:
    #         # gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         bFrame = cv2.bilateralFilter(frame,9,75,75)
    #
    #         # kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype = float)
    #         # mask = cv2.filter2D(gFrame, ddepth = -1, kernel = kernel)
    #
    #         # # INPAINT_NS or INPAINT_TELEA
    #         # newFrame = cv2.inpaint(src = frame, inpaintMask = mask, inpaintRadius = 1, flags = cv2.INPAINT_TELEA)
    #
    #         newFrames.append(bFrame)
    #
    #     self.frames = newFrames

    def phaseCorrelate(self):
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

            (xDif, yDif), _ = cv2.phaseCorrelate(src1 = gFirstFrame32, src2 = gSecondFrame32)

            xDif = int(round(-xDif))
            yDif = int(round(-yDif))

            # Image has shifted
            if xDif != 0 or yDif != 0:
                newFrame = np.zeros((self.height, self.width, 3), np.uint8)

                if xDif < 0:
                    startOldX = - xDif
                    endOldX = self.width
                    startNewX = 0
                    endNewX = self.width + xDif
                else:
                    startOldX = 0
                    endOldX = self.width - xDif
                    startNewX = xDif
                    endNewX = self.width

                if yDif < 0:
                    startOldY = - yDif
                    endOldY = self.height
                    startNewY = 0
                    endNewY = self.height + yDif
                else:
                    startOldY = 0
                    endOldY = self.height - yDif
                    startNewY = yDif
                    endNewY = self.height

                print xDif, yDif
                newFrame[startNewY : endNewY, startNewX : endNewX] = secondFrame[startOldY : endOldY, startOldX : endOldX]
            else:
                newFrame = secondFrame.copy()

            self.frames[frameNo + 1] = newFrame

            if frameNo % 100 == 0:
                print int(float(frameNo) / len(self.frames) * 100.0), '%'

    # def morph(self):
    #     # Doesn't do much
    #     newFrames = []
    #     for frame in self.frames:
    #         shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #         # frame = cv2.erode(frame, shape)
    #         # frame = cv2.dilate(frame, shape)
    #         # frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, shape)
    #         frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, shape)
    #         shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #         frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, shape)
    #         newFrames.append(frame)
    #     self.frames = newFrames

    # def gammaCorrection(self):
    #     # Reference: http://www.normankoren.com/makingfineprints1A.html#Gammabox
    #     newFrames = []
    #     for frame in self.frames:
    #         gamma = 2
    #         normalisedFrame = frame / 255.0
    #         enhancedFrame = normalisedFrame ** (1 / gamma)
    #         newFrame = enhancedFrame * 255
    #         newFrames.append(newFrame)
    #     self.frames = newFrames

inputPath = 'videos/'
inputFile = inputPath + 'Zorro.mp4'

video = Video(inputFile)
# video.removeGraininess()
# video.morph()
# video.stabilise()
# video.removeArtifacts()
# video.denoise()
# video.opticalFlow()
# video.display()
video.phaseCorrelate()
video.sharpen()
video.display()
# video.write('output')
# print 'Denoising'
# video.denoise()
# print 'Normalising'
# video.normalise()
# video.display(compare = True)

# video.sharpen()
# frame = video.frames[50]
# eqHist(frame)
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# newFrame = cv2.equalizeHist(frame)
# newFrame = frame.copy()
# newFrame = cv2.normalize(frame, newFrame, 0.0, 255.0, cv2.NORM_MINMAX)
# values = newFrame.ravel()
# hist = plt.hist(x = values, bins = 256, range = [0, 256])
# plt.show(hist)

# video.sharpen()
# video.gammaCorrection()
# video.display(compare = True)
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# gamma = 1.0
# normalisedFrame = frame / 255.0
# enhancedFrame = normalisedFrame ** (1.0 / gamma)
# newFrame = enhancedFrame * 255.0

# shape = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# frame = cv2.erode(frame, shape)
# frame = cv2.dilate(frame, shape)
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
