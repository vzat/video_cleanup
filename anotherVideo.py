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
        # self.fourcc = int(inVid.get(cv2.CAP_PROP_FOURCC))
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = inVid.get(cv2.CAP_PROP_FPS)
        self.width = int(inVid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(inVid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        startFrame = 100
        endFrame = startFrame + 10
        frameNo = 0

        # Read Frames
        self.rawFrames = []
        validFrame, frame = inVid.read()
        while validFrame:
            if frameNo > startFrame and frameNo < endFrame:
                self.rawFrames.append(frame)
            validFrame, frame = inVid.read()
            frameNo += 1
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

        extension = '.avi'
        return extension

    def write(self, name):
        filename = self.path + name + self.extension
        outVid = cv2.VideoWriter(filename, self.fourcc, self.fps, (self.width, self.height))

        for frame in self.frames:
            outVid.write(frame)

        outVid.release()

    def stretchHist(self):
        # Constrast Stretching
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

            # Reference: https://docs.opencv.org/3.2.0/da/d6e/tutorial_py_geometric_transformations.html
            translationMatrix = np.float32([[1, 0, xDif], [0, 1, yDif]])
            newFrame = cv2.warpAffine(secondFrame, translationMatrix, (self.width, self.height))

            self.frames[frameNo + 1] = newFrame

            if frameNo % 100 == 0:
                print int(float(frameNo) / len(self.frames) * 100.0), '%'

    def enhanceDetail(self):
        for frameNo, frame in enumerate(self.frames):
            bFrame = cv2.medianBlur(frame, 9)

            gFrame = cv2.cvtColor(bFrame, cv2.COLOR_BGR2GRAY)
            threshold, _ = cv2.threshold(src = gFrame, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cannyFrame = cv2.Canny(image = gFrame, threshold1 = 0.5 * threshold, threshold2 = threshold)

            # Make the details more pronounced
            shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            cannyFrame = cv2.dilate(cannyFrame, shape)
            rCannyFrame = cv2.bitwise_not(cannyFrame)

            # Blur background to denoise it
            bg = cv2.bitwise_and(bFrame, bFrame, mask = rCannyFrame)

            # Reference: https://bohr.wlu.ca/hfan/cp467/12/notes/cp467_12_lecture6_sharpening.pdf
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype = float)
            sharpenedFrame = cv2.filter2D(frame, ddepth = -1, kernel = kernel)

            # Detail
            roi = cv2.bitwise_and(sharpenedFrame, sharpenedFrame, mask = cannyFrame)

            self.frames[frameNo] = cv2.bitwise_or(bg, roi)

    def findWatermark(self):
        diffs = []
        for frameNo in range(len(self.frames) - 1):
            frame1 = self.frames[frameNo]
            frame2 = self.frames[frameNo + 1]

            diffs.append(cv2.absdiff(frame1, frame2))
            cv2.imshow('mask', diffs[frameNo])
            cv2.waitKey(0)

        watermarkMask = diffs[0]
        for diff in diffs:
            watermarkMask = cv2.bitwise_and(watermarkMask, diff)

        cv2.imshow('Watermark', watermarkMask)
        cv2.waitKey(0)

    def fourier(self):
        for frameNo, frame in enumerate(self.frames):
            gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # TODO NO https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
            ctp = cv2.cartToPolar(gFrame, (self.width / 2, self.height / 2), 1)

            cv2.imshow('Fourier', ctp)
            cv2.waitKey(0)

    def denoise(self):
        temporalWindowSize = 3
        totalFrames = len(self.frames)
        for frameNo in range(temporalWindowSize / 2):
            cv2.fastNlMeansDenoisingColored(self.frames[frameNo])

        for frameNo in range(totalFrames - temporalWindowSize / 2, totalFrames):
            cv2.fastNlMeansDenoisingColored(self.frames[frameNo])

        for frameNo in range(temporalWindowSize / 2, totalFrames - temporalWindowSize / 2):
            newFrame = cv2.fastNlMeansDenoisingColoredMulti(srcImgs = self.frames, imgToDenoiseIndex = frameNo, temporalWindowSize = temporalWindowSize)
            self.frames[frameNo] = newFrame
            print frameNo

inputPath = 'videos/'
inputFile = inputPath + 'Zorro.mp4'

video = Video(inputFile)

# video.stretchHist()
# video.stabilise()
# video.denoise()
# video.enhanceDetail()

# video.display(compare = True)
video.write('shortZorro')
cv2.waitKey(0)
