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

    def testLab(self):
        for frameNo, frame in enumerate(self.frames):
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            l = lab[:, :, 0]
            (minI, maxI, _, _) = cv2.minMaxLoc(l)
            lab[:, :, 0] = 255 * ((l - minI) / (maxI - minI))
            self.frames[frameNo] = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

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


inputPath = 'videos/'
inputFile = inputPath + 'Zorro.mp4'

video = Video(inputFile)

# video.stretchHist();
video.sharpen()
video.normalise()
video.stabilise()
video.display(compare = True)
cv2.waitKey(0)
