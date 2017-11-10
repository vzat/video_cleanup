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
#
#   Structure:
#   1. Reading the video
#       1.1 Extract video metadata
#           * The codec (fourcc), fps, width and height [1] of the video are
#             needed to be able to write the video in its original form
#           * Because most codecs and extensions are causing problems on Windows,
#             the fourcc has been set to 'XVID' and the extension used is '.avi'
#       1.2 Append the frames to a list
#   2. Stretch Contrast
#       2.1 Extract luminance from frame
#           * For colour images the luminance can be used to scale the intensity
#       2.2 Find the minimum and maximum intensity of the image
#           * To be able to stretch the histogram, the range of the insensities
#             need to be known
#       2.3 Scale the range from the initial intensities to 0 - 255
#           * By applying the formula 255 * (In - Imin) / (Imax - Imin) the
#             initial intensities of the frame are scaled to the whole range
#             from 0 - 255
#           * This function does not improve the frame very much if the minimum
#             and maximum intensities of the frame are already close to 0 - 255
#   3. Stabilise video
#       3.1 Iterate through the frames in pairs
#           * Stabilisation is done by finding the global movement between each
#             pair of frames
#           * The frames are updated at each step to provide a smooth stabilisation
#       3.2 Blur frames
#           * The images are blurred to remove most of the noise from the image
#           * Median Blurring is used as it provides a good trade-off between
#             speed and noise removal
#       3.3 Get edges
#           * The edges of the frame are extracted using Canny
#           * The thresholds used for Canny are calculated based on Otsu's
#             theshold that is beforehand
#           * Values between 0.5 * Otsu's theshold and the value of the threshold
#             provides good results for most images [2]
#       3.4 Find the phase correlation between the two frames
#           * Phase correlation is a way of determening an offset between two
#             similar images
#           * It provides faster and more precise results than
#             using a feature detection algorithm
#           * It returns a subpixel offset for the x and y coordinates
#           * The OpenCV function requires the images to be in a specific format
#             (CV_32FC1 or CV_64FC1) [3]
#       3.5 Translate the second frame to match the first one
#           * A transformation matrix is created using the values from the
#             previous step [4]
#           * By using the translation matrix, the second frame is shifted
#             to match the first image
#           * The frame is updated so the next pair uses the stabilised frame
#   4. Enhance Details and Denoise
#       4.1 Blur frame
#           * This is used for both edge detection and denoising
#           * The frame is blurred using medianBlur which reduces noise considerably
#             while preserving edges
#       4.2 Find edges
#           * Extract the edges using Canny, using the same method as in 3.3
#           * As these edges are extracted from a denoised image, it should
#             contain only the important details of the image
#       4.3 Emphasise details
#           * Closing the mask created in the previous step using a round shape
#           * This makes the details more pronounced
#       4.4 Extract background from the blurred frame
#           * Using a reverse mask from the previous step a denoised background
#             is extracted
#       4.5 Sharpen the original frame
#           * The original frame is sharpened by convolving the frame with a
#             hard sharpening kernel
#           * The kernel is created by combining a Laplacian mask with the
#             the original image [5]
#           * This enhances the details but increases the noise
#       4.6 Blur the sharpen image
#           * The sharpened image is blurred using a Gaussian Blur
#           * The Gaussian Blur is useful for reducing the Gaussian noise
#             created by sharpening the image
#           * This will keep the details enhanced while reducing the extra noise
#       4.7 Extract the edges from the sharpen image
#           * Only the edges of the sharpened frame are extracted as the rest
#             of the frame would be deteriorated
#       4.8 Combine the modified background and edges together
#           * Combine the denoised background with the sharpened edges to create
#             a denoised frame while emphasising the important details
#   5. Write the video
#
#   Extra Notes:
#       * While some of the functions described above could be combined
#         which would improve performance, they have been separated to improve
#         the readability of the code
#       * The enhanceDetail() functions has not been split as blurring is an
#         expensive operation which would increase the processing time considerably
#
#   Experiments:
#       * Use an adaptive normalisation to improve the quality of the frames.
#         This was done using CLAHE. While it make the persons and objects in
#         the video more clear, it deteriorates the frames too much to be used.
#       * Use Optical Flow to find moving objects. This would help in
#         separating the background and foreground. Because there are several
#         scenes in the video and because there are not enough frames it didn't
#         provide better results than single frame processing
#       * Use a Fast Non-Local Means Denoising. There are several functions
#         for this included in OpenCV. While it does slightly improve the image
#         quality is takes too much time to process a frame to be usable.
#
#   References:
#   [1] OpenCV 3.2.0 Documentation, 2016, [Online].
#       Available: https://docs.opencv.org/3.2.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d.
#       [Accessed: 2017-11-09]
#   [2] M.Fang, GX.Yue1, QC.Yu, 'The Study on An Application of Otsu Method in Canny Operator',
#       International Symposium on Information Processing, Huangshan, P. R. China,
#       August 21-23, 2009, pp. 109-112
#   [3] OpenCV 3.2.0 Documentation, 2016, [Online].
#       Available: https://docs.opencv.org/3.2.0/d7/df3/group__imgproc__motion.html#ga552420a2ace9ef3fb053cd630fdb4952.
#       [Accessed: 2017-11-09]
#   [4] OpenCV 3.2.0 Tutorials, 2016, [Online].
#       Available: https://docs.opencv.org/3.2.0/da/d6e/tutorial_py_geometric_transformations.html.
#       [Accessed: 2017-11-09]
#   [5] Sharpening Filters, [Online].
#       Available: https://bohr.wlu.ca/hfan/cp467/12/notes/cp467_12_lecture6_sharpening.pdf.
#       [Accessed: 2017-11-09]

import numpy as np
import cv2
import easygui

class Video:
    def __init__(self, file):
        print 'Reading video'
        (self.path, self.originalName, self.extension) = self.getFileDetails(file)

        inVid = cv2.VideoCapture(file)

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

            # Sharpen details
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype = float)
            sharpenedFrame = cv2.filter2D(frame, ddepth = -1, kernel = kernel)
            sharpenedFrame = cv2.GaussianBlur(sharpenedFrame, (5, 5), 0)

            # Extract only the details from the frame
            roi = cv2.bitwise_and(sharpenedFrame, sharpenedFrame, mask = cannyFrame)

            self.frames[frameNo] = cv2.bitwise_or(bg, roi)

    def stabilise(self):
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

            translationMatrix = np.float32([[1, 0, xDif], [0, 1, yDif]])
            newFrame = cv2.warpAffine(secondFrame, translationMatrix, (self.width, self.height))

            self.frames[frameNo + 1] = newFrame

# Open file
inputFile = easygui.fileopenbox()

# Read file
video = Video(inputFile)

# Process Video
video.stretchHist()
video.stabilise()
video.enhanceDetail()

# Write File
video.write('Restored_' + video.originalName)
