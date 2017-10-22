import numpy as np
import cv2
import easygui

# print cv2.getBuildInformation()
# help(cv2.VideoWriter_fourcc)

def getFileExtension(filename):
    dotPos = filename.rfind('.')
    extension = '.mp4'

    if dotPos != -1 and dotPos < len(filename) - 1:
        extension = filename[dotPos :]

    return extension

def getFilePath(filename):
    slashPos = filename.rfind('/')
    filePath = ''

    # For Windows
    if slashPos == -1:
        slashPos = filename.rfind('\\')

    if slashPos != -1:
        filePath = filename[: slashPos + 1]

    return filePath

# TODO use easygui
inputPath = 'videos/'

inputFile = inputPath + 'Zorro.mp4'
inVid = cv2.VideoCapture(inputFile)



# # Display and Write video
# # Refernce - https://docs.opencv.org/3.2.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
# outputFile = getFilePath(inputFile) + 'output' + getFileExtension(inputFile)
# fourcc = int(inVid.get(cv2.CAP_PROP_FOURCC))
# fps = inVid.get(cv2.CAP_PROP_FPS)
# wVid = int(inVid.get(cv2.CAP_PROP_FRAME_WIDTH))
# hVid = int(inVid.get(cv2.CAP_PROP_FRAME_HEIGHT))
# outVid = cv2.VideoWriter(outputFile, fourcc, fps, (wVid, hVid))
#
# validFrame, frame = inVid.read()
# while validFrame:
#     cv2.imshow('Zorro', frame)
#     # TODO Clean frame - use functions
#     outVid.write(frame)
#
#     key = cv2.waitKey(int(round(fps)))
#     if key == ord('q'):
#         break
#
#     validFrame, frame = inVid.read()
# outVid.release()



# Read 100th frame
for i in range(100):
    retval, frame = inVid.read()
if retval:
    # TODO Do something with the image, use functions
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)

inVid.release()
cv2.destroyAllWindows()
