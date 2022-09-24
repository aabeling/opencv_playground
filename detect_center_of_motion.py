# from https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# from https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity as compare_ssim
import imutils

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

previousFrame = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    #cv.imshow('frame', frame)
    copy = frame.copy()

    if previousFrame is not None:
        # perform comparison

        # convert the images to grayscale
        grayA = cv.cvtColor(previousFrame, cv.COLOR_BGR2GRAY)
        grayB = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        cv.imshow('difference', diff)
        
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv.threshold(diff, 0, 255,
            cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        cv.imshow('thresh', thresh)
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours to draw rectangles into the image
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
        # calculate the center of the motion
        # each rectangle is weighed by its size
        # x and y of the center can be calculated independently
        centerX = 0
        centerY = 0
        countX = 0
        countY = 0
        for c in cnts:
            (x, y, w, h) = cv.boundingRect(c)
            # calculate the horizontal center of the rectangle and weigh it with the width
            centerX += (x + w / 2) * w**2
            countX += w**2
            # calculate the vertical center of the rectangle and weigh it with the height
            centerY += (y + h / 2) * h**2
            countY += h**2
        centerX /= countX
        centerY /= countY
        #print("center: ({},{})".format(centerX, centerY))
        cv.circle(copy, (int(centerX), int(centerY)), 40, (0,255,0), 10)

        cv.imshow('copy', copy)

    previousFrame = frame

    # stop application by pressing q
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
