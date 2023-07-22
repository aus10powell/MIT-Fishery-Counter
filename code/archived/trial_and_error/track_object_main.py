import cv2
import sys

cap = cv2.VideoCapture("/Users/apowell/Downloads/1_2016-04-13_13-57-11.mp4")

if not cap.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = cap.read()
if not ok:
    print("Cannot read video file")
    sys.exit()

print(cap)
# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # Extract Region of interest
    roi = frame[340:720, 500:800]

    # 1. Object Detection
    mask = object_detector.apply(frame)

    ## Uncomment to display
    # cv2.imshow("Output", mask)
    # cv2.waitKey(0)

    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
