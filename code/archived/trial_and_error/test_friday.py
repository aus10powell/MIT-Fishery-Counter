#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:53:27 2021

@author: jamisonmeindl
"""

import cv2 as cv
import numpy as np

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
)


def addVideo(f):
    color = (255, 0, 0)
    thickness = 2

    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((1, 3), np.uint8)
    kernel3 = np.ones((3, 1), np.uint8)

    counter = 0
    wait = 0

    backSub = cv.createBackgroundSubtractorMOG2()
    # backSub = cv.createBackgroundSubtractorKNN()
    capture = cv.VideoCapture(f)

    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    size = (frame_width, frame_height)

    result = cv.VideoWriter("sampleMask.avi", cv.VideoWriter_fourcc(*"MJPG"), 20, size)

    color = np.random.randint(0, 255, (100, 3))
    first = True
    ret, old_frame = capture.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)

    frame_count = 0
    change = []

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        ret, thresh = cv.threshold(fgMask, 50, 255, cv.THRESH_BINARY)

        # Erode and dilate image for optical flow
        mask1 = cv.erode(thresh, kernel, iterations=3)
        mask1 = cv.dilate(mask1, kernel2, iterations=2)
        mask1 = cv.dilate(mask1, kernel3, iterations=2)

        # Erode and dilate image for bounding boxes
        mask2 = cv.erode(thresh, kernel, iterations=2)
        mask2 = cv.dilate(mask2, kernel2, iterations=2)
        mask2 = cv.dilate(mask2, kernel3, iterations=2)
        mask2 = cv.erode(mask2, kernel, iterations=1)

        contours, hierarchy = cv.findContours(mask2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        final = cv.bitwise_and(frame, frame, mask=mask1)

        frame_gray = cv.cvtColor(final, cv.COLOR_BGR2GRAY)

        mask = np.zeros_like(old_frame)
        img = frame
        if wait == 0:
            if int(len(change) // 45) < 1:
                change = change[1:]
            else:
                change = change[int(len(change) // 45) :]
        frame_count += 1
        # calculate optical flow
        if not first:
            # Select good points for optical flow

            try:
                p1, st, err = cv.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None, **lk_params
                )

                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
                img = cv.add(frame, mask)
                # cv.imshow('frame',img)
                p0 = good_new.reshape(-1, 1, 2)

                # Check bounding boxes pf contours

                if len(contours) != 0:
                    for c in contours:
                        rect = cv.boundingRect(c)
                        height, width = fgMask.shape[:2]

                        if (
                            rect[2] > 0.025 * height
                            and rect[2] < 0.7 * height
                            and rect[3] > 0.045 * width
                            and rect[3] < 0.7 * width
                        ):
                            x, y, w, h = cv.boundingRect(c)

                            cv.drawContours(img, c, -1, (255, 0, 0), thickness)
                            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                            for i, (new, old) in enumerate(zip(good_new, good_old)):
                                a, b = new.ravel()
                                c, d = old.ravel()
                                if (
                                    x - 10 < a
                                    and a < x + w + 10
                                    and y - 5 < b
                                    and b < y + h + 5
                                    and x - 10 < c
                                    and c < x + w + 10
                                    and y - 5 < d
                                    and d < y + h + 5
                                ):
                                    change.append(a - c)

                            # If box is in middle of frame, analyze previous movement of good optical flow points
                            if abs(x + (w / 2) - (width / 2)) < 35:
                                if wait == 0:
                                    print(sum(change))
                                    if sum(change) <= 0:
                                        counter += 1
                                        change = []

                                wait = 18

            except:
                p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

        else:
            first = False
            p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

        if wait > 0:
            wait -= 1

        # Show results

        cv.rectangle(img, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(
            img,
            str(capture.get(cv.CAP_PROP_POS_FRAMES)),
            (15, 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )
        cv.rectangle(img, (495, 2), (605, 20), (255, 255, 255), -1)
        cv.putText(
            img,
            "Counter: " + str(counter),
            (500, 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )
        cv.imshow("frame", img)

        cv.imshow("frame2", mask2)

        result.write(mask2)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()

        # keyboard = cv.waitKey(3)
        """
        if keyboard == 'q' or keyboard == 27:
            break
        """

    # result.release()

    capture.release()
    cv.destroyAllWindows()
    return counter


if __name__ == "__main__":
    addVideo(f="/Users/apowell/Downloads/2_2018-04-27_15-50-53.mp4")
