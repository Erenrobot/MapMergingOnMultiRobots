# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:16:28 2020

@author: Eren
"""

import numpy as np
import cv2 as cv
import time
img = cv.imread('a.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
start = time.time()
kp = sift.detect(gray,None)
print(kp)

pts = cv.KeyPoint_convert(kp)
print(pts[0][0])
img=cv.drawKeypoints(gray,kp,img)
end = time.time()
print(end - start)
cv.imwrite('sift_keypoints3.jpg',img)