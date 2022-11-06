import numpy as np
import cv2 as cv
#from matplotlib import pyplot as plt
def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1] = np.dstack([img1])
    out[:rows2, cols1:] = np.dstack([img2])
    distance_x=[]
    distance_y= []
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        distance=int(x1-x2)
        #print(x1-x2)
        distance_x.append(distance)


        distance=int(y1-y2)
        distance_y.append(distance)
        cv.circle(out, (int(x1), int(y1)), 4, (255, 0, 0, 1), 1)
        cv.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0, 1), 1)
        cv.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0, 1), 1)




    return out

def compare(img1, img2):


    # Initiate SIFT detector
    sift = cv.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)

    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda val: val.distance)

    img3 = drawMatches(img1, kp1, img2, kp2, matches[:len(matches)-1])

    # Show the image
    cv.imshow('Matched Features', img3)


img1 = cv.imread('Capture.png')
img2 = cv.imread('Capture1.png')
gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray,100,0.1,10)
corners = np.int0(corners)
compare(img1,img2)
for i in corners:
    print(corners)
    x,y = i.ravel()
    cv.circle(img1,(x,y),3,255,-1)
cv.imshow("a",img1)
cv.waitKey(0)
#plt.imshow(img),plt.show()