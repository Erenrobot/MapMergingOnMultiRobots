import numpy as np
import cv2 as cv
from scipy import stats as s

def odometry_x(focal_length,distance,pixel_length,magnitude):
    x=magnitude*pixel_length*distance/focal_length
    return x

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

    #mode=statistics.multimode(distance_x)
    mode_x=int(s.mode(distance_x)[0])
    mode_y= int(s.mode(distance_y)[0])

    value=odometry_x(25, 7500, 0.0625, mode_x)

    print("change in X: ",mode_x)
    text1 = "change in X: "
    text_x = str(mode_x)
    cv.putText(out, text1 + text_x + "," , (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255),2)

    text2 = "change in Y: "
    text_y = str(mode_y)
    cv.putText(out, text2 + text_y + ",", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 2)
    return out
img1 = cv.imread("v1.jpeg")
img2 = cv.imread("v2.jpeg")
img1=cv.resize(img1,(640,480))
img2=cv.resize(img2,(640,480))
# Initiate ORB detector
orb = cv.ORB_create()
print(orb)
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)
bf = cv.BFMatcher()
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda val: val.distance)

img3 = drawMatches(img1, kp1, img2, kp2, matches[:len(matches)-1])

    # Show the image
cv.imshow('Matched Features', img3)
k = cv.waitKey(0) & 0xff



if k == 27:
    cv.destroyWindow('Matched Features')

#points2f=cv.KeyPoint_convert(kp,[])
#print(points2f)


# draw only keypoints location,not size and orientation
#img2 = cv.drawKeypoints(img, kp, None, color=(0,0,255), flags=0)
#cv.imshow("frame",img2)
