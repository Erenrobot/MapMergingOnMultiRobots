import cv2
import numpy as np
# read images
img1 = cv2.imread('v2.jpeg')
img2 = cv2.imread('v1.jpeg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#sift
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)
print(matches[0].distance)
print(matches[0].imgIdx)
print(matches[0].queryIdx)
print(matches[0].trainIdx)

x=keypoints_1[matches[1].queryIdx].pt[0]
y=keypoints_1[matches[1].queryIdx].pt[1]

x2=keypoints_2[matches[1].trainIdx].pt[0]
y2=keypoints_2[matches[1].trainIdx].pt[1]

center_coordinates = (int(x), int(y))
radius = 30
color = (0, 255, 0)
iyi_feature=[]
center_coordinates2 = (int(x2), int(y2))
for i in range(0,len(matches)):
    x = keypoints_1[matches[i].queryIdx].pt[0]
    y = keypoints_1[matches[i].queryIdx].pt[1]

    x2 = keypoints_2[matches[i].trainIdx].pt[0]
    y2 = keypoints_2[matches[i].trainIdx].pt[1]
    center_coordinates = (int(x), int(y))
    center_coordinates2 = (int(x2), int(y2))
    #img1 = cv2.circle(img1, center_coordinates, radius, color, 2)
    #img2 = cv2.circle(img2, center_coordinates2, radius, color, 2)
    print(img1.shape[1])
    print(img1.shape[0])
    egim=(y2-y)/((img1.shape[1]+x2)-x)
    #print(matches[i].distance)
    if(egim<0.005 and egim>-0.005):
        iyi_feature.append(matches[i])
        print(matches[i].distance)

"""print(x)
print(y)
print(keypoints_1[matches[0].queryIdx].pt[0])
print(keypoints_1[matches[0].queryIdx].pt[1])
print(keypoints_2[matches[0].queryIdx].pt[0])
print(keypoints_2[matches[0].queryIdx].pt[1])
print("xxxxxxxxxxxxxxx")
print(matches[1].distance)
print(matches[1].imgIdx)
print(matches[1].queryIdx)
print(matches[1].trainIdx)
print(keypoints_1[matches[1].queryIdx].pt[0])
print(keypoints_1[matches[1].queryIdx].pt[1])
print(keypoints_2[matches[1].queryIdx].pt[0])
print(keypoints_2[matches[1].queryIdx].pt[1])
print("xxxxxxxxxxxxxxx")
print(matches[2].distance)
print(matches[2].imgIdx)
print(matches[2].queryIdx)
print(matches[2].trainIdx)
print("xxxxxxxxxxxxxxx")
print(matches[3].distance)
print(matches[3].imgIdx)
print(matches[3].queryIdx)
print(matches[3].trainIdx)
print("xxxxxxxxxxxxxxx")
print(matches[3].distance)
print(matches[3].imgIdx)
print(matches[3].queryIdx)
print(matches[3].trainIdx)"""
img1 = cv2.circle(img1, center_coordinates, radius, color, 2)
img2 = cv2.circle(img2, center_coordinates2, radius, color, 2)
img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, iyi_feature, img2, flags=2)
#img3 = cv2.circle(img3, center_coordinates, radius, color, 2)
cv2.imwrite("deneme.png", img3)
cv2.imshow("feature match",img3)
cv2.waitKey(0)