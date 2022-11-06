import cv2
import numpy as np
import statistics

# read images
img1 = cv2.imread('v1.jpeg')
img2 = cv2.imread('v2.jpeg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sutun_uzunlugu=64
#sift
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)
#print(matches[0].distance)
#print(matches[0].imgIdx)
#print(matches[0].queryIdx)
#print(matches[0].trainIdx)

x=keypoints_1[matches[1].queryIdx].pt[0]
y=keypoints_1[matches[1].queryIdx].pt[1]

x2=keypoints_2[matches[1].trainIdx].pt[0]
y2=keypoints_2[matches[1].trainIdx].pt[1]

center_coordinates = (int(x), int(y))
radius = 30
color = (0, 255, 0)
iyi_feature=[]
grup=[[] for x in range(32)]
grup2=[[] for x in range(32)]
center_coordinates2 = (int(x2), int(y2))

tum_egimler=[len(matches)]

for i in range(0,len(matches)): #noktalarin egiminin ortalamasini bulma
    x = keypoints_1[matches[i].queryIdx].pt[0]
    y = keypoints_1[matches[i].queryIdx].pt[1]

    x2 = keypoints_2[matches[i].trainIdx].pt[0]
    y2 = keypoints_2[matches[i].trainIdx].pt[1]

    center_coordinates = (int(x), int(y))
    center_coordinates2 = (int(x2), int(y2))

    egim = (y2 - y) / ((img1.shape[1] + x2) - x)
    tum_egimler.append(egim)
egim_ortalama=statistics.mean(tum_egimler)
print(egim_ortalama)

for i in range(0,len(matches)):
    x = keypoints_1[matches[i].queryIdx].pt[0]
    y = keypoints_1[matches[i].queryIdx].pt[1]

    grup_no_1=x//sutun_uzunlugu

    x2 = keypoints_2[matches[i].trainIdx].pt[0]
    y2 = keypoints_2[matches[i].trainIdx].pt[1]

    grup_no_2 = x2 // sutun_uzunlugu

    center_coordinates = (int(x), int(y))
    center_coordinates2 = (int(x2), int(y2))
    #img1 = cv2.circle(img1, center_coordinates, radius, color, 2)
    #img2 = cv2.circle(img2, center_coordinates2, radius, color, 2)

    egim=(y2-y)/((img1.shape[1]+x2)-x)
    #print(egim)
    #print(matches[i].distance)
    if(egim<0.005 and egim>-0.005):
        grup[int(grup_no_1)].append(matches[i])
        grup2[int(grup_no_2)].append(matches[i])
        #print(grup_no_1)
        #print(grup_no_2)
        #print("xxxxxxxx")
        #print(matches[i].distance)

for i in range(0,32): #medyan ile iyi feature'ler bulma
    train_grid = []
    for j in range(0,len(grup[i])):
        #print(grup[i][j])
        train_grid.append(keypoints_2[grup[i][j].trainIdx].pt[0]//64)

    print("xxxxxxxx")
    print(train_grid)
    if not len(train_grid):
        continue
    medyan=int(statistics.median(train_grid))
    print("kolon medyani:"+str(statistics.median(train_grid)))
    for j in range(0,len(grup[i])):
        medyan_farki=int(keypoints_2[grup[i][j].trainIdx].pt[0] // 64) - medyan
        if(medyan_farki<=1 and medyan_farki>=-1):
            iyi_feature.append(grup[i][j])

print("eslestirilen feature sayisi: "+str(len(iyi_feature)))
if(len(iyi_feature)<15):
    print("iki resim birbirinden farkli")
else:
    print("resimlerde ortak feature'lar var")
#img1 = cv2.circle(img1, center_coordinates, radius, color, 2)
#img2 = cv2.circle(img2, center_coordinates2, radius, color, 2)
img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, iyi_feature, img2, flags=2)
#img3 = cv2.circle(img3, center_coordinates, radius, color, 2)
cv2.imwrite("deneme4.png", img3)
cv2.imshow("feature match",img3)
cv2.waitKey(0)