import numpy as np
from  matplotlib import pyplot as plt
import cv2
'''
filename = "hanamura.jpg"
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.uint8(dst)

#find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

#define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

#draw corners
res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:, 1], res[:,0]]=[0,0,255]
img[res[:, 3], res[:,2]]=[0,255,0]

cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()

#Shi-tomasi corners
img2 = cv2.imread('hanamura.jpg')
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

corners2 = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners2 = np.int0(corners2)

for i in corners2:
    x, y = i.ravel()
    cv2.circle(img2, (x, y), 3, 255, -1)

cv2.imshow('dst', img2)
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()

#SIFT (SIFT was removed from default version of OpenCV. Need to compile it again with opencv_contrib

filename = "hanamura.jpg"
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray, kp, img)

cv2.imshow('sift key points', img)
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
'''

#Feature matching

sample = cv2.imread('icon.jpg', 0)
test = cv2.imread('desktop.png', 0)

#ORB detector
orb = cv2.ORB_create()

#find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(sample, None)
kp2, des2 = orb.detectAndCompute(test, None)

'''
#BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)

#draw first 10 matches
img3 = cv2.drawMatches(sample, kp1, test, kp2, matches[:5000],None,  flags = 2)

plt.imshow(img3), plt.show()
'''
#Using FLANN based Matcher

#initiate ORB detector
FLANN_INDEX_KDTREE = 0

#1st dictionnary needed by the FLANN algorithm. Various parameters explained in FLANN doc
index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    table_number = 6 ,
                    key_size = 12,
                    multi_probe_level = 1)

#2nd dictonnary : number of time the tree in the index should be recursively traversed
search_params = dict(checks=100)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

#Draw only good matches
matchesMask = [[0, 0] for i in xrange(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0, 255, 0),
                    singlePointColor = (255, 0, 0),
                    matchesMask = matchesMask,
                    flags = 0)

img3 = cv2.drawMatchesKnn(sample, kp1, test, kp2, matches, None, **draw_params)

plt.imshow(img3, ), plt.show()