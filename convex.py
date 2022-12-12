# https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html

import cv2 as cv
import numpy as np

drawing = None
contours = None

# 1 
k = cv.isContourConvex(contours)

rect = cv.minAreaRect(contours)
box = cv.boxPoints(rect)
box = np.int0(box)

# 2
hull = cv.convexHull(contours[0])

# 3
defects = cv.convexityDefects(contours,hull)

def anglef(start, end, far):
    import math
    #finding the angle of the defect using cosine law
    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

    cv.line(drawing,start,end,[0,255,0],2)
    return angle
count_defects = 0 #defaults initially set to 0
#finding defects and displaying them on the image
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0] #defect returns 4 arguments
    #using start, end, far to find the defects location
    start = tuple(contours[s][0])
    end = tuple(contours[e][0])
    far = tuple(contours[f][0])

    angle = anglef(start, end, far)
    #we know, angle between 2 fingers is within 90 degrees.
    #so anything greater than that isn;t considered
    if angle <= 90:
        count_defects += 1
        cv.circle(drawing,far,5,[0,0,255],-1) #displaying defect
