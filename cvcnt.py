#  https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593#48553593

# minEnclosingCircle
 
 # APIs
 
import numpy as np
import cv2 as cv

image = None
imgray = None

import imutils
# imutils: A series of convenience functions to make basic image processing functions
# such as translation, rotation, resizing, skeletonization, displaying Matplotlib images
# sorting contours, detecting edges, and much more easier with OpenCV.

cnts = None
cnts = imutils.grab_contours(cnts)

c = max(cnts, key = cv.contourArea)

# contours, hierarchy = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# ### First Parameter: 

# # "Mode" -> hierachy


# # "cv2.RETR_TREE" retrieves all the contours,creates a full family hierarchy list.

# # "cv2.RETR_LIST" retrieves all the contours, but doesn't create any parent-child relationship.

# # "cv2.RETR_EXTERNAL" Only external contour,it doesn't care other internal contours.

# # "cv2.RETR_CCOMP" retrieves relative relationship.only relativly internal/external secondary relationship. Not so usual.



# ### Second Parameter: 

# # "method" -> ContourApproximationModes -> the contour approximation algorithm -> the Teh-Chin chain approximation algorithm


# # "cv.CHAIN_APPROX_SIMPLE" compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.


# # "cv2.CHAIN_APPROX_NONE" STORES ABSOLUTLY ALL THE CONTOUR POINTS. 


# # "cv.CHAIN_APPROX_TC89_L1" & "cv.CHAIN_APPROX_TC89_KCOS"


import random as rng

# 1
def findCnts(src):# "cv2.contourArea" not accurate for canny contour But treat BINARY image well.

    contours, hierarchy = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy

# select contours by thresholds
# 2.1
def select(contours):
    length = []
    area = []
    for contour in contours:
        length.append(cv.arcLength(contour))
        area.append(cv.contourArea(contour))
    return length, area
    # seq = [0, 1, 2, 3, 5, 8, 13]
    # result = filter(lambda x: x % 2 != 0, seq)
# 2.2
def findCnts(src, threshold = None):
    contours, _ = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    selected_cnt = []
    
    if threshold is not None:
        for cnt in contours:
            area = int(cv.contourArea(cnt))
            if area > threshold[0] and area < threshold[1]:
                selected_cnt.append(cnt)
        return selected_cnt
    else:
        return contours

# 2.3 The small epsilon is , the more accuracy the approxPoly is
def contour_poly(contour, epsilon = 0.1):
    # epsilon is maximum distance from contour to approximated contour.
    epsilon = cv.arcLength(contour, True) * epsilon
    poly = cv.approxPolyDP(contour, 3, True)
    return poly

# 2.4
hull = cv.convexHull(cnts)
hullarea = cv.contourArea(hull)

# 3 M = cv.moments(cnt)

# # 3
# def minRec(contour_poly):
#     center_xy, widthheight, angle_cv = cv.minAreaRect(contour_poly)
#     return center_xy, widthheight, angle_cv
# def minRec_area(widthheight):
#     return widthheight[0] * widthheight[1]
# def box(minRec):
#     box = np.int0(cv.boxPoints(minRec))
#     return box

# 3
def bounding_rect(contour):
    x,y,w,h = cv.boundingRect(contour)
    aspect_ratio = float(w)/h

# 3.1
def bounding_rects(contour):
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    return rect, box


# 3.2 extreme point
leftmost = tuple(cnts[cnts[:,:,0].argmin()][0])
rightmost = tuple(cnts[cnts[:,:,0].argmax()][0])
topmost = tuple(cnts[cnts[:,:,1].argmin()][0])
bottommost = tuple(cnts[cnts[:,:,1].argmax()][0])

# Orientation
(x,y),(MA,ma),angle = cv.fitEllipse(cnts)

# 4 minEnclosingCircle
def bounding_circle(contour):
    center, radius = cv.minEnclosingCircle(contour)
    return center, radius
# 4 maxEnclosingCircle
gray = None
dist = cv.distanceTransform(gray, cv.DIST_L2, 3)
dist = cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(dist)

# 5 Mask
mask = np.zeros(imgray.shape,np.uint8)
cv.drawContours(mask,[cnts],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv.findNonZero(mask)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(imgray,mask = mask)

mean_val = cv.mean(image,mask = mask)

# 5-1
# minRec & bounding box
def bound_clip(image, contour):
    x,y,w,h = cv.boundingRect(contour)
    # x, y, w, h = int(x), int(y), int(w), int(h)
    dim = image.shape
    deepth = len(dim)
    if deepth == 3:
        roi = image[y: y+h, x: x+w,:]
        return roi
    elif deepth == 2:
        roi = image[y: y+h, x: x+w]
        return roi

# 5-2
def clip_minRec(image):
    pass
    # google it ! 
    # 1.rotate 2.clip

img = None
img = cv.circle(img, max_loc, int(abs(max_val)), (255, 255, 255), 10)
# center, radius = cv.minEnclosingCircle(contour)


# room: how many dots in a cell :::::
#                               :::::
#                               :::::
def sample_inside_contour(grids, room, height_width, contour):
    result = np.zeros((height_width), dtype = np.uint8)

    for index_row_grid, row in enumerate(grids):
        if index_row_grid % room:

            for index_col_grid, col in enumerate(row):
                if index_col_grid % room:

                    sample_point = grids[index_row_grid, index_col_grid]
                    # if the return value of pointPolygonTest greater than 0
                    # that means the point is inside the polygon contour 
                    if cv.pointPolygonTest(contour, sample_point, True) > 0:
                        result[index_row_grid // room][index_col_grid // room] += 1
    return result


# 1 draw_text
def draw_text(src, text, org, fontScale=0.4, color=(255, 0, 255), thickness=1):
    return cv.putText(src, text, [int(xy) for xy in org], cv.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv.LINE_AA)

# [x,y]
def rect_corner(box): #>    left_top, right_top, right_bottom, left_bottom
    assert len(box), "box is empty"
    left_top, right_top, right_bottom, left_bottom = None, None, None, None
    x = np.array(sorted(box, key = lambda k:k[0])).tolist()
    y = np.array(sorted(box, key = lambda k:k[1])).tolist()
    for i in x[:2]:
        if i in y[:2]:
            left_top = i
            break # otherwise could be right_top
    for i in x[:2]:
        if i in y[2:]:
            left_bottom = i
    for i in x[2:]:
        if i in y[:2]:
            right_top = i
    for i in x[2:]:
        if i in y[2:]:
            right_bottom = i

    if left_top is None and right_bottom is None:
        assert x[0] in y[2:] and x[1] in y[2:], "Error! Unexpected!"
        left_bottom, left_top, right_top, right_bottom = box
    if left_bottom is None and right_top is None:
        assert x[0] in y[:2] and x[1] in y[:2], "Error! Unexpected!"
        left_top, right_top, right_bottom, left_bottom = box      
    return left_top, right_top, right_bottom, left_bottom


def draw_text_at_contour_moment_center(src, contour, text, **kwargs):
    M = cv.moments(contour)
    if M['m00'] != 0:
        x_center = int(M['m10']/M['m00'])
        y_center = int(M['m01']/M['m00'])

    draw_text(src, text, (x_center, y_center), **kwargs)


# build a contour class!
def draw_text_corners(src, points, color = (0, 125, 255)):
    for point in points:
        draw_text(src, str(point), (point[0], point[1]), 0.6, color)


def components(binary_image):
    connectivity = 4 
    num, labels, stats, centroids = cv.connectedComponentsWithStats(binary_image, connectivity, cv.CV_32S)

    return num, labels, stats, centroids


def components_sizeFilter(num, labels, stats, image_shape, threshold = 20):
    sizes = stats[:, -1][1:]

    num -= 1

    result = np.zeros(image_shape)

    for blob in range(num):

        if sizes[blob] >= threshold:
            result[labels == blob + 1] = 255

    return result
