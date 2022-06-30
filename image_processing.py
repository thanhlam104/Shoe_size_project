import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def preprocess(img):
    #enhence contrast
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, structuringElement)
    imgGrayscalePlusTopHat = cv2.add(img, imgTopHat)
    img = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    #denoise
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.GaussianBlur(img, (13, 13), 0)
    img = cv2.dilate(img, kernel, iterations=5)
    img = cv2.erode(img, kernel, iterations=5)

    return img


def img_segmentation(img):
    # For clustering the image using k-means, we first need to convert it into a 2-dimensional array
    image_2D = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

    # tweak the cluster size and see what happens to the Output
    cluster = GaussianMixture(n_components=2, random_state=0).fit(image_2D)
    clustOut = cluster.means_[cluster.predict(image_2D)]

    # Reshape back the image from 2D to 3D image
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])
    clusteredImg = np.uint8(clustered_3D * 255)

    return clusteredImg

def edge_detection(img):
    img_edge = cv2.Canny(img, 5, 15)
    img_edge = cv2.dilate(img_edge, None, iterations=3)
    img_edge = cv2.erode(img_edge, None, iterations=2)

    img_edge_bgr = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    img_line = img_edge_bgr.copy()

    lines = cv2.HoughLines(img_edge, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img_edge_bgr, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(img_edge, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img_line, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    img_line = cv2.cvtColor(img_line, cv2.COLOR_BGR2GRAY)
    return img_line


def box_detection(img_edge):
    contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    contours_poly = [None] * len(contours)
    # bbox = [None] * len(contours)
    rect = [None] * len(contours)
    bbox_min = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        # bbox[i] = cv2.boundingRect(contours_poly[i])
        rect[i] = cv2.minAreaRect(c)
        point = cv2.boxPoints(rect[i])
        point_ = np.array([point[1], point[0], point[3], point[2]])
        bbox_min[i] = np.int0(point_)

    return bbox_min

def crop_image(bbox, img):
    h, w = img.shape[:2]
    pts1 = np.float32(bbox)
    pts2 = np.float32([[0, 0],
                       [0, h],
                       [w, h],
                       [w, 0]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(img, matrix, (w, h))

    h, w = result.shape[:2]
    result = result[int(h * 0.05): int(h * 0.95), int(w * 0.05): int(w * 0.95)]
    return result


def measure_foot_size(bbox_crop, img_ori):
    foot_1 = max(bbox_crop[..., 0]) - min(bbox_crop[..., 0])
    foot_2 = max(bbox_crop[..., 1]) - min(bbox_crop[..., 1])

    h, w = img_ori.shape[:2]

    long = max(foot_1, foot_2) / w * 297 * 0.1
    width = min(foot_1, foot_2) / h * 210 * 0.1
    return long, width

def e2e(img):
    img_process = preprocess(img)
    img_seg = img_segmentation(img_process)
    img_edge = edge_detection(img_seg)
    bbox = box_detection(img_edge)[0]
    img_crop = crop_image(bbox, img_process)
    img_crop_seg = img_segmentation(img_crop)
    bbox_crop = box_detection(edge_detection(img_crop_seg))[0]
    long, width = measure_foot_size(bbox_crop, img_process)

    size_map = {24.4: 40,
                24.8: 40.5,
                25.2: 41,
                25.7: 41.5,
                26: 42,
                26.5: 42.5,
                26.8: 43,
                27.3: 43.5,
                27.8: 44,
                28.3: 44.5,
                28.6: 45,
                29.4: 46}

    if long < 24.4 or long > 29.4:
        size = 'Nope'

    else:
        for i, long_size in enumerate(size_map.keys()):
            if long < long_size:
                size = size_map[long_size]
                break

    return long, width, size


