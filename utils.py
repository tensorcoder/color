import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


def findAverageColor(img):
    img = img[:, :, :-1]
    average = img.mean(axis=0).mean(axis=0)
    # avg = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)
    return average



def findDominantColors(img, number_of_colors=2):

    pixels = np.float32(img.reshape(-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)

    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(
        pixels, number_of_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    return dominant, palette, counts



def plotColors(img, palette, counts):

    # dominant, palette, counts = findDominantColors(img)
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
    rows = np.int_(img.shape[0]*freqs)
    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)

    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.imshow(img)
    ax0.set_title('original')
    ax0.axis('off')

    ax1.imshow(dom_patch)
    ax1.set_title('Dominant colors')
    ax1.axis('off')
    print(type(fig))
    plt.show()


def find_circle(img):
    # gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = cv2.medianBlur(img, 5)
    # gimg = cv2.GaussianBlur(gimg, (5, 5), 0)
    circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, 4, 2, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('detected circles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return circles 

def find_circle2(img):
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = cv2.medianBlur(gimg, 5)
    # gimg = cv2.GaussianBlur(gimg, (5, 5), 0)
    # circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, 1.5, 50, param1=50, param2=30, minRadius=0, maxRadius=0)
    # circles = np.uint16(np.around(circles))
    # for i in circles[0, :]:
    #     cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    edges = auto_canny(img)


    return edges
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def distance(color1, color2):
    """
    returns the distance between the two colors

    """
    d = np.sqrt(np.square((color1[0]-color2[0])) + np.square(
        (color1[1]-color2[1])) + np.square((color1[2]-color2[2])))
    return d
